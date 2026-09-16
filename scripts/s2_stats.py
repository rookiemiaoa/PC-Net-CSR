# -*- coding: utf-8 -*-
"""s2_stats.py —— 统计显著性检验引擎（逐样本 + 跨 seed）。

方法选择依据（近五年文献）
  1) **配对随机化检验（randomization / sign-flip permutation）** 作为主检验。
     Ihemelandu & Ekstrand, SIGIR 2023《Inference at Scale》实测：在 N 达数千时
     Wilcoxon / Sign 的 Type-I 错误率显著膨胀（远高于名义 α），而
     **randomization / bootstrap / t 检验** 的错误率与名义 α 一致。
     故本脚本以随机化检验为主，Wilcoxon 仅作稳健性参照并明示其在大样本下的缺陷。
  2) **配对 bootstrap 95% CI**（percentile）作为 Δ 的区间估计（与既有 harness 同口径）。
  3) **McNemar 精确检验**（二项精确而非 χ² 近似）用于阈值化指标 @0.5 / @0.7。
  4) **等价检验 TOST + 无差异区间（indifference zone）** 用于「Δ≈0 且不显著」的负结果。
     Sakai, SIGIR 2025《My System Is As Effective As Yours》：不显著≠无差异，
     要主张「两轴皆无效」必须做等价检验并预先声明无差异区间。
  5) **效应量**（Cohen's d_z、Cliff's δ、最小可检测效应 MDE）——大样本下几乎所有检验
     都会显著，SIGIR'23 明确指出此时应以效应量判定实践意义。
  6) **Holm–Bonferroni** 控制同一族内的多重比较。
  7) 跨 seed：**层级 bootstrap**（先重采样 seed、再在 seed 内重采样样本）+ 逐 seed 配对 t 检验，
     使「种子方差」与「评测方差」两者都被计入 Δ 的置信区间。

用法:
  python s2_stats.py within  <npz> [--label X] [--iz 0.005]
  python s2_stats.py across  <npz_seed1> <npz_seed2> ... [--label X] [--iz 0.005]
可选 --json /path/out.json
"""
import json
import math
import os
import sys

import numpy as np

try:
    from scipy import stats as _st
except Exception:
    _st = None

B_BOOT = int(os.environ.get('B_BOOT', 10000))
B_RAND = int(os.environ.get('B_RAND', 10000))
THR = float(os.environ.get('THR', 0.35))
ALPHA = 0.05

argv = [a for a in sys.argv[1:]]
MODE = argv[0]
JSON_OUT = None
IZ = 0.005
LABEL = ''
files = []
i = 1
while i < len(argv):
    a = argv[i]
    if a == '--json':
        i += 1; JSON_OUT = argv[i]
    elif a == '--iz':
        i += 1; IZ = float(argv[i])
    elif a == '--label':
        i += 1; LABEL = argv[i]
    else:
        files.append(a)
    i += 1

OPS = ['CSR', 'ISR(g.5,T2)', 'RBR k=0.15']


# ------------------------------------------------------------------ 基础统计
def boot_means(x, B, rng, chunk=500):
    """分块 bootstrap 的均值分布，避免 (B,n) 大矩阵占内存。"""
    n = len(x); out = []
    done = 0
    while done < B:
        c = min(chunk, B - done)
        idx = rng.integers(0, n, size=(c, n))
        out.append(x[idx].mean(1))
        done += c
    return np.concatenate(out)


def rand_p(x, B, rng, chunk=500):
    """配对随机化（符号翻转）检验，双侧。H0: 差值的期望为 0。"""
    n = len(x); obs = abs(float(x.mean()))
    ge = 0; tot = 0
    done = 0
    while done < B:
        c = min(chunk, B - done)
        sgn = rng.integers(0, 2, size=(c, n)) * 2 - 1
        st = np.abs((sgn * x).mean(1))
        ge += int((st >= obs - 1e-15).sum()); tot += c
        done += c
    return (1.0 + ge) / (1.0 + tot)


def mdd(z):
    return float(np.mean(z))


def mde(x, alpha=ALPHA, power=0.80):
    """配对设计的最小可检测效应（双侧）。"""
    n = len(x); sd = float(np.std(x, ddof=1)) if n > 1 else 0.0
    if n < 2 or sd == 0:
        return float('nan')
    z_a = 1.959963985 if alpha == 0.05 else _norm_ppf(1 - alpha / 2)
    z_b = _norm_ppf(power)
    return (z_a + z_b) * sd / math.sqrt(n)


def _norm_ppf(p):
    # Acklam 逆正态近似（无 scipy 时兜底）
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00]
    pl = 0.02425
    if p < pl:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > 1 - pl:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5; r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def _betacf(a, b, x, itmax=200, eps=3e-16):
    qab = a + b; qap = a + 1.0; qam = a - 1.0
    c = 1.0; d = 1.0 - qab * x / qap
    if abs(d) < 1e-300:
        d = 1e-300
    d = 1.0 / d; h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d; h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d; de = d * c; h *= de
        if abs(de - 1.0) < eps:
            break
    return h


def _betainc(a, b, x):
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lb = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(lb) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lb) * _betacf(b, a, 1.0 - x) / b


def t_p(x):
    """配对 t 检验 p（双侧）。scipy 可用时直接用；否则用正则化不完全 Beta 兜底。"""
    n = len(x)
    if n < 2:
        return float('nan')
    if _st is not None:
        return float(_st.ttest_1samp(x, 0.0).pvalue)
    m = float(x.mean()); s = float(np.std(x, ddof=1))
    if s == 0:
        return 0.0 if m != 0 else 1.0
    t = m / (s / math.sqrt(n)); df = n - 1
    return float(_betainc(df / 2.0, 0.5, df / (df + t * t)))


def wilcoxon_p(x):
    if _st is None:
        return float('nan'), float('nan')
    nz = x[x != 0]
    if len(nz) < 1:
        return float('nan'), float('nan')
    try:
        r = _st.wilcoxon(x, zero_method='wilcox', alternative='two-sided',
                         method='approx' if len(nz) > 25 else 'auto')
        return float(r.pvalue), float(r.statistic)
    except Exception:
        return float('nan'), float('nan')


def mcnemar_exact(a, b):
    """a=base通过&method未通过, b=base未通过&method通过。返回双侧精确 p。"""
    n = a + b
    if n == 0:
        return 1.0
    if _st is not None:
        return float(min(1.0, 2.0 * _st.binomtest(min(a, b), n, 0.5).pvalue))
    # 兜底：精确二项
    k = min(a, b); s = sum(math.comb(n, j) for j in range(0, k + 1))
    return float(min(1.0, 2.0 * s / (2.0 ** n)))


def cliffs_delta(x):
    nz = x[x != 0]
    if len(nz) == 0:
        return 0.0
    return float((np.sum(nz > 0) - np.sum(nz < 0)) / len(nz))


def holm(items):
    """items: [(label, p)] -> [(label, p, p_adj)]"""
    m = len(items)
    order = sorted(range(m), key=lambda j: (items[j][1] if items[j][1] == items[j][1] else 1.0))
    out = [None] * m; run = 0.0
    for rank, j in enumerate(order):
        p = items[j][1]
        p = 1.0 if p != p else p
        v = min(1.0, (m - rank) * p)
        run = max(run, v)
        out[j] = (items[j][0], p, run)
    return out


def pooled(V, m):
    """V:(K,n) 逐样本 IoU，m:(n,) 桶掩码。
    ⚠️ 阈值指标必须先逐 rep 卡阈值再汇总，不可'跨 rep 平均后再卡阈值'——
    后者会因 Jensen 效应系统性压低 R@0.5/R@0.7（实测 0.5307 vs 正确 0.5417）。
    """
    Vm = V[:, m]
    return float(Vm.mean()), float((Vm >= 0.5).mean()), float((Vm >= 0.7).mean())


def dvec(Vb, Vc, m):
    """返回 (逐样本 ΔmIoU, Δ@0.5, Δ@0.7)。三者的样本均值 = 池化后的正确 Δ。"""
    b, c = Vb[:, m], Vc[:, m]
    return ((c - b).mean(0),
            ((c >= 0.5).astype(np.float64) - (b >= 0.5).astype(np.float64)).mean(0),
            ((c >= 0.7).astype(np.float64) - (b >= 0.7).astype(np.float64)).mean(0))


# ------------------------------------------------------------------ within
def load(npz):
    z = np.load(npz, allow_pickle=True)
    d = {k[5:]: z[k] for k in z.files if k.startswith('iou__')}
    d['_gs'] = z['gs']; d['_ge'] = z['ge']
    return d


def within(npz, label):
    z = load(npz)
    gs, ge = z['_gs'], z['_ge']
    gl = ge - gs
    base = z['base']                       # (K_REP, n)
    K = base.shape[0]; n = base.shape[1]
    rng = np.random.default_rng(20260914)
    buckets = {'ALL': np.ones(n, bool), 'SHORT': gl <= THR, 'LONG': gl > THR}
    rep = {'label': label, 'npz': npz, 'N': int(n), 'K_REP': int(K), 'iz': IZ, 'tests': {}}
    raw = []
    for op in OPS:
        if op not in z:
            continue
        v = z[op]
        for bn, m in buckets.items():
            x, x5, x7 = dvec(base, v, m)
            bm, b5, b7 = pooled(base, m); mm, m5, m7 = pooled(v, m)
            bm_mean = float(x.mean())
            bdist = boot_means(x, B_BOOT, rng)
            blo = float(np.percentile(bdist, 2.5)); bhi = float(np.percentile(bdist, 97.5))
            p_rand = rand_p(x, B_RAND, rng)
            p_t = t_p(x); p_w, wstat = wilcoxon_p(x)
            sd = float(np.std(x, ddof=1)); dz = float(bm_mean / sd) if sd > 0 else float('nan')
            cd = cliffs_delta(x)
            mde_v = mde(x)
            # McNemar 精确（把 rep 汇总）
            up5 = dn5 = up7 = dn7 = 0
            for k in range(K):
                ib, ic = base[k][m], v[k][m]
                up5 += int(((ib < 0.5) & (ic >= 0.5)).sum()); dn5 += int(((ib >= 0.5) & (ic < 0.5)).sum())
                up7 += int(((ib < 0.7) & (ic >= 0.7)).sum()); dn7 += int(((ib >= 0.7) & (ic < 0.7)).sum())
            p_mc5 = mcnemar_exact(dn5, up5); p_mc7 = mcnemar_exact(dn7, up7)
            # TOST 等价检验（90% CI 是否落在 ±IZ 内）
            lo90 = float(np.percentile(bdist, 5.0)); hi90 = float(np.percentile(bdist, 95.0))
            equiv = bool(lo90 > -IZ and hi90 < IZ)
            rec = dict(op=op, bucket=bn, n=int(m.sum()),
                       base_mIoU=bm, op_mIoU=mm, d_mIoU=mm - bm,
                       base_r5=b5, op_r5=m5, d_r5=m5 - b5,
                       base_r7=b7, op_r7=m7, d_r7=m7 - b7,
                       ci_lo=blo, ci_hi=bhi, p_rand=p_rand, p_t=p_t, p_wilcoxon=p_w,
                       d_z=dz, cliffs_delta=cd, mde=mde_v, sd_delta=sd,
                       p_mcnemar_05=p_mc5, p_mcnemar_07=p_mc7,
                       up5=up5, dn5=dn5, up7=up7, dn7=dn7,
                       lo90=lo90, hi90=hi90, equiv_tost=equiv)
            rep['tests']['%s|%s' % (op, bn)] = rec
            raw.append(('%s|%s' % (op, bn), p_rand))
    rep['holm'] = [dict(test=t, p=p, p_adj=pa) for t, p, pa in holm(raw)]
    return rep


# ------------------------------------------------------------------ across
def across(npzs, label):
    """跨 seed：逐 seed Δ + 层级 bootstrap + 逐 seed 检验。"""
    Z = [load(p) for p in npzs]
    S = len(Z)
    rng = np.random.default_rng(20260914)
    rep = {'label': label, 'n_seeds': S, 'seeds_files': list(npzs), 'iz': IZ, 'ops': {}}
    n = Z[0]['base'].shape[1]
    gs = Z[0]['_gs']; gl = Z[0]['_ge'] - gs
    buckets = {'ALL': np.ones(n, bool), 'SHORT': gl <= THR, 'LONG': gl > THR}
    for op in OPS:
        if any(op not in z for z in Z):
            continue
        rep['ops'][op] = {}
        for bn, m in buckets.items():
            per_seed = []
            for z in Z:
                bm_, b5_, b7_ = pooled(z['base'], m)
                mm_, m5_, m7_ = pooled(z[op], m)
                x, x5, x7 = dvec(z['base'], z[op], m)
                per_seed.append(dict(base_mIoU=bm_, op_mIoU=mm_, d_mIoU=mm_ - bm_,
                                     base_r5=b5_, op_r5=m5_, base_r7=b7_, op_r7=m7_,
                                     delta=x, delta5=x5, delta7=x7))
            dm = np.array([r['d_mIoU'] for r in per_seed])
            d5 = np.array([r['op_r5'] - r['base_r5'] for r in per_seed])
            d7 = np.array([r['op_r7'] - r['base_r7'] for r in per_seed])
            bm = np.array([r['base_mIoU'] for r in per_seed])
            mm = np.array([r['op_mIoU'] for r in per_seed])
            # 逐 seed 配对 t / Wilcoxon
            def tp(a):
                if _st is None or len(a) < 2:
                    return float('nan')
                return float(_st.ttest_1samp(a, 0.0).pvalue)
            def wp(a):
                try:
                    return float(_st.wilcoxon(a, alternative='two-sided',
                                              method='approx' if len(a) > 25 else 'auto').pvalue) \
                        if _st is not None else float('nan')
                except Exception:
                    return float('nan')
            # 层级 bootstrap：先抽 seed 再抽样本
            Hs = []
            for _ in range(2000):
                si = rng.integers(0, S, size=S)
                acc = []
                for s_ in si:
                    x = per_seed[s_]['delta']; n_ = len(x)
                    idx = rng.integers(0, n_, size=n_)
                    acc.append(x[idx].mean())
                Hs.append(float(np.mean(acc)))
            Hs = np.array(Hs)
            rec = dict(bucket=bn, n_per_seed=int(m.sum()), n_seeds=S,
                       base_mIoU_mean=float(bm.mean()), base_mIoU_sd=float(bm.std(ddof=1)),
                       op_mIoU_mean=float(mm.mean()), op_mIoU_sd=float(mm.std(ddof=1)),
                       d_mIoU_mean=float(dm.mean()), d_mIoU_sd=float(dm.std(ddof=1)),
                       d_mIoU_min=float(dm.min()), d_mIoU_max=float(dm.max()),
                       n_seed_pos=int((dm > 0).sum()), n_seed_neg=int((dm < 0).sum()),
                       p_t_seed=tp(dm), p_wilcoxon_seed=wp(dm),
                       p_t_seed_r5=tp(d5), p_t_seed_r7=tp(d7),
                       hier_lo=float(np.percentile(Hs, 2.5)), hier_hi=float(np.percentile(Hs, 97.5)),
                       hier_p=float(2 * min((Hs <= 0).mean(), (Hs >= 0).mean())),
                       hier_sd=float(Hs.std(ddof=1)),
                       per_seed=per_seed)
            rep['ops'][op][bn] = rec
    return rep


def pr(rep, mode):
    print('=' * 110, flush=True)
    print('[%s] label=%s  N=%s  K_REP=%s  iz=±%.4f' % (
        mode, rep['label'], rep.get('N', 'per-seed'),
        rep.get('K_REP', '-'), rep['iz']), flush=True)
    if mode == 'within':
        print('\n%-16s %-6s %6s %9s %9s %9s | %22s | %10s %10s | %10s' % (
            'op', 'bucket', 'n', 'base', 'op', 'ΔmIoU', '95% CI (ΔmIoU)', 'p_rand', 'p_t', 'MDE'), flush=True)
        for k, r in rep['tests'].items():
            print('%-16s %-6s %6d %9.4f %9.4f %+9.4f | [%+8.4f,%+8.4f] | %10.3e %10.3e | %10.5f' % (
                r['op'], r['bucket'], r['n'], r['base_mIoU'], r['op_mIoU'], r['d_mIoU'],
                r['ci_lo'], r['ci_hi'], r['p_rand'], r['p_t'], r['mde']), flush=True)
        print('\n%-16s %-6s %10s %10s %10s %8s %10s %10s %8s %7s' % (
            'op', 'bucket', 'p_rand', 'p_t', 'p_wilcox', 'd_z', 'cliffsδ', 'p_McN@.5', 'p_McN@.7', 'TOST'), flush=True)
        for k, r in rep['tests'].items():
            print('%-16s %-6s %10.3e %10.3e %10.3e %8.4f %10.4f %10.3e %8.3e %7s' % (
                r['op'], r['bucket'], r['p_rand'], r['p_t'], r['p_wilcoxon'],
                r['d_z'], r['cliffs_delta'], r['p_mcnemar_05'], r['p_mcnemar_07'],
                'EQ' if r['equiv_tost'] else '-'), flush=True)
        print('\n[Holm–Bonferroni]', flush=True)
        for h in rep['holm']:
            print('   %-28s p=%.3e  p_adj=%.3e  %s' % (
                h['test'], h['p'], h['p_adj'], 'SIG' if h['p_adj'] < ALPHA else 'ns'), flush=True)
    else:
        print('seeds=%d' % rep['n_seeds'], flush=True)
        for op, bks in rep['ops'].items():
            print('\n--- %s ---' % op, flush=True)
            for bn, r in bks.items():
                print(' %-6s n/seed=%-5d  base mIoU %.4f±%.4f | op %.4f±%.4f | ΔmIoU %+.4f±%.4f '
                      '[min %+.4f, max %+.4f]  +seed:%d -seed:%d' % (
                          bn, r['n_per_seed'], r['base_mIoU_mean'], r['base_mIoU_sd'],
                          r['op_mIoU_mean'], r['op_mIoU_sd'], r['d_mIoU_mean'], r['d_mIoU_sd'],
                          r['d_mIoU_min'], r['d_mIoU_max'], r['n_seed_pos'], r['n_seed_neg']), flush=True)
                print('        hier 95%%CI [%+.4f,%+.4f]  hier_p=%.3e | p_t(seed)=%.3e  p_W(seed)=%.3e | p_t@.5=%.3e p_t@.7=%.3e' % (
                    r['hier_lo'], r['hier_hi'], r['hier_p'], r['p_t_seed'], r['p_wilcoxon_seed'],
                    r['p_t_seed_r5'], r['p_t_seed_r7']), flush=True)
    print('=' * 110, flush=True)


if __name__ == '__main__':
    if MODE == 'within':
        rep = within(files[0], LABEL or os.path.basename(files[0]))
    elif MODE == 'across':
        rep = across(files, LABEL or 'across')
    else:
        raise SystemExit('mode must be within|across')
    pr(rep, MODE)
    if JSON_OUT:
        def enc(o):
            if isinstance(o, (np.floating,)):
                return float(o)
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            raise TypeError(str(type(o)))
        with open(JSON_OUT, 'w') as f:
            json.dump(rep, f, ensure_ascii=False, indent=1, default=enc)
        print('WROTE %s' % JSON_OUT, flush=True)
