"""diag37：对称生长 (Symmetric Growth) + 框内外相关性对比度门控 (Contrast Gate)

动机（d36 的教训，逐条对应）
  * d36 用 EdgeGrow 的"边界外是否存在平台"做门控，把 Charades 短伤从 -2.98 压到 -0.99
    （EGS_a0.60_m1），但**仍为负**；且校准集上"短不回归"约束下最优解退化到 alpha=0.90
    （近乎不操作，长 @0.7 只剩 +2.04）⇒ 布尔"平台/非平台"判据不够。
  * 根因假设：d36 的阈值是**绝对**的 (alpha * max(z))。当整段视频相关性普遍偏高
    （背景场景与文本语义接近）时，短动作两侧也会被误判成"平台"从而过扩。

两个新机制（零训练、纯推理期、不使用 gt）
  (1) 对比度门控 CTR —— 尺度无关
      rho = mean(z_out) / mean(z_in)，z_out=框外延伸段的相关性，z_in=框内相关性。
      长动作的框切在平台内部 -> rho≈1；短动作的框已盖住峰值 -> rho 明显<1。
      比值天然抵消"全局高基线"，绝对阈值做不到这一点。
  (2) 对称生长 SYM —— 保持中心不动
      Delta = min(dL, dR)，两侧同时生长同样的量。
      理由：diag15 证明 PC-Net 的**中心是可靠的、宽度是塌缩的**；过扩伤害主要来自
      **单侧泄露**。对称生长被较短的一侧限制，且不移动中心。

族（命名）
  U_k0.15        : 参照，标准 RBR 并集
  CTR_<thr>_m<m>_b<beta> : 逐侧独立：平台判定(>=m 帧) + 对比度 rho>=beta -> 该侧用 EG 边界
                            (beta=0.0 即 d35/d36 的 EGS，作为对照)
  SYM_<thr>_m<m>_b<beta> : 对称 + 对比度：d = min(允许侧)，两侧各长 d
  SYM2_<thr>_m<m>        : 仅对称（不看对比度），用于分离"对称"的独立贡献
  ORACLE_gate / ORACLE_side / ORACLE_sym : 上界诊断（使用 gt，仅作说明）

机理诊断（使用 gt，**仅作说明，不参与方法**）
  * 短动作受害样本中"单侧泄露"的占比（验证机制 2 的动机）
  * 对比度 rho 预测"扩展是否有益"的 AUC（全体 / 短 / 长）
  * rho 在短/长两桶的均值差（验证机制 1 的可分性）

校准纪律（同 d34~d36）
  阈值在校准集上选，受"短动作不回归"约束后取长 d@0.7 最大；
  test 不参与选择，且把全部变体在 test 上透明打印。

用法: python diag37_gen.py <test_config> <ckpt>
环境变量: BS=16 CALIB_JSON=data/charades/train.json THR=0.35 LIMIT=0 K_REP=1 NO_CACHE=0
"""
import os
import sys
import pickle
import time

import numpy as np
import torch

# ---- 路径解析：默认自动定位本仓库根目录（含 models/pcnet.py），可用 PCNET_ROOT 覆盖 ----
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = _HERE
while _REPO != os.path.dirname(_REPO) and not os.path.exists(os.path.join(_REPO, 'models', 'pcnet.py')):
    _REPO = os.path.dirname(_REPO)
ROOT = os.environ.get('PCNET_ROOT') or os.environ.get('LM_ROOT') or _REPO
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from csr.paths import DATA_ROOT            # noqa: E402

from utils import load_json            # noqa: E402
import datasets as da                  # noqa: E402
from models import PCNet               # noqa: E402
from models.loss import cal_nll_loss   # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

cfg = load_json(sys.argv[1]); ckpt = sys.argv[2]
dargs = cfg['dataset']; DS_NAME = dargs['dataset']
vocab = pickle.load(open(dargs['vocab_path'], 'rb'))
_probe = getattr(da, DS_NAME)(data_path=dargs['test_data'], vocab=vocab, args=dargs, split='test')
VOCAB_SIZE = _probe.vocab_size
COLLATE = _probe.collate_data
del _probe

FP = os.environ.get('FEAT_PATH', dargs['feature_path'])
if not os.path.isabs(FP) and not os.path.exists(FP):
    for cand in (os.path.join(DATA_ROOT, os.path.basename(FP)),
                 os.path.join(ROOT, os.path.basename(FP))):
        if os.path.exists(cand):
            FP = cand
            break
dargs['feature_path'] = FP
print('feature_path=%s exists=%s' % (FP, os.path.exists(FP)), flush=True)

sd = torch.load(ckpt, map_location='cuda')
sp = sd['model_parameters']
ck_cfg = sd.get('config', {})
mcfg = dict(ck_cfg.get('model', {}).get('config', cfg['model']['config']))
P = int(sp['slot_atten.slots'].shape[0]) if 'slot_atten.slots' in sp else mcfg.get('num_props', 8)
mcfg['num_props'] = P; mcfg['vocab_size'] = VOCAB_SIZE
mcfg['max_epoch'] = cfg['train']['max_num_epochs']; mcfg.setdefault('num_iteration', 4)

L = dargs['max_num_frames'] // 4
GRID = np.linspace(0, 1, L)
BS = int(os.environ.get('BS', 16))
THR = float(os.environ.get('THR', 0.35))
K_REP = int(os.environ.get('K_REP', 1))
CALIB_JSON = os.environ.get('CALIB_JSON', 'data/charades/train.json')
LIMIT = int(os.environ.get('LIMIT', 0))
NO_CACHE = os.environ.get('NO_CACHE', '0') == '1'
CACHE_DIR = os.environ.get('CACHE_DIR') or os.path.join(_REPO, 'csr', 'analysis', 'feat_cache')

# 阈值网格：rel = tval * max(z)；z = mean(z) + tval * std(z)
THR_GRID = (('rel', 0.45), ('rel', 0.60), ('z', 0.15), ('z', 0.50))
M_GRID = (1, 2)
BETA_GRID = (0.0, 0.5, 0.6, 0.7, 0.8)


def iou(a0, a1, b0, b1):
    inter = np.maximum(0.0, np.minimum(a1, b1) - np.maximum(a0, b0))
    union = np.maximum(a1, b1) - np.minimum(a0, b0)
    return np.where(union > 1e-9, inter / np.maximum(union, 1e-9), 0.0)


def metrics(i):
    return dict(mIoU=float(i.mean()), i3=float((i >= 0.3).mean()),
                i5=float((i >= 0.5).mean()), i7=float((i >= 0.7).mean()))


def zs(R):
    return (R - R.mean(1, keepdims=True)) / (R.std(1, keepdims=True) + 1e-8)


def flood(m, pc):
    n, l = m.shape
    s = np.zeros(n); e = np.ones(n)
    for i in range(n):
        j = pc[i]; a = j
        while a > 0 and m[i, a - 1]:
            a -= 1
        b = j
        while b < l - 1 and m[i, b + 1]:
            b += 1
        s[i] = GRID[a]; e[i] = GRID[b]
    return s, e


def collect(data_json, tag, reps=1):
    ds = getattr(da, DS_NAME)(data_path=data_json, vocab=vocab, args=dargs, split='test')
    if LIMIT > 0:
        ds = Subset(ds, list(range(min(LIMIT, len(ds)))))
    loader = DataLoader(ds, batch_size=BS, shuffle=False, collate_fn=COLLATE, num_workers=8)
    out = []
    for rep in range(reps):
        np.random.seed(7 + rep); torch.manual_seed(7 + rep)
        model = PCNet(mcfg).cuda(); model.load_state_dict(sp); model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        _orig_pg = model.proposal_generator
        _c_pt = torch.linspace(0.5 / L, 1 - 0.5 / L, L)
        _w_pt = torch.full((L,), max(2.0 / L, 0.02))
        N8, C8, W8, REL, G = [], [], [], [], []
        t0 = time.time()
        with torch.no_grad():
            for bi, batch in enumerate(loader, 1):
                net_input = {k: v.cuda() for k, v in batch['net_input'].items()}
                bsz = len(batch['raw'])
                o = model(epoch=0, **net_input)
                wm = o['words_mask'].unsqueeze(1).expand(bsz, P, -1).contiguous().view(bsz * P, -1)
                wi = o['words_id'].unsqueeze(1).expand(bsz, P, -1).contiguous().view(bsz * P, -1)
                n, _ = cal_nll_loss(o['words_logit'], wi, wm)
                N8.append(n.view(bsz, P).cpu().numpy())
                C8.append(o['center'].view(bsz, P).cpu().numpy())
                W8.append(o['width'].view(bsz, P).cpu().numpy())
                model.proposal_generator = lambda h, b: (_c_pt.repeat(b).cuda(), _w_pt.repeat(b).cuda())
                model.num_props = L
                o2 = model(epoch=0, **net_input)
                wm2 = o2['words_mask'].unsqueeze(1).expand(bsz, L, -1).contiguous().view(bsz * L, -1)
                wi2 = o2['words_id'].unsqueeze(1).expand(bsz, L, -1).contiguous().view(bsz * L, -1)
                n2, _ = cal_nll_loss(o2['words_logit'], wi2, wm2)
                REL.append((-n2.view(bsz, L)).cpu().numpy())
                model.proposal_generator = _orig_pg; model.num_props = P
                dur = np.asarray([i[1] for i in batch['raw']], dtype=np.float64)
                ts = np.asarray([i[2] for i in batch['raw']], dtype=np.float64)
                G.append(ts / dur[:, None])
                if bi % 100 == 0:
                    print('    [%s rep%d] batch %d/%d  %.1fs' % (
                        tag, rep, bi, len(loader), time.time() - t0), flush=True)
        del model; torch.cuda.empty_cache()
        G = np.concatenate(G); N8 = np.concatenate(N8); C8 = np.concatenate(C8)
        W8 = np.concatenate(W8); REL = np.concatenate(REL)
        n = len(G); r = np.arange(n)
        GS, GE = G[:, 0], G[:, 1]
        order = np.argsort(N8, axis=1); j1 = order[:, 0]
        cb, wb = C8[r, j1], W8[r, j1]
        bL = np.clip(cb - wb / 2, 0, 1); bH = np.clip(cb + wb / 2, 0, 1)
        Z = zs(REL)
        pc = np.clip((cb * (L - 1)).round().astype(int), 0, L - 1)
        m15 = Z > (Z.mean(1) + 0.15 * Z.std(1))[:, None]
        m15[np.arange(n), pc] = True
        s15, e15 = flood(m15, pc)
        out.append(dict(GS=GS, GE=GE, GL=GE - GS, cb=cb, wb=wb, bL=bL, bH=bH, Z=Z,
                        s15=s15, e15=e15, n=n,
                        bli=np.clip((bL * (L - 1)).round().astype(int), 0, L - 1),
                        bhi=np.clip((bH * (L - 1)).round().astype(int), 0, L - 1)))
    return out


def collect_cached(data_json, tag, reps=1):
    """特征提取 + 曲线计算的结果缓存到 pickle —— 与 diag36 同 key，可直接命中。"""
    os.makedirs(CACHE_DIR, exist_ok=True)
    key = '%s__%s__r%d__lim%d__L%d.pkl' % (
        DS_NAME, os.path.basename(data_json).replace('.json', ''), reps, LIMIT, L)
    fp = os.path.join(CACHE_DIR, key)
    if os.path.exists(fp) and not NO_CACHE:
        with open(fp, 'rb') as f:
            out = pickle.load(f)
        print('    [%s] 缓存命中 %s  (N=%d)' % (tag, fp, out[0]['n']), flush=True)
        return out
    out = collect(data_json, tag, reps)
    with open(fp, 'wb') as f:
        pickle.dump(out, f)
    print('    [%s] 已缓存 -> %s' % (tag, fp), flush=True)
    return out


def side_run(z, start, th, gap, direction):
    """从 start 沿 direction 走，返回最远的达标下标（允许 gap 个连续落点）。"""
    n = len(z); idx = start; miss = 0; best = None
    while 0 <= idx < n:
        if z[idx] >= th:
            miss = 0; best = idx
        else:
            miss += 1
            if miss > gap:
                break
        idx += direction
    return best


def side_info(D, tmode, tval, g=0):
    """逐侧返回：是否有平台(ok)、外延量(ext, 归一化)、框外/框内对比度(rho)。"""
    n = D['n']; Z = D['Z']; bli = D['bli']; bhi = D['bhi']
    if tmode == 'rel':
        th = tval * Z.max(1)
    else:
        th = Z.mean(1) + tval * Z.std(1)
    okL = np.zeros(n, dtype=bool); okR = np.zeros(n, dtype=bool)
    extL = np.zeros(n); extR = np.zeros(n)
    conL = np.ones(n); conR = np.ones(n)
    for i in range(n):
        z = Z[i]; bl, bh = bli[i], bhi[i]
        mi = max(float(z[bl:bh + 1].mean()), 1e-8)
        if bl - 1 >= 0:
            b = side_run(z, bl - 1, th[i], g, -1)
            if b is not None:
                okL[i] = True
                extL[i] = (bl - b) / (L - 1.0)
                conL[i] = float(z[b:bl].mean()) / mi
        if bh + 1 <= L - 1:
            b = side_run(z, bh + 1, th[i], g, +1)
            if b is not None:
                okR[i] = True
                extR[i] = (b - bh) / (L - 1.0)
                conR[i] = float(z[bh + 1:b + 1].mean()) / mi
    return okL, okR, extL, extR, conL, conR


def tkey_of(tmode, tval):
    return ('t%02d' % int(round(tval * 100))) if tmode == 'rel' else ('z%02d' % int(round(tval * 100)))


def build_variants(D):
    """返回 {名称: (lo, hi, 启用率)}"""
    bL = D['bL']; bH = D['bH']
    u_lo = np.minimum(D['s15'], bL); u_hi = np.maximum(D['e15'], bH)
    out = {'U_k0.15': (u_lo, u_hi, 1.0)}
    for tmode, tval in THR_GRID:
        tk = tkey_of(tmode, tval)
        okL, okR, extL, extR, conL, conR = side_info(D, tmode, tval, 0)
        lenL = extL * (L - 1.0); lenR = extR * (L - 1.0)
        for m in M_GRID:
            fL = okL & (lenL >= m); fR = okR & (lenR >= m)
            base = '%s_m%d' % (tk, m)
            # 仅对称（不看对比度）
            d2 = np.where(fL & fR, np.minimum(extL, extR), 0.0)
            out['SYM2_' + base] = (bL - d2, bH + d2, float((d2 > 1e-9).mean()))
            for beta in BETA_GRID:
                aL = fL & (conL >= beta); aR = fR & (conR >= beta)
                # 逐侧独立（beta=0 即 EGS 对照）
                out['CTR_%s_b%.1f' % (base, beta)] = (
                    np.where(aL, bL - extL, bL), np.where(aR, bH + extR, bH),
                    float((aL | aR).mean()))
                # 对称 + 对比度
                d1 = np.where(aL & aR, np.minimum(extL, extR), 0.0)
                out['SYM_%s_b%.1f' % (base, beta)] = (
                    bL - d1, bH + d1, float((d1 > 1e-9).mean()))
    return out


def ev(D, lo, hi):
    SHORT = D['GL'] <= THR; LONG = ~SHORT
    i = iou(np.clip(lo, 0, 1), np.clip(hi, 0, 1), D['GS'], D['GE'])
    return {'all': metrics(i), 'short': metrics(i[SHORT]), 'long': metrics(i[LONG]), '_iou': i}


def base_metrics(D):
    return ev(D, D['bL'].copy(), D['bH'].copy())


def avg_metrics(list_of_m):
    o = {}
    for k in ('all', 'short', 'long'):
        o[k] = {kk: float(np.mean([m[k][kk] for m in list_of_m])) for kk in ('mIoU', 'i3', 'i5', 'i7')}
    return o


def dpt(cur, base_, bucket, key):
    return 100.0 * (cur[bucket][key] - base_[bucket][key])


def scan(DL):
    per = [build_variants(D) for D in DL]
    names = list(per[0].keys())
    res, rate = {}, {}
    for nm in names:
        ms = [ev(D, per[k][nm][0], per[k][nm][1]) for k, D in enumerate(DL)]
        res[nm] = avg_metrics(ms)
        rate[nm] = float(np.mean([per[k][nm][2] for k in range(len(DL))]))
    base = avg_metrics([base_metrics(D) for D in DL])
    return res, base, rate


# ------------------------- ORACLE 上界 -------------------------
def oracle_gate(D):
    bL = D['bL']; bH = D['bH']
    u_lo = np.minimum(D['s15'], bL); u_hi = np.maximum(D['e15'], bH)
    i_b = iou(bL, bH, D['GS'], D['GE']); i_u = iou(u_lo, u_hi, D['GS'], D['GE'])
    on = i_u > i_b
    return np.where(on, u_lo, bL), np.where(on, u_hi, bH), float(on.mean())


def oracle_side(D):
    bL = D['bL']; bH = D['bH']; GS = D['GS']; GE = D['GE']
    okL, okR, extL, extR, _, _ = side_info(D, 'rel', 0.20, 0)
    loL = np.where(okL, bL - extL, bL)
    i0 = iou(bL, bH, GS, GE)
    lo = np.where(iou(loL, bH, GS, GE) > i0, loL, bL)
    hiR = np.where(okR, bH + extR, bH)
    i1 = iou(lo, bH, GS, GE)
    hi = np.where(iou(lo, hiR, GS, GE) > i1, hiR, bH)
    return lo, hi, float(((lo < bL - 1e-9) | (hi > bH + 1e-9)).mean())


def oracle_sym(D):
    bL = D['bL']; bH = D['bH']; GS = D['GS']; GE = D['GE']
    okL, okR, extL, extR, _, _ = side_info(D, 'rel', 0.20, 0)
    d = np.where(okL & okR, np.minimum(extL, extR), 0.0)
    on = iou(bL - d, bH + d, GS, GE) > iou(bL, bH, GS, GE)
    return np.where(on, bL - d, bL), np.where(on, bH + d, bH), float(on.mean())


def auc(score, label):
    label = np.asarray(label, dtype=bool)
    n1 = float(label.sum()); n0 = float(len(label) - n1)
    if n1 < 1 or n0 < 1:
        return float('nan')
    order = np.argsort(score, kind='mergesort')
    ranks = np.empty(len(score), dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1)
    return float((ranks[label].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


print('\n' + '=' * 112, flush=True)
print('diag37  对称生长 + 对比度门控   dataset=%s  校准集=%s  test=%s  K_REP=%d' % (
    DS_NAME, CALIB_JSON, dargs['test_data'], K_REP), flush=True)
print('=' * 112, flush=True)

print('\n[1/4] 校准集特征 ...', flush=True)
DC = collect_cached(CALIB_JSON, 'calib', reps=1)[0]
print('      N=%d' % DC['n'], flush=True)

print('\n[2/4] 测试集特征 ...', flush=True)
DT_list = collect_cached(dargs['test_data'], 'test', reps=K_REP)
print('      N=%d  K_REP=%d' % (DT_list[0]['n'], K_REP), flush=True)

print('\n[3/4] 复现校验（test base 应与 diag30 一致）', flush=True)
bT = avg_metrics([base_metrics(D) for D in DT_list])
print('   mIoU %.4f  R1@0.5 %.4f  R1@0.7 %.4f' % (
    bT['all']['mIoU'], bT['all']['i5'], bT['all']['i7']), flush=True)

# ------------------------- 机理诊断（用 gt，不参与方法） -------------------------
D0 = DT_list[0]
SH = D0['GL'] <= THR
bL0, bH0 = D0['bL'], D0['bH']
i_b0 = iou(bL0, bH0, D0['GS'], D0['GE'])
i_u0 = iou(np.minimum(D0['s15'], bL0), np.maximum(D0['e15'], bH0), D0['GS'], D0['GE'])
okL, okR, extL, extR, conL, conR = side_info(D0, 'rel', 0.20, 0)
benefit = i_u0 > i_b0
harmed = i_u0 < i_b0
os_mask = ((extL > 1e-9) ^ (extR > 1e-9))
payL = ((extL > 1e-9) & (extR <= 1e-9))
print('\n[诊断] 过扩伤害归因（用于验证机制动机，仅作说明）', flush=True)
for nm, msk in (('短动作受害样本', SH & harmed), ('长动作受益样本', (~SH) & benefit)):
    k = int(msk.sum())
    if k:
        print('  %-12s n=%-5d 单侧泄露占比=%.1f%%' % (
            nm, k, 100.0 * float(os_mask[msk].mean())), flush=True)
sigC = np.minimum(conL, conR)
sigE = np.minimum(extL, extR)
print('  对比度 rho(min)  mean: 短=%.3f 长=%.3f   (越可分越好)' % (
    float(sigC[SH].mean()), float(sigC[~SH].mean())), flush=True)
print('  AUC(对比度 -> 扩展有益):  全体=%.3f  短=%.3f  长=%.3f' % (
    auc(sigC, benefit), auc(sigC[SH], benefit[SH]), auc(sigC[~SH], benefit[~SH])), flush=True)
print('  AUC(对称外延量 -> 扩展有益): 全体=%.3f  短=%.3f  长=%.3f' % (
    auc(sigE, benefit), auc(sigE[SH], benefit[SH]), auc(sigE[~SH], benefit[~SH])), flush=True)

# ------------------------- 参照 -------------------------
rC, bC, rateC = scan([DC])
oT = avg_metrics([ev(D, *oracle_gate(D)[:2]) for D in DT_list])
oS = avg_metrics([ev(D, *oracle_side(D)[:2]) for D in DT_list])
oSY = avg_metrics([ev(D, *oracle_sym(D)[:2]) for D in DT_list])
print('\n[参照]', flush=True)
print('%-24s | %7s %7s | %7s %7s | %7s' % (
    '方法', 'd短@0.5', 'd长@0.5', 'd长@0.7', 'd全@0.5', 'd全mIoU'), flush=True)
for nm, m in (('U_k0.15(标准并集)', rC['U_k0.15']), ('ORACLE_gate', oT),
              ('ORACLE_side(逐侧)', oS), ('ORACLE_sym(对称)', oSY)):
    print('%-24s | %+7.2f %+7.2f | %+7.2f %+7.2f | %+7.2f' % (
        nm, dpt(m, bT, 'short', 'i5'), dpt(m, bT, 'long', 'i5'), dpt(m, bT, 'long', 'i7'),
        dpt(m, bT, 'all', 'i5'), dpt(m, bT, 'all', 'mIoU')), flush=True)

# ------------------------- 校准 -------------------------
print('\n' + '=' * 112, flush=True)
print('[4/4] 在【校准集】上扫描，受"短动作不回归"约束', flush=True)
print('%-24s | %7s %7s | %7s %7s | %7s %7s | %6s' % (
    '变体', 'd短@0.5', 'd短@0.7', 'd长@0.5', 'd长@0.7', 'd全@0.5', 'd全mIoU', '启用率'), flush=True)
cand = []
for nm, m in rC.items():
    if nm == 'U_k0.15':
        continue
    cand.append(dict(nm=nm,
                     d5s=dpt(m, bC, 'short', 'i5'), d7s=dpt(m, bC, 'short', 'i7'),
                     d5l=dpt(m, bC, 'long', 'i5'), d7l=dpt(m, bC, 'long', 'i7'),
                     d5a=dpt(m, bC, 'all', 'i5'), d7a=dpt(m, bC, 'all', 'i7'),
                     dma=dpt(m, bC, 'all', 'mIoU')))
for c in sorted(cand, key=lambda x: -x['d7l'])[:16]:
    print('%-24s | %+7.2f %+7.2f | %+7.2f %+7.2f | %+7.2f %+7.2f | %5.1f%%' % (
        c['nm'], c['d5s'], c['d7s'], c['d5l'], c['d7l'], c['d5a'], c['dma'],
        100.0 * rateC[c['nm']]), flush=True)
okc = [c for c in cand if (c['d5s'] >= -0.2) and (c['d7s'] >= -0.3) and (c['dma'] >= -0.2)]
if okc:
    sel = max(okc, key=lambda c: c['d7l'])
    print('\n  => 校准选择(严格): %s  (calib d短@0.5=%+.2f, d长@0.7=%+.2f, 通过=%d/%d)' % (
        sel['nm'], sel['d5s'], sel['d7l'], len(okc), len(cand)), flush=True)
else:
    sel = max(cand, key=lambda c: c['d7l'] + 3.0 * c['d5s'])
    print('\n  => 校准集无严格解，退化取 Pareto 拐点 argmax(d长@0.7 + 3*d短@0.5): %s' % sel['nm'], flush=True)

print('\n' + '=' * 112, flush=True)
print('【测试集】选定 %s（test 未参与选择）' % sel['nm'], flush=True)
print('%-24s | %7s %7s | %7s %7s | %7s %7s' % (
    '方法', 'd短@0.5', 'd短@0.7', 'd长@0.5', 'd长@0.7', 'd全@0.5', 'd全mIoU'), flush=True)
rT, bT, rateT = scan(DT_list)
for nm, m in (('选定(' + sel['nm'] + ')', rT[sel['nm']]), ('ORACLE_gate(上界)', oT),
              ('ORACLE_side(上界)', oS), ('ORACLE_sym(上界)', oSY)):
    print('%-24s | %+7.2f %+7.2f | %+7.2f %+7.2f | %+7.2f %+7.2f' % (
        nm, dpt(m, bT, 'short', 'i5'), dpt(m, bT, 'short', 'i7'), dpt(m, bT, 'long', 'i5'),
        dpt(m, bT, 'long', 'i7'), dpt(m, bT, 'all', 'i5'), dpt(m, bT, 'all', 'mIoU')), flush=True)

print('\n【透明呈现】全部变体在【测试集】上（按 长 d@0.7 降序，前 45）', flush=True)
print('%-24s | %7s %7s | %7s %7s | %7s %7s | %7s %s' % (
    '变体', 'd短@0.5', 'd短@0.7', 'd长@0.5', 'd长@0.7', 'd全@0.5', 'd全mIoU', '启用率', '达标?'), flush=True)
rows = []
for nm, m in rT.items():
    d5s = dpt(m, bT, 'short', 'i5'); d7s = dpt(m, bT, 'short', 'i7')
    d5l = dpt(m, bT, 'long', 'i5'); d7l = dpt(m, bT, 'long', 'i7')
    d5a = dpt(m, bT, 'all', 'i5'); dma = dpt(m, bT, 'all', 'mIoU')
    ok = (d5s >= -0.2) and (d7s >= -0.3) and (d5l >= 2.0) and (d7l >= 5.0) \
         and (d5a >= 0.0) and (dma > 0.0)
    rows.append((d7l, nm, d5s, d7s, d5l, d5a, dma, rateT[nm], ok))
nok = 0
for d7l, nm, d5s, d7s, d5l, d5a, dma, rt, ok in sorted(rows, reverse=True)[:45]:
    nok += int(ok)
    print('%-24s | %+7.2f %+7.2f | %+7.2f %+7.2f | %+7.2f %+7.2f | %6.1f%% %s' % (
        nm, d5s, d7s, d5l, d7l, d5a, dma, 100.0 * rt, '**达标**' if ok else ''), flush=True)
print('  (达标变体总数 = %d / %d)' % (sum(1 for r in rows if r[-1]), len(rows)), flush=True)

print('\n[基线绝对值] base 短 mIoU %.4f R1@0.5 %.4f R1@0.7 %.4f  (n=%d)' % (
    bT['short']['mIoU'], bT['short']['i5'], bT['short']['i7'], int(SH.sum())), flush=True)
print('             base 长 mIoU %.4f R1@0.5 %.4f R1@0.7 %.4f  (n=%d)' % (
    bT['long']['mIoU'], bT['long']['i5'], bT['long']['i7'], int((~SH).sum())), flush=True)
print('             base 全 mIoU %.4f R1@0.5 %.4f R1@0.7 %.4f' % (
    bT['all']['mIoU'], bT['all']['i5'], bT['all']['i7']), flush=True)

print('\n' + '=' * 112, flush=True)
print('DONE', flush=True)
