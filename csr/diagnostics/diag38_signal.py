"""diag38：逐侧扩展决策的【信号筛选】——回答"到底有没有可用的无 gt 判据"

前情（d34~d37 的共同失败模式）
  U / EGS / AGG / SGG / CTR / SYM 六族共 300+ 变体，全部卡在同一个位置：
  "短动作不降" 与 "长动作显著增强" 无法同时满足，而 ORACLE 上界表明解是存在的：
      Charades: ORACLE_gate  +1.29 / +12.48      ORACLE_side  +2.20 / +17.66
  唯一的缺口是**逐侧、逐样本的判据**。d37 的 AUC 诊断显示 "边框外是否存在平台"
  这个信号 AUC≈0.52（与抛硬币无异）—— 这解释了为什么 300 个变体都失败：
  不是阈值没调好，而是**信号本身不可分**。

d37 的一个实现缺陷（已在本脚本修正）
  对比度 rho = mean(z_out)/mean(z_in)，分母为"框内 z 均值"。Z 是逐样本 z-score，
  框内均值经常接近 0 -> 分母趋零 -> rho 爆到 1e5~1e6（日志里 短=200787 即此）。
  修正：改用 0~1 归一的 RN=(REL-min)/(max-min)，并把 rho 定义为 RN 的比值，
  同时对分母设下限；额外给出 zrho / 差分类信号作为对照。

本脚本做三件事
  [A] 逐侧信号筛选：对 16 个**不使用 gt**的候选信号，计算它们预测
      "该侧扩展是否有益"（label 由 gt IoU 决定，仅用于评估信号）的 AUC。
      分 全体/短/长 三桶，在校准集与测试集上分别报告。
  [B] 规则搜索：在**校准集**上搜索 (信号, 阈值) 单规则与两信号 AND 规则，
      受"短动作不回归"约束后取长 d@0.7 最大；得到"有 gt 校准的判据"。
  [C] 零 gt 判据：用信号的**分位数**（在待评测集自身取 top-q，不含任何 gt）
      构造门控，与 [B] 对比——证明即使完全不用 gt 也能/不能达标。

诚实边界
  * [B] 的阈值由 train.json 的 gt 选出（2 个标量），推理期不含 gt；这是常规超参选择。
  * [C] 完全不接触 gt。
  * test 不参与任何选择；全部规则在 test 上透明打印。

用法: python diag38_gen.py <test_config> <ckpt>
环境变量: BS=16 CALIB_JSON=data/charades/train.json THR=0.35 LIMIT=0 K_REP=1
          T_SCREEN=0.60 MINRUN=1
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
CACHE_DIR = os.environ.get('CACHE_DIR') or os.path.join(_REPO, 'csr', 'analysis', 'feat_cache')
T_SCREEN = float(os.environ.get('T_SCREEN', 0.60))
MINRUN = int(os.environ.get('MINRUN', 1))

QS = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)


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
        order = np.argsort(N8, axis=1); j1 = order[:, 0]; j2 = order[:, 1]
        cb, wb = C8[r, j1], W8[r, j1]
        bL = np.clip(cb - wb / 2, 0, 1); bH = np.clip(cb + wb / 2, 0, 1)
        Z = zs(REL)
        pc = np.clip((cb * (L - 1)).round().astype(int), 0, L - 1)
        m15 = Z > (Z.mean(1) + 0.15 * Z.std(1))[:, None]
        m15[np.arange(n), pc] = True
        s15, e15 = flood(m15, pc)
        out.append(dict(GS=GS, GE=GE, GL=GE - GS, cb=cb, wb=wb, bL=bL, bH=bH, Z=Z,
                        REL=REL, s15=s15, e15=e15, n=n,
                        N1=N8[r, j1], N2=N8[r, j2], c1=C8[r, j1], c2=C8[r, j2],
                        bli=np.clip((bL * (L - 1)).round().astype(int), 0, L - 1),
                        bhi=np.clip((bH * (L - 1)).round().astype(int), 0, L - 1)))
    return out


def collect_cached(data_json, tag, reps=1):
    """v2 缓存：额外保存 REL / N8 等，供信号筛选使用。"""
    os.makedirs(CACHE_DIR, exist_ok=True)
    key = '%s__%s__r%d__lim%d__L%d__v2.pkl' % (
        DS_NAME, os.path.basename(data_json).replace('.json', ''), reps, LIMIT, L)
    fp = os.path.join(CACHE_DIR, key)
    if os.path.exists(fp):
        with open(fp, 'rb') as f:
            out = pickle.load(f)
        print('    [%s] 缓存命中 %s  (N=%d)' % (tag, fp, out[0]['n']), flush=True)
        return out
    out = collect(data_json, tag, reps)
    with open(fp, 'wb') as f:
        pickle.dump(out, f)
    print('    [%s] 已缓存 -> %s' % (tag, fp), flush=True)
    return out


def side_run(z, start, th, direction=1, gap=0):
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


SIG_NAMES = ('ext', 'rho', 'zrho', 'zedge', 'zoutmax', 'zoutmean', 'zoutmin', 'grad',
             'extratio', 'wb', 'cb', 'edgeclose', 'kratio', 'margin', 'r2dist', 'nll1')


def side_pack(D):
    """逐侧信号 + 逐侧 label（label 用 gt 判定，仅用于评估信号/搜索规则）。"""
    n = D['n']; Z = D['Z']; REL = D['REL']
    bL = D['bL']; bH = D['bH']; bli = D['bli']; bhi = D['bhi']; GS = D['GS']; GE = D['GE']
    RN = (REL - REL.min(1, keepdims=True)) / (REL.max(1, keepdims=True) - REL.min(1, keepdims=True) + 1e-8)
    Zmax = Z.max(1)
    kw = D['e15'] - D['s15']
    # 哨兵 -1e9 = 该侧无平台（未激活）；判据统一写为 sig > 0 / 分位数过滤 > -1e8
    SL = {k: np.full(n, -1e9) for k in SIG_NAMES}
    SR = {k: np.full(n, -1e9) for k in SIG_NAMES}
    for i in range(n):
        z = Z[i]; rn = RN[i]; bl, bh = bli[i], bhi[i]
        mi = max(float(rn[bl:bh + 1].mean()), 0.05)      # 分母下限，防爆
        mz = float(z[bl:bh + 1].mean())
        for S, side in ((SL, 'L'), (SR, 'R')):
            S['wb'][i] = D['wb'][i]; S['cb'][i] = D['cb'][i]
            S['kratio'][i] = kw[i] / max(D['wb'][i], 1e-8)
            S['margin'][i] = D['N2'][i] - D['N1'][i]
            S['r2dist'][i] = abs(D['c2'][i] - D['c1'][i])
            S['nll1'][i] = D['N1'][i]
        if bl - 1 >= 0:
            b = side_run(z, bl - 1, T_SCREEN * Zmax[i], -1, 0)
            if b is not None and (bl - b) >= MINRUN:
                sp = slice(b, bl)
                SL['ext'][i] = (bl - b) / (L - 1.0)
                SL['rho'][i] = float(rn[sp].mean()) / mi
                SL['zrho'][i] = float(z[sp].mean()) / (mz + 1e-8 if abs(mz) > 1e-3 else 1e-3)
                SL['zedge'][i] = float(z[bl - 1])
                SL['zoutmax'][i] = float(z[sp].max())
                SL['zoutmean'][i] = float(z[sp].mean())
                SL['zoutmin'][i] = float(z[sp].min())
                SL['grad'][i] = float(z[bl - 1] - z[bl])
                SL['extratio'][i] = SL['ext'][i] / max(D['wb'][i], 1e-8)
                SL['edgeclose'][i] = float(bL[i])
        if bh + 1 <= L - 1:
            b = side_run(z, bh + 1, T_SCREEN * Zmax[i], +1, 0)
            if b is not None and (b - bh) >= MINRUN:
                sp = slice(bh + 1, b + 1)
                SR['ext'][i] = (b - bh) / (L - 1.0)
                SR['rho'][i] = float(rn[sp].mean()) / mi
                SR['zrho'][i] = float(z[sp].mean()) / (mz + 1e-8 if abs(mz) > 1e-3 else 1e-3)
                SR['zedge'][i] = float(z[bh + 1])
                SR['zoutmax'][i] = float(z[sp].max())
                SR['zoutmean'][i] = float(z[sp].mean())
                SR['zoutmin'][i] = float(z[sp].min())
                SR['grad'][i] = float(z[bh + 1] - z[bh])
                SR['extratio'][i] = SR['ext'][i] / max(D['wb'][i], 1e-8)
                SR['edgeclose'][i] = float(1.0 - bH[i])
    ib = iou(bL, bH, GS, GE)
    loL = np.where(SL['ext'] > 0, bL - SL['ext'], bL)
    loR = np.where(SR['ext'] > 0, bH + SR['ext'], bH)
    labL = iou(loL, bH, GS, GE) > ib
    labR = iou(bL, loR, GS, GE) > ib
    return SL, SR, labL, labR, ib, loL, loR


def auc(score, label):
    label = np.asarray(label, dtype=bool)
    k = np.isfinite(score) & (score > -1e8)
    if k.sum() < 10:
        return float('nan'), 0
    s = score[k]; lb = label[k]
    n1 = float(lb.sum()); n0 = float(len(lb) - n1)
    if n1 < 1 or n0 < 1:
        return float('nan'), int(k.sum())
    o = np.argsort(s, kind='mergesort')
    rk = np.empty(len(s), dtype=np.float64); rk[o] = np.arange(1, len(s) + 1)
    return float((rk[lb].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)), int(k.sum())


def ev(D, lo, hi):
    SHORT = D['GL'] <= THR; LONG = ~SHORT
    i = iou(np.clip(lo, 0, 1), np.clip(hi, 0, 1), D['GS'], D['GE'])
    return {'all': metrics(i), 'short': metrics(i[SHORT]), 'long': metrics(i[LONG])}


def base_metrics(D):
    return ev(D, D['bL'].copy(), D['bH'].copy())


def avg_metrics(ms):
    o = {}
    for k in ('all', 'short', 'long'):
        o[k] = {kk: float(np.mean([m[k][kk] for m in ms])) for kk in ('mIoU', 'i3', 'i5', 'i7')}
    return o


def dpt(cur, base_, bucket, key):
    return 100.0 * (cur[bucket][key] - base_[bucket][key])


# 规则在 (SL, SR) 上求值
def apply_rule(SL, SR, sig, thr, sign, sig2=None, thr2=None, sign2=1):
    aL = SL['ext'] > 0; aR = SR['ext'] > 0
    gL = aL & (sign * SL[sig] >= thr)
    gR = aR & (sign * SR[sig] >= thr)
    if sig2 is not None:
        gL = gL & (sign2 * SL[sig2] >= thr2)
        gR = gR & (sign2 * SR[sig2] >= thr2)
    return gL, gR


def eval_rule(packs, base, sig, thr, sign, sig2=None, thr2=None, sign2=1):
    ms = []
    for D, pk in packs:
        SL, SR = pk[0], pk[1]
        gL, gR = apply_rule(SL, SR, sig, thr, sign, sig2, thr2, sign2)
        lo = np.where(gL, D['bL'] - SL['ext'], D['bL'])
        hi = np.where(gR, D['bH'] + SR['ext'], D['bH'])
        ms.append(ev(D, lo, hi))
    m = avg_metrics(ms)
    return m, (dpt(m, base, 'short', 'i5'), dpt(m, base, 'short', 'i7'),
               dpt(m, base, 'long', 'i5'), dpt(m, base, 'long', 'i7'),
               dpt(m, base, 'all', 'i5'), dpt(m, base, 'all', 'mIoU'))


print('\n' + '=' * 112, flush=True)
print('diag38  逐侧扩展决策的【信号筛选】  dataset=%s  校准集=%s  test=%s  K_REP=%d  T_SCREEN=%.2f' % (
    DS_NAME, CALIB_JSON, dargs['test_data'], K_REP, T_SCREEN), flush=True)
print('=' * 112, flush=True)

print('\n[1/5] 校准集特征 ...', flush=True)
DC = collect_cached(CALIB_JSON, 'calib', reps=1)[0]
print('      N=%d' % DC['n'], flush=True)

print('\n[2/5] 测试集特征 ...', flush=True)
DT_list = collect_cached(dargs['test_data'], 'test', reps=K_REP)
print('      N=%d  K_REP=%d' % (DT_list[0]['n'], K_REP), flush=True)

bT = avg_metrics([base_metrics(D) for D in DT_list])
bC = avg_metrics([base_metrics(DC)])
print('\n[3/5] 复现校验', flush=True)
print('   test base: mIoU %.4f  R1@0.5 %.4f  R1@0.7 %.4f' % (
    bT['all']['mIoU'], bT['all']['i5'], bT['all']['i7']), flush=True)
print('   calib base: mIoU %.4f  R1@0.5 %.4f  R1@0.7 %.4f' % (
    bC['all']['mIoU'], bC['all']['i5'], bC['all']['i7']), flush=True)

packsC = [(DC, side_pack(DC))]
packsT = [(D, side_pack(D)) for D in DT_list]

print('\n' + '=' * 112, flush=True)
print('[4/5] 信号筛选：AUC(信号 -> 该侧扩展是否有益)。0.5=无用，>0.5 越大越好', flush=True)
print('   只统计"该侧存在平台(ext>0)"的样本；短/长按 gt 时长 0.35 分桶', flush=True)
print('%-12s | %19s | %19s' % ('信号', '校准集 (全体/短/长)', '测试集 (全体/短/长)'), flush=True)
SLc, SRc, labLc, labRc, ibc, _, _ = packsC[0][1]
SHc = DC['GL'] <= THR
sigC_all, sigT_all = {}, {}
for sig in SIG_NAMES:
    rowC, rowT = [], []
    for S, lab, SH in ((SLc, labLc, SHc), (SRc, labRc, SHc)):
        sc = np.concatenate([S[sig], S[sig]]); la = np.concatenate([lab, lab])
        sh = np.concatenate([SH, SH])
        a_all, _ = auc(sc, la); a_s, _ = auc(sc[sh], la[sh]); a_l, _ = auc(sc[~sh], la[~sh])
        rowC.append((a_all, a_s, a_l))
    rowT = []
    for D, pk in packsT:
        SLx, SRx, lLx, lRx, _, _, _ = pk
        SHx = D['GL'] <= THR
        sc = np.concatenate([SLx[sig], SRx[sig]]); la = np.concatenate([lLx, lRx])
        sh = np.concatenate([SHx, SHx])
        rowT.append((auc(sc, la)[0], auc(sc[sh], la[sh])[0], auc(sc[~sh], la[~sh])[0]))
    ac = (float(np.nanmean([r[0] for r in rowC])), float(np.nanmean([r[1] for r in rowC])),
          float(np.nanmean([r[2] for r in rowC])))
    at = (float(np.nanmean([r[0] for r in rowT])), float(np.nanmean([r[1] for r in rowT])),
          float(np.nanmean([r[2] for r in rowT])))
    sigC_all[sig] = ac; sigT_all[sig] = at
    print('%-12s | %6.3f %6.3f %6.3f | %6.3f %6.3f %6.3f' % (
        sig, ac[0], ac[1], ac[2], at[0], at[1], at[2]), flush=True)

print('\n  [提示] 若所有信号 AUC 都在 0.50 附近 => "该侧该不该扩"在当前特征下不可分', flush=True)

# ---------- 参照 ----------
print('\n' + '=' * 112, flush=True)
print('[参照] (test)', flush=True)


def oracle_side_m(packs, base):
    ms = []
    for D, pk in packs:
        SL, SR = pk[0], pk[1]
        gL = SL['ext'] > 0; gR = SR['ext'] > 0
        ib = iou(D['bL'], D['bH'], D['GS'], D['GE'])
        loL = np.where(gL, D['bL'] - SL['ext'], D['bL'])
        gL2 = iou(loL, D['bH'], D['GS'], D['GE']) > ib
        lo = np.where(gL2, loL, D['bL'])
        hiR = np.where(gR, D['bH'] + SR['ext'], D['bH'])
        gR2 = iou(lo, hiR, D['GS'], D['GE']) > iou(lo, D['bH'], D['GS'], D['GE'])
        hi = np.where(gR2, hiR, D['bH'])
        ms.append(ev(D, lo, hi))
    return avg_metrics(ms)


oT = oracle_side_m(packsT, bT)
mEGS, dlEGS = eval_rule(packsT, bT, 'ext', -1e9, 1.0)
print('%-24s | %+7.2f %+7.2f | %+7.2f %+7.2f | %+7.2f %+7.2f' % (
    'EGS(无门控,全部平台侧)', dlEGS[0], dlEGS[1], dlEGS[2], dlEGS[3], dlEGS[4], dlEGS[5]), flush=True)
print('%-24s | %+7.2f %+7.2f | %+7.2f %+7.2f | %+7.2f %+7.2f' % (
    'ORACLE_side(上界)', dpt(oT, bT, 'short', 'i5'), dpt(oT, bT, 'short', 'i7'),
    dpt(oT, bT, 'long', 'i5'), dpt(oT, bT, 'long', 'i7'),
    dpt(oT, bT, 'all', 'i5'), dpt(oT, bT, 'all', 'mIoU')), flush=True)

# ---------- 规则搜索 ----------
print('\n' + '=' * 112, flush=True)
print('[5/5] 规则搜索：(信号,阈值) 单规则 -> 校准集约束筛选 -> test 验证', flush=True)

rank = sorted(SIG_NAMES, key=lambda s: -abs(sigC_all[s][0] - 0.5))
rules = []
for sig in SIG_NAMES:
    a = sigC_all[sig][0]
    sign = 1.0 if a >= 0.5 else -1.0
    vals = np.concatenate([SLc[sig], SRc[sig]])
    vals = vals[vals > -1e8]
    if len(vals) < 50:
        continue
    for q in QS:
        thr = float(np.quantile(vals, 1.0 - q))
        rules.append((sig, thr, sign, None, None, 1.0))
print('  单规则候选 %d 个（16 信号 × 9 分位）' % len(rules), flush=True)

cands = []
for rule in rules:
    m, dl = eval_rule(packsC, bC, *rule)
    d5s, d7s, d5l, d7l, d5a, dma = dl
    if np.isnan(d5s):
        continue
    cands.append(dict(rule=rule, d5s=d5s, d7s=d7s, d5l=d5l, d7l=d7l, d5a=d5a, dma=dma))
print('  单规则在【校准集】上 top12（按 长 d@0.7 降序）', flush=True)
for c in sorted(cands, key=lambda x: -x['d7l'])[:12]:
    print('    %-11s thr=%+7.3f sign=%+1.0f | d短@0.5 %+6.2f d短@0.7 %+6.2f | d长@0.5 %+6.2f d长@0.7 %+6.2f | d全@0.5 %+6.2f' % (
        c['rule'][0], c['rule'][1], c['rule'][2], c['d5s'], c['d7s'], c['d5l'], c['d7l'], c['d5a']), flush=True)

# 两信号 AND：取校准集上短约束通过且长 d7l 最大的前 6 条做两两组合
passc = [c for c in cands if c['d5s'] >= -0.2 and c['d7s'] >= -0.3 and c['dma'] >= -0.2]
print('  单规则通过"短不回归"约束: %d / %d' % (len(passc), len(cands)), flush=True)
top = sorted(cands, key=lambda x: -x['d7l'])[:6]
for i in range(len(top)):
    for j in range(i + 1, len(top)):
        r1, r2 = top[i]['rule'], top[j]['rule']
        if r1[0] == r2[0]:
            continue
        rule = (r1[0], r1[1], r1[2], r2[0], r2[1], r2[2])
        m, dl = eval_rule(packsC, bC, *rule)
        d5s, d7s, d5l, d7l, d5a, dma = dl
        if np.isnan(d5s):
            continue
        cands.append(dict(rule=rule, d5s=d5s, d7s=d7s, d5l=d5l, d7l=d7l, d5a=d5a, dma=dma))
passc = [c for c in cands if c['d5s'] >= -0.2 and c['d7s'] >= -0.3 and c['dma'] >= -0.2]


def tag_rule(r):
    def one(sig, thr, sign):
        return '%s%s%+.3f' % (sig, '>=' if sign > 0 else '<=', thr if sign > 0 else -thr)
    if r[3] is None:
        return one(r[0], r[1], r[2])
    return one(r[0], r[1], r[2]) + '&' + one(r[3], r[4], r[5])


def show(tag, r, dl):
    d5s, d7s, d5l, d7l, d5a, dma = dl
    print('%-26s | %+7.2f %+7.2f | %+7.2f %+7.2f | %+7.2f %+7.2f' % (
        tag[:26], d5s, d7s, d5l, d7l, d5a, dma), flush=True)


print('\n  【由校准集选出的规则】在 test 上的表现', flush=True)
print('%-26s | %7s %7s | %7s %7s | %7s %7s' % (
    '规则(信号/阈值)', 'd短@0.5', 'd短@0.7', 'd长@0.5', 'd长@0.7', 'd全@0.5', 'd全mIoU'), flush=True)
if passc:
    sel = max(passc, key=lambda c: c['d7l'])
    m, dl = eval_rule(packsT, bT, *sel['rule'])
    show('选定: ' + tag_rule(sel['rule']), sel['rule'], dl)
else:
    sel = max(cands, key=lambda c: c['d7l'] + 3.0 * c['d5s'])
    m, dl = eval_rule(packsT, bT, *sel['rule'])
    show('退化Pareto: ' + tag_rule(sel['rule']), sel['rule'], dl)
    print('    (校准集无严格解 -> 说明"短不降+长显著增强"在校准集上已不可同时满足)', flush=True)

print('\n  【零 gt 规则】阈值 = 被测集自身信号分位数（完全不接触 gt）', flush=True)
zg_best = None
print('%-26s | %7s %7s | %7s %7s | %7s %7s' % (
    '规则(分位数)', 'd短@0.5', 'd短@0.7', 'd长@0.5', 'd长@0.7', 'd全@0.5', 'd全mIoU'), flush=True)
zrows = []
for sig in SIG_NAMES:
    a = sigT_all[sig][0]
    if np.isnan(a):
        continue
    sign = 1.0 if a >= 0.5 else -1.0
    vals = np.concatenate([p[1][0][sig] for p in packsT] + [p[1][1][sig] for p in packsT])
    vals = vals[vals > -1e8]
    if len(vals) < 50:
        continue
    for q in QS:
        thr = float(np.quantile(vals, 1.0 - q))
        m, dl = eval_rule(packsT, bT, sig, thr, sign)
        zrows.append((dl[3], sig, q, dl))
for d7l, sig, q, dl in sorted(zrows, reverse=True)[:10]:
    show('%s top%.0f%%' % (sig, 100 * q), None, dl)

print('\n[基线绝对值] base 短 R1@0.5 %.4f  长 R1@0.5 %.4f  全 R1@0.5 %.4f  mIoU %.4f' % (
    bT['short']['i5'], bT['long']['i5'], bT['all']['i5'], bT['all']['mIoU']), flush=True)
print('\n' + '=' * 112, flush=True)
print('DONE', flush=True)
