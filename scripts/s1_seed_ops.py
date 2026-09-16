# -*- coding: utf-8 -*-
"""s1_seed_ops.py —— 为「多 seed 训练」「统计显著性检验」导出逐样本 IoU 向量。

与 w30_main_ops.py **同一 harness、同一算子定义、同一数据/权重**；唯一差别：
把每个 rep 的**逐样本 IoU 向量**落盘为 npz，供 s2_stats.py 做正式检验与跨 seed 聚合。
（不落盘逐样本就只能在服务器上算，无法跨 seed 复用同一份检验代码。）

被测算子（全部推理期、无 gt、零训练、零参数）
  base     : PC-Net top-1 提案框（8 槽 NLL argmin）
  CSR      : 单窗质心中心重定位（已发表，β=0.20）
  ISR      : 迭代收缩窗中心估计（mean-shift 式，γ=0.5, T=2）
  RBR      : 相关性边界泛洪扩框（k=0.15，宽度轴对照）

用法: python s1_seed_ops.py <config> <ckpt> <tag>
环境: BS=16 K_REP=3 OUT_NPZ=/path/to.npz
"""
import os
import pickle
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = _HERE
while _ROOT != os.path.dirname(_ROOT) and not os.path.exists(os.path.join(_ROOT, 'models', 'pcnet.py')):
    _ROOT = os.path.dirname(_ROOT)
ROOT = os.environ.get('PCNET_ROOT') or _ROOT
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from csr.paths import DATA_ROOT        # noqa: E402

from utils import load_json            # noqa: E402
import datasets as da                  # noqa: E402
from models import PCNet               # noqa: E402
from models.loss import cal_nll_loss   # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

cfg = load_json(sys.argv[1]); ckpt = sys.argv[2]
TAG = sys.argv[3] if len(sys.argv) > 3 else 's1'
dargs = cfg['dataset']; DS_NAME = dargs['dataset']
if os.environ.get('TEST_DATA'):
    dargs['test_data'] = os.environ['TEST_DATA']
vocab = pickle.load(open(dargs['vocab_path'], 'rb'))
# 与官方 runners/main_runner.py:284 保持一致的类名解析（'Charades-CG' -> 'CharadesCG'）
_DS_CLS = getattr(da, DS_NAME.replace('-', ''), None)
if _DS_CLS is None:
    raise AttributeError('datasets 中找不到 %s (尝试 %s)' % (DS_NAME, DS_NAME.replace('-', '')))
ds = _DS_CLS(data_path=dargs['test_data'], vocab=vocab, args=dargs, split='test')
COLLATE = ds.collate_data
FP = os.environ.get('FEAT_PATH', dargs['feature_path'])
if not os.path.isabs(FP) and not os.path.exists(FP):
    for cand in (os.path.join(DATA_ROOT, os.path.basename(FP)),
                 os.path.join(ROOT, os.path.basename(FP))):
        if os.path.exists(cand):
            FP = cand; break
dargs['feature_path'] = FP
VOCAB = ds.vocab_size
LIMIT = int(os.environ.get('LIMIT', 0))
if LIMIT > 0:
    ds = Subset(ds, list(range(min(LIMIT, len(ds)))))

sd = torch.load(ckpt, map_location='cuda'); sp = sd['model_parameters']
mcfg = dict(sd.get('config', {}).get('model', {}).get('config', cfg['model']['config']))
P = int(sp['slot_atten.slots'].shape[0]) if 'slot_atten.slots' in sp else 8
mcfg.update(num_props=P, vocab_size=VOCAB, max_epoch=cfg['train']['max_num_epochs'])
mcfg.setdefault('num_iteration', 4)

L = dargs['max_num_frames'] // 4
CG = (np.arange(L) + 0.5) / L
BS = int(os.environ.get('BS', 16)); K_REP = int(os.environ.get('K_REP', 3))
KS = [float(x) for x in os.environ.get('KS', '0.15').split(',')]
# 默认落到 <repo>/csr/outputs/；可用 OUT_NPZ 或 CSR_OUTPUT_DIR 覆盖。
_OUTDIR = os.environ.get('CSR_OUTPUT_DIR') or os.path.join(ROOT, 'csr', 'outputs')
os.makedirs(_OUTDIR, exist_ok=True)
OUT_NPZ = os.environ.get('OUT_NPZ', os.path.join(_OUTDIR, 's1_%s.npz' % TAG))
loader = DataLoader(ds, batch_size=BS, shuffle=False, collate_fn=COLLATE, num_workers=8)
print('[GPU] %s' % torch.cuda.get_device_name(0), flush=True)
print('[DATA] %s split=%s N=%d P=%d L=%d K_REP=%d' % (
    DS_NAME, os.path.basename(dargs['test_data']), len(ds), P, L, K_REP), flush=True)

centers_g = CG


# ---------------------------------------------------------------- 算子（与 w30 逐字一致）
def iou(a0, a1, b0, b1):
    a0, a1, b0, b1 = map(np.asarray, (a0, a1, b0, b1))
    inter = np.maximum(0.0, np.minimum(a1, b1) - np.maximum(a0, b0))
    union = np.maximum(a1, b1) - np.minimum(a0, b0)
    return np.where(union > 1e-9, inter / np.maximum(union, 1e-9), 0.0)


def smooth_curve(z, window):
    z = np.asarray(z, dtype=np.float64)
    if window <= 1:
        return z
    pad = window // 2
    zp = np.concatenate([np.repeat(z[:, :1], pad, 1), z, np.repeat(z[:, -1:], pad, 1)], 1)
    cs = np.concatenate([np.zeros((len(z), 1)), np.cumsum(zp, 1)], 1)
    return (cs[:, window:] - cs[:, :-window]) / float(window)


def zscore(rel):
    rel = np.asarray(rel, dtype=np.float64)
    return (rel - rel.mean(1, keepdims=True)) / (rel.std(1, keepdims=True) + 1e-8)


def inside_score(z, lo, hi, centers):
    z = np.asarray(z); centers = np.asarray(centers); T = z.shape[1]
    cs = np.concatenate([np.zeros((len(z), 1)), np.cumsum(z, 1)], 1)
    li = np.clip(np.searchsorted(centers, lo), 0, T - 1)
    hi_i = np.clip(np.searchsorted(centers, hi, side='right') - 1, 0, T - 1)
    cnt = np.maximum(hi_i - li + 1, 1)
    return (cs[np.arange(len(z)), hi_i + 1] - cs[np.arange(len(z)), li]) / cnt


def est_centroid(z, cb, beta):
    m = np.abs(centers_g[None, :] - np.asarray(cb)[:, None]) <= beta
    pm = np.where(m, np.maximum(z, 0.0), 0.0)
    s = pm.sum(1)
    c = (pm @ centers_g) / np.maximum(s, 1e-12)
    return np.where(s < 1e-12, cb, c)


def csr_center(z_raw, cb, wb, beta=0.20, rho=0.10, alpha=1.0, dmin=0.01):
    zs = smooth_curve(zscore(z_raw), 7)
    c1 = est_centroid(zs, cb, beta)
    d = c1 - cb
    d = np.where(np.abs(d) < rho * wb, 0.0, d) * alpha
    c2 = np.clip(cb + d, 0, 1)
    if dmin is not None:
        din = inside_score(zs, c2 - wb / 2, c2 + wb / 2, centers_g) - inside_score(zs, cb - wb / 2, cb + wb / 2, centers_g)
        c2 = np.where(din >= dmin, c2, cb)
    return c2


def isr_delta(z_raw, cb, wb, beta0=0.20, gamma=0.5, T=2, rho=0.10, alpha=1.0, dmin=0.01):
    zs = smooth_curve(zscore(z_raw), 7)
    c = np.asarray(cb).copy(); b = beta0
    for _ in range(T):
        c = est_centroid(zs, c, b)
        b = b * gamma
    d = c - cb
    d = np.where(np.abs(d) < rho * wb, 0.0, d) * alpha
    c2 = np.clip(cb + d, 0, 1)
    if dmin is not None:
        din = inside_score(zs, c2 - wb / 2, c2 + wb / 2, centers_g) - inside_score(zs, cb - wb / 2, cb + wb / 2, centers_g)
        d = np.where(din >= dmin, d, 0.0)
    return d


def flood_expand(z_raw, cb, k):
    """RBR：自框中心向外泛洪，相关性 z>=k 处继续扩张；返回并集边界。"""
    zs = smooth_curve(zscore(z_raw), 7)
    n = len(zs)
    lo = np.zeros(n); hi = np.zeros(n)
    for i in range(n):
        t0 = int(np.clip(round(cb[i] * L - 0.5), 0, L - 1))
        a = t0
        while a - 1 >= 0 and zs[i, a - 1] >= k:
            a -= 1
        b = t0
        while b + 1 < L and zs[i, b + 1] >= k:
            b += 1
        lo[i] = a / L; hi[i] = (b + 1) / L
    return lo, hi


def boxes(c, w):
    return np.clip(np.asarray(c) - np.asarray(w) / 2, 0, 1), np.clip(np.asarray(c) + np.asarray(w) / 2, 0, 1)


METHODS = ['base', 'CSR', 'ISR(g.5,T2)'] + ['RBR k=%.2f' % k for k in KS]


def run_rep(rep):
    np.random.seed(7 + rep); torch.manual_seed(7 + rep)
    model = PCNet(mcfg).cuda(); model.load_state_dict(sp); model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    _pg = model.proposal_generator
    _c = torch.linspace(0.5 / L, 1 - 0.5 / L, L); _w = torch.full((L,), max(2.0 / L, 0.02))
    REL, G, N8, C8, W8 = [], [], [], [], []
    t0 = time.time()
    with torch.no_grad():
        for bi, batch in enumerate(loader, 1):
            ni = {k: v.cuda() for k, v in batch['net_input'].items()}
            bsz = len(batch['raw'])
            o = model(epoch=0, **ni)
            wm = o['words_mask'].unsqueeze(1).expand(bsz, P, -1).contiguous().view(bsz * P, -1)
            wi = o['words_id'].unsqueeze(1).expand(bsz, P, -1).contiguous().view(bsz * P, -1)
            n, _ = cal_nll_loss(o['words_logit'], wi, wm)
            N8.append(n.view(bsz, P).cpu().numpy())
            C8.append(o['center'].view(bsz, P).cpu().numpy())
            W8.append(o['width'].view(bsz, P).cpu().numpy())
            model.proposal_generator = lambda h, b: (_c.repeat(b).cuda(), _w.repeat(b).cuda())
            model.num_props = L
            o2 = model(epoch=0, **ni)
            wm2 = o2['words_mask'].unsqueeze(1).expand(bsz, L, -1).contiguous().view(bsz * L, -1)
            wi2 = o2['words_id'].unsqueeze(1).expand(bsz, L, -1).contiguous().view(bsz * L, -1)
            n2, _ = cal_nll_loss(o2['words_logit'], wi2, wm2)
            REL.append((-n2.view(bsz, L)).cpu().numpy())
            model.proposal_generator = _pg; model.num_props = P
            dur = np.asarray([i[1] for i in batch['raw']], dtype=np.float64)
            ts = np.asarray([i[2] for i in batch['raw']], dtype=np.float64)
            G.append(ts / dur[:, None])
            if bi % 100 == 0:
                print('    rep%d %d/%d %.1fs' % (rep, bi, len(loader), time.time() - t0), flush=True)
    G = np.concatenate(G); N8 = np.concatenate(N8); C8 = np.concatenate(C8); W8 = np.concatenate(W8)
    REL = np.concatenate(REL)
    n = len(G); r = np.arange(n)
    GS, GE = G[:, 0], G[:, 1]
    j1 = np.argsort(N8, 1)[:, 0]
    cb, wb = C8[r, j1], W8[r, j1]
    bL, bH = np.clip(cb - wb / 2, 0, 1), np.clip(cb + wb / 2, 0, 1)

    out = {'base': (bL, bH)}
    out['CSR'] = boxes(csr_center(REL, cb, wb), wb)
    d1 = isr_delta(REL, cb, wb, 0.20, 0.50, 2)
    out['ISR(g.5,T2)'] = boxes(np.clip(cb + d1, 0, 1), wb)
    for k in KS:
        s, e = flood_expand(REL, cb, k)
        out['RBR k=%.2f' % k] = (np.minimum(s, bL), np.maximum(e, bH))
    res = {k: iou(lo, hi, GS, GE) for k, (lo, hi) in out.items()}
    return {k: np.asarray(v, dtype=np.float64) for k, v in res.items()}, \
           dict(cb=cb, wb=wb, gs=GS, ge=GE)


ACC = {}; DIAG = None
for rep in range(K_REP):
    res, dg = run_rep(rep)
    DIAG = dg
    for k, v in res.items():
        ACC.setdefault(k, []).append(v)
    print('  rep%d done | base %.4f | CSR %.4f | ISR %.4f | RBR %.4f' % (
        rep, res['base'].mean(), res['CSR'].mean(), res['ISR(g.5,T2)'].mean(),
        res['RBR k=%.2f' % KS[0]].mean()), flush=True)

save = {}
for k, vs in ACC.items():
    save['iou__' + k] = np.stack(vs, 0)          # (K_REP, n)
save['gs'] = DIAG['gs']; save['ge'] = DIAG['ge']
save['cb'] = DIAG['cb']; save['wb'] = DIAG['wb']
save['meta__N'] = np.array([len(DIAG['gs'])])
save['meta__K_REP'] = np.array([K_REP])
os.makedirs(os.path.dirname(OUT_NPZ), exist_ok=True)
np.savez_compressed(OUT_NPZ, **save)
print('[mIoU over %d rep] ' % K_REP + '  '.join(
    '%s=%.4f' % (k, np.mean([v.mean() for v in ACC[k]])) for k in ACC), flush=True)
print('WROTE %s' % OUT_NPZ, flush=True)
print('DONE', flush=True)
