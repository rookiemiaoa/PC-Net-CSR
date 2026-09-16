# -*- coding: utf-8 -*-
"""w11_eval_final.py —— 最终算子的权威评测（服务器 GPU，同权重/harness/数据）。

被测算子（全部推理期、无 gt、零训练、零参数）
  base   : PC-Net top-1 提案框
  CSR    : 已发表的中心重定位（单窗质心，β=0.20）
  ISR    : 迭代收缩窗中心估计（mean-shift 式：窗宽 β_{t+1}=γ·β_t）
  ISR+W  : ISR 之上再加「宽度增益 ω·d」（ω 在 train 上选定，见 W10）

为什么 ISR 而不是放大位移
  W6-C1 实测：δ = gt中心−框中心 对 d0=c1−cb 的回归斜率 = 1.76 ⇒ 窗内质心把真实位移
  压缩到 57%（窗中心拖动造成的收缩偏置）。直接放大（σ=1.76）会同时放大噪声、
  实测更差；缩小窗宽迭代则同时降偏置、不增噪。

输出：mean±std over K_REP、长短分桶、配对 bootstrap 95%CI、McNemar 跨阈值计数。
用法: python w11_eval_final.py <config> <ckpt>
环境变量: BS=16 K_REP=3 THR=0.35 LIMIT=0
"""
import math
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

REF = {'CharadesSTA': dict(mIoU=0.4712, r1_05=0.5484, r1_07=0.2668)}

cfg = load_json(sys.argv[1]); ckpt = sys.argv[2]
dargs = cfg['dataset']; DS_NAME = dargs['dataset']
vocab = pickle.load(open(dargs['vocab_path'], 'rb'))
ds = getattr(da, DS_NAME)(data_path=dargs['test_data'], vocab=vocab, args=dargs, split='test')
COLLATE = ds.collate_data
FP = os.environ.get('FEAT_PATH', dargs['feature_path'])
if not os.path.isabs(FP) and not os.path.exists(FP):
    for cand in (os.path.join(DATA_ROOT, os.path.basename(FP)),
                 os.path.join(ROOT, os.path.basename(FP))):
        if os.path.exists(cand):
            FP = cand; break
dargs['feature_path'] = FP
LIMIT = int(os.environ.get('LIMIT', 0))
if LIMIT > 0:
    ds = Subset(ds, list(range(min(LIMIT, len(ds)))))

sd = torch.load(ckpt, map_location='cuda'); sp = sd['model_parameters']
mcfg = dict(sd.get('config', {}).get('model', {}).get('config', cfg['model']['config']))
P = int(sp['slot_atten.slots'].shape[0]) if 'slot_atten.slots' in sp else 8
mcfg.update(num_props=P, vocab_size=ds.vocab_size, max_epoch=cfg['train']['max_num_epochs'])
mcfg.setdefault('num_iteration', 4)

L = dargs['max_num_frames'] // 4
CG = (np.arange(L) + 0.5) / L
BS = int(os.environ.get('BS', 16)); K_REP = int(os.environ.get('K_REP', 3))
THR = float(os.environ.get('THR', 0.35))
loader = DataLoader(ds, batch_size=BS, shuffle=False, collate_fn=COLLATE, num_workers=8)
print('[GPU] %s' % torch.cuda.get_device_name(0), flush=True)
print('N=%d  num_props=%d  L=%d  K_REP=%d' % (len(ds), P, L, K_REP), flush=True)

# ---------------------------------------------------------------- 算子（自包含）


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


centers_g = CG


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
    """迭代收缩窗的中心位移（返回 d）。门控仍以最终框做自洽检验。"""
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


def boxes(c, w):
    return np.clip(np.asarray(c) - np.asarray(w) / 2, 0, 1), np.clip(np.asarray(c) + np.asarray(w) / 2, 0, 1)


METHODS = ['base', 'CSR', 'ISR(g.5,T2)', 'ISR(g.7,T3)', 'ISR+W(w=-.25)']


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
    GS, GE = G[:, 0], G[:, 1]; GL = GE - GS
    j1 = np.argsort(N8, 1)[:, 0]
    cb, wb = C8[r, j1], W8[r, j1]
    bL, bH = np.clip(cb - wb / 2, 0, 1), np.clip(cb + wb / 2, 0, 1)

    out = {'base': (bL, bH)}
    c_csr = csr_center(REL, cb, wb)
    out['CSR'] = boxes(c_csr, wb)
    d1 = isr_delta(REL, cb, wb, 0.20, 0.50, 2)
    out['ISR(g.5,T2)'] = boxes(np.clip(cb + d1, 0, 1), wb)
    d2 = isr_delta(REL, cb, wb, 0.20, 0.70, 3)
    out['ISR(g.7,T3)'] = boxes(np.clip(cb + d2, 0, 1), wb)
    # 宽度增益 ω=-0.25：a=σ-ω/2, b=σ+ω/2, σ=1
    a_, b_ = 1.0 - (-0.25) / 2.0, 1.0 + (-0.25) / 2.0
    out['ISR+W(w=-.25)'] = (np.clip((cb - wb / 2) + a_ * d1, 0, 1), np.clip((cb + wb / 2) + b_ * d1, 0, 1))
    res = {k: iou(lo, hi, GS, GE) for k, (lo, hi) in out.items()}
    return GL, res, dict(cb=cb, wb=wb)


RES = {}; GLS = None; DIAG = None
for rep in range(K_REP):
    GL, res, dg = run_rep(rep)
    GLS = GL; DIAG = dg
    for k, v in res.items():
        RES.setdefault(k, []).append(v)
    print('  rep%d done | base mIoU=%.4f | CSR %.4f | ISR(g.5,T2) %.4f' % (
        rep, res['base'].mean(), res['CSR'].mean(), res['ISR(g.5,T2)'].mean()), flush=True)


def mets(v):
    return dict(mIoU=float(v.mean()), r5=float((v >= 0.5).mean()), r7=float((v >= 0.7).mean()))


print('\n' + '=' * 118, flush=True)
print('[复现校验] base vs 官方日志', flush=True)
for nm, key, ref in (('mIoU', 'mIoU', REF['CharadesSTA']['mIoU']), ('R1@0.5', 'r5', 0.5484), ('R1@0.7', 'r7', 0.2668)):
    got = float(np.mean([mets(v)[key] for v in RES['base']]))
    print('  %-8s 官方=%.4f  本脚本=%.4f  (%+.4f)  %s' % (nm, ref, got, got - ref, 'OK' if abs(got - ref) < 0.02 else 'DIFF'), flush=True)

print('\n[主表] mean±std over %d rep（论文口径三列）+ Δ vs base' % K_REP, flush=True)
print('%-18s %16s %16s %16s | %9s %9s %9s' % ('', 'R1@0.5', 'R1@0.7', 'mIoU', 'Δ@0.5', 'Δ@0.7', 'ΔmIoU'), flush=True)
b = {k: float(np.mean([mets(v)[k] for v in RES['base']])) for k in ('mIoU', 'r5', 'r7')}
for k in METHODS:
    vs = RES[k]
    ms = {kk: (float(np.mean([mets(v)[kk] for v in vs])), float(np.std([mets(v)[kk] for v in vs]))) for kk in ('mIoU', 'r5', 'r7')}
    print('%-18s %8.4f±%.4f %8.4f±%.4f %8.4f±%.4f | %+9.2f %+9.2f %+9.2f' % (
        k, ms['r5'][0], ms['r5'][1], ms['r7'][0], ms['r7'][1], ms['mIoU'][0], ms['mIoU'][1],
        100 * (ms['r5'][0] - b['r5']), 100 * (ms['r7'][0] - b['r7']), 100 * (ms['mIoU'][0] - b['mIoU'])), flush=True)

print('\n[长短分桶] gt 时长占比 <= %.2f 为短' % THR, flush=True)
print('%-18s %-6s %6s %9s %9s %9s' % ('方法', '桶', 'n', 'mIoU', 'R1@0.5', 'R1@0.7'), flush=True)
for k in METHODS:
    for nm, m in (('短', GLS <= THR), ('长', GLS > THR), ('全', np.ones(len(GLS), bool))):
        v = np.concatenate([RES[k][i][m] for i in range(K_REP)])
        print('%-18s %-6s %6d %9.4f %9.4f %9.4f' % (k, nm, int(m.sum()), v.mean(), (v >= 0.5).mean(), (v >= 0.7).mean()), flush=True)

print('\n[显著性] 配对 bootstrap B=2000 + McNemar', flush=True)
rng = np.random.default_rng(0); B = 2000
for k in ['CSR', 'ISR(g.5,T2)', 'ISR(g.7,T3)', 'ISR+W(w=-.25)']:
    for bucket, m in (('ALL', np.ones(len(GLS), bool)), ('SHORT', GLS <= THR), ('LONG', GLS > THR)):
        dm, d5, d7 = [], [], []
        cnt = [0, 0, 0, 0]
        for i in range(K_REP):
            ib = RES['base'][i][m]; ic = RES[k][i][m]
            n_ = len(ib); idx = rng.integers(0, n_, size=(B, n_))
            dm.append(ic[idx].mean(1) - ib[idx].mean(1))
            d5.append((ic[idx] >= 0.5).mean(1) - (ib[idx] >= 0.5).mean(1))
            d7.append((ic[idx] >= 0.7).mean(1) - (ib[idx] >= 0.7).mean(1))
            cnt[0] += int(((ib < 0.5) & (ic >= 0.5)).sum()); cnt[1] += int(((ib >= 0.5) & (ic < 0.5)).sum())
            cnt[2] += int(((ib < 0.7) & (ic >= 0.7)).sum()); cnt[3] += int(((ib >= 0.7) & (ic < 0.7)).sum())
        dm = np.concatenate(dm); d5 = np.concatenate(d5); d7 = np.concatenate(d7)
        print('  %-16s %-6s ΔmIoU %+6.3f [%+6.3f,%+6.3f] | Δ@.5 %+6.3f [%+6.3f,%+6.3f] | Δ@.7 %+6.3f [%+6.3f,%+6.3f]' % (
            k, bucket, 100 * dm.mean(), 100 * np.percentile(dm, 2.5), 100 * np.percentile(dm, 97.5),
            100 * d5.mean(), 100 * np.percentile(d5, 2.5), 100 * np.percentile(d5, 97.5),
            100 * d7.mean(), 100 * np.percentile(d7, 2.5), 100 * np.percentile(d7, 97.5)), flush=True)
        for tn, up, dn in (('@0.5', cnt[0], cnt[1]), ('@0.7', cnt[2], cnt[3])):
            st = ((abs(up - dn) - 1) ** 2 / float(up + dn)) if (up + dn) > 0 else 0.0
            print('        McNemar %s 升(%d) 降(%d) chi2=%.2f p=%.2e' % (tn, up, dn, st, math.erfc(math.sqrt(st / 2.0))), flush=True)

# ISR vs CSR 直接配对（关键：迭代是否显著优于已发表的 CSR）
print('\n[配对] ISR(g.5,T2) vs CSR（同一 top-1 框，唯一差别是中心估计器）', flush=True)
for bucket, m in (('ALL', np.ones(len(GLS), bool)), ('SHORT', GLS <= THR), ('LONG', GLS > THR)):
    dm, d5, d7 = [], [], []; up5 = dn5 = up7 = dn7 = 0
    for i in range(K_REP):
        ib = RES['CSR'][i][m]; ic = RES['ISR(g.5,T2)'][i][m]
        n_ = len(ib); idx = rng.integers(0, n_, size=(B, n_))
        dm.append(ic[idx].mean(1) - ib[idx].mean(1)); d5.append((ic[idx] >= 0.5).mean(1) - (ib[idx] >= 0.5).mean(1)); d7.append((ic[idx] >= 0.7).mean(1) - (ib[idx] >= 0.7).mean(1))
        up5 += int(((ib < 0.5) & (ic >= 0.5)).sum()); dn5 += int(((ib >= 0.5) & (ic < 0.5)).sum())
        up7 += int(((ib < 0.7) & (ic >= 0.7)).sum()); dn7 += int(((ib >= 0.7) & (ic < 0.7)).sum())
    dm = np.concatenate(dm); d5 = np.concatenate(d5); d7 = np.concatenate(d7)
    print('  %-6s ΔmIoU %+6.3f [%+6.3f,%+6.3f] | Δ@.5 %+6.3f [%+6.3f,%+6.3f] | Δ@.7 %+6.3f [%+6.3f,%+6.3f]' % (
        bucket, 100 * dm.mean(), 100 * np.percentile(dm, 2.5), 100 * np.percentile(dm, 97.5),
        100 * d5.mean(), 100 * np.percentile(d5, 2.5), 100 * np.percentile(d5, 97.5),
        100 * d7.mean(), 100 * np.percentile(d7, 2.5), 100 * np.percentile(d7, 97.5)), flush=True)
    for tn, up, dn in (('@0.5', up5, dn5), ('@0.7', up7, dn7)):
        st = ((abs(up - dn) - 1) ** 2 / float(up + dn)) if (up + dn) > 0 else 0.0
        print('        McNemar %s 升(%d) 降(%d) chi2=%.2f p=%.2e' % (tn, up, dn, st, math.erfc(math.sqrt(st / 2.0))), flush=True)

print('\n[诊断] 框宽与时长', flush=True)
print('  基线框宽中位=%.4f  gt 时长中位=%.4f  corr(w,g)=%.4f' % (
    np.median(DIAG['wb']), np.median(GLS), np.corrcoef(DIAG['wb'], GLS)[0, 1]), flush=True)
print('=' * 118, flush=True)
print('DONE', flush=True)
