# -*- coding: utf-8 -*-
"""diag41_gen.py —— ANET-CG 上 CSR 的参数敏感性根因诊断 + 宽度自适应修复

问题（diag40 / csr_anet_sens 已实测）
  ANET-CG 上 CSR 10 档标量扫描 ΔmIoU 全负（10/10），Δ@0.5 在 -0.06~+0.26 间抖动，
  短桶 -0.60 mIoU / 长桶 +0.57 mIoU。=> 不是"没调好"，而是**结构性失配**。
  几何：基线 top-1 框宽中位 0.9908，gt 时长中位 0.2750（框宽 ≈ 3.6× 目标）。
  框远宽于目标时 IoU≈gt_len/box_len 由**宽度**决定，宽度保持型算子无从下手。

本脚本要回答三件事（全部用真实 ckpt、真实特征、服务器实跑）
  Q1 该 regime 的可改进空间在中心还是在宽度？—— ORACLE_c / ORACLE_w / ORACLE_cw
  Q2 若在宽度：从相关性曲线估计"片段跨度"能否复现该增益？—— 支撑区间估计器
  Q3 该修复是否**在 Charades 上退化为 no-op**（不破坏既有 CSR 增益）？—— 门控 kappa

新算子家族（零训练、推理期、不使用 gt）
  sup_grow : 从相关性峰值向两侧生长，只要 Z >= a*Z[peak]           （峰值相对阈值）
  sup_mass : 从峰值向两侧生长，直到累计正质量 >= q*总正质量        （质量分位区间）
  CSRW_*   : 宽度自适应 —— 当 wb > kappa*w_r 时把宽度换成 w_r（中心沿用 CSR 中心）
  CSRWc_*  : 同上，但中心也换成支撑区间中心（对照）

用法:
  python diag41_gen.py <config_path> <ckpt_path>
环境变量:
  BS=16 K_REP=3 LIMIT=0 SMOOTH=7 BETA=0.20 RHO=0.10 ALPHA=1.0 DMIN=0.01 THR=0.35
"""
import math
import os
import pickle
import sys
import time

import numpy as np
import torch

# ---- 路径解析：自动定位本仓库根目录；可用 LM_ROOT 覆盖（与 oracle_gate.py 一致）----
_HERE = os.path.dirname(os.path.abspath(__file__))
_AUTO = _HERE
while _AUTO != os.path.dirname(_AUTO) and not os.path.exists(os.path.join(_AUTO, 'models', 'pcnet.py')):
    _AUTO = os.path.dirname(_AUTO)
ROOT = os.environ.get('LM_ROOT', _AUTO)
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from utils import load_json            # noqa: E402
import datasets as da                  # noqa: E402
from models import PCNet               # noqa: E402
from models.loss import cal_nll_loss   # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

REF = {
    'CharadesSTA': dict(mIoU=0.4712, r1_05=0.5484, r1_07=0.2668),
    'ActivityNet': dict(mIoU=0.3645, r1_05=0.2962, r1_07=0.1435),
    'TACoS': dict(mIoU=0.0979, r1_05=0.0400, r1_07=0.0117),
}

cfg = load_json(sys.argv[1])
ckpt = sys.argv[2]
dargs = cfg['dataset']
DS_NAME = dargs['dataset']
vocab = pickle.load(open(dargs['vocab_path'], 'rb'))
ds = getattr(da, DS_NAME)(data_path=dargs['test_data'], vocab=vocab, args=dargs, split='test')
VOCAB_SIZE = ds.vocab_size
COLLATE = ds.collate_data

FP = os.environ.get('FEAT_PATH', dargs['feature_path'])
if not os.path.isabs(FP) and not os.path.exists(FP):
    for cand in (os.path.join(os.environ.get('FEAT_DIR', ROOT), os.path.basename(FP)),
                 os.path.join(ROOT, os.path.basename(FP))):
        if os.path.exists(cand):
            FP = cand
            break
dargs['feature_path'] = FP
print('feature_path=%s  exists=%s' % (FP, os.path.exists(FP)), flush=True)

LIMIT = int(os.environ.get('LIMIT', 0))
if LIMIT > 0:
    ds = Subset(ds, list(range(min(LIMIT, len(ds)))))

sd = torch.load(ckpt, map_location='cuda')
sp = sd['model_parameters']
ck_cfg = sd.get('config', {})
mcfg = dict(ck_cfg.get('model', {}).get('config', cfg['model']['config']))
P = int(sp['slot_atten.slots'].shape[0]) if 'slot_atten.slots' in sp else mcfg.get('num_props', 8)
mcfg['num_props'] = P
mcfg['vocab_size'] = VOCAB_SIZE
mcfg['max_epoch'] = cfg['train']['max_num_epochs']
mcfg.setdefault('num_iteration', 4)

L = dargs['max_num_frames'] // 4
CG = (np.arange(L) + 0.5) / L
BS = int(os.environ.get('BS', 16))
K_REP = int(os.environ.get('K_REP', 3))
SMOOTH = int(os.environ.get('SMOOTH', 7))
BETA = float(os.environ.get('BETA', 0.20))
RHO = float(os.environ.get('RHO', 0.10))
ALPHA = float(os.environ.get('ALPHA', 1.0))
DMIN = float(os.environ.get('DMIN', 0.01))
THR = float(os.environ.get('THR', 0.35))
loader = DataLoader(ds, batch_size=BS, shuffle=False, collate_fn=COLLATE, num_workers=8)
N = len(ds)
print('cfg=%s' % sys.argv[1], flush=True)
print('ckpt=%s' % ckpt, flush=True)
print('dataset=%s  N=%d  num_props=%d  L=%d  BS=%d  K_REP=%d' % (DS_NAME, N, P, L, BS, K_REP), flush=True)


# ---------------- 基础工具 ----------------
def iou(a0, a1, b0, b1):
    inter = np.maximum(0.0, np.minimum(a1, b1) - np.maximum(a0, b0))
    union = np.maximum(a1, b1) - np.minimum(a0, b0)
    return np.where(union > 1e-9, inter / np.maximum(union, 1e-9), 0.0)


def smoothZ(Z, w):
    if w <= 1:
        return Z
    pad = w // 2
    Zp = np.concatenate([np.repeat(Z[:, :1], pad, axis=1), Z,
                         np.repeat(Z[:, -1:], pad, axis=1)], axis=1)
    cs = np.concatenate([np.zeros((len(Z), 1)), np.cumsum(Zp, axis=1)], axis=1)
    return (cs[:, w:] - cs[:, :-w]) / float(w)


def inside_score(Z, lo, hi):
    cs = np.concatenate([np.zeros((len(Z), 1)), np.cumsum(Z, axis=1)], axis=1)
    li = np.clip(np.searchsorted(CG, lo), 0, L - 1)
    hii = np.clip(np.searchsorted(CG, hi, side='right') - 1, 0, L - 1)
    cnt = np.maximum(hii - li + 1, 1)
    tot = cs[np.arange(len(Z)), hii + 1] - cs[np.arange(len(Z)), li]
    return tot / cnt


def est_win_vec(Z, cb, beta):
    m = np.abs(CG[None, :] - cb[:, None]) <= beta
    Pmat = np.where(m, np.maximum(Z, 0.0), 0.0)
    s = Pmat.sum(1)
    c = (Pmat @ CG) / np.maximum(s, 1e-12)
    bad = s < 1e-12
    if bad.any():
        c[bad] = cb[bad]
    return c


def boxes_cw(c, w):
    return np.clip(c - w / 2.0, 0, 1), np.clip(c + w / 2.0, 0, 1)


def csr_center(Z_raw, cb, wb):
    """CSR 的中心修正（宽度保持），返回 (c2, moved, shift)。"""
    Zr = (Z_raw - Z_raw.mean(1, keepdims=True)) / (Z_raw.std(1, keepdims=True) + 1e-8)
    Z = smoothZ(Zr, SMOOTH)
    c1 = est_win_vec(Z, cb, BETA)
    d = c1 - cb
    d = np.where(np.abs(d) < RHO * wb, 0.0, d)
    c2 = np.clip(cb + ALPHA * d, 0, 1)
    if DMIN is not None:
        din = inside_score(Z, c2 - wb / 2, c2 + wb / 2) - inside_score(Z, cb - wb / 2, cb + wb / 2)
        c2 = np.where(din >= DMIN, c2, cb)
    moved = np.abs(c2 - cb) > 1e-9
    return c2, moved, np.abs(c2 - cb), Z


# ---------------- 支撑区间估计器 ----------------
def peak_seed(Z):
    """峰值种子 = 最高值平台的质心索引（避免 argmax 在平台上偏向一侧）。"""
    mx = Z.max(1, keepdims=True)
    m = Z >= mx - 1e-12
    idx = np.arange(L)[None, :]
    cnt = m.sum(1)
    return np.clip((m * idx).sum(1) / np.maximum(cnt, 1), 0, L - 1).round().astype(int)


def sup_grow(Z, a):
    """从峰值向两侧生长，只要 Z >= a*Z[peak]。"""
    n = len(Z)
    p = peak_seed(Z)
    l = p.copy()
    r = p.copy()
    for i in range(n):
        thr = a * Z[i, p[i]]
        li = int(p[i])
        while li > 0 and Z[i, li - 1] >= thr:
            li -= 1
        ri = int(p[i])
        while ri < L - 1 and Z[i, ri + 1] >= thr:
            ri += 1
        l[i] = li
        r[i] = ri
    return l, r


def sup_mass(Z, q):
    """以峰值为中心**对称**生长，直到区间内累计正质量 >= q*总正质量。

    对称生长避免了"平顶曲线被 argmax 偏向一侧"的偏置，区间中心即为峰值中心。
    """
    n = len(Z)
    Pz = np.maximum(Z, 0.0)
    tot = Pz.sum(1)
    p = peak_seed(Z)
    l = p.copy()
    r = p.copy()
    for i in range(n):
        if tot[i] <= 1e-12:
            continue
        target = q * tot[i]
        acc = Pz[i, p[i]]
        h = 0
        while acc < target and (p[i] - h > 0 or p[i] + h < L - 1):
            h += 1
            if p[i] - h >= 0:
                acc += Pz[i, p[i] - h]
            if p[i] + h <= L - 1:
                acc += Pz[i, p[i] + h]
        l[i] = max(0, p[i] - h)
        r[i] = min(L - 1, p[i] + h)
    return l, r


def sup_center_width(l, r):
    c = (l + r + 1) / (2.0 * L)
    w = (r - l + 1) / float(L)
    return c, w


# ---------------- 指标累积 ----------------
ACC = {}
SPLIT = {}
STOR = {}
WRMED = []


def add(tag, i, gl=None, extra=None):
    d = {'mIoU': float(i.mean())}
    for t in (0.1, 0.3, 0.5, 0.7, 0.9):
        d['iou%.1f' % t] = float((i >= t).mean())
    ACC.setdefault(tag, []).append(d)
    if gl is not None:
        SPLIT.setdefault(tag, []).append((i, gl))
    if extra is not None:
        STOR.setdefault(tag, []).append(extra)


def run_rep(rep):
    np.random.seed(7 + rep)
    torch.manual_seed(7 + rep)
    model = PCNet(mcfg).cuda()
    model.load_state_dict(sp)
    model.eval()
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
            model.proposal_generator = _orig_pg
            model.num_props = P
            dur = np.asarray([i[1] for i in batch['raw']], dtype=np.float64)
            ts = np.asarray([i[2] for i in batch['raw']], dtype=np.float64)
            G.append(ts / dur[:, None])
            if bi % 100 == 0:
                print('    rep%d  batch %d/%d  %.1fs' % (rep, bi, len(loader), time.time() - t0), flush=True)

    G = np.concatenate(G)
    N8 = np.concatenate(N8)
    C8 = np.concatenate(C8)
    W8 = np.concatenate(W8)
    REL = np.concatenate(REL)
    n = len(G)
    r = np.arange(n)
    GS, GE = G[:, 0], G[:, 1]
    GL = GE - GS
    GC = (GS + GE) / 2.0
    order = np.argsort(N8, axis=1)
    j1 = order[:, 0]
    cb, wb = C8[r, j1], W8[r, j1]
    bL, bH = boxes_cw(cb, wb)

    i_base = iou(bL, bH, GS, GE)
    add('base', i_base, GL)

    # ---- ORACLE 上界（使用 gt，仅作可达性诊断）----
    oL, oH = boxes_cw(GC, wb)
    i_oc = iou(oL, oH, GS, GE)
    add('ORACLE_c', i_oc, GL)
    oL, oH = boxes_cw(cb, GL)
    i_ow = iou(oL, oH, GS, GE)
    add('ORACLE_w', i_ow, GL)
    oL, oH = boxes_cw(GC, GL)
    i_ocw = iou(oL, oH, GS, GE)
    add('ORACLE_cw', i_ocw, GL)

    # ---- CSR（宽度保持，现状）----
    c2, moved, shift, Z = csr_center(REL, cb, wb)
    lo, hi = boxes_cw(c2, wb)
    i_csr = iou(lo, hi, GS, GE)
    add('CSR', i_csr, GL)

    # ---- 宽度自适应家族 ----
    # 关键风险：相关性曲线若是尖峰，支撑区间宽度 -> 1/L，会把框压得过窄。
    # 因此引入宽度下限 FLOOR（占视频比例），并同时报告 w_r 分布。
    fam = [('g0.5', sup_grow(Z, 0.5)), ('m0.8', sup_mass(Z, 0.8))]
    WRmed = []
    for nm, (l, rr) in fam:
        cr, wr = sup_center_width(l, rr)
        WRmed.append(wr)
        ok = Z.max(1) > 1e-9
        for kap in (1.5, 2.5):
            for fl in (0.0, 0.12):
                wrf = np.maximum(wr, fl)
                use = ok & (wb > kap * wrf) & (wb > wrf + 1e-6)
                wout = np.where(use, wrf, wb)
                # 只换宽度，中心沿用 CSR 中心
                L2, H2 = boxes_cw(c2, wout)
                add('CW_%s_k%.1f_f%.2f' % (nm, kap, fl), iou(L2, H2, GS, GE), GL, extra=use)
                # 中心与宽度都换成支撑区间
                L3, H3 = boxes_cw(np.where(use, cr, c2), wout)
                add('CWc_%s_k%.1f_f%.2f' % (nm, kap, fl), iou(L3, H3, GS, GE), GL, extra=use)
    WRMED.append(np.concatenate(WRmed))

    return dict(n=n, moved=moved, shift=shift, GL=GL, wb=wb, cb=cb, GL2=GL,
                i_base=i_base, i_csr=i_csr, i_oc=i_oc, i_ow=i_ow, i_ocw=i_ocw,
                bL=bL, bH=bH, lo=lo, hi=hi)


diag = None
for rep in range(K_REP):
    d = run_rep(rep)
    if diag is None:
        diag = d
    print('  rep %d done | base mIoU=%.4f | CSR mIoU=%.4f | ORACLE_c mIoU=%.4f | ORACLE_w mIoU=%.4f' % (
        rep, ACC['base'][-1]['mIoU'], ACC['CSR'][-1]['mIoU'],
        ACC['ORACLE_c'][-1]['mIoU'], ACC['ORACLE_w'][-1]['mIoU']), flush=True)


def mean_of(tag, key):
    return float(np.mean([m[key] for m in ACC[tag]]))


def std_of(tag, key):
    return float(np.std([m[key] for m in ACC[tag]]))


line = '=' * 112
print('\n' + line, flush=True)
print('diag41  %s  N=%d  K_REP=%d  ckpt=%s' % (
    DS_NAME, diag['n'], K_REP, os.path.basename(os.path.dirname(ckpt))), flush=True)
print(line, flush=True)

if DS_NAME in REF:
    ref = REF[DS_NAME]
    print('[复现校验] base vs 官方日志 Best results', flush=True)
    for nm, key, v in (('R1,mIoU', 'mIoU', ref['mIoU']), ('R1@0.5', 'iou0.5', ref['r1_05']),
                       ('R1@0.7', 'iou0.7', ref['r1_07'])):
        got = mean_of('base', key)
        print('  %-4s %-9s 官方=%.4f  本脚本=%.4f  (%+.4f)' % (
            'OK ' if abs(got - v) <= 0.02 else 'DIFF', nm, v, got, got - v), flush=True)
print(line, flush=True)

print('[Q1 上界诊断] 中心还是宽度？（ORACLE 用 gt，仅作可达性上界）', flush=True)
print('%-16s %14s %14s %14s' % ('', 'R1@0.5', 'R1@0.7', 'mIoU'), flush=True)
for tag in ['base', 'ORACLE_c', 'ORACLE_w', 'ORACLE_cw']:
    print('%-16s %8.4f±%.4f %8.4f±%.4f %8.4f±%.4f' % (
        tag, mean_of(tag, 'iou0.5'), std_of(tag, 'iou0.5'),
        mean_of(tag, 'iou0.7'), std_of(tag, 'iou0.7'),
        mean_of(tag, 'mIoU'), std_of(tag, 'mIoU')), flush=True)
print('-' * 112, flush=True)
print('%-16s %+14s %+14s %+14s' % ('Δ vs base', 'ΔR1@0.5', 'ΔR1@0.7', 'ΔmIoU'), flush=True)
for tag in ['ORACLE_c', 'ORACLE_w', 'ORACLE_cw']:
    print('%-16s %+14.4f %+14.4f %+14.4f' % (
        tag, 100 * (mean_of(tag, 'iou0.5') - mean_of('base', 'iou0.5')),
        100 * (mean_of(tag, 'iou0.7') - mean_of('base', 'iou0.7')),
        100 * (mean_of(tag, 'mIoU') - mean_of('base', 'mIoU'))), flush=True)

print('\n' + line, flush=True)
print('[Q2 可复现性] CSR 现状 + 宽度自适应家族（零训练、不用 gt）', flush=True)
print('%-22s %13s %13s %13s' % ('', 'R1@0.5', 'R1@0.7', 'mIoU'), flush=True)
alltags = ['base', 'CSR'] + [t for t in ACC.keys() if t.startswith('CW')]
for tag in alltags:
    print('%-22s %7.4f±%.4f %7.4f±%.4f %7.4f±%.4f' % (
        tag, mean_of(tag, 'iou0.5'), std_of(tag, 'iou0.5'),
        mean_of(tag, 'iou0.7'), std_of(tag, 'iou0.7'),
        mean_of(tag, 'mIoU'), std_of(tag, 'mIoU')), flush=True)
print('-' * 112, flush=True)
print('%-22s %+13s %+13s %+13s' % ('Δ vs base', 'ΔR1@0.5', 'ΔR1@0.7', 'ΔmIoU'), flush=True)
for tag in alltags[1:]:
    print('%-22s %+13.4f %+13.4f %+13.4f' % (
        tag, 100 * (mean_of(tag, 'iou0.5') - mean_of('base', 'iou0.5')),
        100 * (mean_of(tag, 'iou0.7') - mean_of('base', 'iou0.7')),
        100 * (mean_of(tag, 'mIoU') - mean_of('base', 'mIoU'))), flush=True)

# 最佳宽度自适应变体
cand = [t for t in alltags[1:] if t != 'CSR']
best = max(cand, key=lambda t: mean_of(t, 'mIoU')) if cand else 'CSR'
print('\n[Q2 best] %s  ΔmIoU=%+.4f  Δ@0.5=%+.4f  Δ@0.7=%+.4f' % (
    best, 100 * (mean_of(best, 'mIoU') - mean_of('base', 'mIoU')),
    100 * (mean_of(best, 'iou0.5') - mean_of('base', 'iou0.5')),
    100 * (mean_of(best, 'iou0.7') - mean_of('base', 'iou0.7'))), flush=True)

print('\n' + line, flush=True)
print('[长短分桶] gt 时长占比 <= %.2f 为短' % THR, flush=True)
print('%-22s %-6s %6s %9s %9s %9s' % ('方法', '桶', 'n', 'mIoU', 'R1@0.5', 'R1@0.7'), flush=True)
btags = ['base', 'CSR', best]
for tag in btags:
    ib = np.concatenate([v[0] for v in SPLIT[tag]])
    gb = np.concatenate([v[1] for v in SPLIT[tag]])
    for nm, m in (('短', gb <= THR), ('长', gb > THR), ('全', np.ones(len(gb), bool))):
        print('%-22s %-6s %6d %9.4f %9.4f %9.4f' % (
            tag, nm, int(m.sum()), ib[m].mean(), (ib[m] >= 0.5).mean(), (ib[m] >= 0.7).mean()), flush=True)
print('-' * 112, flush=True)
print('%-22s %-6s %6s %9s %9s %9s' % ('Δ vs base', '桶', 'n', 'ΔmIoU', 'ΔR1@0.5', 'ΔR1@0.7'), flush=True)
ib = np.concatenate([v[0] for v in SPLIT['base']])
gb = np.concatenate([v[1] for v in SPLIT['base']])
for tag in ['CSR', best]:
    ic = np.concatenate([v[0] for v in SPLIT[tag]])
    for nm, m in (('短', gb <= THR), ('长', gb > THR), ('全', np.ones(len(gb), bool))):
        print('%-22s %-6s %6d %+9.4f %+9.4f %+9.4f' % (
            tag + '-base', nm, int(m.sum()),
            100 * (ic[m].mean() - ib[m].mean()),
            100 * ((ic[m] >= 0.5).mean() - (ib[m] >= 0.5).mean()),
            100 * ((ic[m] >= 0.7).mean() - (ib[m] >= 0.7).mean())), flush=True)

print('\n' + line, flush=True)
print('[显著性] 配对 bootstrap B=2000（95%% CI）+ McNemar', flush=True)
rng = np.random.default_rng(0)
B = 2000
for tag in ['CSR', best]:
    ic = np.concatenate([v[0] for v in SPLIT[tag]])
    for nm, m in (('short', gb <= THR), ('long', gb > THR), ('all', np.ones(len(gb), bool))):
        ibb, icc = ib[m], ic[m]
        nn = len(ibb)
        idx = rng.integers(0, nn, size=(B, nn))
        dm = icc[idx].mean(1) - ibb[idx].mean(1)
        d5 = (icc[idx] >= 0.5).mean(1) - (ibb[idx] >= 0.5).mean(1)
        d7 = (icc[idx] >= 0.7).mean(1) - (ibb[idx] >= 0.7).mean(1)
        b5 = int(((ibb < 0.5) & (icc >= 0.5)).sum())
        c5 = int(((ibb >= 0.5) & (icc < 0.5)).sum())
        b7 = int(((ibb < 0.7) & (icc >= 0.7)).sum())
        c7 = int(((ibb >= 0.7) & (icc < 0.7)).sum())
        print('  %-7s %-6s n=%5d ΔmIoU %+6.3f [%+6.3f,%+6.3f] | Δ@0.5 %+6.3f [%+6.3f,%+6.3f] | Δ@0.7 %+6.3f [%+6.3f,%+6.3f]' % (
            tag, nm, nn, 100 * dm.mean(), 100 * np.percentile(dm, 2.5), 100 * np.percentile(dm, 97.5),
            100 * d5.mean(), 100 * np.percentile(d5, 2.5), 100 * np.percentile(d5, 97.5),
            100 * d7.mean(), 100 * np.percentile(d7, 2.5), 100 * np.percentile(d7, 97.5)), flush=True)
        for tn, bb, cc in (('@0.5', b5, c5), ('@0.7', b7, c7)):
            stat = ((abs(bb - cc) - 1) ** 2 / float(bb + cc)) if (bb + cc) > 0 else 0.0
            p = math.erfc(math.sqrt(stat / 2.0))
            print('           McNemar %s: 升(%d) 降(%d)  chi2=%.2f  p=%.2e' % (tn, bb, cc, stat, p), flush=True)

print('\n' + line, flush=True)
print('[诊断] 几何', flush=True)
wb = diag['wb']
gl = diag['GL2']
print('  基线框宽: 中位=%.4f  均值=%.4f  >0.9 占比=%.3f' % (
    np.median(wb), wb.mean(), float((wb > 0.9).mean())), flush=True)
print('  gt 时长  : 中位=%.4f  均值=%.4f' % (np.median(gl), gl.mean()), flush=True)
print('  wb/gt 比 : 中位=%.3f' % np.median(wb / np.maximum(gl, 1e-6)), flush=True)
if WRMED:
    wr = np.concatenate(WRMED)
    print('  支撑区间宽 w_r: 中位=%.4f  均值=%.4f  分位(10/50/90)=%.3f/%.3f/%.3f' % (
        np.median(wr), wr.mean(), np.percentile(wr, 10), np.percentile(wr, 50), np.percentile(wr, 90)),
        flush=True)
for tag in ['CSR', best]:
    if tag in STOR:
        use = np.concatenate(STOR[tag])
        print('  %-22s 触发率=%.3f' % (tag, use.mean()), flush=True)
print('  CSR 移动率=%.3f  平均|Δ|(移动)=%.4f' % (
    diag['moved'].mean(), diag['shift'][diag['moved']].mean() if diag['moved'].any() else 0.0), flush=True)
print(line, flush=True)
print('DONE', flush=True)
