# -*- coding: utf-8 -*-
"""attrib_csr.py —— CSR 定版配置下的「伤害归因」逐样本分解（本地离线）

按 |c_b - c_gt| 分箱，统计短桶内各箱的：n / 移动率 / ΔmIoU / ΔR1@0.5 / ΔR1@0.7。
用于报告 §4.2。
"""
import os
import pickle
import sys

import numpy as np

# ---- 路径解析：默认自动定位本仓库根目录；特征缓存目录可用 FEAT_CACHE_DIR / CACHE_DIR 覆盖 ----
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = _HERE
while _REPO != os.path.dirname(_REPO) and not os.path.exists(os.path.join(_REPO, 'models', 'pcnet.py')):
    _REPO = os.path.dirname(_REPO)
sys.path.insert(0, _REPO)

from csr.paths import CACHE_DIR               # noqa: E402
from csr import ops                           # noqa: E402  （单一权威算子实现）

L = 50
cg = (np.arange(L) + 0.5) / L
REPS = pickle.load(open(os.path.join(CACHE_DIR, 'charades_test_trivial_r3.pkl'), 'rb'))
THR = 0.35
SMOOTH, BETA, RHO, ALPHA, DMIN = 7, 0.20, 0.10, 1.00, 0.01


def iou(a0, a1, b0, b1):
    return ops.iou(a0, a1, b0, b1)


def smoothZ(Z, w):
    return ops.smooth_curve(Z, w)


def inside_score(Z, lo, hi):
    return ops.inside_score(Z, lo, hi, cg)


def est_win_vec(Z, cb, beta):
    return ops.estimate_window_center(Z, cb, beta, cg)


def csr(D):
    Z = smoothZ(D['Z'], SMOOTH)
    cb, wb = D['cb'], D['wb']
    c1 = est_win_vec(Z, cb, BETA)
    d = c1 - cb
    d = np.where(np.abs(d) < RHO * wb, 0.0, d)
    c2 = np.clip(cb + ALPHA * d, 0, 1)
    din = inside_score(Z, c2 - wb / 2, c2 + wb / 2) - inside_score(Z, D['bL'], D['bH'])
    c2 = np.where(din >= DMIN, c2, cb)
    moved = np.abs(c2 - cb) > 0
    return np.clip(c2 - wb / 2, 0, 1), np.clip(c2 + wb / 2, 0, 1), moved


# 合并 3 个 rep
IB, IC, MOVED, SHORT, CENERR = [], [], [], [], []
for D in REPS:
    lb, hb = D['bL'], D['bH']
    lc, hc, mv = csr(D)
    IB.append(iou(lb, hb, D['GS'], D['GE']))
    IC.append(iou(lc, hc, D['GS'], D['GE']))
    MOVED.append(mv)
    SHORT.append(D['GL'] <= THR)
    cgt = 0.5 * (D['GS'] + D['GE'])
    CENERR.append(np.abs(D['cb'] - cgt))
IB = np.concatenate(IB); IC = np.concatenate(IC)
MOVED = np.concatenate(MOVED); SHORT = np.concatenate(SHORT)
CENERR = np.concatenate(CENERR)

D_MIOU = IC - IB
D_5 = (IC >= 0.5).astype(float) - (IB >= 0.5).astype(float)
D_7 = (IC >= 0.7).astype(float) - (IB >= 0.7).astype(float)

print('=' * 96)
print('CSR 定版（sm%d β%.2f ρ%.2f α%.2f dmin%.2f）伤害归因 —— 短桶（gt<=%.2f）' % (
    SMOOTH, BETA, RHO, ALPHA, DMIN, THR))
print('=' * 96)
bins = [(0.00, 0.03), (0.03, 0.08), (0.08, 0.15), (0.15, 0.30), (0.30, 1.01)]
print('%-14s %6s %8s %10s %10s %10s' % ('|cb-cgt|', 'n', 'move', 'd_mIoU', 'd@0.5', 'd@0.7'))
for lo, hi in bins:
    sel = SHORT & (CENERR >= lo) & (CENERR < hi)
    n = int(sel.sum())
    if n == 0:
        continue
    print('%-14s %6d %8.2f %+10.3f %+10.3f %+10.3f' % (
        '[%.2f,%.2f)' % (lo, hi), n, MOVED[sel].mean(),
        100 * D_MIOU[sel].mean(), 100 * D_5[sel].mean(), 100 * D_7[sel].mean()))

# 汇总：中心本来就准 vs 中心确实偏
for name, sel in (('短·中心准(<0.08)', SHORT & (CENERR < 0.08)),
                  ('短·中心偏(>=0.08)', SHORT & (CENERR >= 0.08)),
                  ('短·合计', SHORT)):
    n = int(sel.sum())
    print('%-18s n=%5d move=%.3f  d_mIoU=%+.3f  d@0.5=%+.3f  d@0.7=%+.3f' % (
        name, n, MOVED[sel].mean(), 100 * D_MIOU[sel].mean(),
        100 * D_5[sel].mean(), 100 * D_7[sel].mean()))

# 跨阈值计数（短桶）
for t in (0.5, 0.7):
    Ib = IB[SHORT] >= t
    Ic = IC[SHORT] >= t
    up = int(((~Ib) & Ic).sum()); dn = int((Ib & (~Ic)).sum())
    print('短桶 %s 跨线：上穿 %d / 下穿 %d  (净 %+d)' % (t, up, dn, up - dn))
