# -*- coding: utf-8 -*-
"""ablate_gate.py —— 定版配置下「关闭自洽门控」的消融（本地离线）"""
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


def iou(a0, a1, b0, b1):
    return ops.iou(a0, a1, b0, b1)


def smoothZ(Z, w):
    return ops.smooth_curve(Z, w)


def inside_score(Z, lo, hi):
    return ops.inside_score(Z, lo, hi, cg)


def est_win_vec(Z, cb, beta):
    return ops.estimate_window_center(Z, cb, beta, cg)


def run(dmin):
    out = {'all': [], 'short': [], 'long': []}
    for D in REPS:
        Z = smoothZ(D['Z'], 7)
        cb, wb = D['cb'], D['wb']
        c1 = est_win_vec(Z, cb, 0.20)
        d = c1 - cb
        d = np.where(np.abs(d) < 0.10 * wb, 0.0, d)
        c2 = np.clip(cb + 1.00 * d, 0, 1)
        if dmin is not None:
            din = inside_score(Z, c2 - wb / 2, c2 + wb / 2) - inside_score(Z, D['bL'], D['bH'])
            c2 = np.where(din >= dmin, c2, cb)
        i = iou(np.clip(c2 - wb / 2, 0, 1), np.clip(c2 + wb / 2, 0, 1), D['GS'], D['GE'])
        SH = D['GL'] <= THR
        for b, s in (('all', np.ones(len(i), bool)), ('short', SH), ('long', ~SH)):
            out[b].append(i[s])
    return {b: np.mean([x.mean() for x in v]) for b, v in out.items()}, \
           {b: np.mean([(x >= 0.5).mean() for x in v]) for b, v in out.items()}, \
           {b: np.mean([(x >= 0.7).mean() for x in v]) for b, v in out.items()}


def run_base():
    out = {'all': [], 'short': [], 'long': []}
    for D in REPS:
        i = iou(D['bL'], D['bH'], D['GS'], D['GE'])
        SH = D['GL'] <= THR
        for b, s in (('all', np.ones(len(i), bool)), ('short', SH), ('long', ~SH)):
            out[b].append(i[s])
    return {b: np.mean([x.mean() for x in v]) for b, v in out.items()}, \
           {b: np.mean([(x >= 0.5).mean() for x in v]) for b, v in out.items()}, \
           {b: np.mean([(x >= 0.7).mean() for x in v]) for b, v in out.items()}


print('=' * 70)
print('消融：自洽门控开/关（其余参数一致，sm7 β0.20 ρ0.10 α1.00）')
print('=' * 70)
base_m, base_5, base_7 = run_base()
print('base             ALL %.4f/%.4f/%.4f | 短 %.4f/%.4f/%.4f | 长 %.4f/%.4f/%.4f' % (
    base_m['all'], base_5['all'], base_7['all'], base_m['short'], base_5['short'], base_7['short'],
    base_m['long'], base_5['long'], base_7['long']))
for tag, dm in (('门控开 dmin=0.01', 0.01), ('门控关', None)):
    m, i5, i7 = run(dm)
    print('%-16s  ALL %+.3f/%+.3f/%+.3f | 短 %+.3f/%+.3f/%+.3f | 长 %+.3f/%+.3f/%+.3f (pt)' % (
        tag,
        100 * (m['all'] - base_m['all']), 100 * (i5['all'] - base_5['all']), 100 * (i7['all'] - base_7['all']),
        100 * (m['short'] - base_m['short']), 100 * (i5['short'] - base_5['short']), 100 * (i7['short'] - base_7['short']),
        100 * (m['long'] - base_m['long']), 100 * (i5['long'] - base_5['long']), 100 * (i7['long'] - base_7['long'])))
