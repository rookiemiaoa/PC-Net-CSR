# -*- coding: utf-8 -*-
"""oracle_gate.py —— 定版配置下的 ORACLE 门控上界（需 gt，仅作上界）

完美门控：仅当 |cb - c_gt| > thr 才平移（其余参数同定版：sm7 β0.20 ρ0.10 α1.00）。
同时给出 ORACLE_c（只换 gt 中心、保持框宽）作对照。
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


def iou(a0, a1, b0, b1):
    return ops.iou(a0, a1, b0, b1)


def smoothZ(Z, w):
    return ops.smooth_curve(Z, w)


def est_win_vec(Z, cb, beta):
    return ops.estimate_window_center(Z, cb, beta, cg)


def agg(fn):
    res = {}
    for bucket in ('all', 'short', 'long'):
        vals = []
        for D in REPS:
            i = fn(D)
            SH = D['GL'] <= THR
            sel = slice(None) if bucket == 'all' else (SH if bucket == 'short' else ~SH)
            vals.append([float(i[sel].mean()), float((i[sel] >= 0.5).mean()), float((i[sel] >= 0.7).mean())])
        res[bucket] = np.mean(vals, 0)
    return res


def oracle_gate(thr):
    def f(D):
        c1 = est_win_vec(smoothZ(D['Z'], 7), D['cb'], 0.20)
        d = c1 - D['cb']
        d = np.where(np.abs(d) < 0.10 * D['wb'], 0.0, d)
        err = np.abs(D['cb'] - 0.5 * (D['GS'] + D['GE']))
        d = np.where(err > thr, d, 0.0)
        c2 = np.clip(D['cb'] + 1.00 * d, 0, 1)
        w = D['wb']
        return iou(np.clip(c2 - w / 2, 0, 1), np.clip(c2 + w / 2, 0, 1), D['GS'], D['GE'])
    return f


def oracle_c(D):
    c2 = 0.5 * (D['GS'] + D['GE'])
    w = D['wb']
    return iou(np.clip(c2 - w / 2, 0, 1), np.clip(c2 + w / 2, 0, 1), D['GS'], D['GE'])


base = agg(lambda D: iou(D['bL'], D['bH'], D['GS'], D['GE']))
print('%-24s %s' % ('', ' '.join(['%9s' % x for x in
      ['ALL_mIoU', 'ALL@.5', 'ALL@.7', 'S_mIoU', 'S@.5', 'S@.7', 'L_mIoU', 'L@.5', 'L@.7']])))


def show(tag, r):
    s = '%-24s' % tag
    for b in ('all', 'short', 'long'):
        for k in range(3):
            s += ' %+9.3f' % (100 * (r[b][k] - base[b][k]))
    print(s)


show('ORACLE_c(gt中心保宽)', agg(oracle_c))
for thr in (0.08, 0.15):
    show('ORACLE_gate thr=%.2f' % thr, agg(oracle_gate(thr)))
