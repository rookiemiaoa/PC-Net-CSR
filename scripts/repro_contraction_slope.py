# -*- coding: utf-8 -*-
"""复现 §3.3 命题 1 的实测收缩斜率 1.763（质心只兑现约 57% 的所需位移）。

定义（2026-10-13 原诊断 ``w6_centre_bias.py`` 的 [C1] 段，2026-10-19 逐位复核）
------------------------------------------------------------------------------
* 曲线：``feat_cache/charades_test_trivial_r3.pkl`` 的 ``Z``，先 ``smooth_curve(Z, 7)``
* 中心估计：窗口内正部加权质心，``beta = 0.20``
* 回归：``y = c_g - c_b``（所需位移）对 ``x = c_1 - c_b``（实际位移）做带截距最小二乘，
  斜率 ``k`` > 1 表示估计器欠校正；兑现比例 ``= 1 / k``
* 论文引的是 rep 0 的 1.763（K_REP = 3 的第一次），三 rep 一并打印以便审计

用法
----
    python scripts/repro_contraction_slope.py --check
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(_HERE)
sys.path.insert(0, REPO)

from csr import ops                  # noqa: E402
from csr.paths import CACHE_DIR      # noqa: E402

CACHE = os.path.join(CACHE_DIR, 'charades_test_trivial_r3.pkl')
L = 50
CG = (np.arange(L) + 0.5) / L
SMOOTH = 7
BETA = 0.20
PAPER_SLOPE = 1.763


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--json', default=None)
    args = ap.parse_args()

    reps = pickle.load(open(CACHE, 'rb'))
    rows = []
    for i, D in enumerate(reps):
        Z = ops.smooth_curve(D['Z'], SMOOTH)
        cb = D['cb']
        c1 = ops.estimate_window_center(Z, cb, BETA, CG)
        gc = (D['GS'] + D['GE']) / 2.0
        x = c1 - cb
        y = gc - cb
        ok = np.isfinite(x) & np.isfinite(y)
        k, b = np.polyfit(x[ok], y[ok], 1)
        rows.append({'rep': i, 'slope': float(k), 'intercept': float(b),
                     'realized_fraction': float(1.0 / k), 'n': int(ok.sum())})

    print('cache : %s' % CACHE)
    print('定义  : smooth=%d, beta=%.2f, 回归 y=(c_g-c_b) ~ x=(c_1-c_b)，带截距' % (SMOOTH, BETA))
    print('')
    print('%5s %10s %12s %12s' % ('rep', 'slope', '1/slope', 'n'))
    for r in rows:
        print('%5d %10.4f %12.4f %12d' % (r['rep'], r['slope'], r['realized_fraction'], r['n']))
    print('')
    print('论文值 1.763 ← rep 0，偏差 %.6f' % abs(rows[0]['slope'] - PAPER_SLOPE))
    print('三 rep 均值 %.4f，极差 %.4f' % (
        np.mean([r['slope'] for r in rows]),
        max(r['slope'] for r in rows) - min(r['slope'] for r in rows)))

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        json.dump({'cache': CACHE, 'smooth': SMOOTH, 'beta': BETA,
                   'definition': 'polyfit(c1-cb, c_g-c_b, deg=1) with intercept',
                   'paper_slope': PAPER_SLOPE, 'per_rep': rows}, open(args.json, 'w'),
                  indent=2, ensure_ascii=False)
        print('wrote %s' % args.json)

    return 0 if (not args.check or abs(rows[0]['slope'] - PAPER_SLOPE) <= 5e-4) else 1


if __name__ == '__main__':
    sys.exit(main())
