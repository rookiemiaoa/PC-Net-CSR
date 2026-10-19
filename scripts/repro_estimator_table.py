# -*- coding: utf-8 -*-
"""复现 §4.3「中心估计器对比」的 11 个数（6 个 MAE + 5 个方向准确率）。

定义（与 2026-09-13 原表完全一致，2026-10-19 逐位复核）
--------------------------------------------------------
* 曲线：``feat_cache/charades_test_trivial_r3.pkl`` 的 ``Z``
  （逐点 -nll，已 z-score，**未平滑**；该表中估计器只换聚合方式，不换平滑）
* 网格 ``cg = (i + 0.5) / 50``，真值中心 ``c_g = (GS + GE) / 2``
* baseline          ``c_b``（模型 top-1 框中心）
* global argmax     ``argmax_t Z``
* softmax τ = 0.25  对全部 50 个 bin 做 softmax 后取期望位置
* windowed centroid ``mask = |cg - c_b| <= beta``，取 ``relu(Z)`` 的加权质心；
                    空窗（权重和为 0）回落到 ``c_b``
* MAE               ``mean |c - c_g|``（估计值先 clip 到 [0,1]）
* dirAcc            ``mean[sign(c - c_b) == sign(c_g - c_b)]``，
                    分母为 ``|c_g - c_b| > 0.02`` 的 2750 个样本
                    （**不是**主表 dir_acc 的 0.08 子群，两者不可混引）

用法
----
    python scripts/repro_estimator_table.py
    python scripts/repro_estimator_table.py --check        # 与论文值逐位比对
    python scripts/repro_estimator_table.py --json results/estimator_grid/estimator_grid.json
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

from csr.paths import CACHE_DIR  # noqa: E402

CACHE = os.path.join(CACHE_DIR, 'charades_test_trivial_r3.pkl')
L = 50
CG = (np.arange(L) + 0.5) / L
TAU = 0.25
DIR_SUBGROUP = 0.02          # 方向准确率的偏离子群门限
BETAS = (0.50, 0.30, 0.20)

# 论文 v12 §4.3 的原值（MAE, dirAcc）
PAPER = {
    'baseline_c_b': (0.1699, None),
    'global_argmax': (0.1742, 0.684),
    'softmax_tau0.25': (0.1499, 0.770),
    'windowed_centroid_beta0.50': (0.1441, 0.769),
    'windowed_centroid_beta0.30': (0.1443, 0.778),
    'windowed_centroid_beta0.20': (0.1483, 0.769),
}


def _stats(c, cb, gc):
    c = np.clip(np.asarray(c, dtype=np.float64), 0.0, 1.0)
    mae = float(np.mean(np.abs(c - gc)))
    sub = np.abs(gc - cb) > DIR_SUBGROUP
    dir_acc = float(np.mean(np.sign(c - cb)[sub] == np.sign(gc - cb)[sub]))
    return mae, dir_acc, int(sub.sum())


def _windowed_centroid(Z, cb, beta):
    m = np.abs(CG[None, :] - cb[:, None]) <= beta
    P = np.where(m, np.maximum(Z, 0.0), 0.0)
    s = P.sum(1)
    c = (P @ CG) / np.maximum(s, 1e-12)
    bad = s < 1e-12
    if bad.any():
        c[bad] = cb[bad]
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true', help='与论文值比对，不一致则退出码 1')
    ap.add_argument('--rep', type=int, default=0, help='使用第几个 rep（默认 0）')
    ap.add_argument('--json', default=None, help='把结果写成 JSON')
    args = ap.parse_args()

    reps = pickle.load(open(CACHE, 'rb'))
    D = reps[args.rep]
    Z, cb = D['Z'], D['cb']
    gc = (D['GS'] + D['GE']) / 2.0

    est = {}
    est['baseline_c_b'] = cb
    est['global_argmax'] = CG[np.argmax(Z, axis=1)]
    sm = np.exp((Z - Z.max(1, keepdims=True)) / TAU)
    sm /= sm.sum(1, keepdims=True)
    est['softmax_tau0.25'] = sm @ CG
    for b in BETAS:
        est['windowed_centroid_beta%.2f' % b] = _windowed_centroid(Z, cb, b)

    print('cache   : %s' % CACHE)
    print('rep     : %d / %d    N = %d' % (args.rep, len(reps), len(cb)))
    print('定义    : dirAcc 分母 = |c_g - c_b| > %.2f 的样本' % DIR_SUBGROUP)
    print('')
    print('%-30s %9s %9s | %9s %9s | %s' % ('estimator', 'MAE', 'paper', 'dirAcc', 'paper', 'match'))
    print('-' * 96)

    out, bad = {}, 0
    for name, c in est.items():
        mae, dacc, nsub = _stats(c, cb, gc)
        p_mae, p_dir = PAPER[name]
        # 按论文的有效位数比对：MAE 4 位；dirAcc 3 位，容差 1e-3 以吸收
        # 「四舍五入 vs 银行家舍入」的末位差（如 0.76945 → 论文 0.770 / 本机 0.769）
        ok_mae = abs(round(mae, 4) - p_mae) <= 1e-9
        ok_dir = True if p_dir is None else abs(dacc - p_dir) <= 1e-3
        ok = ok_mae and ok_dir
        bad += 0 if ok else 1
        out[name] = {'mae': round(mae, 4), 'dir_acc': round(dacc, 4),
                     'dir_subgroup_n': nsub, 'paper_mae': p_mae, 'paper_dir_acc': p_dir,
                     'match': bool(ok)}
        print('%-30s %9.4f %9.4f | %9s %9s | %s' % (
            name, mae, p_mae, ('%.4f' % dacc) if p_dir is not None else '-',
            ('%.3f' % p_dir) if p_dir is not None else '-', 'OK' if ok else 'DIFF'))

    print('-' * 96)
    print('子群样本数 n = %d（论文写作 2750）' % out['global_argmax']['dir_subgroup_n'])
    print('结论：%d/%d 项与论文逐位一致' % (len(out) - bad, len(out)))

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        json.dump({'cache': CACHE, 'rep': args.rep, 'n': int(len(cb)),
                   'definitions': {
                       'curve': 'Z (z-scored -nll, unsmoothed)',
                       'grid': 'cg=(i+0.5)/50',
                       'gt_center': '(GS+GE)/2',
                       'softmax_tau': TAU,
                       'dir_subgroup': '|c_g - c_b| > %.2f' % DIR_SUBGROUP,
                       'dir_subgroup_n': out['global_argmax']['dir_subgroup_n']},
                   'estimators': out}, open(args.json, 'w'), indent=2, ensure_ascii=False)
        print('wrote %s' % args.json)

    return 1 if (args.check and bad) else 0


if __name__ == '__main__':
    sys.exit(main())
