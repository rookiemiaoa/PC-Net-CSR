# -*- coding: utf-8 -*-
"""W6：用「未 clip 原始边缘」重做 (σ, ω) 族 + 中心估计器的收缩偏置诊断。

W5 的实现用了缓存里已被 clip 到 [0,1] 的 bL/bH 加位移，导致 σ=1,ω=0 与真正的 CSR
（由 centre±w/2 构造）不一致（短桶 Δ@.5 差 0.4pt）。本脚本改用未 clip 边缘。

另一个更重要的诊断：窗内正部质心 c1 被**搜索窗中心 cb 拖动**（经典收缩偏置）
⇒ E[c1 - cb | 真位移] 的回归斜率可能 > 1 ⇒ 可以「去收缩」放大幅度增益 σ。
本脚本给出该斜率，并给出「按方向 + 平均幅度」的天花板。
"""
import os
import pickle
import sys

import numpy as np

# ---- 路径解析：自动定位本仓库根目录（与 oracle_gate.py 一致）----
_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = _HERE
while REPO != os.path.dirname(REPO) and not os.path.exists(os.path.join(REPO, 'models', 'pcnet.py')):
    REPO = os.path.dirname(REPO)
sys.path.insert(0, REPO)
from csr import ops                       # noqa: E402
from csr.paths import CACHE_DIR           # noqa: E402

L = 50
cg = (np.arange(L) + 0.5) / L
REPS = pickle.load(open(os.path.join(CACHE_DIR, 'charades_test_trivial_r3.pkl'), 'rb'))
THR = 0.35


def met(i):
    return dict(mIoU=float(i.mean()), i5=float((i >= 0.5).mean()), i7=float((i >= 0.7).mean()))


def agg(per):
    return {k: float(np.mean([d[k] for d in per])) for k in per[0]}


def raw_edges(D):
    return D['cb'] - D['wb'] / 2.0, D['cb'] + D['wb'] / 2.0


def signals(D, beta=0.20, rho=0.10, gate=True, dmin=0.01):
    """返回 (d0 未门控未死区, d 已死区已门控)"""
    Z = ops.smooth_curve(D['Z'], 7)
    cb, wb = D['cb'], D['wb']
    c1 = ops.estimate_window_center(Z, cb, beta, cg)
    d0 = c1 - cb
    d = np.where(np.abs(d0) < rho * wb, 0.0, d0)
    if gate:
        c2 = np.clip(cb + d, 0, 1)
        din = ops.inside_score(Z, c2 - wb / 2, c2 + wb / 2, cg) - ops.inside_score(Z, D['bL'], D['bH'], cg)
        d = np.where(din >= dmin, d, 0.0)
    return d0, d


def fam(D, d, sigma, omega):
    """new_lo = bL_raw + a d ; new_hi = bH_raw + b d ; a=σ-ω/2, b=σ+ω/2"""
    L0, H0 = raw_edges(D)
    a = sigma - omega / 2.0
    b = sigma + omega / 2.0
    return np.clip(L0 + a * d, 0, 1), np.clip(H0 + b * d, 0, 1)


base = {}
for bn, sel in (('ALL', None), ('SHORT', 's'), ('LONG', 'l')):
    per = []
    for D in REPS:
        i = ops.iou(D['bL'], D['bH'], D['GS'], D['GE'])
        m = (D['GL'] <= THR) if sel == 's' else ((D['GL'] > THR) if sel == 'l' else np.ones(len(i), bool))
        per.append(met(i[m]))
    base[bn] = agg(per)
print('base: ALL mIoU=%.4f@.5=%.4f@.7=%.4f | SHORT %.4f/%.4f/%.4f | LONG %.4f/%.4f/%.4f' % (
    base['ALL']['mIoU'], base['ALL']['i5'], base['ALL']['i7'],
    base['SHORT']['mIoU'], base['SHORT']['i5'], base['SHORT']['i7'],
    base['LONG']['mIoU'], base['LONG']['i5'], base['LONG']['i7']))

print()
print('=' * 116)
print('[C1] 中心估计器 c1 的收缩偏置：δ = gt中心 − 框中心 对 d0 = c1 − cb 的回归')
print('=' * 116)
D = REPS[0]
d0, d = signals(D)
delta = 0.5 * (D['GS'] + D['GE']) - D['cb']
for nm, sel in (('ALL', np.ones(len(d0), bool)), ('d0≠0', np.abs(d0) > 1e-9), ('|d0|>0.05', np.abs(d0) > 0.05)):
    x, yv = d0[sel], delta[sel]
    if len(x) < 10:
        continue
    sl, ic = np.polyfit(x, yv, 1)
    r = np.corrcoef(x, yv)[0, 1]
    print('  %-10s n=%4d  斜率=%.3f 截距=%+.3f  r=%.3f  同号率=%.4f' % (
        nm, len(x), sl, ic, r, (np.sign(x) == np.sign(yv)).mean()))
x, yv = d0[np.abs(d0) > 1e-9], delta[np.abs(d0) > 1e-9]
print('  ⇒ 斜率 %.2f 表示估计器把真实位移**压缩**到 %.0f%%；σ=1/斜率=%.2f 才是无偏放大' % (
    sl, 100 * sl, 1 / max(sl, 1e-6)))

print()
print('=' * 116)
print('[C2] (σ, ω) 扫描（未 clip 边缘；CSR = σ=1.00, ω=0.00）')
print('=' * 116)
SIGMAS = [0.0, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
OMEGAS = [-1.0, -0.5, 0.0, 0.5, 1.0]
print('  %-6s' % 'σ\\ω', end='')
for om in OMEGAS:
    print(' | %-24s' % ('ω=%+.1f' % om), end='')
print()
res = {}
for sg in SIGMAS:
    print('  %-6.2f' % sg, end='')
    for om in OMEGAS:
        per = {'ALL': [], 'SHORT': [], 'LONG': []}
        for D in REPS:
            _, d = signals(D)
            lo, hi = fam(D, d, sg, om)
            i = ops.iou(lo, hi, D['GS'], D['GE'])
            for bn, sel in (('ALL', None), ('SHORT', 's'), ('LONG', 'l')):
                m = (D['GL'] <= THR) if sel == 's' else ((D['GL'] > THR) if sel == 'l' else np.ones(len(i), bool))
                per[bn].append(met(i[m]))
        A = agg(per['ALL']); S = agg(per['SHORT']); Lg = agg(per['LONG'])
        res[(sg, om)] = (A, S, Lg)
        print(' | %+6.2f %+6.2f %+6.2f     ' % (
            100 * (A['mIoU'] - base['ALL']['mIoU']), 100 * (A['i5'] - base['ALL']['i5']),
            100 * (A['i7'] - base['ALL']['i7'])), end='')
    print()
print()
print('  %-12s %8s %8s %8s | %8s %8s | %8s %8s' % ('(σ,ω)', 'ΔmIoU', 'Δ@.5', 'Δ@.7', 'SΔmIoU', 'SΔ@.5', 'LΔmIoU', 'LΔ@.5'))
for (sg, om), (A, S, Lg) in sorted(res.items(), key=lambda t: -t[1][0]['mIoU'])[:10]:
    print('  (%.2f,%+.2f) %+8.2f %+8.2f %+8.2f | %+8.2f %+8.2f | %+8.2f %+8.2f' % (
        sg, om, 100 * (A['mIoU'] - base['ALL']['mIoU']), 100 * (A['i5'] - base['ALL']['i5']),
        100 * (A['i7'] - base['ALL']['i7']), 100 * (S['mIoU'] - base['SHORT']['mIoU']),
        100 * (S['i5'] - base['SHORT']['i5']), 100 * (Lg['mIoU'] - base['LONG']['mIoU']),
        100 * (Lg['i5'] - base['LONG']['i5'])))
print()
print('=' * 116)
print('[C3] 中心量的天花板：不同「知道多少」的位移方案（诊断，用 gt）')
print('=' * 116)
schemes = {
    'ORACLE 完美中心(逐样本 gt 中心)': lambda D: 0.5 * (D['GS'] + D['GE']),
    'ORACLE 方向+全局平均幅度': lambda D: None,
    'ORACLE 方向+逐样本最优幅度': lambda D: None,
}
per = []
for D in REPS:
    c = 0.5 * (D['GS'] + D['GE'])
    i = ops.iou(*ops.boxes_from_center_width(c, D['wb']), D['GS'], D['GE'])
    per.append(met(i))
r = agg(per)
print('  %-30s mIoU=%.4f @.5=%.4f @.7=%.4f | ΔmIoU %+.2f' % (
    '完美中心(框宽不变)', r['mIoU'], r['i5'], r['i7'], 100 * (r['mIoU'] - base['ALL']['mIoU'])))
D = REPS[0]
delta = 0.5 * (D['GS'] + D['GE']) - D['cb']
_, d = signals(D)
sd = np.sign(d)
gt_sgn = np.sign(delta)
# 方向+全局平均幅度
per = []
for D in REPS:
    _, d = signals(D)
    sd = np.sign(d)
    mag = np.abs(0.5 * (D['GS'] + D['GE']) - D['cb'])
    mag = np.where(np.abs(d) > 1e-9, mag.mean(), 0.0)
    c = np.clip(D['cb'] + sd * mag, 0, 1)
    i = ops.iou(*ops.boxes_from_center_width(c, D['wb']), D['GS'], D['GE'])
    per.append(met(i))
r = agg(per)
print('  %-30s mIoU=%.4f @.5=%.4f @.7=%.4f | ΔmIoU %+.2f' % (
    '方向(估计)+全局平均幅度', r['mIoU'], r['i5'], r['i7'], 100 * (r['mIoU'] - base['ALL']['mIoU'])))
# 方向(估计)+逐样本最优幅度（在幅度网格上搜）
MAGS = np.round(np.arange(0.0, 0.301, 0.02), 3)
per = []
for D in REPS:
    _, d = signals(D)
    sd = np.sign(d)
    cand = []
    for mg in MAGS:
        c = np.clip(D['cb'] + sd * mg, 0, 1)
        cand.append(ops.iou(*ops.boxes_from_center_width(c, D['wb']), D['GS'], D['GE']))
    per.append(met(np.stack(cand, 1).max(1)))
r = agg(per)
print('  %-30s mIoU=%.4f @.5=%.4f @.7=%.4f | ΔmIoU %+.2f' % (
    '方向(估计)+逐样本最优幅度', r['mIoU'], r['i5'], r['i7'], 100 * (r['mIoU'] - base['ALL']['mIoU'])))
print('DONE')
