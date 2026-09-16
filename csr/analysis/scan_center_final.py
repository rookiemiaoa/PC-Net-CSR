# -*- coding: utf-8 -*-
"""scan_center_final.py —— CSR 最终定版：稳健性 + 配对显著性检验（本地离线）

CSR (Center-Shift Refinement) 定版配置（与服务器 eval_csr.py / 历史名 diag40_gen.py 完全一致）：
  1) 相关性曲线 Z（逐点提案的 -nll，先 z-score 再平滑）
  2) 平滑窗口 7
  3) 中心估计器：窗口内正部加权质心，窗口半径 beta=0.20
  4) dead-zone：|d| < 0.10*框宽 则不平移
  5) 阻尼 alpha=1.00
  6) 自洽门控：仅当平移后「框内 Z 均值」提升 >= 0.01 时才接受
  宽度保持不变（这是与 RBR 的本质区别）
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


def inside_score(Z, lo, hi):
    return ops.inside_score(Z, lo, hi, cg)


def est_win_vec(Z, cb, beta):
    return ops.estimate_window_center(Z, cb, beta, cg)


def csr(D, smooth=7, beta=0.20, rho=0.10, alpha=1.00, dmin=0.01):
    Z = smoothZ(D['Z'], smooth)
    cb, wb = D['cb'], D['wb']
    c1 = est_win_vec(Z, cb, beta)
    d = c1 - cb
    d = np.where(np.abs(d) < rho * wb, 0.0, d)
    c2 = np.clip(cb + alpha * d, 0, 1)
    if dmin is not None:
        din = inside_score(Z, c2 - wb / 2, c2 + wb / 2) - inside_score(Z, D['bL'], D['bH'])
        c2 = np.where(din >= dmin, c2, cb)
    return np.clip(c2 - wb / 2, 0, 1), np.clip(c2 + wb / 2, 0, 1)


def rbr(D):
    return np.minimum(D['s15'], D['bL']), np.maximum(D['e15'], D['bH'])


def met(i):
    return dict(mIoU=float(i.mean()), i5=float((i >= 0.5).mean()), i7=float((i >= 0.7).mean()))


METHODS = {'base': lambda D: (D['bL'], D['bH']), 'RBR': rbr, 'CSR': csr}

print('=' * 118)
print('[A] 逐 rep 指标与 mean±std（K_REP=%d, N=%d）' % (len(REPS), REPS[0]['n']))
print('=' * 118)
print('%-28s %s' % ('', ' '.join(['%16s' % x for x in
      ['ALL_mIoU', 'ALL@.5', 'ALL@.7', 'S_mIoU', 'S@.5', 'S@.7', 'L_mIoU', 'L@.5', 'L@.7']])))

store = {}
for name, fn in METHODS.items():
    for bucket in ('all', 'short', 'long'):
        vals = {k: [] for k in ('mIoU', 'i5', 'i7')}
        for D in REPS:
            lo, hi = fn(D)
            i = iou(lo, hi, D['GS'], D['GE'])
            SH = D['GL'] <= THR
            sel = slice(None) if bucket == 'all' else (SH if bucket == 'short' else ~SH)
            m = met(i[sel])
            for k in vals:
                vals[k].append(m[k])
        store[(name, bucket)] = {k: (np.mean(v), np.std(v)) for k, v in vals.items()}

for name in ('base', 'RBR', 'CSR'):
    s = '%-28s' % name
    for bucket in ('all', 'short', 'long'):
        for k in ('mIoU', 'i5', 'i7'):
            mu, sd = store[(name, bucket)][k]
            s += ' %7.4f±%.4f' % (mu, sd)
    print(s)

print('\n%-28s %s' % ('Δ vs base', ' '.join(['%16s' % x for x in
      ['ALL_mIoU', 'ALL@.5', 'ALL@.7', 'S_mIoU', 'S@.5', 'S@.7', 'L_mIoU', 'L@.5', 'L@.7']])))
for name in ('RBR', 'CSR'):
    s = '%-28s' % name
    for bucket in ('all', 'short', 'long'):
        for k in ('mIoU', 'i5', 'i7'):
            d = 100 * (store[(name, bucket)][k][0] - store[('base', bucket)][k][0])
            s += ' %+16.3f' % d
    print(s)

print('\n' + '=' * 118)
print('[B] 配对 bootstrap（B=2000）+ McNemar（跨阈值计数）—— CSR vs base')
print('=' * 118)
rng = np.random.default_rng(0)
B = 2000
for bucket, sel_fn in (('all', lambda D: np.ones(len(D['cb']), bool)),
                       ('short', lambda D: D['GL'] <= THR),
                       ('long', lambda D: D['GL'] > THR)):
    d_miou, d_5, d_7 = [], [], []
    b5 = c5 = b7 = c7 = 0
    for D in REPS:
        lo_b, hi_b = D['bL'], D['bH']
        lo_c, hi_c = csr(D)
        i_b = iou(lo_b, hi_b, D['GS'], D['GE'])
        i_c = iou(lo_c, hi_c, D['GS'], D['GE'])
        sel = sel_fn(D)
        ib, ic = i_b[sel], i_c[sel]
        n = len(ib)
        idx = rng.integers(0, n, size=(B, n))
        d_miou.append(ic[idx].mean(1) - ib[idx].mean(1))
        d_5.append((ic[idx] >= 0.5).mean(1) - (ib[idx] >= 0.5).mean(1))
        d_7.append((ic[idx] >= 0.7).mean(1) - (ib[idx] >= 0.7).mean(1))
        b5 += int(((i_b < 0.5) & (i_c >= 0.5))[sel].sum()); c5 += int(((i_b >= 0.5) & (i_c < 0.5))[sel].sum())
        b7 += int(((i_b < 0.7) & (i_c >= 0.7))[sel].sum()); c7 += int(((i_b >= 0.7) & (i_c < 0.7))[sel].sum())
    dm = np.concatenate(d_miou); d5 = np.concatenate(d_5); d7 = np.concatenate(d_7)

    def ci(v):
        return (100 * np.percentile(v, 2.5), 100 * np.percentile(v, 97.5), 100 * (v <= 0).mean())

    a = ci(dm); b_ = ci(d5); c_ = ci(d7)
    m_dm = float(np.mean(dm)) * 100
    m_d5 = float(np.mean(d5)) * 100
    m_d7 = float(np.mean(d7)) * 100
    print('  %-6s ΔmIoU %+6.3f [%+6.3f,%+6.3f] p(<=0)=%.4f | Δ@0.5 %+6.3f [%+6.3f,%+6.3f] p(<=0)=%.4f | Δ@0.7 %+6.3f [%+6.3f,%+6.3f] p(<=0)=%.4f' % (
        bucket, m_dm, a[0], a[1], a[2], m_d5, b_[0], b_[1], b_[2], m_d7, c_[0], c_[1], c_[2]))
    # McNemar（正态近似，含连续性校正）
    for nm, bb, cc in (('@0.5', b5, c5), ('@0.7', b7, c7)):
        stat = ((abs(bb - cc) - 1) ** 2 / (bb + cc)) if (bb + cc) > 0 else 0.0
        import math
        p = math.erfc(math.sqrt(stat / 2.0))
        print('      McNemar %s: 升(%d) 降(%d)  chi2=%.2f  p≈%.2e' % (nm, bb, cc, stat, p))

print('\n' + '=' * 118)
print('[C] 诊断：平移行为')
print('=' * 118)
for D in REPS[:1]:
    c1 = est_win_vec(smoothZ(D['Z'], 7), D['cb'], 0.20)
    d = c1 - D['cb']
    d = np.where(np.abs(d) < 0.10 * D['wb'], 0.0, d)
    c2 = np.clip(D['cb'] + 1.00 * d, 0, 1)
    Z = smoothZ(D['Z'], 7)
    din = inside_score(Z, c2 - D['wb'] / 2, c2 + D['wb'] / 2) - inside_score(Z, D['bL'], D['bH'])
    acc = din >= 0.01
    mv = np.abs(d) > 0
    both = mv & acc
    CGT = 0.5 * (D['GS'] + D['GE'])
    for nm, sel in (('ALL', np.ones(len(D['cb']), bool)), ('short', D['GL'] <= THR), ('long', D['GL'] > THR)):
        print('  %-6s n=%4d  超dead-zone率=%.3f  自洽接受率=%.3f  最终移动率=%.3f  平均|Δ|=%.4f' % (
            nm, int(sel.sum()), mv[sel].mean(), acc[sel].mean(), both[sel].mean(), np.abs(0.9 * d[sel]).mean()))
