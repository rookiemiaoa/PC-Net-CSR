# -*- coding: utf-8 -*-
"""w47_traj_analysis.py —— 训练轨迹标定（建议 #4）的汇总与门限反解。

读 w45_*.json（逐 epoch 轨迹点，只换检查点、其余口径不变）与 w34/w36 的
收敛工作点，合成一张「第二坐标 a → CSR 符号」的完整表，并做两件事：

 (1) 经验括号：所有 Δ<0 的点里最大的 a，与所有 Δ>0 的点里最小的 a，
     给出符号翻转的实测夹逼区间 —— 这比"0.51–0.70 无工作点"更紧。
 (2) 两参数反解：由 MAE(c1)/MAE(cb) = a(1−κ_e) + (1−a)(1+κ_w) 最小二乘
     解出 (κ_e, κ_w)，再算 a* = κ_w/(κ_e+κ_w)；分别只用收敛点、用全部点。

输出: <CSR_CURVE_DIR>/w47_traj_calibration.json
      （默认 = <repo>/results/curve_availability/w47_traj_calibration.json；
        可用 argv[1] 或环境变量 CSR_CURVE_DIR 指定其它目录。）
"""
import glob
import json
import os
import sys

import numpy as np

# 数据目录：默认取发布包内的 results/curve_availability，可用 argv[1] 覆盖。
_HERE = os.path.dirname(os.path.abspath(__file__))
DIAG = (sys.argv[1] if len(sys.argv) > 1
        else os.environ.get('CSR_CURVE_DIR',
                            os.path.join(_HERE, '..', '..', 'results', 'curve_availability')))

# ---------- 收敛工作点（表 26 口径，来自 w34 / w36） ----------
CONV = [
    ('PC-Net', 'pcnet', 'w34_pcnet.json'),
    ('CNM', 'cnm', 'w36_cnm.json'),
    ('CPL', 'cpl', 'w34_cpl.json'),
    ('QMN', 'qmn', 'w34_qmn.json'),
    ('CCR', 'ccr', 'w34_ccr.json'),
]


def load(fn):
    p = os.path.join(DIAG, fn)
    return json.load(open(p)) if os.path.exists(p) else None


def row(label, stage, a, mb, mc, delta, rho, extra=''):
    return dict(label=label, stage=stage, a=float(a), mae_base=float(mb), mae_c1=float(mc),
                delta=float(delta), rho=float(rho), r=float(mc) / float(mb), extra=extra)


ROWS = []
for name, tag, fn in CONV:
    d = load(fn)
    if d is None:
        print('[warn] 缺 %s' % fn)
        continue
    c = d['curve']
    ROWS.append(row(name, 'converged', c['dir_acc_c1'], c['mae_base'], c['mae_c1'],
                    d['rows']['CSR']['dm'], d['rho_geo'], fn))

for fn in sorted(glob.glob(os.path.join(DIAG, 'w45_*.json'))):
    base = os.path.basename(fn)
    d = json.load(open(fn))
    stem = base[len('w45_'):-len('.json')]          # 形如 cnm_e2 / cnm_best / ccr_e1
    bk, ep = stem.rsplit('_', 1)
    c = d['curve']
    ROWS.append(row(bk.upper() if bk != 'cnm' else 'CNM', 'epoch ' + ep,
                    c['dir_acc_c1'], c['mae_base'], c['mae_c1'],
                    d['rows']['CSR']['dm'], d['rho_geo'], base))

ROWS.sort(key=lambda r: r['a'])

# ---------- 去重：同一检查点的重复测量 ----------
# CPL 的 model-best 既是"收敛点"又在轨迹里单独评过一次；
# CNM 的 model-best 与 model-40 是同一个文件。按 (a, MAE) 精确匹配去重。
seen = {}
DED = []
for r in ROWS:
    key = (round(r['a'], 6), round(r['mae_base'], 6), round(r['mae_c1'], 6), round(r['delta'], 6))
    if key in seen:
        seen[key].append('%s/%s' % (r['label'], r['stage']))
        continue
    seen[key] = ['%s/%s' % (r['label'], r['stage'])]
    DED.append(r)
dup_groups = [v for v in seen.values() if len(v) > 1]

print('=' * 108)
print('原始测量 %d 次；互异工作点 %d 个；重复组 %d 个' % (len(ROWS), len(DED), len(dup_groups)))
for g in dup_groups:
    print('   重复(同一检查点): %s' % ' = '.join(g))

ROWS = DED
print('%-10s %-12s %8s %8s %8s %8s %9s' % ('工作点', '阶段', 'a', 'MAE(cb)', 'MAE(c1)', 'ΔmIoU', 'ρ_geo'))
for r in ROWS:
    print('%-10s %-12s %8.4f %8.4f %8.4f %+8.2f %9.3f' % (
        r['label'], r['stage'], r['a'], r['mae_base'], r['mae_c1'], r['delta'], r['rho']))

neg = [r for r in ROWS if r['delta'] < 0]
pos = [r for r in ROWS if r['delta'] > 0]
a_hi_neg = max(r['a'] for r in neg) if neg else float('nan')
a_lo_pos = min(r['a'] for r in pos) if pos else float('nan')
print('\n[经验括号] 所有 Δ<0 的点中最大 a = %.4f（n=%d, %s）' % (
    a_hi_neg, len(neg), ', '.join('%s/%s' % (r['label'], r['stage']) for r in neg if r['a'] == a_hi_neg)))
print('           所有 Δ>0 的点中最小 a = %.4f（n=%d, %s）' % (
    a_lo_pos, len(pos), ', '.join('%s/%s' % (r['label'], r['stage']) for r in pos if r['a'] == a_lo_pos)))
print('           ⇒ 符号翻转的实测夹逼区间 a* ∈ (%.4f, %.4f)' % (a_hi_neg, a_lo_pos))
print('           序号一致性：按 a 升序排列的符号序列 = %s' % ''.join(
    '+' if r['delta'] > 0 else '-' for r in ROWS))


def fit(rows):
    A = np.array([[(1 - r['a']), -r['a']] for r in rows])
    b = np.array([r['r'] - 1.0 for r in rows])
    sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    k_w, k_e = float(sol[0]), float(sol[1])
    pred = A @ sol
    return k_e, k_w, k_w / (k_e + k_w), float(np.abs(pred - b).max()), len(rows)


OUT = {'points': ROWS, 'a_hi_neg': a_hi_neg, 'a_lo_pos': a_lo_pos,
       'n_distinct': len(ROWS), 'dup_groups': dup_groups}
for nm, sub in (('conv_only', [r for r in ROWS if r['stage'] == 'converged']), ('all', ROWS)):
    k_e, k_w, a_star, mx, n = fit(sub)
    print('\n[两参数反解 · %s]  n=%d  κ_e=%.4f  κ_w=%.4f  ⇒ a*=%.4f  (最大残差 %.4f)' % (
        nm, n, k_e, k_w, a_star, mx))
    OUT[nm] = dict(k_e=k_e, k_w=k_w, a_star=a_star, max_resid=mx, n=n)

with open(os.path.join(DIAG, 'w47_traj_calibration.json'), 'w') as f:
    json.dump(OUT, f, ensure_ascii=False, indent=1)
print('\nWROTE %s/w47_traj_calibration.json' % DIAG)
