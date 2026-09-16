# -*- coding: utf-8 -*-
"""w46_proof_check.py —— 为建议 #2「完整命题证明」做数值核对。

两件事：
 (A) 命题 1 的闭式收缩率。证明给出 κ(h) = 1 − m(h)/h，其中 m(h) 是 z 在
     非对称窗 [h−B, h+B] 上截断后的质心（B = β·w_b，h = c_g − c_b）。
     这里对高斯核数值求 κ(h)，核对 (i) 两端极限（点质量 κ→1、均匀 κ→0）
     与 (ii) 与实测的 1/1.763 = 0.567 是否在同一量级。

 (B) 命题 2 的门限 a* = ḡ₋/(ḡ₊+ḡ₋) 需要一个「方向错时的位移倍数」κ_w 与
     「方向对时的收缩倍数」κ_e。两者不能由几何单独定出，但可由五个工作点
     的实测 (方向准确率 a, 中心 MAE 变化) 反解：
         MAE_c1 / MAE_base = a·(1−κ_e) + (1−a)·(1+κ_w)
     5 个方程 2 个未知数 ⇒ 最小二乘拟合。再算 a* ≈ κ_w/(κ_e+κ_w)，
     核对它是否把五个工作点的 CSR 符号分开。

这是「2 参数拟合 5 点」的一致性检验，不是独立验证 —— 论文里必须照此表述。
"""
import json
import os

import numpy as np

_trapz = getattr(np, 'trapezoid', None) or np.trapz      # numpy 2.x 改名

# 输出目录：默认写到本脚本上一级的 figures/，可用 CSR_OUTPUT_DIR 覆盖。
_HERE = os.path.dirname(os.path.abspath(__file__))
OUTD = os.environ.get('CSR_OUTPUT_DIR') or os.path.join(_HERE, 'figures')
os.makedirs(OUTD, exist_ok=True)

# ---------- (A) 命题 1 闭式 ----------
BETA, WB = 0.20, 0.351122111082077          # Ch/PC-Net 的中位框宽
B = BETA * WB
H_GRID = np.linspace(3e-3, 1.6 * B, 36)
U_GRID = np.linspace(-1, 1, 2001)            # 归一化到窗半宽


def kappa_gauss(h, B, sigma):
    """κ(h) = 1 − m(h)/h，m 为高斯在 [h−B, h+B] 上截断后的质心。"""
    u = h + U_GRID * B
    z = np.exp(-u ** 2 / (2 * sigma ** 2))
    s = z.sum()
    if s <= 1e-300:
        return np.nan
    m = (u * z).sum() / s
    return 1.0 - m / h


# 反解与实测 0.567 相符的 sigma（用 h 的中位趋势代表）
target = 1.0 / 1.763
best = None
for sigma in np.exp(np.linspace(np.log(0.01), np.log(1.5), 220)):
    ks = np.array([kappa_gauss(h, B, sigma) for h in H_GRID])
    ks = ks[np.isfinite(ks) & (ks > 0) & (ks < 1)]
    if ks.size:
        k = float(np.mean(ks))
        d = abs(k - target)
        if best is None or d < best[0]:
            best = (d, sigma, k)
_, sig_hat, k_hat = best
print('=' * 78)
print('[A] 命题 1 闭式核对  B = β·w_b = %.4f' % B)
print('    实测收缩率 1/1.763 = %.4f' % target)
print('    高斯核下 κ 中位数 = %.4f  ⇒ 反解 σ̂ = %.4f' % (k_hat, sig_hat))
# 两端极限（点质量在窗内 ⇒ κ→1；均匀 ⇒ κ→0）
for sg, nm in ((0.006, 'σ→0（近点质量）'), (60.0, 'σ→∞（近均匀）')):
    g = H_GRID[::2] if sg > 1 else H_GRID[H_GRID < 0.9 * B]
    ks = np.array([kappa_gauss(h, B, sg) for h in g])
    ks = ks[np.isfinite(ks)]
    print('    %-16s κ ∈ [%.4f, %.4f]' % (nm, ks.min(), ks.max()))

# ---------- (B) 命题 2 门限 ----------
# 五个工作点：(名称, 方向准确率 a, 基础中心 MAE, 曲线质心 c1 的 MAE, CSR ΔmIoU)
PTS = [
    ('PC-Net', 0.7760416666666666, 0.17092924871967025, 0.14978498195997397, +1.05),
    ('CNM', 0.6999105545617174, 0.19841921611608043, 0.17552370992275132, +3.00),
    ('CPL', 0.5095541401273885, 0.17786749859016607, 0.19056636895428214, -3.76),
    ('QMN', 0.4716565767749037, 0.17670241064147457, 0.18951677376874784, -3.53),
    ('CCR', 0.44668911335578004, 0.17026370693154058, 0.18355780805735397, -3.18),
]
# (1−a)·κ_w − a·κ_e = MAE_c1/MAE_base − 1
A = np.array([[(1 - a), -a] for _, a, _, _, _ in PTS])
b = np.array([mc / mb - 1.0 for _, _, mb, mc, _ in PTS])
sol, res, rank, sv = np.linalg.lstsq(A, b, rcond=None)
k_w, k_e = float(sol[0]), float(sol[1])
a_star = k_w / (k_e + k_w)
print('\n[B] 命题 2 门限反解（5 点最小二乘，2 参数）')
print('    κ_e（方向对时的收缩倍数） = %.4f' % k_e)
print('    κ_w（方向错时的位移倍数） = %.4f' % k_w)
print('    r = MAE_c1/MAE_base  实测 vs 模型:')
for (nm, a, mb, mc, dm) in PTS:
    pred = a * (1 - k_e) + (1 - a) * (1 + k_w)
    print('      %-7s a=%.3f  Δ=%+5.2f  r_obs=%.4f  r_fit=%.4f' % (nm, a, dm, mc / mb, pred))
print('    ⇒ a* = κ_w/(κ_e+κ_w) = %.4f' % a_star)
print('    符号核对（a > a* 应 CSR>0）:')
ok = True
for (nm, a, _, _, dm) in PTS:
    pred_sign = '正' if a > a_star else '负'
    obs_sign = '正' if dm > 0 else '负'
    hit = pred_sign == obs_sign
    ok &= hit
    print('      %-7s a=%.3f  a*=%.3f  预测 %s / 实测 %s  %s' % (
        nm, a, a_star, pred_sign, obs_sign, 'OK' if hit else 'MISS'))
print('    五/五点符号一致:', ok)
_OUT_JSON = os.path.join(OUTD, 'w46_proof.json')
json.dump({'sigma_hat': sig_hat, 'kappa_hat': k_hat, 'kappa_target': target,
           'k_e': k_e, 'k_w': k_w, 'a_star': a_star, 'sign_all_ok': bool(ok)},
          open(_OUT_JSON, 'w'),
          ensure_ascii=False, indent=1)
print('\nWROTE %s' % _OUT_JSON)
