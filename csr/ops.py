# -*- coding: utf-8 -*-
"""csr.ops —— CSR / RBR 的纯 NumPy 算子实现（单一权威来源）。

本模块是「中心重定位（CSR）」与「边界扩框（RBR）」两个推理期精修算子的
**唯一实现处**。所有 eval / analysis 脚本必须从这里 import，禁止各自复制粘贴，
以保证：

- 论文里的数字来自同一份代码（可复现性的根基）；
- 命名统一（snake_case）、接口清晰、超参有唯一默认值。

本模块只依赖 NumPy（`csr/__init__.py` 惰性导出，`import csr.ops` 不引入 torch）。
NumPy 之外的逐点相关性曲线（`-nll(t)`）由调用方在 GPU 上算好后传入。

命名约定
--------
- 函数名与局部变量统一 snake_case。
- 几何量约定（归一化，∈ [0, 1]）：
    `cb`   baseline 框中心（center）
    `wb`   baseline 框宽度（width）
    `lo` / `hi`  框的起 / 止（start / end），lo < hi
    `gs` / `ge`  ground-truth 起 / 止
    `gl`   gt 时长 = ge - gs
- 批量约定：除特别说明外，输入均为 ``[N, L]`` 或 ``[N]`` 的 float64 数组，
  ``N`` 为样本数，``L`` 为逐点提案数（默认 50 = max_num_frames // 4）。

超参默认值（在 data/charades/train.json 上校准，测试集原样使用）
--------
``smooth=7, beta=0.20, rho=0.10, alpha=1.00, dmin=0.01``
"""
import numpy as np

__all__ = [
    'DEFAULT_CSR_HYPERPARAMS',
    'iou',
    'smooth_curve',
    'zscore',
    'inside_score',
    'estimate_window_center',
    'shift_center',
    'refine_center_c2',
    'refine_center',
    'isr_shift',
    'flood_expand',
    'boxes_from_center_width',
    'metrics_from_iou',
]

#: CSR 定版超参（train.json 校准，测试集原样使用；见 csr/README.md §4）。
DEFAULT_CSR_HYPERPARAMS = dict(
    smooth=7,
    beta=0.20,
    rho=0.10,
    alpha=1.00,
    dmin=0.01,
)


def iou(lo_a, hi_a, lo_b, hi_b):
    """向量化 IoU。

    Args
    ----
    lo_a, hi_a : array_like ``[N]`` 或 ``[N, 1]``
        预测框的起 / 止。
    lo_b, hi_b : array_like ``[N]`` 或 ``[N, K]``
        ground-truth（或另一组框）的起 / 止。

    Returns
    -------
    np.ndarray
        形状经广播后逐对 IoU；空并集（union <= 1e-9）置 0。
    """
    lo_a, hi_a, lo_b, hi_b = map(np.asarray, (lo_a, hi_a, lo_b, hi_b))
    inter = np.maximum(0.0, np.minimum(hi_a, hi_b) - np.maximum(lo_a, lo_b))
    union = np.maximum(hi_a, hi_b) - np.minimum(lo_a, lo_b)
    return np.where(union > 1e-9, inter / np.maximum(union, 1e-9), 0.0)


def smooth_curve(z, window):
    """对相关性曲线做滑动平均（边缘用最近值填充，保持长度不变）。

    Args
    ----
    z : np.ndarray ``[N, L]``
        逐样本相关性曲线（已 z-score 化，或任意实数曲线）。
    window : int
        平滑窗口；``<= 1`` 表示不平滑，原样返回。

    Returns
    -------
    np.ndarray ``[N, L]``
        平滑后的曲线。
    """
    z = np.asarray(z, dtype=np.float64)
    w = int(window)
    if w <= 1:
        return z
    pad = w // 2
    zp = np.concatenate([np.repeat(z[:, :1], pad, axis=1), z,
                         np.repeat(z[:, -1:], pad, axis=1)], axis=1)
    cs = np.concatenate([np.zeros((len(z), 1)), np.cumsum(zp, axis=1)], axis=1)
    return (cs[:, w:] - cs[:, :-w]) / float(w)


def zscore(rel):
    """按行（逐样本）z-score 标准化。

    Args
    ----
    rel : np.ndarray ``[N, L]``
        相关性曲线 r(t) = -nll(t)。

    Returns
    -------
    np.ndarray ``[N, L]``
        (rel - mean) / (std + 1e-8)，按行计算。
    """
    rel = np.asarray(rel, dtype=np.float64)
    return (rel - rel.mean(1, keepdims=True)) / (rel.std(1, keepdims=True) + 1e-8)


def inside_score(z, lo, hi, centers):
    """区间 ``[lo, hi]`` 内相关性曲线的平均取值。

    用 ``centers``（逐点提案中心位置网格）做离散化定位，取落在区间内的网格点均值。

    Args
    ----
    z : np.ndarray ``[N, L]``
        相关性曲线。
    lo, hi : np.ndarray ``[N]``
        区间起 / 止（归一化）。
    centers : np.ndarray ``[L]``
        逐点提案中心位置（升序网格，如 ``(arange(L)+0.5)/L``）。

    Returns
    -------
    np.ndarray ``[N]``
        各样本区间内 z 的均值。
    """
    z = np.asarray(z)
    centers = np.asarray(centers)
    L = z.shape[1]
    cs = np.concatenate([np.zeros((len(z), 1)), np.cumsum(z, axis=1)], axis=1)
    li = np.clip(np.searchsorted(centers, lo), 0, L - 1)
    hi_i = np.clip(np.searchsorted(centers, hi, side='right') - 1, 0, L - 1)
    cnt = np.maximum(hi_i - li + 1, 1)
    tot = cs[np.arange(len(z)), hi_i + 1] - cs[np.arange(len(z)), li]
    return tot / cnt


def estimate_window_center(z, cb, beta, centers):
    """窗口 ``[cb - beta, cb + beta]`` 内正部加权质心。

    Args
    ----
    z : np.ndarray ``[N, L]``
        相关性曲线（通常已平滑）。
    cb : np.ndarray ``[N]``
        baseline 框中心。
    beta : float
        搜索窗口半径（归一化时长单位）。
    centers : np.ndarray ``[L]``
        逐点提案中心网格。

    Returns
    -------
    np.ndarray ``[N]``
        估计的中心 c1；若窗口内正质量为零，回退为 cb。
    """
    z = np.asarray(z)
    cb = np.asarray(cb)
    centers = np.asarray(centers)
    mask = np.abs(centers[None, :] - cb[:, None]) <= beta
    pm = np.where(mask, np.maximum(z, 0.0), 0.0)
    ssum = pm.sum(1)
    c = (pm @ centers) / np.maximum(ssum, 1e-12)
    bad = ssum < 1e-12
    if bad.any():
        c = c.copy()
        c[bad] = cb[bad]
    return c


def shift_center(cb, wb, c1, rho, alpha):
    """dead-zone + 阻尼的位移。

    Args
    ----
    cb : np.ndarray ``[N]``
        baseline 中心。
    wb : np.ndarray ``[N]``
        baseline 宽度。
    c1 : np.ndarray ``[N]``
        窗口质心估计的中心。
    rho : float
        dead-zone 阈值（位移幅度 < rho*wb 时置零）。
    alpha : float
        位移阻尼系数。

    Returns
    -------
    np.ndarray ``[N]``
        阻尼后的新中心 c2（clip 到 [0, 1]）。
    """
    d = np.asarray(c1) - np.asarray(cb)
    d = np.where(np.abs(d) < rho * np.asarray(wb), 0.0, d)
    return np.clip(np.asarray(cb) + alpha * d, 0, 1)


def refine_center_c2(z, cb, wb, centers, smooth=DEFAULT_CSR_HYPERPARAMS['smooth'],
                     beta=DEFAULT_CSR_HYPERPARAMS['beta'],
                     rho=DEFAULT_CSR_HYPERPARAMS['rho'],
                     alpha=DEFAULT_CSR_HYPERPARAMS['alpha'],
                     dmin=DEFAULT_CSR_HYPERPARAMS['dmin']):
    """CSR 核心，返回**中心**（不返回框）。需要框时用 :func:`refine_center`。

    与 :func:`refine_center` 的区别只在返回值：这里给出未做 ``[0,1]`` 裁剪的
    ``c2``，因此可以精确回答「中心被移到了哪里」。**不要**用
    ``(lo + hi) / 2`` 反推中心——框一旦被裁剪，该式不再成立。

    完整流水线（与论文 §3 一致）：
        1) 平滑相关性曲线；
        2) 窗口质心估计 c1；
        3) dead-zone + 阻尼 → c2；
        4) 自洽门控：仅当平移后框内 z 均值提升 >= dmin 才接受，否则回退 cb。

    Returns
    -------
    (c2, moved, shift) : tuple of np.ndarray
        c2    精修后的中心（已 clip 到 [0, 1]）。
        moved bool 数组，是否发生平移。
        shift |c2 - cb|，平移幅度。
    """
    z = np.asarray(z)
    cb = np.asarray(cb)
    wb = np.asarray(wb)
    # 顺序不可颠倒：先 z-score 再平滑（见 CSR_METHOD.md；先平滑会引入尺度偏置）。
    zs = smooth_curve(zscore(z), smooth)
    c1 = estimate_window_center(zs, cb, beta, centers)
    c2 = shift_center(cb, wb, c1, rho, alpha)
    if dmin is not None:
        din = inside_score(zs, c2 - wb / 2, c2 + wb / 2, centers) \
            - inside_score(zs, cb - wb / 2, cb + wb / 2, centers)
        c2 = np.where(din >= dmin, c2, cb)
    moved = np.abs(c2 - cb) > 1e-9
    return c2, moved, np.abs(c2 - cb)


def refine_center(z, cb, wb, centers, smooth=DEFAULT_CSR_HYPERPARAMS['smooth'],
                  beta=DEFAULT_CSR_HYPERPARAMS['beta'],
                  rho=DEFAULT_CSR_HYPERPARAMS['rho'],
                  alpha=DEFAULT_CSR_HYPERPARAMS['alpha'],
                  dmin=DEFAULT_CSR_HYPERPARAMS['dmin']):
    """CSR 核心：保持宽度、重定位中心。

    完整流水线（与论文 §3 一致）：
        1) 平滑相关性曲线；
        2) 窗口质心估计 c1；
        3) dead-zone + 阻尼 → c2；
        4) 自洽门控：仅当平移后框内 z 均值提升 >= dmin 才接受，否则回退 cb。

    Args
    ----
    z : np.ndarray ``[N, L]``
        相关性曲线（**未平滑**的 z-score 值；本函数内部会先平滑）。
    cb, wb : np.ndarray ``[N]``
        baseline 框的中心 / 宽度。
    centers : np.ndarray ``[L]``
        逐点提案中心网格。
    smooth, beta, rho, alpha : float
        见 DEFAULT_CSR_HYPERPARAMS。
    dmin : float or None
        自洽门控阈值；``None`` 关闭门控（消融用）。

    Returns
    -------
    (lo, hi, moved, shift) : tuple of np.ndarray
        lo, hi  精修框起 / 止（宽度保持 = wb）。
        moved   bool 数组，是否发生平移。
        shift   |c2 - cb|，平移幅度。

    Note
    ----
    **返回值是 4 元组，顺序为 (lo, hi, moved, shift)**。只要中心时请用
    :func:`refine_center_c2`；写成 ``c2, moved, shift = refine_center(...)[:3]``
    会把左边界当成中心（2026-10 在 eval_anet_support.py 中真实发生过）。
    """
    wb = np.asarray(wb)
    c2, moved, shift = refine_center_c2(z, cb, wb, centers, smooth=smooth,
                                        beta=beta, rho=rho, alpha=alpha, dmin=dmin)
    lo = np.clip(c2 - wb / 2, 0, 1)
    hi = np.clip(c2 + wb / 2, 0, 1)
    return lo, hi, moved, shift


def isr_shift(z, cb, wb, centers, beta0=0.20, gamma=0.5, T=2,
              rho=DEFAULT_CSR_HYPERPARAMS['rho'],
              alpha=DEFAULT_CSR_HYPERPARAMS['alpha'],
              dmin=DEFAULT_CSR_HYPERPARAMS['dmin']):
    """ISR：迭代收缩窗的中心位移（返回位移 ``d``，**不返回中心**）。

    与 :func:`refine_center_c2` 的差别只在中心估计：CSR 用一次固定窗
    ``beta=0.20`` 的质心，ISR 用 mean-shift 式的 ``T`` 次迭代，窗宽按
    ``beta_{t+1} = gamma * beta_t`` 收缩。其余四步（z-score→平滑、dead-zone、
    阻尼、自洽门控）完全一致。

    调用方拿到 ``d`` 后再做 ``c2 = clip(cb + d, 0, 1)``、按基线宽度建框——
    这一点必须保持，否则门控里的「最终框」会算错。

    Returns
    -------
    np.ndarray ``[N]``
        中心位移 ``d``（未通过的样本为 0）。
    """
    z = np.asarray(z)
    cb = np.asarray(cb)
    wb = np.asarray(wb)
    zs = smooth_curve(zscore(z), DEFAULT_CSR_HYPERPARAMS['smooth'])
    c = cb.copy()
    b = beta0
    for _ in range(int(T)):
        c = estimate_window_center(zs, c, b, centers)
        b = b * gamma
    d = c - cb
    d = np.where(np.abs(d) < rho * wb, 0.0, d) * alpha
    c2 = np.clip(cb + d, 0, 1)
    if dmin is not None:
        din = inside_score(zs, c2 - wb / 2, c2 + wb / 2, centers) \
            - inside_score(zs, cb - wb / 2, cb + wb / 2, centers)
        d = np.where(din >= dmin, d, 0.0)
    return d


def flood_expand(rel, cb, k, centers):
    """RBR：从 top-1 中心沿 z-score 化相关性曲线向两侧 flood 扩展。

    Args
    ----
    rel : np.ndarray ``[N, L]``
        相关性曲线（未标准化；内部会先 z-score）。
    cb : np.ndarray ``[N]``
        baseline 框中心。
    k : float
        阈值系数，扩展条件 ``z > mean(z) + k * std(z)``。
    centers : np.ndarray ``[L]``
        逐点提案中心网格（用于映射回 [0,1] 坐标）。

    Returns
    -------
    (s, e) : tuple of np.ndarray ``[N]``
        flood 得到的区间起 / 止（归一化）。
    """
    rel = np.asarray(rel)
    cb = np.asarray(cb)
    centers = np.asarray(centers)
    L = rel.shape[1]
    z = zscore(rel)
    th = z.mean(1) + k * z.std(1)
    pc = np.clip((cb * (L - 1)).round().astype(int), 0, L - 1)
    m = z > th[:, None]
    m[np.arange(len(cb)), pc] = True
    s = np.zeros(len(cb))
    e = np.ones(len(cb))
    for i in range(len(cb)):
        a1 = pc[i]
        while a1 > 0 and m[i, a1 - 1]:
            a1 -= 1
        b1 = pc[i]
        while b1 < L - 1 and m[i, b1 + 1]:
            b1 += 1
        s[i] = centers[a1]
        e[i] = centers[b1]
    return s, e


def boxes_from_center_width(center, width):
    """由中心 / 宽度构造 (lo, hi) 框（clip 到 [0,1]）。

    Args
    ----
    center, width : array_like ``[N]``

    Returns
    -------
    (lo, hi) : tuple of np.ndarray
    """
    center = np.asarray(center)
    width = np.asarray(width)
    return np.clip(center - width / 2.0, 0, 1), np.clip(center + width / 2.0, 0, 1)


def metrics_from_iou(i, thresholds=(0.5, 0.7)):
    """由 IoU 向量计算 mIoU 与各阈值命中率（论文口径）。

    Args
    ----
    i : np.ndarray ``[N]``
        IoU 向量。
    thresholds : tuple of float
        命中率阈值（默认 0.5 / 0.7）。

    Returns
    -------
    dict
        ``{'mIoU': float, 'iou<0.5>': float, 'iou<0.7>': float, ...}``。
    """
    i = np.asarray(i)
    d = {'mIoU': float(i.mean())}
    for t in thresholds:
        d['iou%.1f' % t] = float((i >= t).mean())
    return d
