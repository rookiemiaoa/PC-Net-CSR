# -*- coding: utf-8 -*-
"""csr —— PC-Net + CSR：推理期中心重定位精修算子及其评测/分析代码。

两个算子（CSR / RBR）依赖 torch，采用惰性导出（PEP 562），
因此 ``import csr.paths`` 不会因为本文件而把 torch 拉进来。
``csr.ops`` 是纯 NumPy 算子（单一权威实现），随包直接可用。

    from csr import CenterShiftRefiner          # 核心贡献：中心重定位（无子群偏置）
    from csr import RelevanceBoundaryRefiner    # 对照：边界扩框（RBR，存在方向偏置）
    from csr import ops                         # 纯 NumPy 算子（iou/平滑/质心/重定位/flood）

子模块
------
- ``csr.csr``   CSR 算子（torch 版，需加载模型）。
- ``csr.refine``  RBR 算子（torch 版）。
- ``csr.ops``   纯 NumPy 算子实现（iou / 平滑 / 质心估计 / 中心重定位 / flood），
                供 eval / analysis 脚本共用，避免复制粘贴。
- ``csr.paths``  统一路径解析（仅标准库）。
"""
from . import ops                              # noqa: F401  纯 NumPy，随包直接可用

__all__ = ['CenterShiftRefiner', 'RelevanceBoundaryRefiner', 'ops']


def __getattr__(name):
    if name == 'CenterShiftRefiner':
        from .csr import CenterShiftRefiner
        return CenterShiftRefiner
    if name == 'RelevanceBoundaryRefiner':
        from .refine import RelevanceBoundaryRefiner
        return RelevanceBoundaryRefiner
    raise AttributeError('module %r has no attribute %r' % (__name__, name))


def __dir__():
    return sorted(list(globals().keys()) + __all__)
