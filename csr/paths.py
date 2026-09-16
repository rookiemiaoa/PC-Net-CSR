# -*- coding: utf-8 -*-
"""csr.paths —— 统一路径解析（开源版不依赖任何机器专有路径）。

本仓库在原版 PC-Net 之上新增了推理期精修算子（CSR / RBR）及其评测、分析代码。
为了让这些脚本在任意机器上开箱可跑，所有路径统一遵守「默认自动定位，需要时用
环境变量覆盖」的约定。本模块是这套约定的唯一定义处。

常量
----
REPO_ROOT / PCNET_ROOT
    PC-Net 仓库根目录。默认从本文件位置向上查找同时含 ``models/pcnet.py``
    与 ``train.py`` 的目录；可用环境变量 ``PCNET_ROOT`` 覆盖（兼容旧名 ``LM_ROOT``）。
DATA_ROOT
    大文件所在目录（I3D / C3D 特征 hdf5、额外词表等）。默认 = ``REPO_ROOT``。
    随仓库分发的只有 ``data/*.json`` 与 ``data/*/glove.pkl``；
    ``i3d_features.hdf5`` 等需自行准备（见 README「数据准备」）。
WS_BASE
    外部弱监督基线的工作区，内含 QMN / pps / CCR 等官方仓库的克隆目录。
    默认 = ``REPO_ROOT`` 的同级目录 ``ws_base/``；可用 ``WS_BASE`` 覆盖。
CNM_ROOT
    CNM（AAAI-2022）仓库克隆目录，仅跨骨干迁移实验需要。
    默认 = ``REPO_ROOT`` 的同级目录 ``CNM-CG/``；可用 ``CNM_ROOT`` 覆盖。
CACHE_DIR
    离线分析用的特征缓存目录（``*.pkl``）。默认 = ``<REPO_ROOT>/csr/analysis/feat_cache``；
    可用 ``CACHE_DIR`` 或 ``FEAT_CACHE_DIR`` 覆盖。
OUTPUT_DIR
    评测与扫描结果的落盘目录。默认 = ``<REPO_ROOT>/csr/outputs``；
    可用 ``CSR_OUTPUT_DIR`` 覆盖。

本模块只依赖标准库。``csr/__init__.py`` 对两个算子采用惰性导出（PEP 562），
因此 ``import csr.paths`` 不会把 torch / numpy 拉进来。
"""
import os

__all__ = ['REPO_ROOT', 'PCNET_ROOT', 'DATA_ROOT', 'EXTERNAL_ROOT',
           'WS_BASE', 'CNM_ROOT', 'CACHE_DIR', 'OUTPUT_DIR',
           'find_repo_root', 'env_path']


def find_repo_root(start):
    """从 start 向上查找 PC-Net 仓库根（同时含 models/pcnet.py 与 train.py）。"""
    d = os.path.abspath(start)
    while True:
        if (os.path.exists(os.path.join(d, 'models', 'pcnet.py'))
                and os.path.exists(os.path.join(d, 'train.py'))):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            return os.path.abspath(start)
        d = parent


def env_path(*names, **kw):
    """按顺序取第一个非空环境变量作为路径；都没有则用 default。"""
    default = kw.get('default')
    for n in names:
        v = os.environ.get(n)
        if v:
            return os.path.abspath(os.path.expanduser(v))
    return default


REPO_ROOT = env_path('PCNET_ROOT', 'LM_ROOT',
                     default=find_repo_root(os.path.dirname(os.path.abspath(__file__))))
PCNET_ROOT = REPO_ROOT
EXTERNAL_ROOT = os.path.dirname(REPO_ROOT)

DATA_ROOT = env_path('DATA_ROOT', default=REPO_ROOT)
WS_BASE = env_path('WS_BASE', default=os.path.join(EXTERNAL_ROOT, 'ws_base'))
CNM_ROOT = env_path('CNM_ROOT', default=os.path.join(EXTERNAL_ROOT, 'CNM-CG'))
CACHE_DIR = env_path('CACHE_DIR', 'FEAT_CACHE_DIR',
                     default=os.path.join(REPO_ROOT, 'csr', 'analysis', 'feat_cache'))
OUTPUT_DIR = env_path('CSR_OUTPUT_DIR', default=os.path.join(REPO_ROOT, 'csr', 'outputs'))
