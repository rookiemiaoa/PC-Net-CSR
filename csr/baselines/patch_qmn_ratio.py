# -*- coding: utf-8 -*-
"""Fix QMN ratio: correct default is 1, not 0.

QMN train.py: `--ratio default=1` (proposal fusing ratio) and
`--margin-dis default=0`. The earlier patch wrongly set ratio=0, which
disables proposal fusion and collapses mIoU to ~0.21. Set ratio=1.

路径：默认取与 PC-Net 仓库同级的 ws_base/，可用环境变量 WS_BASE 覆盖。
"""
import io, json, glob, os

import sys                                    # noqa: E402

# ---- 外部基线工作区（内含 QMN/ pps/ CCR/ 克隆目录）：默认与 PC-Net 仓库同级 ----
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = _HERE
while _REPO != os.path.dirname(_REPO) and not os.path.exists(os.path.join(_REPO, 'models', 'pcnet.py')):
    _REPO = os.path.dirname(_REPO)
sys.path.insert(0, _REPO)

from csr.paths import WS_BASE                 # noqa: E402

# 1. Fix the QMN config JSON (ratio 0 -> 1)
for p in glob.glob(os.path.join(WS_BASE, 'QMN/config/charades/*.json')):
    cfg = json.load(io.open(p, encoding='utf-8'))
    mc = cfg.get('model', {}).get('config', {})
    if 'ratio' in mc:
        mc['ratio'] = 1
    if 'margin_dis' in mc:
        mc['margin_dis'] = 0
    io.open(p, 'w', encoding='utf-8').write(json.dumps(cfg, indent=2, ensure_ascii=False))
    print('fixed ratio=1 in', p)

# 2. Sanity print
cfg = json.load(io.open(os.path.join(WS_BASE, 'QMN/config/charades/cg_main.json'), encoding='utf-8'))
print('cg_main ratio =', cfg['model']['config'].get('ratio'),
      'margin_dis =', cfg['model']['config'].get('margin_dis'))
