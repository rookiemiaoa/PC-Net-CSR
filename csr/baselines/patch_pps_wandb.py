# -*- coding: utf-8 -*-
"""patch_pps_wandb.py —— 把 pps/runner/base_runner.py 的 wandb 硬 import 改成可选。"""
import io
import os

import sys                                    # noqa: E402

# ---- 外部基线工作区（内含 QMN/ pps/ CCR/ 克隆目录）：默认与 PC-Net 仓库同级 ----
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = _HERE
while _REPO != os.path.dirname(_REPO) and not os.path.exists(os.path.join(_REPO, 'models', 'pcnet.py')):
    _REPO = os.path.dirname(_REPO)
sys.path.insert(0, _REPO)

from csr.paths import WS_BASE                 # noqa: E402
p = os.path.join(WS_BASE, 'pps/runner/base_runner.py')
s = io.open(p, encoding='utf-8').read()

# 先修复上一次被 /n 写坏的内容
s = s.replace('try:/n    import wandb\nexcept Exception:/n    wandb = None',
              'try:\n    import wandb\nexcept Exception:\n    wandb = None')
# 再做原始替换（幂等）
s = s.replace('import random\nimport wandb\n',
              'import random\ntry:\n    import wandb\nexcept Exception:\n    wandb = None\n')

io.open(p, 'w', encoding='utf-8').write(s)
print('patched OK')
