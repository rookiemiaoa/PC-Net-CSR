# -*- coding: utf-8 -*-
"""Patch QMN main_runner.py: fix UnboundLocalError 'res' in non-vote eval branch.

The else (non-vote) branch ends with `predictions_list.extend(batch['raw'])` at
24-space indent but never assigns `res` (the vote branch assigns
`res = top_1_metric(...)`). The subsequent `for key, v in res.items()` at
20-space indent then raises UnboundLocalError.

Fix (matches CPL lineage): insert `res = top_1_metric(selected_props[:, 0], gt)`
at 24-space indent, right after `predictions_list.extend(...)` in the else branch.
"""
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
p = os.path.join(WS_BASE, 'QMN/runners/main_runner.py')
s = io.open(p, encoding='utf-8').read()

# target: the else-branch tail (24-space `predictions_list.extend` followed by
# the 20-space `for key, v in res.items()`). Use the exact blank-line + indent.
old = (
    "                        predictions_list.extend(batch['raw']) \n"
    "                \n"
    "                \n"
    "                    for key, v in res.items():"
)

new = (
    "                        predictions_list.extend(batch['raw']) \n"
    "                        res = top_1_metric(selected_props[:, 0], gt)\n"
    "                \n"
    "                    for key, v in res.items():"
)

assert old in s, 'target block not found'
assert s.count(old) == 1, 'target block not unique (count=%d)' % s.count(old)
s = s.replace(old, new)
io.open(p, 'w', encoding='utf-8').write(s)
print('PATCHED QMN eval (24-space indent)')
