# -*- coding: utf-8 -*-
"""patch_baselines.py —— 修复 QMN / CCR 的 config 与 loss 签名问题。

QMN : 模型 __init__ 需要 config['ratio'] 与 config['margin_dis']。
      注意 QMN train.py 里 `--ratio` 默认值是 **1**（提案融合系数，不是 0！），
      `--margin-dis` 默认 0。ratio=0 会导致提案融合失效、mIoU 塌到 ~0.21（≈随机）。
      ⇒ 补键时 ratio 默认必须为 1，margin_dis 为 0。
CCR : ivc_loss 把 gpu_idx 声明为必需位置参数，但 runner 未传；给其默认值 0
      （单卡 CUDA_VISIBLE_DEVICES 映射后 device 即 0）。

路径：默认取与 PC-Net 仓库同级的 ws_base/，可用环境变量 WS_BASE 覆盖。
"""
import io
import json
import glob
import os

import sys                                    # noqa: E402

# ---- 外部基线工作区（内含 QMN/ pps/ CCR/ 克隆目录）：默认与 PC-Net 仓库同级 ----
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = _HERE
while _REPO != os.path.dirname(_REPO) and not os.path.exists(os.path.join(_REPO, 'models', 'pcnet.py')):
    _REPO = os.path.dirname(_REPO)
sys.path.insert(0, _REPO)

from csr.paths import WS_BASE                 # noqa: E402

# ---------- QMN: 补 ratio / margin_dis / loss 键 ----------
for p in glob.glob(os.path.join(WS_BASE, 'QMN/config/charades/*.json')):
    cfg = json.load(io.open(p, encoding='utf-8'))
    mc = cfg.get('model', {}).get('config', {})
    if 'ratio' not in mc:
        mc['ratio'] = 1          # 提案融合系数，正确默认值（QMN --ratio 默认 1）
    if 'margin_dis' not in mc:
        mc['margin_dis'] = 0
    ls = cfg.setdefault('loss', {})
    # QMN runner 直接读这些 loss 键（命令行 --vtc-loss-cof/--cc-loss-cof 默认 0，
    # 但不传时 config 里没有该键，runner 里 self.args['loss']['vtc_loss_cof'] 会 KeyError）
    for k in ('vtc_loss_cof', 'cc_loss_cof'):
        if k not in ls:
            ls[k] = 0
    io.open(p, 'w', encoding='utf-8').write(json.dumps(cfg, indent=2, ensure_ascii=False))
    print('patched QMN cfg', p)

# ---------- CCR: ivc_loss gpu_idx 默认值 ----------
p = os.path.join(WS_BASE, 'CCR/models/loss.py')
s = io.open(p, encoding='utf-8').read()
old = 'def ivc_loss(words_logit, words_id, words_mask, num_props,\n             cf_ref_loss, cf_pos_loss, cf_neg_loss_1, cf_neg_loss_2, gpu_idx,'
new = 'def ivc_loss(words_logit, words_id, words_mask, num_props,\n             cf_ref_loss, cf_pos_loss, cf_neg_loss_1, cf_neg_loss_2, gpu_idx=0,'
assert old in s, 'ivc_loss signature not found'
s = s.replace(old, new)
io.open(p, 'w', encoding='utf-8').write(s)
print('patched CCR ivc_loss gpu_idx default=0')

print('ALL_PATCH_DONE')
