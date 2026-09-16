# -*- coding: utf-8 -*-
"""eval_cnm_base.py —— 用 CNM 官方权重/接口做 WSCMR 口径评测（含长短分桶）

（历史文件名 eval_cnm.py。）

用法（在 CNM 仓库根目录下执行，因为要 import 该仓库的 models/datasets）：
  python <本仓库>/csr/eval/eval_cnm_base.py <config.json> <ckpt.pt>
环境变量：
  BS        batch size（默认 128，与官方 runner 一致）
  THR       短/长分桶阈值（gt 时长占比，默认 0.35）
  OFFICIAL  1 = 用官方 runners.main_runner.calculate_IoU_batch2 计算 IoU（默认 1）
输出：ALL / short / long 三桶的 mIoU, R1@0.5, R1@0.7
"""
import json
import os
import sys

import numpy as np
import torch
from torch.utils import data as tdata

sys.path.insert(0, '.')

import datasets as da
import models as md
from utils import load_json

CFG = sys.argv[1]
CKPT = sys.argv[2]
THR = float(os.environ.get('THR', 0.35))
BS = int(os.environ.get('BS', 128))
USE_OFFICIAL = os.environ.get('OFFICIAL', '1') == '1'

args = load_json(CFG)
dargs = args['dataset']

with open(dargs['vocab_path'], 'rb') as fp:
    import pickle
    vocab = pickle.load(fp)

cls = getattr(da, dargs['dataset'])
ds = cls(data_path=dargs['test_data'], vocab=vocab, args=dargs)
loader = tdata.DataLoader(ds, batch_size=BS, shuffle=False,
                          collate_fn=ds.collate_data, num_workers=8)

mcfg = dict(args['model']['config'])
mcfg['max_num_words'] = dargs['max_num_words']
mcfg['frames_input_size'] = dargs['frame_dim']
mcfg['words_input_size'] = dargs['word_dim']
mcfg['vocab_size'] = ds.vocab_size

model = getattr(md, args['model']['name'])(mcfg).cuda()
sd = torch.load(CKPT, map_location='cpu')
model.load_state_dict(sd['model_parameters'])
model.eval()
print('loaded %s | %s | test=%s | N=%d | BS=%d | num_updates=%s' % (
    CKPT.split('/')[-2], args['model']['name'], dargs['test_data'], len(ds), BS,
    sd.get('num_updates', '?')), flush=True)

if USE_OFFICIAL:
    # 官方实现的 IoU（与 runners/main_runner.py 完全一致），用于消除口径歧义
    def iou_fn(p0, p1, g0, g1):
        union = (np.minimum(p0, g0), np.maximum(p1, g1))
        inter = (np.maximum(p0, g0), np.minimum(p1, g1))
        iou = (inter[1] - inter[0] + 1e-10) / (union[1] - union[0] + 1e-10)
        iou[union[1] - union[0] < -1e-5] = 0
        iou[iou < 0] = 0.0
        return iou
else:
    def iou_fn(p0, p1, g0, g1):
        inter = np.maximum(0.0, np.minimum(p1, g1) - np.maximum(p0, g0))
        union = np.maximum(p1, g1) - np.minimum(p0, g0)
        return np.where(union > 1e-9, inter / np.maximum(union, 1e-9), 0.0)


P0, P1, G0, G1, GL = [], [], [], [], []
CTR, WID = [], []
with torch.no_grad():
    for batch in loader:
        raw = batch['raw']
        durations = np.asarray([r[1] for r in raw], dtype=np.float64)
        gt = np.asarray([r[2] for r in raw], dtype=np.float64)
        net_input = {k: (v.cuda() if torch.is_tensor(v) else v)
                     for k, v in batch['net_input'].items()}
        out = model(**net_input)
        bsz = len(durations)
        w = out['width'].view(bsz)
        c = out['center'].view(bsz)
        p0 = torch.clamp(c - w / 2, min=0).cpu().numpy()
        p1 = torch.clamp(c + w / 2, max=1).cpu().numpy()
        g0 = gt[:, 0] / durations
        g1 = gt[:, 1] / durations
        P0.append(p0); P1.append(p1); G0.append(g0); G1.append(g1)
        GL.append(np.maximum(g1 - g0, 0.0))
        CTR.append(c.cpu().numpy()); WID.append(w.cpu().numpy())

P0 = np.concatenate(P0); P1 = np.concatenate(P1)
G0 = np.concatenate(G0); G1 = np.concatenate(G1)
GL = np.concatenate(GL)
CTR = np.concatenate(CTR); WID = np.concatenate(WID)

iou = iou_fn(P0, P1, G0, G1)
short = GL <= THR
print('\n%-8s %6s %10s %10s %10s' % ('bucket', 'n', 'mIoU', 'R1@0.5', 'R1@0.7'), flush=True)
for name, sel in (('ALL', np.ones(len(iou), bool)), ('short', short), ('long', ~short)):
    i = iou[sel]
    print('%-8s %6d %10.4f %10.4f %10.4f' % (name, int(sel.sum()), i.mean(),
                                             (i >= 0.5).mean(), (i >= 0.7).mean()), flush=True)
print('\n[box]  median width of prediction = %.4f | median gt dur = %.4f | mean width = %.4f' % (
    np.median(P1 - P0), np.median(GL), (P1 - P0).mean()), flush=True)
