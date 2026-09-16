# -*- coding: utf-8 -*-
"""perword_cache.py —— 生成「逐词相关性曲线 + 完整提案集」缓存。

为什么需要它
------------
此前所有分析（RBR / CSR / 300+ 变体）用的都是 `cal_nll_loss(...)` 的**词维平均**
相关性 r(t)。这等于假设「整条查询对时刻 t 的支持度」可以用一个标量刻画。
但 W3 的诊断显示：该曲线对**目标时长**几乎无信息（R²≈0.03）—— 它是定位信号，
不是范围信号。而 WSCMR 的查询是**组合式**的（"someone is sitting" + "in the kitchen"），
每个组成词各自的时间证据是不同的：动作词可能只在一小段成立，场景词可能全局成立。
**逐词曲线**里可能含有时长/边界的结构，而词维平均会把它抹掉。

本脚本把每个逐点提案下的**逐词 NLL** 完整落盘，供离线分析：
  RELW [N, L, W] : 点提案 (L=50) 下每个词的 -nll（词维不聚合）
  NW   [N, P, W] : 模型自己的 8 个提案下每个词的 -nll
  C8/W8/N8       : 8 个提案的中心/宽度/聚合 nll
  REL  [N, L]    : 聚合（词平均）曲线，用于与既有 CSR/RBR 口径对齐
  GS/GE/GL       : gt（仅评测与诊断用；框构造不用）
  另存 words_id / words_mask 以便按词聚类分析（动作词 vs 场景词）

用法:
  python csr/diagnostics/perword_cache.py <config> <ckpt> <out.pkl> [test|train]
环境变量: K_REP=3  BS=16  LIMIT=0
"""
import os
import pickle
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = _HERE
while _REPO != os.path.dirname(_REPO) and not os.path.exists(os.path.join(_REPO, 'models', 'pcnet.py')):
    _REPO = os.path.dirname(_REPO)
ROOT = os.environ.get('PCNET_ROOT') or _REPO
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from csr.paths import DATA_ROOT        # noqa: E402

from utils import load_json            # noqa: E402
import datasets as da                  # noqa: E402
from models import PCNet               # noqa: E402
from models.loss import cal_nll_loss   # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

cfg_p, ckpt, out_p = sys.argv[1], sys.argv[2], sys.argv[3]
split = sys.argv[4] if len(sys.argv) > 4 else 'test'
cfg = load_json(cfg_p)
dargs = cfg['dataset']
DS_NAME = dargs['dataset']
vocab = pickle.load(open(dargs['vocab_path'], 'rb'))
data_path = dargs.get('test_data') if split == 'test' else dargs.get('train_data')
ds = getattr(da, DS_NAME)(data_path=data_path, vocab=vocab, args=dargs, split=split)
VOCAB_SIZE = ds.vocab_size
COLLATE = ds.collate_data

FP = os.environ.get('FEAT_PATH', dargs['feature_path'])
if not os.path.isabs(FP) and not os.path.exists(FP):
    for cand in (os.path.join(DATA_ROOT, os.path.basename(FP)),
                 os.path.join(ROOT, os.path.basename(FP))):
        if os.path.exists(cand):
            FP = cand
            break
dargs['feature_path'] = FP

LIMIT = int(os.environ.get('LIMIT', 0))
if LIMIT > 0:
    ds = Subset(ds, list(range(min(LIMIT, len(ds)))))

sd = torch.load(ckpt, map_location='cuda')
sp = sd['model_parameters']
ck_cfg = sd.get('config', {})
mcfg = dict(ck_cfg.get('model', {}).get('config', cfg['model']['config']))
P = int(sp['slot_atten.slots'].shape[0]) if 'slot_atten.slots' in sp else mcfg.get('num_props', 8)
mcfg['num_props'] = P
mcfg['vocab_size'] = VOCAB_SIZE
mcfg['max_epoch'] = cfg['train']['max_num_epochs']
mcfg.setdefault('num_iteration', 4)

L = dargs['max_num_frames'] // 4
BS = int(os.environ.get('BS', 16))
K_REP = int(os.environ.get('K_REP', 3))
loader = DataLoader(ds, batch_size=BS, shuffle=False, collate_fn=COLLATE, num_workers=8)

print('[GPU] %s  count=%d' % (torch.cuda.get_device_name(0), torch.cuda.device_count()), flush=True)
print('config=%s\nckpt=%s\nsplit=%s  N=%d  num_props=%d  L=%d  BS=%d  K_REP=%d\nout=%s' % (
    cfg_p, ckpt, split, len(ds), P, L, BS, K_REP, out_p), flush=True)


def perword_nll(logit, idx, mask):
    """逐词 NLL（不做词维平均），形状 [B, W]，padding 位置置 0。"""
    eps = 0.1
    logit = logit.log_softmax(dim=-1)
    nll = -logit.gather(dim=-1, index=idx.unsqueeze(-1)).squeeze(-1)
    smooth = -logit.sum(dim=-1)
    nll = (1 - eps) * nll + eps / logit.size(-1) * smooth
    return nll.masked_fill(mask == 0, 0.0)


def pad_stack(chunks):
    """把块沿**最后一维**（词维）补零到公共宽度后拼接。
    Charades 查询词数逐 batch 不同，collate 只 pad 到 batch 内最大长度。"""
    Wm = max(a.shape[-1] for a in chunks)
    out = []
    for a in chunks:
        if a.shape[-1] < Wm:
            pad = np.zeros(a.shape[:-1] + (Wm - a.shape[-1],), dtype=a.dtype)
            a = np.concatenate([a, pad], axis=-1)
        out.append(a)
    return np.concatenate(out, axis=0)


def run_rep(rep):
    np.random.seed(7 + rep); torch.manual_seed(7 + rep)
    model = PCNet(mcfg).cuda()
    model.load_state_dict(sp)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    _orig_pg = model.proposal_generator
    _c_pt = torch.linspace(0.5 / L, 1 - 0.5 / L, L)
    _w_pt = torch.full((L,), max(2.0 / L, 0.02))

    A = {k: [] for k in ('C8', 'W8', 'N8', 'NW', 'REL', 'RELW', 'WID', 'WMK', 'G')}
    t0 = time.time()
    with torch.no_grad():
        for bi, batch in enumerate(loader, 1):
            net_input = {k: v.cuda() for k, v in batch['net_input'].items()}
            bsz = len(batch['raw'])
            o = model(epoch=0, **net_input)
            W = o['words_id'].shape[1]
            wm = o['words_mask'].unsqueeze(1).expand(bsz, P, -1).contiguous().view(bsz * P, -1)
            wi = o['words_id'].unsqueeze(1).expand(bsz, P, -1).contiguous().view(bsz * P, -1)
            n, _ = cal_nll_loss(o['words_logit'], wi, wm)
            A['N8'].append(n.view(bsz, P).cpu().numpy())
            A['NW'].append(perword_nll(o['words_logit'], wi, wm).view(bsz, P, W).cpu().numpy())
            A['C8'].append(o['center'].view(bsz, P).cpu().numpy())
            A['W8'].append(o['width'].view(bsz, P).cpu().numpy())

            model.proposal_generator = lambda h, b: (_c_pt.repeat(b).cuda(), _w_pt.repeat(b).cuda())
            model.num_props = L
            o2 = model(epoch=0, **net_input)
            wm2 = o2['words_mask'].unsqueeze(1).expand(bsz, L, -1).contiguous().view(bsz * L, -1)
            wi2 = o2['words_id'].unsqueeze(1).expand(bsz, L, -1).contiguous().view(bsz * L, -1)
            n2, _ = cal_nll_loss(o2['words_logit'], wi2, wm2)
            A['REL'].append((-n2.view(bsz, L)).cpu().numpy())
            A['RELW'].append((-perword_nll(o2['words_logit'], wi2, wm2)).view(bsz, L, W).cpu().numpy())
            A['WID'].append(o2['words_id'].cpu().numpy())
            A['WMK'].append(o2['words_mask'].cpu().numpy())
            model.proposal_generator = _orig_pg
            model.num_props = P

            dur = np.asarray([i[1] for i in batch['raw']], dtype=np.float64)
            ts = np.asarray([i[2] for i in batch['raw']], dtype=np.float64)
            A['G'].append(ts / dur[:, None])
            if bi % 50 == 0:
                print('   rep%d batch %d/%d  %.1fs' % (rep, bi, len(loader), time.time() - t0), flush=True)

    G = np.concatenate(A['G'])
    n = len(G)
    r = np.arange(n)
    N8 = np.concatenate(A['N8']); C8 = np.concatenate(A['C8']); W8 = np.concatenate(A['W8'])
    order = np.argsort(N8, axis=1)
    j1 = order[:, 0]
    cb, wb = C8[r, j1], W8[r, j1]
    return dict(
        GS=G[:, 0], GE=G[:, 1], GL=G[:, 1] - G[:, 0],
        cb=cb, wb=wb, bL=np.clip(cb - wb / 2, 0, 1), bH=np.clip(cb + wb / 2, 0, 1),
        C8=C8, W8=W8, N8=N8, NW=pad_stack(A['NW']),
        REL=np.concatenate(A['REL']), RELW=pad_stack(A['RELW']),
        WID=pad_stack(A['WID']), WMK=pad_stack(A['WMK']),
        n=n, num_props=P, L=L, split=split,
    )


OUT = []
for rep in range(K_REP):
    d = run_rep(rep)
    OUT.append(d)
    print('  rep %d done  N=%d  RELW=%s' % (rep, d['n'], d['RELW'].shape), flush=True)

os.makedirs(os.path.dirname(os.path.abspath(out_p)), exist_ok=True)
pickle.dump(OUT, open(out_p, 'wb'), protocol=4)
print('SAVED %s  (%.1f MB)' % (out_p, os.path.getsize(out_p) / 1e6), flush=True)
print('DONE', flush=True)
