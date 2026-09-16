# -*- coding: utf-8 -*-
"""reproduce_base.py —— 跨数据集验证 RBR 的基线复现 harness（Charades-CG / ActivityNet-CG / TACoS）

（历史文件名 diag30_gen.py，实验日志与论文 Table 索引仍以该编号对应。）

目的
  Charades-CG 上的结论是：RBR 让长动作三指标全涨（R1@0.7 +10.5pt），代价是短动作小幅退步、
  整体 R1@0.5 掉 ~1pt。本脚本把同一套协议搬到**动作更长**的数据集上，检验这是
  「Charades 特有现象」还是「长动作受益、短动作受损」的普遍规律。

纪律（verify-then-report）
  先复现官方 ckpt 的基线指标（官方日志 REF 表逐列比对），复现一致后才看 RBR 的增量。
  若 base 与 REF 不符 → 说明评测口径有差异，RBR 数字一律不可信。

协议
  - 权重：仓库自带官方 ckpt（不重训），base 与 RBR 用同一套权重 → 严格同源对照
  - 无 gt 参与框构造：gt 只在最后算 IoU / 命中率
  - K_REP>=3 取 mean±std（PC-Net 的 _mask_words 未固定随机种子，单次评测噪声 ~±0.005）
  - 报告论文口径三列（R1@0.5 / R1@0.7 / mIoU）+ 仓库额外 R@5 列，便于与官方日志对齐

用法:
  python csr/eval/reproduce_base.py <config_path> <ckpt_path>   （在仓库根目录执行）
环境变量:
  BS=16  K_REP=3  LIMIT=0(不限样本数, 冒烟测试用 512)  KS=0.10,0.15,0.20,0.30,0.50
"""
import os
import sys
import pickle
import time

import numpy as np
import torch

# ---- 路径解析：默认自动定位本仓库根目录（含 models/pcnet.py），可用 PCNET_ROOT 覆盖 ----
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = _HERE
while _REPO != os.path.dirname(_REPO) and not os.path.exists(os.path.join(_REPO, 'models', 'pcnet.py')):
    _REPO = os.path.dirname(_REPO)
ROOT = os.environ.get('PCNET_ROOT') or os.environ.get('LM_ROOT') or _REPO
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from csr.paths import DATA_ROOT            # noqa: E402
from csr import ops                        # noqa: E402  （单一权威算子实现）

from utils import load_json            # noqa: E402
import datasets as da                  # noqa: E402
from models import PCNet               # noqa: E402
from models.loss import cal_nll_loss   # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

# 官方日志里 Best results 那一行（用于复现校验），键为 config 里的 dataset 名
REF = {
    'CharadesSTA': dict(mIoU=0.4712, r1_05=0.5484, r1_07=0.2668, r5_mIoU=0.6919,
                        r5_05=0.8608, r5_07=0.5433),
    'ActivityNet': dict(mIoU=0.3645, r1_05=0.2962, r1_07=0.1435, r5_mIoU=0.4189,
                        r5_05=0.3617, r5_07=0.1968),
    'TACoS': dict(mIoU=0.0979, r1_05=0.0400, r1_07=0.0117, r5_mIoU=0.1993,
                  r5_05=0.1032, r5_07=0.0435),
}

cfg = load_json(sys.argv[1])
ckpt = sys.argv[2]
dargs = cfg['dataset']
DS_NAME = dargs['dataset']
vocab = pickle.load(open(dargs['vocab_path'], 'rb'))
ds = getattr(da, DS_NAME)(data_path=dargs['test_data'], vocab=vocab, args=dargs, split='test')
VOCAB_SIZE = ds.vocab_size
COLLATE = ds.collate_data

# 特征文件定位：config 中的 feature_path 为相对路径时，依次在 DATA_ROOT 与仓库根目录查找。
# 大特征文件（i3d_features.hdf5 等）不随仓库分发，见 README「数据准备」。这里只改内存中的路径，不动仓库。
FP = os.environ.get('FEAT_PATH', dargs['feature_path'])
if not os.path.isabs(FP) and not os.path.exists(FP):
    for cand in (os.path.join(DATA_ROOT, os.path.basename(FP)),
                 os.path.join(ROOT, os.path.basename(FP))):
        if os.path.exists(cand):
            FP = cand
            break
dargs['feature_path'] = FP
print('feature_path=%s  exists=%s' % (FP, os.path.exists(FP)), flush=True)

LIMIT = int(os.environ.get('LIMIT', 0))
if LIMIT > 0:
    ds = Subset(ds, list(range(min(LIMIT, len(ds)))))

sd = torch.load(ckpt, map_location='cuda')
sp = sd['model_parameters']
# 以 ckpt 自带 config 为准（保证架构与训练时完全一致），再用 slots 形状校正 num_props
ck_cfg = sd.get('config', {})
mcfg = dict(ck_cfg.get('model', {}).get('config', cfg['model']['config']))
P = int(sp['slot_atten.slots'].shape[0]) if 'slot_atten.slots' in sp else mcfg.get('num_props', 8)
mcfg['num_props'] = P
mcfg['vocab_size'] = VOCAB_SIZE
mcfg['max_epoch'] = cfg['train']['max_num_epochs']
mcfg.setdefault('num_iteration', 4)
print('mcfg=%s' % {k: v for k, v in mcfg.items() if k != 'DualTransformer'}, flush=True)

L = dargs['max_num_frames'] // 4          # 逐点提案数（PC-Net 在 50 帧的重建分辨率上工作）
GRID = np.linspace(0, 1, L)
BS = int(os.environ.get('BS', 16))
K_REP = int(os.environ.get('K_REP', 3))
KS = [float(x) for x in os.environ.get('KS', '0.10,0.15,0.20,0.30,0.50').split(',')]
loader = DataLoader(ds, batch_size=BS, shuffle=False,
                    collate_fn=COLLATE, num_workers=8)

N = len(ds)
print('cfg=%s' % sys.argv[1], flush=True)
print('ckpt=%s' % ckpt, flush=True)
print('dataset=%s  N=%d  num_props=%d  L=%d  BS=%d  K_REP=%d' % (
    DS_NAME, N, P, L, BS, K_REP), flush=True)


def iou(a0, a1, b0, b1):
    return ops.iou(a0, a1, b0, b1)


def flood(REL, cb, k):
    """从 top-1 中心出发，沿 z-score 化的相关性曲线向两侧 flood 扩展。"""
    return ops.flood_expand(REL, cb, k, GRID)


def metrics(i):
    """从一个 IoU 向量算全套指标（论文口径 + 仓库 R@5 列）。"""
    d = {'mIoU': float(i.mean())}
    for t in (0.1, 0.3, 0.5, 0.7, 0.9):
        d['iou%.1f' % t] = float((i >= t).mean())
    return d


ACC = {}          # tag -> [metrics dict per rep]
SPLIT = {}        # tag -> list of (iou_vec, gt_len)


def add(tag, i, gl=None):
    ACC.setdefault(tag, []).append(metrics(i))
    if gl is not None:
        SPLIT.setdefault(tag, []).append((i, gl))


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

    N8, C8, W8, REL, G = [], [], [], [], []
    t0 = time.time()
    with torch.no_grad():
        for bi, batch in enumerate(loader, 1):
            net_input = {k: v.cuda() for k, v in batch['net_input'].items()}
            bsz = len(batch['raw'])
            o = model(epoch=0, **net_input)
            wm = o['words_mask'].unsqueeze(1).expand(bsz, P, -1).contiguous().view(bsz * P, -1)
            wi = o['words_id'].unsqueeze(1).expand(bsz, P, -1).contiguous().view(bsz * P, -1)
            n, _ = cal_nll_loss(o['words_logit'], wi, wm)
            N8.append(n.view(bsz, P).cpu().numpy())
            C8.append(o['center'].view(bsz, P).cpu().numpy())
            W8.append(o['width'].view(bsz, P).cpu().numpy())
            # 逐点提案 -> 逐帧相关性曲线
            model.proposal_generator = lambda h, b: (_c_pt.repeat(b).cuda(), _w_pt.repeat(b).cuda())
            model.num_props = L
            o2 = model(epoch=0, **net_input)
            wm2 = o2['words_mask'].unsqueeze(1).expand(bsz, L, -1).contiguous().view(bsz * L, -1)
            wi2 = o2['words_id'].unsqueeze(1).expand(bsz, L, -1).contiguous().view(bsz * L, -1)
            n2, _ = cal_nll_loss(o2['words_logit'], wi2, wm2)
            REL.append((-n2.view(bsz, L)).cpu().numpy())
            model.proposal_generator = _orig_pg
            model.num_props = P
            dur = np.asarray([i[1] for i in batch['raw']], dtype=np.float64)
            ts = np.asarray([i[2] for i in batch['raw']], dtype=np.float64)
            G.append(ts / dur[:, None])
            if bi % 50 == 0:
                print('    rep%d  batch %d/%d  %.1fs' % (rep, bi, len(loader), time.time() - t0), flush=True)

    G = np.concatenate(G); N8 = np.concatenate(N8); C8 = np.concatenate(C8)
    W8 = np.concatenate(W8); REL = np.concatenate(REL)
    n = len(G); r = np.arange(n)
    GS, GE = G[:, 0], G[:, 1]; GL = GE - GS

    order = np.argsort(N8, axis=1)
    j1 = order[:, 0]
    cb, wb = C8[r, j1], W8[r, j1]
    bL, bH = np.clip(cb - wb / 2, 0, 1), np.clip(cb + wb / 2, 0, 1)

    iouAll = iou(np.clip(C8 - W8 / 2, 0, 1), np.clip(C8 + W8 / 2, 0, 1), GS[:, None], GE[:, None])
    iouOrd = np.take_along_axis(iouAll, order, axis=1)
    r5base = iouOrd[:, :5].max(1)

    add('base', iouOrd[:, 0], GL)
    add('base_R5', r5base, GL)
    for k in KS:
        s, e = flood(REL, cb, k)
        l2 = np.minimum(s, bL); h2 = np.maximum(e, bH)
        i2 = iou(l2, h2, GS, GE)
        add('RBR k=%.2f' % k, i2, GL)
        add('RBR k=%.2f_R5' % k, np.maximum(i2, iouOrd[:, 1:5].max(1)), GL)
    return dict(cb=cb, wb=wb, GL=GL, n=n, C8=C8, W8=W8)


diag = None
for rep in range(K_REP):
    d = run_rep(rep)
    if diag is None:
        diag = d
    print('  rep %d done | base mIoU=%.4f R1@0.5=%.4f R1@0.7=%.4f' % (
        rep, ACC['base'][-1]['mIoU'], ACC['base'][-1]['iou0.5'], ACC['base'][-1]['iou0.7']), flush=True)


def mean_of(tag, key):
    return float(np.mean([m[key] for m in ACC[tag]]))


def std_of(tag, key):
    return float(np.std([m[key] for m in ACC[tag]]))


line = '=' * 96
print('\n' + line, flush=True)
print('diag30  %s   N=%d  K_REP=%d  ckpt=%s' % (DS_NAME, diag['n'], K_REP, os.path.basename(os.path.dirname(ckpt))), flush=True)
print(line, flush=True)

# ---------- 1) 复现校验 ----------
if DS_NAME in REF:
    ref = REF[DS_NAME]
    print('[复现校验] 本脚本 vs 官方日志 Best results', flush=True)
    chk = [('mIoU', 'mIoU', ref['mIoU']), ('R@1,IoU@0.5', 'iou0.5', ref['r1_05']),
           ('R@1,IoU@0.7', 'iou0.7', ref['r1_07']), ('R@5,mIoU', 'mIoU', ref['r5_mIoU']),
           ('R@5,IoU@0.5', 'iou0.5', ref['r5_05']), ('R@5,IoU@0.7', 'iou0.7', ref['r5_07'])]
    tag_for = {'mIoU': 'base', 'R@1,IoU@0.5': 'base', 'R@1,IoU@0.7': 'base',
               'R@5,mIoU': 'base_R5', 'R@5,IoU@0.5': 'base_R5', 'R@5,IoU@0.7': 'base_R5'}
    ok = True
    for name, key, v in chk:
        got = mean_of(tag_for[name], key)
        flag = 'OK ' if abs(got - v) <= 0.006 else 'DIFF'
        if flag == 'DIFF':
            ok = False
        print('  %-4s %-14s 官方=%.4f  本脚本=%.4f  (%+.4f)' % (flag, name, v, got, got - v), flush=True)
    print('  => %s' % ('复现一致（差异均在评测噪声 ±0.005 内），后续 RBR 增量可信'
                       if ok else '复现不一致！RBR 数字不可信，需先查评测口径'), flush=True)
print(line, flush=True)

# ---------- 2) 主表 ----------
print('[主表] 论文口径三列 + 仓库 R@5 列（mean±std over %d rep）' % K_REP, flush=True)
print('%-20s %10s %10s %10s | %10s %10s %10s' % (
    '', 'R1@0.5', 'R1@0.7', 'mIoU', 'R5@0.5', 'R5@0.7', 'R5,mIoU'), flush=True)
tags = ['base'] + ['RBR k=%.2f' % k for k in KS]
for tag in tags:
    if tag not in ACC:
        continue
    r5tag = tag + '_R5'
    print('%-20s %10.4f %10.4f %10.4f | %10.4f %10.4f %10.4f' % (
        tag,
        mean_of(tag, 'iou0.5'), mean_of(tag, 'iou0.7'), mean_of(tag, 'mIoU'),
        mean_of(r5tag, 'iou0.5'), mean_of(r5tag, 'iou0.7'), mean_of(r5tag, 'mIoU')), flush=True)
print('-' * 96, flush=True)
print('%-20s %10s %10s %10s | %10s %10s %10s' % ('vs base', 'ΔR1@0.5', 'ΔR1@0.7', 'ΔmIoU',
                                                 'ΔR5@0.5', 'ΔR5@0.7', 'ΔR5,mIoU'), flush=True)
for tag in tags[1:]:
    print('%-20s %+10.4f %+10.4f %+10.4f | %+10.4f %+10.4f %+10.4f' % (
        tag,
        mean_of(tag, 'iou0.5') - mean_of('base', 'iou0.5'),
        mean_of(tag, 'iou0.7') - mean_of('base', 'iou0.7'),
        mean_of(tag, 'mIoU') - mean_of('base', 'mIoU'),
        mean_of(tag + '_R5', 'iou0.5') - mean_of('base_R5', 'iou0.5'),
        mean_of(tag + '_R5', 'iou0.7') - mean_of('base_R5', 'iou0.7'),
        mean_of(tag + '_R5', 'mIoU') - mean_of('base_R5', 'mIoU')), flush=True)

# ---------- 3) 提案塌缩诊断（跨数据集可比） ----------
print(line, flush=True)
print('[诊断] 提案几何（跨数据集可比）', flush=True)
C8, W8 = diag['C8'], diag['W8']
print('  8提案宽度        : mean=%.4f  median=%.4f  p10=%.4f  p90=%.4f' % (
    W8.mean(), np.median(W8), np.percentile(W8, 10), np.percentile(W8, 90)), flush=True)
print('  样本内中心离散度  : mean=%.4f   （8 个提案在时间轴上铺开的程度；均匀网格 ≈0.30，全挤一处 ≈0）' % (
    C8.std(1).mean()), flush=True)
Cs = np.sort(C8, axis=1)
print('  跨样本塌缩度      : mean=%.4f   （按 rank 对齐后，同一 rank 的中心在不同输入间的波动；与输入无关的固定网格 ≈0）' % (
    Cs.std(0).mean()), flush=True)
print('  top-1 中心跨样本std: %.4f' % (diag['cb'].std()), flush=True)
print('  gt 时长占比      : mean=%.4f  median=%.4f' % (diag['GL'].mean(), np.median(diag['GL'])), flush=True)
print('  基线 top-1 框宽  : mean=%.4f  median=%.4f' % (diag['wb'].mean(), np.median(diag['wb'])), flush=True)

# ---------- 4) 按 gt 时长分桶 ----------
print(line, flush=True)
print('[分桶] 按 gt 时长占比分桶（k=0.15）', flush=True)
buckets = [(0.0, 0.20), (0.20, 0.35), (0.35, 0.50), (0.50, 1.01)]
print('%-14s %7s %9s %9s %9s | %9s %9s %9s' % (
    '桶', 'n', 'base_mIoU', 'base@0.5', 'base@0.7', 'RBR_mIoU', 'RBR@0.5', 'RBR@0.7'), flush=True)
ib = np.concatenate([v[0] for v in SPLIT['base']])
gb = np.concatenate([v[1] for v in SPLIT['base']])
ir = np.concatenate([v[0] for v in SPLIT['RBR k=0.15']])
gr = np.concatenate([v[1] for v in SPLIT['RBR k=0.15']])
for lo, hi in buckets:
    m = (gb >= lo) & (gb < hi)
    mr = (gr >= lo) & (gr < hi)
    if m.sum() == 0:
        continue
    print('%-14s %7d %9.4f %9.4f %9.4f | %9.4f %9.4f %9.4f' % (
        '[%.2f,%.2f)' % (lo, hi), int(m.sum()),
        ib[m].mean(), (ib[m] >= 0.5).mean(), (ib[m] >= 0.7).mean(),
        ir[mr].mean(), (ir[mr] >= 0.5).mean(), (ir[mr] >= 0.7).mean()), flush=True)
print('%-14s %7d %9.4f %9.4f %9.4f | %9.4f %9.4f %9.4f' % (
    'ALL', len(ib), ib.mean(), (ib >= 0.5).mean(), (ib >= 0.7).mean(),
    ir.mean(), (ir >= 0.5).mean(), (ir >= 0.7).mean()), flush=True)

# 长短二分（与 Charades 报告口径一致，阈值 0.35）
print('-' * 96, flush=True)
print('[长短二分] 阈值 gt>0.35（与 Charades 报告一致）', flush=True)
for nm, i_, g_ in (('base', ib, gb), ('RBR k=0.15', ir, gr)):
    lo = g_ <= 0.35; hi = ~lo
    print('  %-12s mIoU 短=%.4f 长=%.4f | R1@0.5 短=%.4f 长=%.4f | R1@0.7 短=%.4f 长=%.4f  (短 n=%d, 长 n=%d)' % (
        nm, i_[lo].mean(), i_[hi].mean(),
        (i_[lo] >= 0.5).mean(), (i_[hi] >= 0.5).mean(),
        (i_[lo] >= 0.7).mean(), (i_[hi] >= 0.7).mean(), int(lo.sum()), int(hi.sum())), flush=True)
print(line, flush=True)
print('DONE', flush=True)
