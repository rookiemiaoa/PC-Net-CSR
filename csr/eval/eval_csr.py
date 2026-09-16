# -*- coding: utf-8 -*-
"""eval_csr.py —— CSR（Center-Shift Refinement）在 PC-Net 官方权重上的完整验证。

（历史文件名 diag40_gen.py，实验日志与论文 Table 索引仍以该编号对应。）

背景
  历史路线 RBR 是「边界扩框」，存在结构性方向偏置：扩框对「基线框已比 gt 短」的样本有益，
  对多数样本有害 —— 实测短动作 R1@0.5 -2.98pt（headline 转负）。
  本工作把主战场移到「中心」：IoU 对中心偏移是偶函数（δ=0 取极大），故中心修正无子群偏置。
  ORACLE_c（只把中心换成 gt、保持框宽）在 Charades-CG 上把 mIoU 从 0.4720 抬到 0.7040
  （短 +20.7pt、长 +30.4pt），证明中心是真正的瓶颈。

方法（零训练、纯推理期、不使用 gt）
  1) 逐点提案 -> 相关性曲线 r(t) = -nll(t)，z-score 得 Z
  2) 平滑（窗口 SMOOTH）
  3) 中心估计：窗口 [c_b-beta, c_b+beta] 内正部加权质心
  4) dead-zone：|d| < RHO*框宽 则不平移（保护「中心本来就准」的样本）
  5) 阻尼：c = c_b + ALPHA*d
  6) 自洽门控：仅当平移后「框内 Z 均值」不下降（>= DMIN）才接受
  7) **宽度保持不变**（与 RBR 的本质区别 —— 短样本不会因宽度变化跨阈值）

超参来自 train.json（开发集）校准：SMOOTH=7 BETA=0.20 RHO=0.10 ALPHA=1.0 DMIN=0.01

用法:
  python csr/eval/eval_csr.py <config_path> <ckpt_path>   （在仓库根目录执行）
环境变量:
  BS=16 K_REP=3 LIMIT=0 SMOOTH=7 BETA=0.20 RHO=0.10 ALPHA=1.0 DMIN=0.01 KS=0.10,0.15,0.20
"""
import math
import os
import pickle
import sys
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

REF = {
    'CharadesSTA': dict(mIoU=0.4712, r1_05=0.5484, r1_07=0.2668),
    'ActivityNet': dict(mIoU=0.3645, r1_05=0.2962, r1_07=0.1435),
    'TACoS': dict(mIoU=0.0979, r1_05=0.0400, r1_07=0.0117),
}

cfg = load_json(sys.argv[1])
ckpt = sys.argv[2]
dargs = cfg['dataset']
DS_NAME = dargs['dataset']
vocab = pickle.load(open(dargs['vocab_path'], 'rb'))
ds = getattr(da, DS_NAME)(data_path=dargs['test_data'], vocab=vocab, args=dargs, split='test')
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
print('feature_path=%s  exists=%s' % (FP, os.path.exists(FP)), flush=True)

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
CG = (np.arange(L) + 0.5) / L          # 逐点提案的中心位置
GRID = np.linspace(0.0, 1.0, L)
BS = int(os.environ.get('BS', 16))
K_REP = int(os.environ.get('K_REP', 3))
KS = [float(x) for x in os.environ.get('KS', '0.10,0.15,0.20').split(',')]
SMOOTH = int(os.environ.get('SMOOTH', 7))
BETA = float(os.environ.get('BETA', 0.20))
RHO = float(os.environ.get('RHO', 0.10))
ALPHA = float(os.environ.get('ALPHA', 1.0))
DMIN = float(os.environ.get('DMIN', 0.01))
THR = float(os.environ.get('THR', 0.35))    # 短/长分桶阈值（gt 时长占比）
loader = DataLoader(ds, batch_size=BS, shuffle=False, collate_fn=COLLATE, num_workers=8)

N = len(ds)
print('cfg=%s' % sys.argv[1], flush=True)
print('ckpt=%s' % ckpt, flush=True)
print('dataset=%s  N=%d  num_props=%d  L=%d  BS=%d  K_REP=%d' % (DS_NAME, N, P, L, BS, K_REP), flush=True)
print('CSR: SMOOTH=%d BETA=%.2f RHO=%.2f ALPHA=%.2f DMIN=%.2f THR=%.2f' % (
    SMOOTH, BETA, RHO, ALPHA, DMIN, THR), flush=True)


# ---------------- 工具（算子实现统一在 csr/ops.py，此处仅做薄封装/别名） ----------------
def iou(a0, a1, b0, b1):
    return ops.iou(a0, a1, b0, b1)


def smoothZ(Z, w):
    return ops.smooth_curve(Z, w)


def inside_score(Z, lo, hi):
    return ops.inside_score(Z, lo, hi, CG)


def est_win_vec(Z, cb, beta):
    return ops.estimate_window_center(Z, cb, beta, CG)


def flood(REL, cb, k):
    return ops.flood_expand(REL, cb, k, GRID)


def csr_boxes(Z_raw, cb, wb):
    """返回 CSR 精修框 (lo, hi, moved, shift)，宽度与基线一致。"""
    lo, hi, moved, shift = ops.refine_center(
        Z_raw, cb, wb, CG, smooth=SMOOTH, beta=BETA, rho=RHO, alpha=ALPHA, dmin=DMIN)
    return lo, hi, moved, shift


def metrics(i):
    d = {'mIoU': float(i.mean())}
    for t in (0.1, 0.3, 0.5, 0.7, 0.9):
        d['iou%.1f' % t] = float((i >= t).mean())
    return d


ACC = {}
SPLIT = {}


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
            if bi % 100 == 0:
                print('    rep%d  batch %d/%d  %.1fs' % (rep, bi, len(loader), time.time() - t0), flush=True)

    G = np.concatenate(G); N8 = np.concatenate(N8); C8 = np.concatenate(C8)
    W8 = np.concatenate(W8); REL = np.concatenate(REL)
    n = len(G); r = np.arange(n)
    GS, GE = G[:, 0], G[:, 1]; GL = GE - GS
    order = np.argsort(N8, axis=1)
    j1 = order[:, 0]
    cb, wb = C8[r, j1], W8[r, j1]
    bL, bH = np.clip(cb - wb / 2, 0, 1), np.clip(cb + wb / 2, 0, 1)

    i_base = iou(bL, bH, GS, GE)
    add('base', i_base, GL)
    for k in KS:
        s, e = flood(REL, cb, k)
        i_r = iou(np.minimum(s, bL), np.maximum(e, bH), GS, GE)
        add('RBR k=%.2f' % k, i_r, GL)
    lo, hi, moved, shift = csr_boxes(REL, cb, wb)
    i_csr = iou(lo, hi, GS, GE)
    add('CSR', i_csr, GL)
    # R@5 框池（CSR 顶替 top-1）
    iouAll = iou(np.clip(C8 - W8 / 2, 0, 1), np.clip(C8 + W8 / 2, 0, 1), GS[:, None], GE[:, None])
    iouOrd = np.take_along_axis(iouAll, order, axis=1)
    add('base_R5', iouOrd[:, :5].max(1), GL)
    add('CSR_R5', np.maximum(i_csr, iouOrd[:, 1:5].max(1)), GL)
    return dict(n=n, moved=moved, shift=shift, GL=GL,
                i_base=i_base, i_csr=i_csr, lo=lo, hi=hi, bL=bL, bH=bH, wb=wb, cb=cb)


diag = None
for rep in range(K_REP):
    d = run_rep(rep)
    if diag is None:
        diag = d
    print('  rep %d done | base mIoU=%.4f R1@0.5=%.4f | CSR mIoU=%.4f R1@0.5=%.4f R1@0.7=%.4f' % (
        rep, ACC['base'][-1]['mIoU'], ACC['base'][-1]['iou0.5'],
        ACC['CSR'][-1]['mIoU'], ACC['CSR'][-1]['iou0.5'], ACC['CSR'][-1]['iou0.7']), flush=True)


def mean_of(tag, key):
    return float(np.mean([m[key] for m in ACC[tag]]))


def std_of(tag, key):
    return float(np.std([m[key] for m in ACC[tag]]))


line = '=' * 104
print('\n' + line, flush=True)
print('diag40  %s  N=%d  K_REP=%d  ckpt=%s' % (
    DS_NAME, diag['n'], K_REP, os.path.basename(os.path.dirname(ckpt))), flush=True)
print(line, flush=True)

if DS_NAME in REF:
    ref = REF[DS_NAME]
    print('[复现校验] base vs 官方日志 Best results', flush=True)
    ok = True
    for nm, key, v in (('R1,mIoU', 'mIoU', ref['mIoU']), ('R1@0.5', 'iou0.5', ref['r1_05']),
                       ('R1@0.7', 'iou0.7', ref['r1_07'])):
        got = mean_of('base', key)
        flag = 'OK ' if abs(got - v) <= 0.02 else 'DIFF'
        if flag == 'DIFF':
            ok = False
        print('  %-4s %-9s 官方=%.4f  本脚本=%.4f  (%+.4f)' % (flag, nm, v, got, got - v), flush=True)
    print('  => %s' % ('一致' if ok else '不一致（novel 切分基线本就不同，或存在特征溯源差异）'), flush=True)
print(line, flush=True)

print('[主表] mean±std over %d rep（论文口径三列）' % K_REP, flush=True)
print('%-16s %16s %16s %16s' % ('', 'R1@0.5', 'R1@0.7', 'mIoU'), flush=True)
tags = ['base'] + ['RBR k=%.2f' % k for k in KS] + ['CSR']
for tag in tags:
    print('%-16s %8.4f±%.4f %8.4f±%.4f %8.4f±%.4f' % (
        tag, mean_of(tag, 'iou0.5'), std_of(tag, 'iou0.5'),
        mean_of(tag, 'iou0.7'), std_of(tag, 'iou0.7'),
        mean_of(tag, 'mIoU'), std_of(tag, 'mIoU')), flush=True)
print('-' * 104, flush=True)
print('%-16s %16s %16s %16s' % ('Δ vs base', 'ΔR1@0.5', 'ΔR1@0.7', 'ΔmIoU'), flush=True)
for tag in tags[1:]:
    print('%-16s %+16.4f %+16.4f %+16.4f' % (
        tag, 100 * (mean_of(tag, 'iou0.5') - mean_of('base', 'iou0.5')),
        100 * (mean_of(tag, 'iou0.7') - mean_of('base', 'iou0.7')),
        100 * (mean_of(tag, 'mIoU') - mean_of('base', 'mIoU'))), flush=True)

print('\n' + line, flush=True)
print('[长短分桶] gt 时长占比 <= %.2f 为短' % THR, flush=True)
print('%-16s %-6s %6s %9s %9s %9s' % ('方法', '桶', 'n', 'mIoU', 'R1@0.5', 'R1@0.7'), flush=True)
for tag in ('base', 'CSR'):
    ib = np.concatenate([v[0] for v in SPLIT[tag]])
    gb = np.concatenate([v[1] for v in SPLIT[tag]])
    for nm, m in (('短', gb <= THR), ('长', gb > THR), ('全', np.ones(len(gb), bool))):
        print('%-16s %-6s %6d %9.4f %9.4f %9.4f' % (
            tag, nm, int(m.sum()), ib[m].mean(), (ib[m] >= 0.5).mean(), (ib[m] >= 0.7).mean()), flush=True)
print('-' * 104, flush=True)
print('%-16s %-6s %6s %9s %9s %9s' % ('Δ (CSR-base)', '桶', 'n', 'ΔmIoU', 'ΔR1@0.5', 'ΔR1@0.7'), flush=True)
ib = np.concatenate([v[0] for v in SPLIT['base']]); gb = np.concatenate([v[1] for v in SPLIT['base']])
ic = np.concatenate([v[0] for v in SPLIT['CSR']])
for nm, m in (('短', gb <= THR), ('长', gb > THR), ('全', np.ones(len(gb), bool))):
    print('%-16s %-6s %6d %+9.4f %+9.4f %+9.4f' % (
        'CSR-base', nm, int(m.sum()),
        100 * (ic[m].mean() - ib[m].mean()),
        100 * ((ic[m] >= 0.5).mean() - (ib[m] >= 0.5).mean()),
        100 * ((ic[m] >= 0.7).mean() - (ib[m] >= 0.7).mean())), flush=True)

# ---------------- 显著性（配对 bootstrap + McNemar） ----------------
print('\n' + line, flush=True)
print('[显著性] 配对 bootstrap B=2000（95%% CI）+ McNemar（跨阈值计数）', flush=True)
print(line, flush=True)
rng = np.random.default_rng(0)
B = 2000
for nm, m in (('short', diag['GL'] <= THR), ('long', diag['GL'] > THR), ('all', np.ones(diag['n'], bool))):
    ib, ic = diag['i_base'][m], diag['i_csr'][m]
    n = len(ib)
    idx = rng.integers(0, n, size=(B, n))
    dm = ic[idx].mean(1) - ib[idx].mean(1)
    d5 = (ic[idx] >= 0.5).mean(1) - (ib[idx] >= 0.5).mean(1)
    d7 = (ic[idx] >= 0.7).mean(1) - (ib[idx] >= 0.7).mean(1)
    b5 = int(((ib < 0.5) & (ic >= 0.5)).sum()); c5 = int(((ib >= 0.5) & (ic < 0.5)).sum())
    b7 = int(((ib < 0.7) & (ic >= 0.7)).sum()); c7 = int(((ib >= 0.7) & (ic < 0.7)).sum())
    print('  %-6s n=%5d  ΔmIoU %+6.3f [%+6.3f,%+6.3f] | Δ@0.5 %+6.3f [%+6.3f,%+6.3f] | Δ@0.7 %+6.3f [%+6.3f,%+6.3f]' % (
        nm, n, 100 * dm.mean(), 100 * np.percentile(dm, 2.5), 100 * np.percentile(dm, 97.5),
        100 * d5.mean(), 100 * np.percentile(d5, 2.5), 100 * np.percentile(d5, 97.5),
        100 * d7.mean(), 100 * np.percentile(d7, 2.5), 100 * np.percentile(d7, 97.5)), flush=True)
    for tn, bb, cc in (('@0.5', b5, c5), ('@0.7', b7, c7)):
        stat = ((abs(bb - cc) - 1) ** 2 / float(bb + cc)) if (bb + cc) > 0 else 0.0
        p = math.erfc(math.sqrt(stat / 2.0))
        print('         McNemar %s: 升(%d) 降(%d)  chi2=%.2f  p=%.2e' % (tn, bb, cc, stat, p), flush=True)

print('\n' + line, flush=True)
print('[诊断] 平移行为', flush=True)
gb = diag['GL']
for nm, m in (('短', gb <= THR), ('长', gb > THR), ('全', np.ones(diag['n'], bool))):
    print('  %-4s n=%5d  移动率=%.3f  平均|Δ|(移动样本)=%.4f  平均|Δ|(全体)=%.4f' % (
        nm, int(m.sum()), diag['moved'][m].mean(),
        diag['shift'][m & diag['moved']].mean() if (m & diag['moved']).any() else 0.0,
        diag['shift'][m].mean()), flush=True)
print('  基线框宽中位=%.4f  gt 时长中位=%.4f' % (np.median(diag['wb']), np.median(diag['GL'])), flush=True)
print(line, flush=True)
print('DONE', flush=True)
