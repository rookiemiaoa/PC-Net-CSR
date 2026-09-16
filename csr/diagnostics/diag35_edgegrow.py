"""diag35：边界外推（Edge-Grow）—— 从框边界往外长，相关性仍接近峰值才长

动机（来自 diag32 的伤害归因）
  短样本被伤害比例 13.6%，平均损失 -0.116 IoU —— 都是"过扩"。
  现有 RBR 从**框中心**沿宽松阈值 flood，一旦泄露到邻域就会把框拉宽；
  它对"框边界到底切在动作内部还是动作外部"毫无判断。

本方法（可证伪的核心假设）
  从**框边界外侧**往外走，只要相关性 z 仍 >= a * max(z) 就继续长，一旦掉下去就停。
    * 长动作：框边界切在动作**内部** -> 边界外侧相关性仍接近峰值 -> 长出去（补上漏掉的起止点）；
    * 短动作：框已比动作宽 -> 边界外侧只是一般相关性的噪声/邻域 -> 立刻掉下去 -> **不长**（零伤害）。
  峰值相对阈值自动适配"平台 vs 尖峰"，无需任何绝对刻度。
  允许 g 帧连续低于阈值（填洞），避免长动作内部低相关性凹陷把扩展截断。

族
  EG_a_g     : 边界外推（阈值 = a*max(z)，允许 g 帧洞）
  EG1_a      : 只外推 1 个网格步（最保守）
  EGG_a_R    : 边界外推，但每侧至少要有 R 帧连续达标才采纳该侧
  全部与基线框取并集（只扩不缩），零训练、纯推理期、**不使用 gt**。

校准纪律（与 diag34 一致）
  a/g 在 train.json 上选（受"短动作不回归"约束后取长 Δ@0.7 最大）；
  **test_trivial.json 不参与选择**，只用于报告；同时把全部网格在 test 上的曲线打印出来。

用法: python diag35_gen.py <test_config> <ckpt>
环境变量: BS=16  CALIB_JSON=data/charades/train.json  THR=0.35  LIMIT=0  K_REP=1
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

from utils import load_json            # noqa: E402
import datasets as da                  # noqa: E402
from models import PCNet               # noqa: E402
from models.loss import cal_nll_loss   # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

cfg = load_json(sys.argv[1]); ckpt = sys.argv[2]
dargs = cfg['dataset']; DS_NAME = dargs['dataset']
vocab = pickle.load(open(dargs['vocab_path'], 'rb'))
_probe = getattr(da, DS_NAME)(data_path=dargs['test_data'], vocab=vocab, args=dargs, split='test')
VOCAB_SIZE = _probe.vocab_size
COLLATE = _probe.collate_data
del _probe

FP = os.environ.get('FEAT_PATH', dargs['feature_path'])
if not os.path.isabs(FP) and not os.path.exists(FP):
    for cand in (os.path.join(DATA_ROOT, os.path.basename(FP)),
                 os.path.join(ROOT, os.path.basename(FP))):
        if os.path.exists(cand):
            FP = cand
            break
dargs['feature_path'] = FP
print('feature_path=%s exists=%s' % (FP, os.path.exists(FP)), flush=True)

sd = torch.load(ckpt, map_location='cuda')
sp = sd['model_parameters']
ck_cfg = sd.get('config', {})
mcfg = dict(ck_cfg.get('model', {}).get('config', cfg['model']['config']))
P = int(sp['slot_atten.slots'].shape[0]) if 'slot_atten.slots' in sp else mcfg.get('num_props', 8)
mcfg['num_props'] = P; mcfg['vocab_size'] = VOCAB_SIZE
mcfg['max_epoch'] = cfg['train']['max_num_epochs']; mcfg.setdefault('num_iteration', 4)

L = dargs['max_num_frames'] // 4
GRID = np.linspace(0, 1, L)
BS = int(os.environ.get('BS', 16))
THR = float(os.environ.get('THR', 0.35))
K_REP = int(os.environ.get('K_REP', 1))
CALIB_JSON = os.environ.get('CALIB_JSON', 'data/charades/train.json')
LIMIT = int(os.environ.get('LIMIT', 0))

A_GRID = (0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.80)
G_GRID = (0, 1, 2, 3, 5)
R_GRID = (1, 2, 3)


def iou(a0, a1, b0, b1):
    inter = np.maximum(0.0, np.minimum(a1, b1) - np.maximum(a0, b0))
    union = np.maximum(a1, b1) - np.minimum(a0, b0)
    return np.where(union > 1e-9, inter / np.maximum(union, 1e-9), 0.0)


def metrics(i):
    return dict(mIoU=float(i.mean()), i3=float((i >= 0.3).mean()),
                i5=float((i >= 0.5).mean()), i7=float((i >= 0.7).mean()))


def zs(R):
    return (R - R.mean(1, keepdims=True)) / (R.std(1, keepdims=True) + 1e-8)


def flood(m, pc):
    n, l = m.shape
    s = np.zeros(n); e = np.ones(n)
    for i in range(n):
        j = pc[i]; a = j
        while a > 0 and m[i, a - 1]:
            a -= 1
        b = j
        while b < l - 1 and m[i, b + 1]:
            b += 1
        s[i] = GRID[a]; e[i] = GRID[b]
    return s, e


def collect(data_json, tag, reps=1):
    ds = getattr(da, DS_NAME)(data_path=data_json, vocab=vocab, args=dargs, split='test')
    if LIMIT > 0:
        ds = Subset(ds, list(range(min(LIMIT, len(ds)))))
    loader = DataLoader(ds, batch_size=BS, shuffle=False, collate_fn=COLLATE, num_workers=8)
    out = []
    for rep in range(reps):
        np.random.seed(7 + rep); torch.manual_seed(7 + rep)
        model = PCNet(mcfg).cuda(); model.load_state_dict(sp); model.eval()
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
                model.proposal_generator = _orig_pg; model.num_props = P
                dur = np.asarray([i[1] for i in batch['raw']], dtype=np.float64)
                ts = np.asarray([i[2] for i in batch['raw']], dtype=np.float64)
                G.append(ts / dur[:, None])
                if bi % 100 == 0:
                    print('    [%s rep%d] batch %d/%d  %.1fs' % (
                        tag, rep, bi, len(loader), time.time() - t0), flush=True)
        del model; torch.cuda.empty_cache()
        G = np.concatenate(G); N8 = np.concatenate(N8); C8 = np.concatenate(C8)
        W8 = np.concatenate(W8); REL = np.concatenate(REL)
        n = len(G); r = np.arange(n)
        GS, GE = G[:, 0], G[:, 1]
        order = np.argsort(N8, axis=1); j1 = order[:, 0]
        cb, wb = C8[r, j1], W8[r, j1]
        bL = np.clip(cb - wb / 2, 0, 1); bH = np.clip(cb + wb / 2, 0, 1)
        Z = zs(REL)
        pc = np.clip((cb * (L - 1)).round().astype(int), 0, L - 1)
        m15 = Z > (Z.mean(1) + 0.15 * Z.std(1))[:, None]
        m15[np.arange(n), pc] = True
        s15, e15 = flood(m15, pc)
        out.append(dict(GS=GS, GE=GE, GL=GE - GS, cb=cb, wb=wb, bL=bL, bH=bH, Z=Z,
                        s15=s15, e15=e15, n=n,
                        bli=np.clip((bL * (L - 1)).round().astype(int), 0, L - 1),
                        bhi=np.clip((bH * (L - 1)).round().astype(int), 0, L - 1)))
    return out


def edge_row(z, start, th, gap, direction):
    """从 start 出发按 direction 走，只要 z>=th 就继续；允许 gap 帧连续低于阈值。返回最远达标下标，无则 None。"""
    n = len(z); idx = start; miss = 0; best = None
    while 0 <= idx < n:
        if z[idx] >= th:
            miss = 0; best = idx
        else:
            miss += 1
            if miss > gap:
                break
        idx += direction
    return best


def edge_grow(D, a, g, minrun=1):
    """边界外推：返回 lo, hi（已与基线框并集）。"""
    n = D['n']; Z = D['Z']; bli = D['bli']; bhi = D['bhi']
    bL = D['bL']; bH = D['bH']
    th = a * Z.max(1)
    lo = bL.copy(); hi = bH.copy()
    for i in range(n):
        z = Z[i]
        bl, bh = bli[i], bhi[i]
        # 左侧：从 bl-1 往左
        if bl - 1 >= 0:
            b = edge_row(z, bl - 1, th[i], g, -1)
            if b is not None:
                run = bl - b
                if run >= minrun:
                    lo[i] = min(bL[i], GRID[b])
        # 右侧：从 bh+1 往右
        if bh + 1 <= L - 1:
            b = edge_row(z, bh + 1, th[i], g, +1)
            if b is not None:
                run = b - bh
                if run >= minrun:
                    hi[i] = max(bH[i], GRID[b])
    return lo, hi


def ev(D, lo, hi):
    SHORT = D['GL'] <= THR; LONG = ~SHORT
    i = iou(np.clip(lo, 0, 1), np.clip(hi, 0, 1), D['GS'], D['GE'])
    return {'all': metrics(i), 'short': metrics(i[SHORT]), 'long': metrics(i[LONG]), '_iou': i}


def base_metrics(D):
    return ev(D, D['bL'].copy(), D['bH'].copy())


def avg_metrics(list_of_m):
    keys = ('all', 'short', 'long')
    o = {}
    for k in keys:
        o[k] = {kk: float(np.mean([m[k][kk] for m in list_of_m])) for kk in ('mIoU', 'i3', 'i5', 'i7')}
    return o


def dpt(cur, base_, bucket, key):
    return 100.0 * (cur[bucket][key] - base_[bucket][key])


print('=' * 112, flush=True)
print('diag35  边界外推(Edge-Grow)  dataset=%s  校准=%s  test=%s' % (
    DS_NAME, CALIB_JSON, dargs['test_data']), flush=True)
print('=' * 112, flush=True)
print('[1/3] 校准集特征 ...', flush=True)
DC = collect(CALIB_JSON, 'calib', reps=1)[0]
print('      N=%d' % DC['n'], flush=True)
print('[2/3] 测试集特征 ...', flush=True)
DT_list = collect(dargs['test_data'], 'test', reps=K_REP)
print('      N=%d  K_REP=%d' % (DT_list[0]['n'], K_REP), flush=True)

bC = base_metrics(DC)
bT = avg_metrics([base_metrics(D) for D in DT_list])

# 参照：标准并集 RBR
uC = ev(DC, np.minimum(DC['s15'], DC['bL']), np.maximum(DC['e15'], DC['bH']))
uT = avg_metrics([ev(D, np.minimum(D['s15'], D['bL']), np.maximum(D['e15'], D['bH'])) for D in DT_list])
# ORACLE 门控（上界）
oC = ev(DC, np.where(iou(np.minimum(DC['s15'], DC['bL']), np.maximum(DC['e15'], DC['bH']),
                         DC['GS'], DC['GE']) > iou(DC['bL'], DC['bH'], DC['GS'], DC['GE']),
                     np.minimum(DC['s15'], DC['bL']), DC['bL']),
        np.where(iou(np.minimum(DC['s15'], DC['bL']), np.maximum(DC['e15'], DC['bH']),
                     DC['GS'], DC['GE']) > iou(DC['bL'], DC['bH'], DC['GS'], DC['GE']),
                 np.maximum(DC['e15'], DC['bH']), DC['bH']))
oT = avg_metrics([ev(D, *[np.where(
    iou(np.minimum(D['s15'], D['bL']), np.maximum(D['e15'], D['bH']), D['GS'], D['GE'])
    > iou(D['bL'], D['bH'], D['GS'], D['GE']),
    np.minimum(D['s15'], D['bL']), D['bL']),
    np.where(iou(np.minimum(D['s15'], D['bL']), np.maximum(D['e15'], D['bH']), D['GS'], D['GE'])
             > iou(D['bL'], D['bH'], D['GS'], D['GE']),
             np.maximum(D['e15'], D['bH']), D['bH'])]) for D in DT_list])

print('\n[复现校验] test base（应 = diag30：mIoU 0.4720 / R1@0.5 0.5522 / R1@0.7 0.2658）', flush=True)
print('   mIoU %.4f  R1@0.5 %.4f  R1@0.7 %.4f' % (
    bT['all']['mIoU'], bT['all']['i5'], bT['all']['i7']), flush=True)
print('\n[参照]', flush=True)
print('%-24s | %7s %7s | %7s %7s | %7s %7s' % (
    '方法', 'Δ短@0.5', 'Δ长@0.5', 'Δ长@0.7', 'Δ全@0.5', 'Δ全@0.7', 'Δ全mIoU'), flush=True)
for nm, m in (('U_k0.15(标准并集)', uT), ('ORACLE_gate(上界)', oT)):
    print('%-24s | %+7.2f %+7.2f | %+7.2f %+7.2f | %+7.2f %+7.2f' % (
        nm, dpt(m, bT, 'short', 'i5'), dpt(m, bT, 'long', 'i5'), dpt(m, bT, 'long', 'i7'),
        dpt(m, bT, 'all', 'i5'), dpt(m, bT, 'all', 'i7'), dpt(m, bT, 'all', 'mIoU')), flush=True)

# ---------------- 校准 ----------------
print('\n' + '=' * 112, flush=True)
print('[3/3] 在【校准集 train.json】上扫描 (a, g) —— 受"短动作不回归"约束', flush=True)
print('%-18s | %7s %7s | %7s %7s | %7s %7s | %s' % (
    '(a, g)', 'Δ短@0.5', 'Δ长@0.5', 'Δ长@0.7', 'Δ全@0.5', 'Δ全@0.7', 'Δ全mIoU', '约束'), flush=True)
cands = []
for a in A_GRID:
    for g in G_GRID:
        lo, hi = edge_grow(DC, a, g)
        r = ev(DC, lo, hi)
        d5s = dpt(r, bC, 'short', 'i5'); d7s = dpt(r, bC, 'short', 'i7')
        d5l = dpt(r, bC, 'long', 'i5'); d7l = dpt(r, bC, 'long', 'i7')
        d5a = dpt(r, bC, 'all', 'i5'); d7a = dpt(r, bC, 'all', 'i7')
        dma = dpt(r, bC, 'all', 'mIoU')
        okc = (d5s >= -0.05) and (d7s >= -0.05) and (d5a >= -0.05) and (dma >= -0.05)
        cands.append(dict(a=a, g=g, d5s=d5s, d7s=d7s, d5l=d5l, d7l=d7l,
                          d5a=d5a, d7a=d7a, dma=dma, ok=okc))
        print('%-18s | %+7.2f %+7.2f | %+7.2f %+7.2f | %+7.2f %+7.2f | %s' % (
            '(%.2f, %d)' % (a, g), d5s, d5l, d7l, d5a, d7a, dma,
            '<-通过' if okc else ''), flush=True)
okc = [c for c in cands if c['ok']]
if okc:
    sel = max(okc, key=lambda c: c['d7l'])
    print('\n  => 校准选择: a=%.2f, g=%d  (calib 长 Δ@0.7=%+.2f，短 Δ@0.5=%+.2f)' % (
        sel['a'], sel['g'], sel['d7l'], sel['d5s']), flush=True)
else:
    sel = max(cands, key=lambda c: c['d7l'] - 3.0 * max(0.0, -c['d5s']))
    print('\n  => 无候选满足硬约束，退化加权目标: a=%.2f, g=%d' % (sel['a'], sel['g']), flush=True)

# ---------------- 在 test 上报告选定方案 + 全网格 ----------------
loT, hiT = edge_grow(DT_list[0], sel['a'], sel['g'])
selT = avg_metrics([ev(D, *edge_grow(D, sel['a'], sel['g'])) for D in DT_list])
print('\n' + '=' * 112, flush=True)
print('【测试集】选定 a=%.2f, g=%d（test 未参与选择）' % (sel['a'], sel['g']), flush=True)
print('%-24s | %7s %7s | %7s %7s | %7s %7s' % (
    '方法', 'Δ短@0.5', 'Δ长@0.5', 'Δ长@0.7', 'Δ全@0.5', 'Δ全@0.7', 'Δ全mIoU'), flush=True)
print('%-24s | %+7.2f %+7.2f | %+7.2f %+7.2f | %+7.2f %+7.2f' % (
    'EdgeGrow(选定)', dpt(selT, bT, 'short', 'i5'), dpt(selT, bT, 'long', 'i5'),
    dpt(selT, bT, 'long', 'i7'), dpt(selT, bT, 'all', 'i5'),
    dpt(selT, bT, 'all', 'i7'), dpt(selT, bT, 'all', 'mIoU')), flush=True)

print('\n【透明呈现】全部 (a,g) 在【测试集】上的表现', flush=True)
print('%-18s | %7s %7s | %7s %7s | %7s %7s | %s' % (
    '(a, g)', 'Δ短@0.5', 'Δ短@0.7', 'Δ长@0.5', 'Δ长@0.7', 'Δ全@0.5', 'Δ全@0.7', '达标?'), flush=True)
rows = []
for a in A_GRID:
    for g in G_GRID:
        m = avg_metrics([ev(D, *edge_grow(D, a, g)) for D in DT_list])
        d5s = dpt(m, bT, 'short', 'i5'); d7s = dpt(m, bT, 'short', 'i7')
        d5l = dpt(m, bT, 'long', 'i5'); d7l = dpt(m, bT, 'long', 'i7')
        d5a = dpt(m, bT, 'all', 'i5'); d7a = dpt(m, bT, 'all', 'i7')
        dma = dpt(m, bT, 'all', 'mIoU')
        ok = (d5s >= -0.2) and (d7s >= -0.3) and (d5l >= 2.0) and (d7l >= 5.0) \
             and (d5a >= 0.0) and (d7a > 0.0) and (dma > 0.0)
        rows.append((d7l, a, g, d5s, d7s, d5l, d5a, d7a, dma, ok))
for d7l, a, g, d5s, d7s, d5l, d5a, d7a, dma, ok in sorted(rows, reverse=True):
    print('%-18s | %+7.2f %+7.2f | %+7.2f %+7.2f | %+7.2f %+7.2f | %s' % (
        '(%.2f, %d)' % (a, g), d5s, d7s, d5l, d7l, d5a, d7a, '**达标**' if ok else ''), flush=True)

print('\n[基线绝对值] base 短 mIoU %.4f R1@0.5 %.4f R1@0.7 %.4f' % (
    bT['short']['mIoU'], bT['short']['i5'], bT['short']['i7']), flush=True)
print('             base 长 mIoU %.4f R1@0.5 %.4f R1@0.7 %.4f' % (
    bT['long']['mIoU'], bT['long']['i5'], bT['long']['i7']), flush=True)
print('\n' + '=' * 112, flush=True)
print('DONE', flush=True)
