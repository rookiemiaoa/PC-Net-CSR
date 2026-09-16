"""diag36：把 EdgeGrow 的"平台检测"当**门控**，把 RBR 的 flood 当**扩展量**

动机（d34/d35 的联合教训）
  * d34：用"扩展区域的相关性强度"(exz/minz/maxz/ratio) 做门控 —— 全部失败。
    原因：RBR 过扩出来的区域，在相关性曲线上看起来同样"合理"，信号不可分。
  * d35：EdgeGrow（从框边界**外侧**往外长）把短伤从 -2.98 压到 -0.45（6.6 倍改善），
    但它的**扩展量**也小（长 @0.7 只 +4~6，而 RBR 能 +10~14）。

本方法的假设（可证伪）
  EdgeGrow 的判断力（"边界外是否存在连续平台"）和 RBR 的扩展力（"宽松阈值 flood 到最远"）
  是两个**正交**的能力。用前者的**布尔判定**去开关后者的**扩展量**，即可
     短动作（边界外立刻掉下去 → 判 False）→ 不动 → 零伤害
     长动作（边界外仍是平台     → 判 True ）→ 吃满 RBR 增益
  关键在于 EdgeGrow 的判定是否比 d34 的相关性强度信号更可分。

族（全部零训练、纯推理期、不使用 gt）
  U_k0.15        : 参照，标准 RBR 并集
  AGG_a{}_{}m    : "任一侧"被 EG 判定为平台 -> 整框用 U_k0.15，否则完全不动
  BGG_a{}_{}m    : "两侧都"被判定 -> 用 U_k0.15（更严格门控）
  SGG_a{}_{}m    : 逐侧门控 + 该侧用 U_k0.15 的边界（最精细）
  EGS_a{}_{}m    : 逐侧门控 + 该侧用 EG 自己的边界（= d35 的行为，作对照）

校准纪律（与 d34/d35 一致）
  a/m 在 train.json 上选，受"短动作不回归"约束后取长 Δ@0.7 最大；
  **test_trivial.json 不参与选择**，只用于报告，且把全部变体在 test 上打印出来。

用法: python diag36_gen.py <test_config> <ckpt>
环境变量: BS=16  CALIB_JSON=data/charades/train.json  THR=0.35  LIMIT=0  K_REP=1  NO_CACHE=0
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
NO_CACHE = os.environ.get('NO_CACHE', '0') == '1'
CACHE_DIR = os.environ.get('CACHE_DIR') or os.path.join(_REPO, 'csr', 'analysis', 'feat_cache')

A_GRID = (0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)
M_GRID = (1, 2, 3, 5)


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


def collect_cached(data_json, tag, reps=1):
    """特征提取 + 曲线计算的结果缓存到 pickle —— 第二轮起迭代只需数秒。"""
    os.makedirs(CACHE_DIR, exist_ok=True)
    key = '%s__%s__r%d__lim%d__L%d.pkl' % (
        DS_NAME, os.path.basename(data_json).replace('.json', ''), reps, LIMIT, L)
    fp = os.path.join(CACHE_DIR, key)
    if os.path.exists(fp) and not NO_CACHE:
        with open(fp, 'rb') as f:
            out = pickle.load(f)
        print('    [%s] 缓存命中 %s  (N=%d)' % (tag, fp, out[0]['n']), flush=True)
        return out
    out = collect(data_json, tag, reps)
    with open(fp, 'wb') as f:
        pickle.dump(out, f)
    print('    [%s] 已缓存 -> %s' % (tag, fp), flush=True)
    return out


def edge_row(z, start, th, gap, direction):
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


def eg_flags(D, a, g=0, minrun=1):
    """返回 (左判定, 右判定, 左边界, 右边界)。判定=该侧存在 >=minrun 帧的连续达标平台。"""
    n = D['n']; Z = D['Z']; bli = D['bli']; bhi = D['bhi']
    bL = D['bL']; bH = D['bH']
    th = a * Z.max(1)
    lok = np.zeros(n, dtype=bool); rok = np.zeros(n, dtype=bool)
    lb = bL.copy(); rb = bH.copy()
    for i in range(n):
        z = Z[i]; bl, bh = bli[i], bhi[i]
        if bl - 1 >= 0:
            b = edge_row(z, bl - 1, th[i], g, -1)
            if b is not None and (bl - b) >= minrun:
                lok[i] = True; lb[i] = GRID[b]
        if bh + 1 <= L - 1:
            b = edge_row(z, bh + 1, th[i], g, +1)
            if b is not None and (b - bh) >= minrun:
                rok[i] = True; rb[i] = GRID[b]
    return lok, rok, lb, rb


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


def build_variants(D):
    """返回 {名称: (lo, hi, 门控启用率 or None)}"""
    bL = D['bL']; bH = D['bH']
    u_lo = np.minimum(D['s15'], bL); u_hi = np.maximum(D['e15'], bH)
    out = {'U_k0.15': (u_lo, u_hi, 1.0)}
    for a in A_GRID:
        for m in M_GRID:
            lok, rok, lb, rb = eg_flags(D, a, 0, m)
            tag = 'a%.2f_m%d' % (a, m)
            # AGG: 任一侧有平台 -> 整框用 RBR
            onA = lok | rok
            out['AGG_' + tag] = (np.where(onA, u_lo, bL), np.where(onA, u_hi, bH), float(onA.mean()))
            # BGG: 两侧都有平台 -> 整框用 RBR
            onB = lok & rok
            out['BGG_' + tag] = (np.where(onB, u_lo, bL), np.where(onB, u_hi, bH), float(onB.mean()))
            # SGG: 逐侧门控 + 该侧用 RBR 边界
            out['SGG_' + tag] = (np.where(lok, u_lo, bL), np.where(rok, u_hi, bH),
                                 float((lok | rok).mean()))
            # EGS: 逐侧门控 + 该侧用 EG 自己的边界（d35 行为，作对照）
            out['EGS_' + tag] = (np.where(lok, np.minimum(lb, bL), bL),
                                 np.where(rok, np.maximum(rb, bH), bH),
                                 float((lok | rok).mean()))
    return out


def scan(DL, tag):
    """对一个数据集(list of reps)求所有变体的平均指标。"""
    per = [build_variants(D) for D in DL]
    names = list(per[0].keys())
    res = {}
    rate = {}
    for nm in names:
        ms = [ev(D, per[k][nm][0], per[k][nm][1]) for k, D in enumerate(DL)]
        res[nm] = avg_metrics(ms)
        rate[nm] = float(np.mean([per[k][nm][2] for k in range(len(DL))]))
    base = avg_metrics([base_metrics(D) for D in DL])
    return res, base, rate


print('\n' + '=' * 112, flush=True)
print('diag36  EdgeGrow 门控 + RBR 扩展   dataset=%s  校准集=%s  test=%s  K_REP=%d' % (
    DS_NAME, CALIB_JSON, dargs['test_data'], K_REP), flush=True)
print('=' * 112, flush=True)

print('\n[1/3] 校准集特征 ...', flush=True)
DC = collect_cached(CALIB_JSON, 'calib', reps=1)[0]
print('      N=%d' % DC['n'], flush=True)

print('\n[2/3] 测试集特征 ...', flush=True)
DT_list = collect_cached(dargs['test_data'], 'test', reps=K_REP)
print('      N=%d  K_REP=%d' % (DT_list[0]['n'], K_REP), flush=True)

print('\n[3/3] 复现校验（test base 应 = diag30：Charades mIoU 0.4720 / R1@0.5 0.5522 / R1@0.7 0.2658）', flush=True)
bT = avg_metrics([base_metrics(D) for D in DT_list])
print('   mIoU %.4f  R1@0.5 %.4f  R1@0.7 %.4f' % (bT['all']['mIoU'], bT['all']['i5'], bT['all']['i7']), flush=True)

print('\n[参照]', flush=True)
print('%-22s | %7s %7s | %7s %7s | %7s %7s' % (
    '方法', 'Δ短@0.5', 'Δ长@0.5', 'Δ长@0.7', 'Δ全@0.5', 'Δ全@0.7', 'Δ全mIoU'), flush=True)
rC, bC, rateC = scan([DC], 'calib')

# ORACLE 门控（逐样本取优）—— 上界
def oracle_gate(D):
    bL = D['bL']; bH = D['bH']
    u_lo = np.minimum(D['s15'], bL); u_hi = np.maximum(D['e15'], bH)
    i_b = iou(bL, bH, D['GS'], D['GE']); i_u = iou(u_lo, u_hi, D['GS'], D['GE'])
    on = i_u > i_b
    return np.where(on, u_lo, bL), np.where(on, u_hi, bH), float(on.mean())


oT = avg_metrics([ev(D, *oracle_gate(D)[:2]) for D in DT_list])
for nm, m in (('U_k0.15(标准并集)', rC['U_k0.15']), ('ORACLE_gate(上界)', oT)):
    print('%-22s | %+7.2f %+7.2f | %+7.2f %+7.2f | %+7.2f %+7.2f' % (
        nm, dpt(m, bT, 'short', 'i5'), dpt(m, bT, 'long', 'i5'), dpt(m, bT, 'long', 'i7'),
        dpt(m, bT, 'all', 'i5'), dpt(m, bT, 'all', 'i7'), dpt(m, bT, 'all', 'mIoU')), flush=True)

print('\n' + '=' * 112, flush=True)
print('[4/4] 在【校准集】上扫描，受"短动作不回归"约束', flush=True)
print('%-22s | %7s %7s | %7s %7s | %7s %7s | %6s %s' % (
    '变体', 'Δ短@0.5', 'Δ长@0.5', 'Δ长@0.7', 'Δ全@0.5', 'Δ全@0.7', 'Δ全mIoU', '启用率', '约束'), flush=True)
cand = []
for nm, m in rC.items():
    if nm == 'U_k0.15':
        continue
    d5s = dpt(m, bC, 'short', 'i5'); d7s = dpt(m, bC, 'short', 'i7')
    d5l = dpt(m, bC, 'long', 'i5'); d7l = dpt(m, bC, 'long', 'i7')
    d5a = dpt(m, bC, 'all', 'i5'); d7a = dpt(m, bC, 'all', 'i7')
    dma = dpt(m, bC, 'all', 'mIoU')
    okc = (d5s >= -0.2) and (d7s >= -0.3) and (dma >= -0.2)
    cand.append(dict(nm=nm, d5s=d5s, d7s=d7s, d5l=d5l, d7l=d7l, d5a=d5a, d7a=d7a, dma=dma, ok=okc))
for c in sorted(cand, key=lambda x: -x['d7l'])[:18]:
    print('%-22s | %+7.2f %+7.2f | %+7.2f %+7.2f | %+7.2f %+7.2f | %6.1f%% %s' % (
        c['nm'], c['d5s'], c['d5l'], c['d7l'], c['d5a'], c['d7a'], c['dma'],
        100.0 * rateC[c['nm']], '<-通过' if c['ok'] else ''), flush=True)
okc = [c for c in cand if c['ok']]
if okc:
    sel = max(okc, key=lambda c: c['d7l'])
    print('\n  => 校准选择: %s  (calib 短 Δ@0.5=%+.2f，长 Δ@0.7=%+.2f)' % (
        sel['nm'], sel['d5s'], sel['d7l']), flush=True)
else:
    sel = max(cand, key=lambda c: c['d7l'])
    print('\n  => 无候选满足硬约束，退化取 calib 长 Δ@0.7 最大: %s' % sel['nm'], flush=True)

print('\n' + '=' * 112, flush=True)
print('【测试集】选定 %s（test 未参与选择）' % sel['nm'], flush=True)
print('%-22s | %7s %7s | %7s %7s | %7s %7s' % (
    '方法', 'Δ短@0.5', 'Δ长@0.5', 'Δ长@0.7', 'Δ全@0.5', 'Δ全@0.7', 'Δ全mIoU'), flush=True)
rT, bT, rateT = scan(DT_list, 'test')
print('%-22s | %+7.2f %+7.2f | %+7.2f %+7.2f | %+7.2f %+7.2f' % (
    '选定(' + sel['nm'] + ')', dpt(rT[sel['nm']], bT, 'short', 'i5'), dpt(rT[sel['nm']], bT, 'long', 'i5'),
    dpt(rT[sel['nm']], bT, 'long', 'i7'), dpt(rT[sel['nm']], bT, 'all', 'i5'),
    dpt(rT[sel['nm']], bT, 'all', 'i7'), dpt(rT[sel['nm']], bT, 'all', 'mIoU')), flush=True)
print('%-22s | %+7.2f %+7.2f | %+7.2f %+7.2f | %+7.2f %+7.2f' % (
    'ORACLE_gate(上界)', dpt(oT, bT, 'short', 'i5'), dpt(oT, bT, 'long', 'i5'),
    dpt(oT, bT, 'long', 'i7'), dpt(oT, bT, 'all', 'i5'),
    dpt(oT, bT, 'all', 'i7'), dpt(oT, bT, 'all', 'mIoU')), flush=True)

print('\n【透明呈现】全部变体在【测试集】上（按 长 Δ@0.7 降序）', flush=True)
print('%-22s | %7s %7s | %7s %7s | %7s %7s | %8s %s' % (
    '变体', 'Δ短@0.5', 'Δ长@0.5', 'Δ长@0.7', 'Δ全@0.5', 'Δ全@0.7', 'ΔmIoU', '启用率', '达标?'), flush=True)
rows = []
for nm, m in rT.items():
    d5s = dpt(m, bT, 'short', 'i5'); d7s = dpt(m, bT, 'short', 'i7')
    d5l = dpt(m, bT, 'long', 'i5'); d7l = dpt(m, bT, 'long', 'i7')
    d5a = dpt(m, bT, 'all', 'i5'); d7a = dpt(m, bT, 'all', 'i7')
    dma = dpt(m, bT, 'all', 'mIoU')
    ok = (d5s >= -0.2) and (d7s >= -0.3) and (d5l >= 2.0) and (d7l >= 5.0) \
         and (d5a >= 0.0) and (d7a > 0.0) and (dma > 0.0)
    rows.append((d7l, nm, d5s, d7s, d5l, d5a, d7a, dma, rateT[nm], ok))
for d7l, nm, d5s, d7s, d5l, d5a, d7a, dma, rt, ok in sorted(rows, reverse=True)[:40]:
    print('%-22s | %+7.2f %+7.2f | %+7.2f %+7.2f | %+7.2f %+7.2f | %7.1f%% %s' % (
        nm, d5s, d5l, d7l, d5a, d7a, dma, 100.0 * rt, '**达标**' if ok else ''), flush=True)

print('\n[基线绝对值] base 短 mIoU %.4f R1@0.5 %.4f R1@0.7 %.4f' % (
    bT['short']['mIoU'], bT['short']['i5'], bT['short']['i7']), flush=True)
print('             base 长 mIoU %.4f R1@0.5 %.4f R1@0.7 %.4f' % (
    bT['long']['mIoU'], bT['long']['i5'], bT['long']['i7']), flush=True)
print('             base 全 mIoU %.4f R1@0.5 %.4f R1@0.7 %.4f' % (
    bT['all']['mIoU'], bT['all']['i5'], bT['all']['i7']), flush=True)

print('\n' + '=' * 112, flush=True)
print('DONE', flush=True)
