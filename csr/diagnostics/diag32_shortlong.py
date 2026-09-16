"""diag32：把「短动作退步、长动作增强」改成「短动作不降、长动作显著增强」

现状（Charades-CG, K_REP=3, 论文口径）：
  短动作(gt<=0.35)  R1@0.5 -2.98pt | R1@0.7 -2.21pt | mIoU -0.17pt
  长动作(gt>0.35)   R1@0.5 +4.77pt | R1@0.7 +10.49pt | mIoU +4.66pt

为什么之前的门控扫描（diag26）"看起来没用"：
  diag26 只看 **mean IoU**，短动作只掉 0.17pt，看起来无害；
  但论文口径是 **阈值命中率**（IoU>=0.5）。扩框把一批短样本的 IoU 从 0.5x 压到 0.4x，
  mean IoU 几乎不动，R1@0.5 却掉 3.0pt —— 这就是"跨线效应"。
  ⇒ 必须在**阈值口径 + 短/长分桶**下做扫描。

本脚本（全部零训练、纯推理期、**不使用任何 gt 参与框构造**）：
  模型只前向一次/样本（逐点提案拿相关性曲线），所有变体都在缓存数组上事后计算。

扫描族
  A. U_k        并集-阈值扫描（k 放宽到 2.0，看严格核是否天然无伤短动作）
  B. CAP_k_b    并集 + 单侧扩框上限（每侧最多外扩 b*wb）
  C. SNAP_k_g   有界收缩（把框向高置信核收，最多收 g*wb）—— 对偶操作
  D. MIX_k_g    宽则并集、窄则有界缩
  E. MIN_k_t    逐侧最小延长量（延长不足 t 就不动该侧）
  F. CORE_k_g   闭运算填洞 + 高置信核并集（防止长动作内部凹陷截断 flood）
  G. ORACLE_*   上界诊断（逐样本最优门控 / 只修中心 / 只修宽度）

指标：mIoU / R1@0.3 / R1@0.5 / R1@0.7，全部按 全部 / 短(gt<=0.35) / 长(gt>0.35) 三档报告。
协议：K_REP>=3，报告 mean±std。

用法: python diag32_gen.py <config> <ckpt>
环境变量: BS=16  K_REP=3  LIMIT=0  THR=0.35
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
GRID = np.linspace(0, 1, L)
BS = int(os.environ.get('BS', 16))
K_REP = int(os.environ.get('K_REP', 3))
THR = float(os.environ.get('THR', 0.35))
loader = DataLoader(ds, batch_size=BS, shuffle=False, collate_fn=COLLATE, num_workers=8)
N = len(ds)

print('=' * 108, flush=True)
print('diag32  数据集无关版  dataset=%s  N=%d  P=%d  L=%d  K_REP=%d  短/长阈值=%.2f' % (
    DS_NAME, N, P, L, K_REP, THR), flush=True)
print('  ckpt=%s' % ckpt, flush=True)
print('  feature_path=%s' % FP, flush=True)
print('=' * 108, flush=True)


# ---------------------------------------------------------------- 基础工具
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
    """从 pc 出发在布尔掩码 m 上向两侧 flood，返回区间 [s,e]（GRID 坐标）。"""
    n, l = m.shape
    s = np.zeros(n); e = np.ones(n)
    for i in range(n):
        j = pc[i]
        a = j
        while a > 0 and m[i, a - 1]:
            a -= 1
        b = j
        while b < l - 1 and m[i, b + 1]:
            b += 1
        s[i] = GRID[a]; e[i] = GRID[b]
    return s, e


def fill_gaps(m, g):
    """闭运算：把长度 < g 的 False 空洞填成 True（防止长动作内部凹陷截断 flood）。"""
    if g <= 1:
        return m
    out = m.copy()
    n, l = m.shape
    for i in range(n):
        row = out[i]
        j = 0
        while j < l:
            if not row[j]:
                k0 = j
                while k0 < l and not row[k0]:
                    k0 += 1
                if j > 0 and k0 < l and (k0 - j) < g:
                    out[i, j:k0] = True
                j = k0
            else:
                j += 1
    return out


# ---------------------------------------------------------------- 前向：缓存逐样本数组
def collect(rep):
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
    del model
    torch.cuda.empty_cache()

    G = np.concatenate(G); N8 = np.concatenate(N8); C8 = np.concatenate(C8)
    W8 = np.concatenate(W8); REL = np.concatenate(REL)
    n = len(G); r = np.arange(n)
    GS, GE = G[:, 0], G[:, 1]; GL = GE - GS
    order = np.argsort(N8, axis=1)
    j1 = order[:, 0]
    cb, wb = C8[r, j1], W8[r, j1]
    bL, bH = np.clip(cb - wb / 2, 0, 1), np.clip(cb + wb / 2, 0, 1)
    return dict(n=n, GS=GS, GE=GE, GL=GL, cb=cb, wb=wb, bL=bL, bH=bH,
                C8=C8, W8=W8, REL=REL, order=order, P=P)


KS = [0.15, 0.30, 0.50, 0.75, 1.00, 1.50, 2.00]
GAPS = [2, 3, 5]
CAPS = [0.05, 0.10, 0.15, 0.25, 0.50, 1.00]
SHRINKS = [0.15, 0.30, 0.50]
MINS = [0.02, 0.05, 0.08, 0.10, 0.15, 0.20]
RATIOS = [1.00, 1.25, 1.50]
# 组合族只挂在最有希望的基座上（k=0.15 宽严适中 / k=0.50 略严 / 闭运算 g=3）
CG_BASE = [(0.15, 0), (0.50, 0), (0.15, 3), (0.50, 3), (0.50, 5)]

PER_REP = []     # list of dict: tag -> {'all':m,'short':m,'long':m}


def build_and_score(D):
    """在缓存数组上构造所有变体的框并打分；返回 dict tag -> 三个桶的 metrics。"""
    GS, GE, GL = D['GS'], D['GE'], D['GL']
    cb, wb, bL, bH = D['cb'], D['wb'], D['bL'], D['bH']
    REL = D['REL']
    SHORT = GL <= THR
    LONG = ~SHORT
    Z = zs(REL)
    pc = np.clip((cb * (L - 1)).round().astype(int), 0, L - 1)

    def put(store, tag, lo, hi):
        lo = np.clip(lo, 0, 1); hi = np.clip(hi, 0, 1)
        hi = np.maximum(hi, lo + 1e-6)
        i = iou(lo, hi, GS, GE)
        store[tag] = {'all': metrics(i), 'short': metrics(i[SHORT]), 'long': metrics(i[LONG])}

    out = {}
    put(out, 'base', bL.copy(), bH.copy())

    # 预计算各 k 的 flood 区间
    FL = {}
    for k in KS:
        m = Z > (Z.mean(1) + k * Z.std(1))[:, None]
        m[np.arange(len(cb)), pc] = True
        FL[k] = flood(m, pc)
    # 闭运算后的 flood 区间
    FLC = {}
    for g in GAPS:
        for k in (0.15, 0.50, 1.00):
            m = Z > (Z.mean(1) + k * Z.std(1))[:, None]
            m[np.arange(len(cb)), pc] = True
            FLC[(k, g)] = flood(fill_gaps(m, g), pc)

    # ---------------- A. 并集-阈值扫描 ----------------
    for k in KS:
        s, e = FL[k]
        put(out, 'U_k%.2f' % k, np.minimum(s, bL), np.maximum(e, bH))

    # ---------------- B. 并集 + 单侧扩框上限 ----------------
    for k in (0.15, 0.50):
        s, e = FL[k]
        for b in CAPS:
            lo = np.maximum(np.minimum(s, bL), cb - (0.5 + b) * wb)
            hi = np.minimum(np.maximum(e, bH), cb + (0.5 + b) * wb)
            lo = np.minimum(lo, bL); hi = np.maximum(hi, bH)
            put(out, 'CAP_k%.2f_b%.2f' % (k, b), lo, hi)

    # ---------------- C. 有界收缩（对偶操作） ----------------
    for k in (0.50, 1.00, 1.50):
        s, e = FL[k]
        for g in SHRINKS:
            lim = g * wb
            lo = bL + np.minimum(np.maximum(0.0, s - bL), lim)
            hi = bH - np.minimum(np.maximum(0.0, bH - e), lim)
            put(out, 'SNAP_k%.2f_g%.2f' % (k, g), lo, hi)

    # ---------------- D. 混合：宽则并集 / 窄则有界缩 ----------------
    for k in (0.50, 1.00):
        s, e = FL[k]
        wide = (e - s) > wb
        for g in SHRINKS:
            lim = g * wb
            lo_s = bL + np.minimum(np.maximum(0.0, s - bL), lim)
            hi_s = bH - np.minimum(np.maximum(0.0, bH - e), lim)
            lo_u = np.minimum(s, bL); hi_u = np.maximum(e, bH)
            put(out, 'MIX_k%.2f_g%.2f' % (k, g),
                np.where(wide, lo_u, lo_s), np.where(wide, hi_u, hi_s))

    # ---------------- E. 逐侧最小延长量 ----------------
    for k in (0.15, 0.50):
        s, e = FL[k]
        for t in MINS:
            extL = np.maximum(0.0, bL - s); extR = np.maximum(0.0, e - bH)
            lo = np.where(extL > t, np.minimum(s, bL), bL)
            hi = np.where(extR > t, np.maximum(e, bH), bH)
            put(out, 'MIN_k%.2f_t%.2f' % (k, t), lo, hi)

    # ---------------- F. 闭运算填洞 + 并集 ----------------
    for k in (0.15, 0.50, 1.00):
        for g in GAPS:
            s, e = FLC[(k, g)]
            put(out, 'CORE_k%.2f_g%d' % (k, g), np.minimum(s, bL), np.maximum(e, bH))

    # ---------------- H/I/J/K. 组合族：挂在最有希望的基座上 ----------------
    for (k, g) in CG_BASE:
        s, e = (FL[k] if g == 0 else FLC[(k, g)])
        t0 = 'k%.2f%s' % (k, ('' if g == 0 else '_c%d' % g))
        # H. 核 + 单侧扩框上限
        for b in (0.05, 0.10, 0.15, 0.25):
            lo = np.maximum(np.minimum(s, bL), cb - (0.5 + b) * wb)
            hi = np.minimum(np.maximum(e, bH), cb + (0.5 + b) * wb)
            lo = np.minimum(lo, bL); hi = np.maximum(hi, bH)
            put(out, 'CCAP_%s_b%.2f' % (t0, b), lo, hi)
        # I. 核 + 逐侧最小延长量
        for t in (0.05, 0.08, 0.10, 0.15):
            extL = np.maximum(0.0, bL - s); extR = np.maximum(0.0, e - bH)
            lo = np.where(extL > t, np.minimum(s, bL), bL)
            hi = np.where(extR > t, np.maximum(e, bH), bH)
            put(out, 'CMIN_%s_t%.2f' % (t0, t), lo, hi)
        # J. 门控：核宽必须显著超过框宽才启用
        for rho in RATIOS:
            on = (e - s) > rho * wb
            lo = np.where(on, np.minimum(s, bL), bL)
            hi = np.where(on, np.maximum(e, bH), bH)
            put(out, 'GATE_%s_r%.2f' % (t0, rho), lo, hi)
        # K. 核明显更窄时向核有界收缩，否则并集
        for gs_ in (0.10, 0.20, 0.30):
            lim = gs_ * wb
            lo_s = bL + np.minimum(np.maximum(0.0, s - bL), lim)
            hi_s = bH - np.minimum(np.maximum(0.0, bH - e), lim)
            narrow = (e - s) < wb
            lo_u = np.minimum(s, bL); hi_u = np.maximum(e, bH)
            put(out, 'CSNAP_%s_n%.2f' % (t0, gs_),
                np.where(narrow, lo_s, lo_u), np.where(narrow, hi_s, hi_u))

    # ---------------- G. ORACLE 上界诊断（仅供参照，不是可用方法） ----------------
    i_base = iou(bL, bH, GS, GE)
    s, e = FL[0.15]
    i_rbr = iou(np.minimum(s, bL), np.maximum(e, bH), GS, GE)
    put(out, 'ORACLE_gate(逐样本取优)', np.where(i_rbr > i_base, np.minimum(s, bL), bL),
        np.where(i_rbr > i_base, np.maximum(e, bH), bH))
    # 只修中心：gt 中心 + 原框宽
    put(out, 'ORACLE_c(gt中心+原框宽)',
        np.clip((GS + GE) / 2 - wb / 2, 0, 1), np.clip((GS + GE) / 2 + wb / 2, 0, 1))
    put(out, 'ORACLE_w(gt宽度+原中心)', np.clip(cb - GL / 2, 0, 1), np.clip(cb + GL / 2, 0, 1))
    put(out, 'ORACLE_cw(gt中心+gt宽)', GS.copy(), GE.copy())

    # 伤害归因（基于默认 RBR k=0.15）
    Di = i_rbr - i_base
    D['DIAG'] = dict(
        harm_short_frac=float((Di[SHORT] < -1e-9).mean()),
        harm_short_mean=float(Di[SHORT][Di[SHORT] < 0].mean()) if (Di[SHORT] < 0).any() else 0.0,
        harm_long_frac=float((Di[LONG] < -1e-9).mean()),
        expL=np.maximum(0.0, bL - s), expR=np.maximum(0.0, e - bH),
        Di=Di, i_base=i_base, i_rbr=i_rbr,
        cross=float(((i_base >= 0.5) & (i_rbr < 0.5)).mean()),
        save=float(((i_base < 0.5) & (i_rbr >= 0.5)).mean()),
        unexp=float(((bL - s <= 0) & (e - bH <= 0)).mean()),
        corew=e - s, boxw=bH - bL)
    return out


for rep in range(K_REP):
    D = collect(rep)
    if rep == 0:
        D0 = D
    res = build_and_score(D)
    PER_REP.append(res)
    m = res['base']['all']
    print('  rep %d done | base mIoU=%.4f R1@0.5=%.4f R1@0.7=%.4f' % (
        rep, m['mIoU'], m['i5'], m['i7']), flush=True)

TAGS = list(PER_REP[0].keys())


def agg(tag, bucket, key):
    return float(np.mean([r[tag][bucket][key] for r in PER_REP]))


def aggs(tag, bucket, key):
    return float(np.std([r[tag][bucket][key] for r in PER_REP]))


line = '=' * 118
print('\n' + line, flush=True)
print('diag32  %s  N=%d  K_REP=%d  短/长阈值=%.2f' % (DS_NAME, D0['n'], K_REP, THR), flush=True)
print(line, flush=True)

# ---------- 总表 ----------
print('[总表] 全部样本（mean over %d rep）' % K_REP, flush=True)
print('%-26s %9s %9s %9s %9s' % ('方法', 'mIoU', 'R1@0.3', 'R1@0.5', 'R1@0.7'), flush=True)
for tag in TAGS:
    print('%-26s %9.4f %9.4f %9.4f %9.4f' % (
        tag, agg(tag, 'all', 'mIoU'), agg(tag, 'all', 'i3'),
        agg(tag, 'all', 'i5'), agg(tag, 'all', 'i7')), flush=True)

# ---------- 短/长分桶 + Δ ----------
print('\n' + line, flush=True)
print('[关键表] 短动作 vs 长动作的 Δ（×100 = 百分点；排序：长动作 ΔR1@0.7 降序）', flush=True)
print('%-26s | %7s %7s | %7s %7s %7s | %7s %7s | %s' % (
    '方法', 'Δ短@0.5', 'Δ长@0.5', 'Δ短mIoU', 'Δ长mIoU', 'Δ全mIoU', 'Δ短@0.7', 'Δ长@0.7', '达标?'), flush=True)
b5s = agg('base', 'short', 'i5'); b5l = agg('base', 'long', 'i5')
bms = agg('base', 'short', 'mIoU'); bml = agg('base', 'long', 'mIoU')
bma = agg('base', 'all', 'mIoU'); b7s = agg('base', 'short', 'i7'); b7l = agg('base', 'long', 'i7')

rows = []
for tag in TAGS:
    d = dict(
        tag=tag,
        d5s=agg(tag, 'short', 'i5') - b5s, d5l=agg(tag, 'long', 'i5') - b5l,
        dms=agg(tag, 'short', 'mIoU') - bms, dml=agg(tag, 'long', 'mIoU') - bml,
        dma=agg(tag, 'all', 'mIoU') - bma,
        d7s=agg(tag, 'short', 'i7') - b7s, d7l=agg(tag, 'long', 'i7') - b7l,
        d5a=agg(tag, 'all', 'i5') - b5s, d7a=agg(tag, 'all', 'i7') - agg('base', 'all', 'i7'))
    ok = (d['d5s'] >= -0.002) and (d['d7s'] >= -0.003) and (d['dms'] >= -0.002) \
         and (d['d5l'] >= 0.020) and (d['d7l'] >= 0.050) \
         and (d['d5a'] >= 0.0) and (d['d7a'] > 0.0) and (d['dma'] > 0.0)
    d['ok'] = ok
    rows.append(d)

for r in sorted(rows, key=lambda x: -x['d7l']):
    if r['tag'] == 'base':
        continue
    mark = ' **达标**' if r['ok'] else ''
    print('%-26s | %+7.2f %+7.2f | %+7.2f %+7.2f %+7.2f | %+7.2f %+7.2f |%s' % (
        r['tag'], 100 * r['d5s'], 100 * r['d5l'], 100 * r['dms'], 100 * r['dml'],
        100 * r['dma'], 100 * r['d7s'], 100 * r['d7l'], mark), flush=True)

# ---------- 达标候选汇总 ----------
print('\n' + line, flush=True)
print('[达标候选] 条件：短动作 Δ@0.5>=-0.2pt 且 Δ@0.7>=-0.3pt 且 ΔmIoU>=-0.2pt；'
      '长动作 Δ@0.5>=+2.0pt 且 Δ@0.7>=+5.0pt；整体 Δ@0.5>=0、Δ@0.7>0、ΔmIoU>0', flush=True)
wins = [r for r in rows if r['ok']]
if not wins:
    print('  无。最高「长动作 ΔR1@0.7」的 5 个变体（未达标，供下一轮设计参考）：', flush=True)
    for r in sorted(rows, key=lambda x: -x['d7l'])[:5]:
        if r['tag'] == 'base':
            continue
        print('    %-26s Δ短@0.5=%+.2f Δ长@0.5=%+.2f Δ长@0.7=%+.2f Δ全@0.5=%+.2f' % (
            r['tag'], 100 * r['d5s'], 100 * r['d5l'], 100 * r['d7l'], 100 * r['d5a']), flush=True)
else:
    for r in sorted(wins, key=lambda x: -x['d7l']):
        print('  ** %-26s Δ短@0.5=%+.2f Δ长@0.5=%+.2f Δ长@0.7=%+.2f Δ全@0.5=%+.2f Δ全@0.7=%+.2f Δ全mIoU=%+.2f' % (
            r['tag'], 100 * r['d5s'], 100 * r['d5l'], 100 * r['d7l'],
            100 * r['d5a'], 100 * r['d7a'], 100 * r['dma']), flush=True)

# ---------- 基线绝对值 ----------
print('\n' + line, flush=True)
print('[基线绝对值，供核对] base 短: mIoU %.4f R1@0.5 %.4f R1@0.7 %.4f  (n=%d)' % (
    bms, b5s, b7s, int((D0['GL'] <= THR).sum())), flush=True)
print('%-24s base 长: mIoU %.4f R1@0.5 %.4f R1@0.7 %.4f  (n=%d)' % (
    '', bml, b5l, b7l, int((D0['GL'] > THR).sum())), flush=True)

# ---------- ORACLE 上界 ----------
print('\n[ORACLE 上界：任何方法都超不过]', flush=True)
print('  逐样本最优门控  Δ短@0.5=%+.2f   Δ长@0.5=%+.2f   Δ长@0.7=%+.2f   Δ全mIoU=%+.2f' % (
    100 * (agg('ORACLE_gate(逐样本取优)', 'short', 'i5') - b5s),
    100 * (agg('ORACLE_gate(逐样本取优)', 'long', 'i5') - b5l),
    100 * (agg('ORACLE_gate(逐样本取优)', 'long', 'i7') - b7l),
    100 * (agg('ORACLE_gate(逐样本取优)', 'all', 'mIoU') - bma)), flush=True)
print('  只修中心(gt中心+原框宽) Δ短@0.5=%+.2f Δ长@0.5=%+.2f Δ全mIoU=%+.2f' % (
    100 * (agg('ORACLE_c(gt中心+原框宽)', 'short', 'i5') - b5s),
    100 * (agg('ORACLE_c(gt中心+原框宽)', 'long', 'i5') - b5l),
    100 * (agg('ORACLE_c(gt中心+原框宽)', 'all', 'mIoU') - bma)), flush=True)
print('  只修宽度(gt宽度+原中心) Δ短@0.5=%+.2f Δ长@0.5=%+.2f Δ全mIoU=%+.2f' % (
    100 * (agg('ORACLE_w(gt宽度+原中心)', 'short', 'i5') - b5s),
    100 * (agg('ORACLE_w(gt宽度+原中心)', 'long', 'i5') - b5l),
    100 * (agg('ORACLE_w(gt宽度+原中心)', 'all', 'mIoU') - bma)), flush=True)
print('  中心+宽度全对     Δ短@0.5=%+.2f Δ长@0.5=%+.2f Δ全mIoU=%+.2f' % (
    100 * (agg('ORACLE_cw(gt中心+gt宽)', 'short', 'i5') - b5s),
    100 * (agg('ORACLE_cw(gt中心+gt宽)', 'long', 'i5') - b5l),
    100 * (agg('ORACLE_cw(gt中心+gt宽)', 'all', 'mIoU') - bma)), flush=True)

# ---------- 伤害归因 ----------
G_ = D0['DIAG']
print('\n[伤害归因，基于 U_k0.15]', flush=True)
print('  短样本被伤害比例 %.4f  被伤害样本的平均 ΔIoU %+.4f' % (
    G_['harm_short_frac'], G_['harm_short_mean']), flush=True)
print('  长样本被伤害比例 %.4f' % G_['harm_long_frac'], flush=True)
print('  IoU 跨线：0.5 上→下 %.4f   0.5 下→上 %.4f' % (G_['cross'], G_['save']), flush=True)
print('  未发生扩展的样本比例 %.4f（框本来就≥flood区间）' % G_['unexp'], flush=True)
print('  基线框宽 中位 %.4f | flood核宽 中位 %.4f' % (
    np.median(D0['DIAG']['boxw']), np.median(D0['DIAG']['corew'])), flush=True)

print('\n' + line, flush=True)
print('DONE', flush=True)
