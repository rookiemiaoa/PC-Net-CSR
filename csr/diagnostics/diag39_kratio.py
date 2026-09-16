"""diag39：聚焦 kratio / margin 两信号 —— 找出"短不降 + 长显著增强"的可用判据

d38 的结论（信号筛选，AUC）
  * "边框外是否有平台"(ext) AUC≈0.55；"对比度"(rho) AUC≈0.45 ⇒ 这两个不可用
  * **kratio = flood核宽 / top-1框宽**：校准 0.675 / 测试 0.653（短 0.636 / 长 0.632）⇒ 四路一致，可用
      直觉：kratio 高 => 相关性核心几乎盖满框 => 框"装不下"该区域 => 该扩
            kratio 低 => 框远宽于核心        => 框已经够大     => 别扩
  * **margin = nll(top2) - nll(top1)，取反后 AUC≈0.65**：前两名提案几乎打平 => 定位不确定 => 该扩
  * wb(框宽) 取反 0.544、edgeclose 0.568、nll1 取反 0.561 —— 弱信号，作组合项

d38 遗留的两个问题（本脚本解决）
  1. d38 的规则排序只按"长 d@0.7"，把 kratio 的最优用法（可能"短保得多、长涨得少"）排在表外
  2. d38 没打印 AND 组合与完整前沿

本脚本
  [A] 单信号完整扫（kratio / margin / wb / edgeclose / zoutmax / ext，各 9 分位）
  [B] 两信号 AND 全组合（6 信号两两 × 9×9 阈值）
  [C] 复合分数 s = a*z(kratio) + b*z(-margin)，阈值取分位数
  [D] 三种口径分别报告：校准集选优 / 零 gt 分位数 / 完整前沿
  全部在 v2 缓存上跑（秒级），K_REP=3。

用法: python diag39_gen.py <test_config> <ckpt>
环境变量: BS=16 CALIB_JSON=data/charades/train.json THR=0.35 T_SCREEN=0.60 MINRUN=1
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
CACHE_DIR = os.environ.get('CACHE_DIR') or os.path.join(_REPO, 'csr', 'analysis', 'feat_cache')
T_SCREEN = float(os.environ.get('T_SCREEN', 0.60))
MINRUN = int(os.environ.get('MINRUN', 1))
QS = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)


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
        order = np.argsort(N8, axis=1); j1 = order[:, 0]; j2 = order[:, 1]
        cb, wb = C8[r, j1], W8[r, j1]
        bL = np.clip(cb - wb / 2, 0, 1); bH = np.clip(cb + wb / 2, 0, 1)
        Z = zs(REL)
        pc = np.clip((cb * (L - 1)).round().astype(int), 0, L - 1)
        m15 = Z > (Z.mean(1) + 0.15 * Z.std(1))[:, None]
        m15[np.arange(n), pc] = True
        s15, e15 = flood(m15, pc)
        out.append(dict(GS=GS, GE=GE, GL=GE - GS, cb=cb, wb=wb, bL=bL, bH=bH, Z=Z,
                        REL=REL, s15=s15, e15=e15, n=n,
                        N1=N8[r, j1], N2=N8[r, j2], c1=C8[r, j1], c2=C8[r, j2],
                        bli=np.clip((bL * (L - 1)).round().astype(int), 0, L - 1),
                        bhi=np.clip((bH * (L - 1)).round().astype(int), 0, L - 1)))
    return out


def collect_cached(data_json, tag, reps=1):
    os.makedirs(CACHE_DIR, exist_ok=True)
    key = '%s__%s__r%d__lim%d__L%d__v2.pkl' % (
        DS_NAME, os.path.basename(data_json).replace('.json', ''), reps, LIMIT, L)
    fp = os.path.join(CACHE_DIR, key)
    if os.path.exists(fp):
        with open(fp, 'rb') as f:
            out = pickle.load(f)
        print('    [%s] 缓存命中 %s  (N=%d)' % (tag, fp, out[0]['n']), flush=True)
        return out
    out = collect(data_json, tag, reps)
    with open(fp, 'wb') as f:
        pickle.dump(out, f)
    print('    [%s] 已缓存 -> %s' % (tag, fp), flush=True)
    return out


def side_run(z, start, th, direction=1, gap=0):
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


SIG_NAMES = ('ext', 'rho', 'zedge', 'zoutmax', 'zoutmean', 'zoutmin', 'grad',
             'extratio', 'wb', 'cb', 'edgeclose', 'kratio', 'margin', 'r2dist', 'nll1')


def side_pack(D):
    n = D['n']; Z = D['Z']; REL = D['REL']
    bL = D['bL']; bH = D['bH']; bli = D['bli']; bhi = D['bhi']; GS = D['GS']; GE = D['GE']
    RN = (REL - REL.min(1, keepdims=True)) / (REL.max(1, keepdims=True) - REL.min(1, keepdims=True) + 1e-8)
    Zmax = Z.max(1)
    kw = D['e15'] - D['s15']
    SL = {k: np.full(n, -1e9) for k in SIG_NAMES}
    SR = {k: np.full(n, -1e9) for k in SIG_NAMES}
    for i in range(n):
        z = Z[i]; rn = RN[i]; bl, bh = bli[i], bhi[i]
        mi = max(float(rn[bl:bh + 1].mean()), 0.05)
        mz = float(z[bl:bh + 1].mean())
        for S in (SL, SR):
            S['wb'][i] = D['wb'][i]; S['cb'][i] = D['cb'][i]
            S['kratio'][i] = kw[i] / max(D['wb'][i], 1e-8)
            S['margin'][i] = D['N2'][i] - D['N1'][i]
            S['r2dist'][i] = abs(D['c2'][i] - D['c1'][i])
            S['nll1'][i] = D['N1'][i]
        if bl - 1 >= 0:
            b = side_run(z, bl - 1, T_SCREEN * Zmax[i], -1, 0)
            if b is not None and (bl - b) >= MINRUN:
                sp = slice(b, bl)
                SL['ext'][i] = (bl - b) / (L - 1.0)
                SL['rho'][i] = float(rn[sp].mean()) / mi
                SL['zedge'][i] = float(z[bl - 1])
                SL['zoutmax'][i] = float(z[sp].max())
                SL['zoutmean'][i] = float(z[sp].mean())
                SL['zoutmin'][i] = float(z[sp].min())
                SL['grad'][i] = float(z[bl - 1] - z[bl])
                SL['extratio'][i] = SL['ext'][i] / max(D['wb'][i], 1e-8)
                SL['edgeclose'][i] = float(bL[i])
        if bh + 1 <= L - 1:
            b = side_run(z, bh + 1, T_SCREEN * Zmax[i], +1, 0)
            if b is not None and (b - bh) >= MINRUN:
                sp = slice(bh + 1, b + 1)
                SR['ext'][i] = (b - bh) / (L - 1.0)
                SR['rho'][i] = float(rn[sp].mean()) / mi
                SR['zedge'][i] = float(z[bh + 1])
                SR['zoutmax'][i] = float(z[sp].max())
                SR['zoutmean'][i] = float(z[sp].mean())
                SR['zoutmin'][i] = float(z[sp].min())
                SR['grad'][i] = float(z[bh + 1] - z[bh])
                SR['extratio'][i] = SR['ext'][i] / max(D['wb'][i], 1e-8)
                SR['edgeclose'][i] = float(1.0 - bH[i])
    ib = iou(bL, bH, GS, GE)
    loL = np.where(SL['ext'] > 0, bL - SL['ext'], bL)
    loR = np.where(SR['ext'] > 0, bH + SR['ext'], bH)
    labL = iou(loL, bH, GS, GE) > ib
    labR = iou(bL, loR, GS, GE) > ib
    return SL, SR, labL, labR


def auc(score, label):
    label = np.asarray(label, dtype=bool)
    k = np.isfinite(score) & (score > -1e8)
    if k.sum() < 10:
        return float('nan')
    s = score[k]; lb = label[k]
    n1 = float(lb.sum()); n0 = float(len(lb) - n1)
    if n1 < 1 or n0 < 1:
        return float('nan')
    o = np.argsort(s, kind='mergesort')
    rk = np.empty(len(s), dtype=np.float64); rk[o] = np.arange(1, len(s) + 1)
    return float((rk[lb].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def ev(D, lo, hi):
    SHORT = D['GL'] <= THR; LONG = ~SHORT
    i = iou(np.clip(lo, 0, 1), np.clip(hi, 0, 1), D['GS'], D['GE'])
    return {'all': metrics(i), 'short': metrics(i[SHORT]), 'long': metrics(i[LONG])}


def base_metrics(D):
    return ev(D, D['bL'].copy(), D['bH'].copy())


def avg_metrics(ms):
    o = {}
    for k in ('all', 'short', 'long'):
        o[k] = {kk: float(np.mean([m[k][kk] for m in ms])) for kk in ('mIoU', 'i3', 'i5', 'i7')}
    return o


def dpt(cur, base_, bucket, key):
    return 100.0 * (cur[bucket][key] - base_[bucket][key])


def eval_gate(D, SL, SR, conds):
    """conds = [(signal, thr, dir)]，全 AND、**逐侧**作用；ext>0 为必要条件。
       dir>0 => sig>=thr；dir<0 => sig<=thr。"""
    gL = SL['ext'] > 0
    gR = SR['ext'] > 0
    for sig, thr, d in conds:
        if d > 0:
            gL = gL & (SL[sig] >= thr); gR = gR & (SR[sig] >= thr)
        else:
            gL = gL & (SL[sig] <= thr); gR = gR & (SR[sig] <= thr)
    lo = np.where(gL, D['bL'] - np.maximum(SL['ext'], 0), D['bL'])
    hi = np.where(gR, D['bH'] + np.maximum(SR['ext'], 0), D['bH'])
    return avg_metrics([ev(D, lo, hi)])


def eval_rule(packs, base, conds):
    ms = [eval_gate(D, pk[0], pk[1], conds) for D, pk in packs]
    m = avg_metrics(ms)
    return m, (dpt(m, base, 'short', 'i5'), dpt(m, base, 'short', 'i7'),
               dpt(m, base, 'long', 'i5'), dpt(m, base, 'long', 'i7'),
               dpt(m, base, 'all', 'i5'), dpt(m, base, 'all', 'mIoU'))


def passes(dl):
    d5s, d7s, d5l, d7l, d5a, dma = dl
    return (d5s >= -0.2) and (d7s >= -0.3) and (d5l >= 2.0) and (d7l >= 5.0) \
        and (d5a >= 0.0) and (dma > 0.0)


print('\n' + '=' * 112, flush=True)
print('diag39  聚焦 kratio / margin  dataset=%s  calib=%s  test=%s  K_REP=%d' % (
    DS_NAME, CALIB_JSON, dargs['test_data'], K_REP), flush=True)
print('=' * 112, flush=True)

DC = collect_cached(CALIB_JSON, 'calib', reps=1)[0]
DT_list = collect_cached(dargs['test_data'], 'test', reps=K_REP)
bC = avg_metrics([base_metrics(DC)])
bT = avg_metrics([base_metrics(D) for D in DT_list])
print('  test base 全 R1@0.5 %.4f / R1@0.7 %.4f / mIoU %.4f' % (
    bT['all']['i5'], bT['all']['i7'], bT['all']['mIoU']), flush=True)
print('  短 R1@0.5 %.4f  长 R1@0.5 %.4f' % (bT['short']['i5'], bT['long']['i5']), flush=True)

packsC = [(DC, side_pack(DC))]
packsT = [(D, side_pack(D)) for D in DT_list]

# ---- [A/B/C] 规则生成 ----
SIG_SET = ('kratio', 'margin', 'wb', 'edgeclose', 'zoutmax', 'ext')
DIRS = {}
SLc, SRc, lLc, lRc = packsC[0][1]
SHc = DC['GL'] <= THR
for sig in SIG_SET:
    sc = np.concatenate([SLc[sig], SRc[sig]]); la = np.concatenate([lLc, lRc])
    aC = auc(sc, la)
    sct, lat = [], []
    for D, pk in packsT:
        sct.append(np.concatenate([pk[0][sig], pk[1][sig]]))
        lat.append(np.concatenate([pk[2], pk[3]]))
    aT = auc(np.concatenate(sct), np.concatenate(lat))
    DIRS[sig] = 1 if aC >= 0.5 else -1
    print('  信号 %-10s 校准AUC=%.3f 测试AUC=%.3f  方向=%s' % (
        sig, aC, aT, '越大越该扩' if DIRS[sig] > 0 else '越小越该扩'), flush=True)

thrs = {}
for sig in SIG_SET:
    vals = np.concatenate([SLc[sig], SRc[sig]])
    vals = vals[vals > -1e8]
    if DIRS[sig] > 0:
        thrs[sig] = [float(np.quantile(vals, 1.0 - q)) for q in QS]
    else:
        thrs[sig] = [float(np.quantile(vals, q)) for q in QS]

rules = []      # (tag, conds)
for sig in SIG_SET:
    for t in thrs[sig]:
        rules.append(('%s%s%.3f' % (sig, '>=' if DIRS[sig] > 0 else '<=', t),
                      [(sig, t, DIRS[sig])]))
pairs = []
for i, s1 in enumerate(SIG_SET):
    for s2 in SIG_SET[i + 1:]:
        for t1 in thrs[s1]:
            for t2 in thrs[s2]:
                pairs.append(('%s&%s' % (s1, s2),
                              [(s1, t1, DIRS[s1]), (s2, t2, DIRS[s2])]))
print('\n  单规则 %d 个 + 两信号AND %d 个' % (len(rules), len(pairs)), flush=True)

allr = rules + pairs
rowsC = []
for tag, conds in allr:
    m, dl = eval_rule(packsC, bC, conds)
    rowsC.append((tag, conds, dl))
print('  校准集评估完成，通过严格达标: %d / %d' % (sum(1 for r in rowsC if passes(r[2])), len(rowsC)), flush=True)

rowsT = []
for tag, conds in allr:
    m, dl = eval_rule(packsT, bT, conds)
    rowsT.append((tag, conds, dl))

npass = sum(1 for r in rowsT if passes(r[2]))
print('\n' + '=' * 112, flush=True)
print('【测试集】达标（短≥−0.2/≥−0.3, 长≥+2/≥+5, 全≥0, mIoU>0）: %d / %d' % (npass, len(rowsT)), flush=True)
print('%-30s | %7s %7s | %7s %7s | %7s %7s' % (
    '规则', 'd短@0.5', 'd短@0.7', 'd长@0.5', 'd长@0.7', 'd全@0.5', 'd全mIoU'), flush=True)
shown = 0
for tag, conds, dl in rowsT:
    if passes(dl) and shown < 25:
        shown += 1
        print('%-30s | %+7.2f %+7.2f | %+7.2f %+7.2f | %+7.2f %+7.2f  **达标**' % (
            tag[:30], dl[0], dl[1], dl[2], dl[3], dl[4], dl[5]), flush=True)
if npass == 0:
    print('  （无）', flush=True)

print('\n【前沿】测试集上"短不降"（d短@0.5≥−0.2）的规则，按 长 d@0.7 降序前 20', flush=True)
print('%-30s | %7s %7s | %7s %7s | %7s %7s | %s' % (
    '规则', 'd短@0.5', 'd短@0.7', 'd长@0.5', 'd长@0.7', 'd全@0.5', 'd全mIoU', '校准同规则'), flush=True)
dC = {r[0]: r[2] for r in rowsC}
sh = [r for r in rowsT if r[2][0] >= -0.2 and r[2][1] >= -0.3]
for tag, conds, dl in sorted(sh, key=lambda x: -x[2][3])[:20]:
    c = dC.get(tag)
    print('%-30s | %+7.2f %+7.2f | %+7.2f %+7.2f | %+7.2f %+7.2f | 短%+.2f 长%+.2f' % (
        tag[:30], dl[0], dl[1], dl[2], dl[3], dl[4], dl[5],
        c[0] if c else float('nan'), c[3] if c else float('nan')), flush=True)
print('  (满足"短不降"的规则共 %d / %d)' % (len([r for r in rowsT if r[2][0] >= -0.2 and r[2][1] >= -0.3]), len(rowsT)), flush=True)

print('\n【校准集选优 -> test 验证】', flush=True)
if any(passes(r[2]) for r in rowsC):
    sel = max([r for r in rowsC if passes(r[2])], key=lambda x: x[2][3])
    print('  严格解存在: %s' % sel[0], flush=True)
else:
    sel = max(rowsC, key=lambda x: x[2][3] + 3.0 * x[2][0])
    print('  校准集无严格解 -> Pareto 拐点 argmax(长@0.7 + 3*短@0.5): %s' % sel[0], flush=True)
dl = {r[0]: r[2] for r in rowsT}[sel[0]]
print('  test: 短 %+.2f/%+.2f | 长 %+.2f/%+.2f | 全 %+.2f | mIoU %+.2f  %s' % (
    dl[0], dl[1], dl[2], dl[3], dl[4], dl[5], '**达标**' if passes(dl) else ''), flush=True)
print('  calib: 短 %+.2f/%+.2f | 长 %+.2f/%+.2f | 全 %+.2f | mIoU %+.2f' % (
    sel[2][0], sel[2][1], sel[2][2], sel[2][3], sel[2][4], sel[2][5]), flush=True)

print('\n[基线绝对值] 短 R1@0.5 %.4f / R1@0.7 %.4f | 长 R1@0.5 %.4f / R1@0.7 %.4f | 全 %.4f / %.4f / mIoU %.4f' % (
    bT['short']['i5'], bT['short']['i7'], bT['long']['i5'], bT['long']['i7'],
    bT['all']['i5'], bT['all']['i7'], bT['all']['mIoU']), flush=True)
print('\n' + '=' * 112, flush=True)
print('DONE', flush=True)
