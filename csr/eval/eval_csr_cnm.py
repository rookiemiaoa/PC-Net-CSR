# -*- coding: utf-8 -*-
"""eval_csr_cnm.py —— CSR（中心重定位精修）在外部弱监督基线 CNM 上的原样移植

（历史文件名 csr_cnm.py。）

目的（审稿人必问项）
  CSR 是在 PC-Net 上校准出来的推理期算子。若它只对 PC-Net 有效，它就是一个针对
  特定权重的后处理技巧；若它在**完全独立的骨干**上也把 headline 指标整体推正，
  它就是一条与骨干无关的规律。本脚本用 CNM（AAAI-2022，与 PC-Net 同源）做这个检验。

为什么可以「零调参」移植
  CNM 与 PC-Net 的几何网格完全相同：两者 max_num_frames=200，且 CNM 的
  _sample_frame_features 会把特征固定重采样到 200 帧 ⇒ 逐点提案数 L=200//4=50，
  逐点归一化中心 CG=(arange(50)+0.5)/50。因此 CSR 的 5 个标量
  (SMOOTH=7, BETA=0.20, RHO=0.10, ALPHA=1.00, DMIN=0.01) 原样使用，一个都不改。

相关性曲线的来源（不改变 CNM 任何权重、不使用 gt）
  CNM.forward 在 eval 下提前 return (center,width)。它的训练路径第二段是
  「给定高斯掩码 → 掩码词重建 → nll」。这里把掩码换成 L 个尖峰（width=0.1），
  复刻该第二段，得到逐点 nll ⇒ r(t) = -nll(t)。这与 PC-Net 的「逐点提案」同构：
  两者都是「把候选时刻喂给重建器，看它把 query 重建得多好」。

指标口径
  完全采用 CNM 官方 runners/main_runner.py 的 calculate_IoU_batch2
  （inter/union 同为 +1e-10、union 宽 < -1e-5 置零、负值截零），因此本脚本的
  base 列应当复现 §6.8 的 38.43 / 37.31 / 15.44（Test-Trivial）。

用法
  python csr/eval/eval_csr_cnm.py <config.json> <ckpt.pt>

环境变量
  BS=32 K_REP=3 LIMIT=0 SMOOTH=7 BETA=0.20 RHO=0.10 ALPHA=1.00 DMIN=0.01
  THR=0.35 KS=0.15 GATE_RUN=1
"""
import math
import os
import pickle
import sys
import time

import numpy as np
import torch

# ---- 路径解析：CNM 克隆目录默认与 PC-Net 仓库同级（CNM-CG/），可用 CNM_ROOT 覆盖 ----
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = _HERE
while _REPO != os.path.dirname(_REPO) and not os.path.exists(os.path.join(_REPO, 'models', 'pcnet.py')):
    _REPO = os.path.dirname(_REPO)
sys.path.insert(0, _REPO)

from csr.paths import CNM_ROOT             # noqa: E402
from csr import ops                        # noqa: E402  （单一权威算子实现）

ROOT = CNM_ROOT
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import models                          # noqa: E402
from models.cnm import _generate_mask  # noqa: E402
from models.loss import cal_nll_loss   # noqa: E402
from utils import load_json            # noqa: E402
import datasets as da                  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

# CNM 在 CG 协议下的官方（自报）best 结果，用于复现校验（仅 CharadesSTA）
REF = {
    'CharadesSTA': dict(mIoU=0.3843, r1_05=0.3731, r1_07=0.1544),
    'ActivityNet': dict(mIoU=0.3237, r1_05=0.0, r1_07=0.0),
}

cfg = load_json(sys.argv[1])
ckpt = sys.argv[2]
dargs = cfg['dataset']
vocab = pickle.load(open(dargs['vocab_path'], 'rb'))
ds = getattr(da, dargs['dataset'])(data_path=dargs['test_data'], vocab=vocab, args=dargs)
VOCAB = ds.vocab_size
COLLATE = ds.collate_data
DS_NAME = dargs['dataset']

sd = torch.load(ckpt, map_location='cpu')
sp = sd['model_parameters']
mcfg = dict(sd.get('config', {}).get('model', {}).get('config', cfg['model']['config']))
mcfg['max_num_words'] = dargs['max_num_words']
mcfg['frames_input_size'] = dargs['frame_dim']
mcfg['words_input_size'] = dargs['word_dim']
mcfg['vocab_size'] = int(sp['fc_rec.weight'].shape[0])
assert mcfg['vocab_size'] == VOCAB, 'vocab_size 不一致: ckpt=%d dataset=%d' % (
    mcfg['vocab_size'], VOCAB)

L = dargs['max_num_frames'] // 4
CG = (np.arange(L) + 0.5) / L
GRID = np.linspace(0.0, 1.0, L)

BS = int(os.environ.get('BS', 32))
K_REP = int(os.environ.get('K_REP', 3))
LIMIT = int(os.environ.get('LIMIT', 0))
SMOOTH = int(os.environ.get('SMOOTH', 7))
BETA = float(os.environ.get('BETA', 0.20))
RHO = float(os.environ.get('RHO', 0.10))
ALPHA = float(os.environ.get('ALPHA', 1.00))
DMIN = float(os.environ.get('DMIN', 0.01))
THR = float(os.environ.get('THR', 0.35))
KS = [float(x) for x in os.environ.get('KS', '0.15').split(',')]
GATE_RUN = int(os.environ.get('GATE_RUN', 1))
WPT = float(os.environ.get('WPT', 0.1))   # 逐点提案的高斯宽度（CNM 官方 clamp 下限为 0.1）
SPLIT_NAME = os.path.basename(dargs['test_data']).replace('.json', '')
IS_MAIN = 'trivial' in SPLIT_NAME

if LIMIT > 0:
    ds = Subset(ds, list(range(min(LIMIT, len(ds)))))
N = len(ds)
loader = DataLoader(ds, batch_size=BS, shuffle=False, collate_fn=COLLATE, num_workers=4)

line = '=' * 108
print('csr_cnm  cfg=%s' % sys.argv[1], flush=True)
print('ckpt=%s' % ckpt, flush=True)
print('dataset=%s  split=%s  N=%d  L=%d  BS=%d  K_REP=%d  alpha=%.2f  max_width=%.2f' % (
    DS_NAME, SPLIT_NAME, N, L, BS, K_REP, mcfg.get('alpha', -1), mcfg.get('max_width', -1)),
      flush=True)
print('CSR: SMOOTH=%d BETA=%.2f RHO=%.2f ALPHA=%.2f DMIN=%.2f  (零调参，与 PC-Net 完全相同)' % (
    SMOOTH, BETA, RHO, ALPHA, DMIN), flush=True)
print('point-mask width WPT=%.2f' % WPT, flush=True)
print(line, flush=True)


# ---------------- IoU / 指标（CNM 官方口径） ----------------
def iou_cnm(p0, p1, g0, g1):
    """复刻 runners/main_runner.py::calculate_IoU_batch2"""
    inter0 = np.maximum(p0, g0)
    inter1 = np.minimum(p1, g1)
    union0 = np.minimum(p0, g0)
    union1 = np.maximum(p1, g1)
    iou = 1.0 * (inter1 - inter0 + 1e-10) / (union1 - union0 + 1e-10)
    iou[(union1 - union0) < -1e-5] = 0
    iou[iou < 0] = 0.0
    return iou


def metrics(i):
    d = {'mIoU': float(i.mean())}
    for t in (0.5, 0.7):
        d['iou%.1f' % t] = float((i >= t).mean())
    return d


# ---------------- CSR 算子（统一实现于 csr/ops.py，此处仅做薄封装） ----------------
def smoothZ(Z, w):
    return ops.smooth_curve(Z, w)


def inside_score(Z, lo, hi):
    return ops.inside_score(Z, lo, hi, CG)


def est_win_vec(Z, cb, beta):
    return ops.estimate_window_center(Z, cb, beta, CG)


def flood(REL, cb, k):
    return ops.flood_expand(REL, cb, k, GRID)


def csr_boxes(Z_raw, cb, wb, dmin):
    """CSR：保宽的中心重定位。dmin=None 关闭自洽门控（消融用）。"""
    lo, hi, _, shift = ops.refine_center(
        Z_raw, cb, wb, CG, smooth=SMOOTH, beta=BETA, rho=RHO, alpha=ALPHA, dmin=dmin)
    return lo, hi, shift


# ---------------- 相关性曲线（复刻 CNM 训练路径第二段） ----------------
@torch.no_grad()
def rel_curve(model, ni):
    ff = ni['frames_feat']
    bsz, n_frames, _ = ff.shape
    ff2 = model.frame_fc(ff)
    frames_mask = _generate_mask(ff2, ni['frames_len'])
    wf = ni['words_feat'].clone()
    wf[:, 0] = model.start_vec
    wpos = model.word_pos_encoder(wf)
    wf = model.word_fc(wf)
    wmask = _generate_mask(wf, ni['words_len'] + 1)
    model.trans(ff2, frames_mask, wf + wpos, wmask, decoding=1)   # 与 forward 同
    pl = n_frames // 4
    keep = torch.linspace(0, n_frames - 1, steps=pl).long()
    pf = ff2[:, keep]
    pm = frames_mask[:, keep]
    wf1, _ = model._mask_words(wf, ni['words_len'], weights=ni['weights'])
    wf1 = wf1 + wpos
    wf1 = wf1[:, :-1]
    wm1 = wmask[:, :-1]
    x = torch.linspace(0.0, 1.0, pl, device=ff.device)
    t = torch.linspace(0.5 / pl, 1 - 0.5 / pl, pl, device=ff.device)
    wpt = torch.full_like(t, WPT)
    gw = torch.exp(-model.alpha * (x[None, :] - t[:, None]).pow(2) / wpt.pow(2)[:, None])
    gw = gw.unsqueeze(0).expand(bsz, -1, -1).reshape(bsz * pl, pl)
    pfe = pf.unsqueeze(1).expand(bsz, pl, pl, -1).reshape(bsz * pl, pl, -1)
    pme = pm.unsqueeze(1).expand(bsz, pl, pl).reshape(bsz * pl, pl)
    wfe = wf1.unsqueeze(1).expand(bsz, pl, wf1.size(1), -1).reshape(bsz * pl, wf1.size(1), -1)
    wme = wm1.unsqueeze(1).expand(bsz, pl, wm1.size(1)).reshape(bsz * pl, wm1.size(1))
    wde = ni['words_id'].unsqueeze(1).expand(bsz, pl, ni['words_id'].size(1)) \
        .reshape(bsz * pl, ni['words_id'].size(1))
    _, h2 = model.trans(pfe, pme, wfe, wme, decoding=2, gauss_weight=gw)
    logit = model.fc_rec(h2)
    nll, _ = cal_nll_loss(logit, wde, wme)
    return (-nll.view(bsz, pl)).float().cpu().numpy()


ACC = {}
SPLIT = {}
TAGS = ['base'] + ['RBR k=%.2f' % k for k in KS] + ['CSR'] \
    + (['CSR_ng'] if GATE_RUN else []) + ['ORACLE_c', 'ORACLE_w']


def add(tag, i, gl):
    ACC.setdefault(tag, []).append(metrics(i))
    SPLIT.setdefault(tag, []).append((i, gl))


def run_rep(rep):
    np.random.seed(11 + rep)
    torch.manual_seed(11 + rep)
    model = getattr(models, cfg['model']['name'])(mcfg).cuda()
    model.load_state_dict(sp)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    II = {t: [] for t in TAGS}
    GL = []
    DIAG = dict(cb=[], wb=[], shift=[], moved=[])
    t0 = time.time()
    with torch.no_grad():
        for bi, batch in enumerate(loader, 1):
            ni = {k: v.cuda() for k, v in batch['net_input'].items()}
            ni['words_feat'] = ni['words_feat'].clone()
            out = model(**ni)
            cb = out['center'].view(-1).float().cpu().numpy()
            wb = out['width'].view(-1).float().cpu().numpy()
            REL = rel_curve(model, ni)
            dur = np.asarray([r[1] for r in batch['raw']], dtype=np.float64)
            ts = np.asarray([r[2] for r in batch['raw']], dtype=np.float64)
            g = ts / dur[:, None]
            g0, g1 = g[:, 0], g[:, 1]
            gl = g1 - g0

            bL = np.clip(cb - wb / 2, 0, 1)
            bH = np.clip(cb + wb / 2, 0, 1)
            II['base'].append(iou_cnm(bL, bH, g0, g1))
            for k in KS:
                s, e = flood(REL, cb, k)
                II['RBR k=%.2f' % k].append(
                    iou_cnm(np.minimum(s, bL), np.maximum(e, bH), g0, g1))
            lo, hi, sh = csr_boxes(REL, cb, wb, DMIN)
            II['CSR'].append(iou_cnm(lo, hi, g0, g1))
            if GATE_RUN:
                lo2, hi2, _ = csr_boxes(REL, cb, wb, None)
                II['CSR_ng'].append(iou_cnm(lo2, hi2, g0, g1))
            gc = (g0 + g1) / 2.0
            oc0 = np.clip(gc - wb / 2, 0, 1)
            oc1 = np.clip(gc + wb / 2, 0, 1)
            II['ORACLE_c'].append(iou_cnm(oc0, oc1, g0, g1))
            ow0 = np.clip(cb - gl / 2, 0, 1)
            ow1 = np.clip(cb + gl / 2, 0, 1)
            II['ORACLE_w'].append(iou_cnm(ow0, ow1, g0, g1))
            GL.append(gl)
            DIAG['cb'].append(cb)
            DIAG['wb'].append(wb)
            DIAG['shift'].append(sh)
            DIAG['moved'].append(sh > 1e-9)
            if bi % 25 == 0:
                print('    rep%d  batch %d/%d  %.1fs' % (rep, bi, len(loader), time.time() - t0),
                      flush=True)

    n_batch = len(GL)
    for t in TAGS:
        assert len(II[t]) == n_batch, '%s 批次数不符: %d != %d' % (t, len(II[t]), n_batch)
    GL = np.concatenate(GL)
    for t in TAGS:
        add(t, np.concatenate(II[t]), GL)
    d = {k: np.concatenate(v) for k, v in DIAG.items()}
    d['GL'] = GL
    d['i_base'] = np.concatenate(II['base'])
    d['i_csr'] = np.concatenate(II['CSR'])
    return d


diag = None
for rep in range(K_REP):
    dd = run_rep(rep)
    if diag is None:
        diag = dd
    print('  rep %d done | base mIoU=%.4f R1@0.5=%.4f R1@0.7=%.4f | CSR mIoU=%.4f R1@0.5=%.4f R1@0.7=%.4f' % (
        rep, ACC['base'][-1]['mIoU'], ACC['base'][-1]['iou0.5'], ACC['base'][-1]['iou0.7'],
        ACC['CSR'][-1]['mIoU'], ACC['CSR'][-1]['iou0.5'], ACC['CSR'][-1]['iou0.7']), flush=True)


def mean_of(tag, key):
    return float(np.mean([m[key] for m in ACC[tag]]))


def std_of(tag, key):
    return float(np.std([m[key] for m in ACC[tag]]))


print('\n' + line, flush=True)
print('csr_cnm  %s  split=%s  N=%d  K_REP=%d  ckpt=%s' % (
    DS_NAME, SPLIT_NAME, diag['GL'].shape[0], K_REP, os.path.basename(ckpt)), flush=True)
print(line, flush=True)

if IS_MAIN and LIMIT == 0:
    ref = REF[DS_NAME]
    print('[复现校验] base vs CNM 官方自报 Best results（仅 test_trivial 切分可比）', flush=True)
    for nm, key, v in (('R1,mIoU', 'mIoU', ref['mIoU']), ('R1@0.5', 'iou0.5', ref['r1_05']),
                       ('R1@0.7', 'iou0.7', ref['r1_07'])):
        got = mean_of('base', key)
        flag = 'OK ' if abs(got - v) <= 0.02 else 'DIFF'
        print('  %-4s %-9s 官方=%.4f  本脚本=%.4f  (%+.4f)' % (flag, nm, v, got, got - v), flush=True)
    print(line, flush=True)

print('[主表] mean±std over %d rep' % K_REP, flush=True)
print('%-16s %16s %16s %16s' % ('', 'R1@0.5', 'R1@0.7', 'mIoU'), flush=True)
for tag in TAGS:
    print('%-14s %8.4f±%.4f %8.4f±%.4f %8.4f±%.4f' % (
        tag, mean_of(tag, 'iou0.5'), std_of(tag, 'iou0.5'),
        mean_of(tag, 'iou0.7'), std_of(tag, 'iou0.7'),
        mean_of(tag, 'mIoU'), std_of(tag, 'mIoU')), flush=True)
print('-' * 108, flush=True)
print('%-16s %16s %16s %16s' % ('Δ vs base', 'ΔR1@0.5', 'ΔR1@0.7', 'ΔmIoU'), flush=True)
for tag in TAGS[1:]:
    print('%-14s %+16.4f %+16.4f %+16.4f' % (
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
        print('%-14s %-6s %6d %9.4f %9.4f %9.4f' % (
            tag, nm, int(m.sum()), ib[m].mean(), (ib[m] >= 0.5).mean(),
            (ib[m] >= 0.7).mean()), flush=True)
print('-' * 108, flush=True)
print('%-16s %-6s %6s %9s %9s %9s' % ('Δ (CSR-base)', '桶', 'n', 'ΔmIoU', 'ΔR1@0.5', 'ΔR1@0.7'),
      flush=True)
ib = np.concatenate([v[0] for v in SPLIT['base']])
gb = np.concatenate([v[1] for v in SPLIT['base']])
ic = np.concatenate([v[0] for v in SPLIT['CSR']])
for nm, m in (('短', gb <= THR), ('长', gb > THR), ('全', np.ones(len(gb), bool))):
    print('%-14s %-6s %6d %+9.4f %+9.4f %+9.4f' % (
        'CSR-base', nm, int(m.sum()),
        100 * (ic[m].mean() - ib[m].mean()),
        100 * ((ic[m] >= 0.5).mean() - (ib[m] >= 0.5).mean()),
        100 * ((ic[m] >= 0.7).mean() - (ib[m] >= 0.7).mean())), flush=True)

print('\n' + line, flush=True)
print('[显著性] 配对 bootstrap B=2000（95%% CI）+ McNemar', flush=True)
rng = np.random.default_rng(0)
B = 2000
for nm, m in (('short', diag['GL'] <= THR), ('long', diag['GL'] > THR),
              ('all', np.ones(len(diag['GL']), bool))):
    a, b = diag['i_base'][m], diag['i_csr'][m]
    n = len(a)
    idx = rng.integers(0, n, size=(B, n))
    dm = b[idx].mean(1) - a[idx].mean(1)
    d5 = (b[idx] >= 0.5).mean(1) - (a[idx] >= 0.5).mean(1)
    d7 = (b[idx] >= 0.7).mean(1) - (a[idx] >= 0.7).mean(1)
    b5 = int(((a < 0.5) & (b >= 0.5)).sum())
    c5 = int(((a >= 0.5) & (b < 0.5)).sum())
    b7 = int(((a < 0.7) & (b >= 0.7)).sum())
    c7 = int(((a >= 0.7) & (b < 0.7)).sum())
    print('  %-5s n=%5d  ΔmIoU %+6.3f [%+6.3f,%+6.3f] | Δ@0.5 %+6.3f [%+6.3f,%+6.3f] | Δ@0.7 %+6.3f [%+6.3f,%+6.3f]' % (
        nm, n, 100 * dm.mean(), 100 * np.percentile(dm, 2.5), 100 * np.percentile(dm, 97.5),
        100 * d5.mean(), 100 * np.percentile(d5, 2.5), 100 * np.percentile(d5, 97.5),
        100 * d7.mean(), 100 * np.percentile(d7, 2.5), 100 * np.percentile(d7, 97.5)), flush=True)
    for tn, bb, cc in (('@0.5', b5, c5), ('@0.7', b7, c7)):
        stat = ((abs(bb - cc) - 1) ** 2 / float(bb + cc)) if (bb + cc) > 0 else 0.0
        p = math.erfc(math.sqrt(stat / 2.0))
        print('         McNemar %s: 升(%d) 降(%d)  chi2=%.2f  p=%.2e' % (tn, bb, cc, stat, p),
              flush=True)

print('\n' + line, flush=True)
print('[诊断] 平移行为 + 两轴图落点', flush=True)
gb = diag['GL']
for nm, m in (('短', gb <= THR), ('长', gb > THR), ('全', np.ones(len(gb), bool))):
    print('  %-4s n=%5d  移动率=%.3f  平均|Δ|(移动样本)=%.4f  平均|Δ|(全体)=%.4f' % (
        nm, int(m.sum()), diag['moved'][m].mean(),
        diag['shift'][m & diag['moved']].mean() if (m & diag['moved']).any() else 0.0,
        diag['shift'][m].mean()), flush=True)
wb_med = float(np.median(diag['wb']))
gl_med = float(np.median(diag['GL']))
print('  基线框宽中位=%.4f  gt 时长中位=%.4f  比值=%.2f' % (wb_med, gl_med, wb_med / max(gl_med, 1e-9)),
      flush=True)
print('  参考 PC-Net: 框宽中位 0.351 / gt 0.261 / 比值 1.35；ActivityNet 0.991 / 0.275 / 3.60', flush=True)
print(line, flush=True)
print('DONE', flush=True)
