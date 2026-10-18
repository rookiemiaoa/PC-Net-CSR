# -*- coding: utf-8 -*-
"""w36_cnm_curve.py —— 表 3 CNM 行的唯一采集器（w34 的 CNM 版）

为什么必须单独一个脚本
  ``w34_curve_usability.py`` 依赖 PC-Net 的 forward 输出 ``center/width/words_logit``，
  以及 ``install_point_mask`` 对 ``fc_gauss`` / ``proposal_generator`` 打钩子。CNM 的
  forward 在 eval 下直接 ``return (center, width)``，既没有高斯提案头，也不返回
  ``words_logit``，因此 w34 在 CNM 上不可用（会 ``RuntimeError: 未找到任何高斯参数挂钩``）。

  本脚本复刻 ``csr/eval/eval_csr_cnm.py`` 的曲线读取路径（L 个点掩码 → 掩码词重建
  nll），但输出与 w34 **严格同构**的 JSON（``rows`` / ``curve`` / ``rho_geo``），使
  表 3 的五行由同一个渲染器 ``scripts/render_table3.py`` 消费。

口径一致性（三条硬约束）
  1. IoU 复刻 CNM 官方 ``runners/main_runner.py::calculate_IoU_batch2``；
  2. CSR 五步走 ``csr/ops.py`` 的单一实现，参数与 PC-Net 完全相同（零调参）；
  3. ``dir_acc`` 的「确有偏移」阈值 ``DDIR=0.08``，与 w34 相同。

用法
  python w36_cnm_curve.py <cnm_root> <config.json> <ckpt.pt> <tag>

环境变量
  BS=32 K_REP=3 LIMIT=0 WPT=0.1 OUT=...
"""
import json
import math
import os
import pickle
import sys
import time

import numpy as np
import torch

# ---- 1) 定位 PC-Net 仓库（为了 import csr.ops）----
_HERE = os.path.dirname(os.path.abspath(__file__))
_PCNET = _HERE
while _PCNET != os.path.dirname(_PCNET) and not os.path.exists(
        os.path.join(_PCNET, 'models', 'pcnet.py')):
    _PCNET = os.path.dirname(_PCNET)
sys.path.insert(0, _PCNET)

from csr import ops                        # noqa: E402  单一权威算子实现

# ---- 2) 定位 CNM 仓库（为了 import CNM 的 models / datasets）----
ROOT = os.environ.get('CNM_ROOT') or (sys.argv[1] if len(sys.argv) > 1 else '')
if not os.path.isdir(ROOT):
    raise SystemExit('CNM 根目录不存在: %r（用 CNM_ROOT 或第一个参数指定）' % ROOT)
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import models                          # noqa: E402  (CNM 仓库的 models)
from models.cnm import _generate_mask  # noqa: E402
from models.loss import cal_nll_loss   # noqa: E402
from utils import load_json            # noqa: E402
import datasets as da                  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

cfg = load_json(sys.argv[2])
ckpt = sys.argv[3]
TAG = sys.argv[4] if len(sys.argv) > 4 else 'cnm'

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
DDIR = float(os.environ.get('DDIR', 0.08))
WPT = float(os.environ.get('WPT', 0.1))
OUT = os.environ.get('OUT') or os.path.join(
    _PCNET, 'results', 'curve_availability', 'w36_%s.json' % TAG)

SPLIT_NAME = os.path.basename(dargs['test_data'])
if LIMIT > 0:
    ds = Subset(ds, list(range(min(LIMIT, len(ds)))))
N = len(ds)
loader = DataLoader(ds, batch_size=BS, shuffle=False, collate_fn=COLLATE, num_workers=4)

line = '=' * 108
print('w36_cnm  cfg=%s' % sys.argv[2], flush=True)
print('ckpt=%s' % ckpt, flush=True)
print('root=%s' % ROOT, flush=True)
print('dataset=%s  split=%s  N=%d  L=%d  BS=%d  K_REP=%d  WPT=%.2f' % (
    DS_NAME, SPLIT_NAME, N, L, BS, K_REP, WPT), flush=True)
print('CSR: SMOOTH=%d BETA=%.2f RHO=%.2f ALPHA=%.2f DMIN=%.2f  (零调参，与 PC-Net 完全相同)'
      % (SMOOTH, BETA, RHO, ALPHA, DMIN), flush=True)
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


def auc(y, s):
    """w34 同款：秩统计 AUC，用于 Δ_inside 的判别力。"""
    y = np.asarray(y).astype(bool)
    s = np.asarray(s, dtype=np.float64)
    if y.all() or (~y).all():
        return float('nan')
    order = np.argsort(s)
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    # 处理并列：取平均秩
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    if (cnt > 1).any():
        sumr = np.zeros(len(cnt))
        np.add.at(sumr, inv, ranks)
        ranks = (sumr / cnt)[inv]
    n1 = int(y.sum())
    n0 = len(y) - n1
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


# ---------------- 相关性曲线：L 个点掩码 + 掩码词重建 nll ----------------
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
    model.trans(ff2, frames_mask, wf + wpos, wmask, decoding=1)
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


# ---------------- 单次重复 ----------------
C_KEYS = ('base', 'CSR', 'ORACLE_c', 'ORACLE_w', 'ORACLE_box')
D_KEYS = ('cb', 'wb', 'c1', 'cstar', 'cgt', 'din', 'gl')


def run_rep(rep):
    np.random.seed(11 + rep)
    torch.manual_seed(11 + rep)
    model = getattr(models, cfg['model']['name'])(mcfg).cuda()
    model.load_state_dict(sp)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    C = {k: [] for k in C_KEYS}
    D = {k: [] for k in D_KEYS}
    t0 = time.time()
    with torch.no_grad():
        for bi, batch in enumerate(loader, 1):
            ni = {k: v.cuda() for k, v in batch['net_input'].items()}
            ni['words_feat'] = ni['words_feat'].clone()
            out = model(**ni)
            cb = out['center'].view(-1).float().cpu().numpy()
            wb = out['width'].view(-1).float().cpu().numpy()

            REL = rel_curve(model, ni)

            dur = np.asarray([it[1] for it in batch['raw']], dtype=np.float64)
            ts = np.asarray([it[2] for it in batch['raw']], dtype=np.float64)
            g = ts / dur[:, None]
            g0, g1 = g[:, 0], g[:, 1]
            cg = (g0 + g1) / 2.0
            gw = g1 - g0

            # CSR 五步（与 PC-Net 完全相同的参数）
            c2, moved, shift = ops.refine_center_c2(
                REL, cb, wb, CG, smooth=SMOOTH, beta=BETA, rho=RHO,
                alpha=ALPHA, dmin=DMIN)
            # c1：平滑后的窗口质心（refine_center_c2 内部同款，单独取出用于 dir_acc）
            zs = ops.smooth_curve(ops.zscore(REL), SMOOTH)
            c1 = ops.estimate_window_center(zs, cb, BETA, CG)
            # Δ_inside：门控所用的自洽信号
            din = ops.inside_score(zs, c2 - wb / 2, c2 + wb / 2, CG) \
                - ops.inside_score(zs, cb - wb / 2, cb + wb / 2, CG)

            bL = np.clip(cb - wb / 2, 0, 1)
            bH = np.clip(cb + wb / 2, 0, 1)
            C['base'].append(iou_cnm(bL, bH, g0, g1))
            sL = np.clip(c2 - wb / 2, 0, 1)
            sH = np.clip(c2 + wb / 2, 0, 1)
            C['CSR'].append(iou_cnm(sL, sH, g0, g1))
            C['ORACLE_c'].append(iou_cnm(
                np.clip(cg - wb / 2, 0, 1), np.clip(cg + wb / 2, 0, 1), g0, g1))
            C['ORACLE_w'].append(iou_cnm(
                np.clip(cb - gw / 2, 0, 1), np.clip(cb + gw / 2, 0, 1), g0, g1))
            C['ORACLE_box'].append(iou_cnm(g0, g1, g0, g1))

            D['cb'].append(cb); D['wb'].append(wb)
            D['c1'].append(c1); D['cstar'].append(c2)
            D['cgt'].append(cg); D['din'].append(din); D['gl'].append(gw)
            if bi % 25 == 0:
                print('    rep%d  batch %d/%d  %.1fs' % (rep, bi, len(loader),
                                                          time.time() - t0), flush=True)

    return ({k: np.concatenate(v) for k, v in C.items()},
            {k: np.concatenate(v) for k, v in D.items()})


RES, DAT = [], None
for rep in range(K_REP):
    r_, d_ = run_rep(rep)
    RES.append(r_)
    DAT = d_
    print('  rep%d | base mIoU=%.4f | CSR %.4f (%+.2f) | ORACLE_c %.4f (%+.2f)' % (
        rep, 100 * r_['base'].mean(), 100 * r_['CSR'].mean(),
        100 * (r_['CSR'].mean() - r_['base'].mean()),
        100 * r_['ORACLE_c'].mean(),
        100 * (r_['ORACLE_c'].mean() - r_['base'].mean())), flush=True)

J = {'tag': TAG, 'dataset': DS_NAME, 'split': SPLIT_NAME,
     'N': int(len(DAT['gl'])), 'L': L, 'K_REP': K_REP, 'WPT': WPT, 'ckpt': ckpt,
     'rho_geo': float(np.median(DAT['wb']) / max(np.median(DAT['gl']), 1e-9)),
     'med_wb': float(np.median(DAT['wb'])), 'med_gl': float(np.median(DAT['gl']))}

print('\n' + '=' * 100, flush=True)
print('[算子表] %s  mean over %d rep' % (TAG, K_REP), flush=True)
b = {k: float(np.mean([metrics(r_['base'])[k] for r_ in RES])) for k in ('mIoU', 'iou0.5', 'iou0.7')}
J['rows'] = {}
for k in C_KEYS:
    ms = {kk: float(np.mean([metrics(r_[k])[kk] for r_ in RES]))
          for kk in ('mIoU', 'iou0.5', 'iou0.7')}
    print('%-12s mIoU %7.2f (%+6.2f) | R1@0.5 %7.2f (%+6.2f) | R1@0.7 %7.2f (%+6.2f)' % (
        k, 100 * ms['mIoU'], 100 * (ms['mIoU'] - b['mIoU']),
        100 * ms['iou0.5'], 100 * (ms['iou0.5'] - b['iou0.5']),
        100 * ms['iou0.7'], 100 * (ms['iou0.7'] - b['iou0.7'])), flush=True)
    J['rows'][k] = dict(mIoU=100 * ms['mIoU'], r5=100 * ms['iou0.5'], r7=100 * ms['iou0.7'],
                        dm=100 * (ms['mIoU'] - b['mIoU']),
                        d5=100 * (ms['iou0.5'] - b['iou0.5']),
                        d7=100 * (ms['iou0.7'] - b['iou0.7']))

# ---------------- 曲线可用性 ----------------
cb_, wb_ = DAT['cb'], DAT['wb']
c1_, cs_, cg_ = DAT['c1'], DAT['cstar'], DAT['cgt']
e_base = np.abs(cb_ - cg_)
e_c1 = np.abs(c1_ - cg_)
e_cs = np.abs(cs_ - cg_)
dirn = np.abs(cb_ - cg_) >= DDIR

dir_acc_1 = float(np.mean(np.sign(c1_ - cb_)[dirn] == np.sign(cg_ - cb_)[dirn])) \
    if dirn.sum() else float('nan')
dir_acc_s = float(np.mean(np.sign(cs_ - cb_)[dirn] == np.sign(cg_ - cb_)[dirn])) \
    if dirn.sum() else float('nan')
mv = (cs_ != cb_)
closer = float(np.mean(e_cs[mv] < e_base[mv])) if mv.sum() else float('nan')
imp = np.concatenate([r_['CSR'] > r_['base'] for r_ in RES])
J['curve'] = dict(
    mae_base=float(e_base.mean()), mae_c1=float(e_c1.mean()), mae_final=float(e_cs.mean()),
    dmae_c1=float(e_c1.mean() - e_base.mean()),
    dmae_final=float(e_cs.mean() - e_base.mean()),
    dir_acc_c1=dir_acc_1, dir_acc_final=dir_acc_s, move_rate=float(mv.mean()),
    closer_when_moved=closer, n_offset=int(dirn.sum()),
    auc_din=float(auc(imp, np.tile(DAT['din'], K_REP))), frac_improved=float(imp.mean()),
)
print('\n[曲线可用性] %s' % TAG, flush=True)
print('  中心 MAE: 基础 %.4f -> 曲线质心 c1 %.4f (%+.4f) -> CSR 输出 %.4f (%+.4f)' % (
    e_base.mean(), e_c1.mean(), e_c1.mean() - e_base.mean(),
    e_cs.mean(), e_cs.mean() - e_base.mean()), flush=True)
print('  方向准确率 (|Δc_b|>=%.2f, n=%d): c1 %.4f | 最终 %.4f' % (
    DDIR, dirn.sum(), dir_acc_1, dir_acc_s), flush=True)
print('  移动率 %.4f | 移动后更接近真值中心 %.4f | Δ_inside AUC %.4f | 改善样本占比 %.4f' % (
    mv.mean(), closer, J['curve']['auc_din'], imp.mean()), flush=True)
print('  几何: 框宽中位 %.4f / gt 中位 %.4f / rho_geo %.4f' % (
    J['med_wb'], J['med_gl'], J['rho_geo']), flush=True)

with open(OUT, 'w') as f:
    json.dump(J, f, ensure_ascii=False, indent=1)
print('\nWROTE %s' % OUT, flush=True)
print('DONE', flush=True)
