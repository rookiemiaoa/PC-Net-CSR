# -*- coding: utf-8 -*-
"""w34_curve_usability.py —— 为什么同一个 CSR 算子在某些骨干上正向、在另一些上负向？

动机（来自 w32 的结果）
  给四个弱监督基线加上同一个 CSR 后处理，结果分裂：
      PC-Net  47.20 -> 48.33  (+1.13 SIG)
      CPL     46.47 -> 42.71  (-3.76 SIG)
      CCR     46.32 -> 43.14  (-3.18 SIG)
      QMN     46.32 -> 42.78  (-3.53 SIG)
  而三者的几何比值 rho_geo 都在 1.33-1.39，即表 3「宽度相当、中心轴可用」的区间内。
  => 单靠 rho_geo 不能预测这一分裂。需要第二个判据。

本脚本测量第二个判据的两个候选：
  (A) 曲线可用性：把该骨干自身的逐位置相关性曲线交给同一个窗内质心估计器，
      它给出的中心 c1 是否比基础中心 c_b 更接近真值中心？MAE 降不降、方向准确率多少。
      —— 由 w32 的 install_point_mask 一次性读出（L=50 个点掩码）。
  (B) 中心可改进空间：ORACLE_c（真值中心 + 基础框宽）相对 base 的 mIoU 增量。
      —— 不需要前向，纯粹的几何量，代表"中心轴还剩多少油水"。

输出 JSON 供论文表/图与文字使用。不修改任何基线源码。

用法: python w34_curve_usability.py <repo_root> <config> <ckpt> <tag>
环境变量: BS=16 K_REP=3 LIMIT=0 WPT=0.1 OUT=...
"""
import json
import math
import os
import pickle
import sys
import time

import numpy as np
import torch
import torch.nn as nn

REPO = sys.argv[1]
TAG = sys.argv[4] if len(sys.argv) > 4 else 'baseline'
sys.path.insert(0, REPO)
os.chdir(REPO)

import models                          # noqa: E402
from utils import load_json            # noqa: E402
import datasets as da                  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402


def find_loss():
    try:
        from models.loss import cal_nll_loss
        return cal_nll_loss
    except Exception:
        pass
    for cand in ('models.loss', 'model.loss', 'loss'):
        try:
            m = __import__(cand, fromlist=['cal_nll_loss'])
            return getattr(m, 'cal_nll_loss')
        except Exception:
            continue
    raise ImportError('cal_nll_loss 未找到')


cal_nll_loss = find_loss()

cfg = load_json(sys.argv[2]); ckpt = sys.argv[3]
dargs = cfg['dataset']; DS_NAME = dargs['dataset']
if os.environ.get('TEST_DATA'):
    dargs['test_data'] = os.environ['TEST_DATA']
vocab = pickle.load(open(dargs['vocab_path'], 'rb'))
try:
    ds = getattr(da, DS_NAME)(data_path=dargs['test_data'], vocab=vocab, args=dargs, split='test')
except TypeError:
    ds = getattr(da, DS_NAME)(data_path=dargs['test_data'], vocab=vocab, args=dargs)
VOCAB = ds.vocab_size
COLLATE = ds.collate_data
LIMIT = int(os.environ.get('LIMIT', 0))
if LIMIT > 0:
    ds = Subset(ds, list(range(min(LIMIT, len(ds)))))

sd = torch.load(ckpt, map_location='cpu')
sp = sd['model_parameters']
mcfg = dict(sd.get('config', {}).get('model', {}).get('config', cfg['model']['config']))
for k in ('max_num_words', 'frames_input_size', 'words_input_size'):
    if k in dargs:
        mcfg[k] = dargs[k]
mcfg.setdefault('frame_feat_dim', dargs.get('frame_dim', mcfg.get('frame_feat_dim')))
mcfg.setdefault('word_feat_dim', dargs.get('word_dim', mcfg.get('word_feat_dim')))
mcfg['vocab_size'] = VOCAB
mcfg['max_epoch'] = cfg['train']['max_num_epochs']

L = dargs['max_num_frames'] // 4
CG = (np.arange(L) + 0.5) / L
BS = int(os.environ.get('BS', 16)); K_REP = int(os.environ.get('K_REP', 3))
WPT = float(os.environ.get('WPT', 0.1))
THR = float(os.environ.get('THR', 0.35))
DDIR = 0.08                                  # 方向准确率的「确有偏移」判定阈值（同表 10）
OUT = os.environ.get('OUT') or os.path.join(REPO, 'results', 'curve_availability', 'w34_%s.json' % TAG)
loader = DataLoader(ds, batch_size=BS, shuffle=False, collate_fn=COLLATE, num_workers=4)

print('[GPU] %s' % torch.cuda.get_device_name(0), flush=True)
print('[BASE] %s  model=%s  split=%s  N=%d  L=%d  BS=%d  K_REP=%d  WPT=%.2f' % (
    TAG, cfg['model']['name'], os.path.basename(dargs['test_data']), len(ds), L, BS, K_REP, WPT), flush=True)


# ------------------------------- 算子（与 w30/w32 逐行一致） -------------------------------
def iou(a0, a1, b0, b1):
    a0, a1, b0, b1 = map(np.asarray, (a0, a1, b0, b1))
    inter = np.maximum(0.0, np.minimum(a1, b1) - np.maximum(a0, b0))
    union = np.maximum(a1, b1) - np.minimum(a0, b0)
    return np.where(union > 1e-9, inter / np.maximum(union, 1e-9), 0.0)


def smooth_curve(z, window):
    z = np.asarray(z, dtype=np.float64)
    if window <= 1:
        return z
    pad = window // 2
    zp = np.concatenate([np.repeat(z[:, :1], pad, 1), z, np.repeat(z[:, -1:], pad, 1)], 1)
    cs = np.concatenate([np.zeros((len(z), 1)), np.cumsum(zp, 1)], 1)
    return (cs[:, window:] - cs[:, :-window]) / float(window)


def zscore(rel):
    rel = np.asarray(rel, dtype=np.float64)
    return (rel - rel.mean(1, keepdims=True)) / (rel.std(1, keepdims=True) + 1e-8)


def inside_score(z, lo, hi):
    z = np.asarray(z); T = z.shape[1]
    cs = np.concatenate([np.zeros((len(z), 1)), np.cumsum(z, 1)], 1)
    li = np.clip(np.searchsorted(CG, lo), 0, T - 1)
    hi_i = np.clip(np.searchsorted(CG, hi, side='right') - 1, 0, T - 1)
    cnt = np.maximum(hi_i - li + 1, 1)
    return (cs[np.arange(len(z)), hi_i + 1] - cs[np.arange(len(z)), li]) / cnt


def est_centroid(z, cb, beta):
    m = np.abs(CG[None, :] - np.asarray(cb)[:, None]) <= beta
    pm = np.where(m, np.maximum(z, 0.0), 0.0)
    s = pm.sum(1)
    c = (pm @ CG) / np.maximum(s, 1e-12)
    return np.where(s < 1e-12, cb, c)


def csr_full(z_raw, cb, wb, beta=0.20, rho=0.10, alpha=1.0, dmin=0.01):
    """返回 (c*, c1, din, moved)。c* 为最终中心（含死区+门控）。"""
    zs = smooth_curve(zscore(z_raw), 7)
    c1 = est_centroid(zs, cb, beta)
    d = c1 - cb
    d = np.where(np.abs(d) < rho * wb, 0.0, d) * alpha
    c2 = np.clip(cb + d, 0, 1)
    din = inside_score(zs, c2 - wb / 2, c2 + wb / 2, ) - inside_score(zs, cb - wb / 2, cb + wb / 2)
    c3 = np.where(din >= dmin, c2, cb) if dmin is not None else c2
    return c3, c1, din, (c3 != cb)


def auc(y, s):
    """秩和法 AUC。y 为 0/1 标签。"""
    y = np.asarray(y, dtype=np.float64); s = np.asarray(s, dtype=np.float64)
    if y.sum() == 0 or y.sum() == len(y):
        return float('nan')
    r = np.argsort(np.argsort(s, kind='stable'), kind='stable') + 1.0
    n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


# ------------------------------- 点掩码桩 -------------------------------
class PointGaussStub(nn.Module):
    def __init__(self, n_props, width):
        super().__init__()
        self.n_props = n_props
        self.width = width
        self.c = None

    def forward(self, h):
        bsz = h.size(0)
        if self.c is None or self.c.size(0) != self.n_props:
            self.c = torch.tensor(CG[:self.n_props], dtype=torch.float32, device=h.device)
        eps = 1e-4
        cc = self.c.clamp(eps, 1 - eps)
        lc = torch.log(cc / (1 - cc))
        lw = torch.full_like(lc, float(math.log(self.width / (1 - self.width))))
        return torch.cat([lc, lw], 0).unsqueeze(0).expand(bsz, -1).contiguous()


_orig_fuse = None
_FUSE_PATCHED = False


def _patch_fuse(mod):
    global _orig_fuse, _FUSE_PATCHED
    if not _FUSE_PATCHED:
        _orig_fuse = mod.match_and_fuse_params
        mod.match_and_fuse_params = lambda gp, mp, r: gp
        _FUSE_PATCHED = True


def _unpatch_fuse(mod):
    global _FUSE_PATCHED
    if _FUSE_PATCHED:
        mod.match_and_fuse_params = _orig_fuse
        _FUSE_PATCHED = False


def install_point_mask(model, n_props, width=0.1):
    saved = {}
    hooked = []
    if hasattr(model, 'fc_gauss'):
        saved['fc_gauss'] = model.fc_gauss
        model.fc_gauss = PointGaussStub(n_props, width).to(next(model.parameters()).device)
        hooked.append('fc_gauss')
    if hasattr(model, 'proposal_generator'):
        saved['proposal_generator'] = model.proposal_generator
        _c = torch.linspace(0.5 / n_props, 1 - 0.5 / n_props, n_props)
        _w = torch.full((n_props,), max(2.0 / n_props, 0.02))
        model.proposal_generator = lambda h, b: (_c.repeat(b).cuda(), _w.repeat(b).cuda())
        hooked.append('proposal_generator')

    mod = sys.modules.get(type(model).__module__)
    fuse_mod = None
    if mod is not None and hasattr(mod, 'match_and_fuse_params'):
        _patch_fuse(mod)
        fuse_mod = mod
        hooked.append('match_and_fuse_params')

    if not hooked:
        raise RuntimeError('未找到任何高斯参数挂钩')
    saved['num_props'] = getattr(model, 'num_props', None)
    model.num_props = n_props

    def restore():
        for k, v in saved.items():
            setattr(model, k, v)
        if fuse_mod is not None:
            _unpatch_fuse(fuse_mod)

    return restore, hooked


def _expand(t, bsz, Pu):
    if t.size(0) == bsz and Pu > 1:
        return t.unsqueeze(1).expand(bsz, Pu, -1).reshape(bsz * Pu, -1)
    return t.reshape(bsz * Pu, -1)


def _nll_per_prop(out, bsz):
    Pu = out['words_logit'].size(0) // bsz
    wm = _expand(out['words_mask'], bsz, Pu)
    wi = _expand(out['words_id'], bsz, Pu)
    n, _ = cal_nll_loss(out['words_logit'], wi, wm)
    return n.reshape(bsz, Pu)


def _call(model, ni, state):
    if hasattr(model, 'use_negative'):
        if state.get('neg') is None:
            model.use_negative = False
            try:
                try:
                    return model(mode='test', epoch=0, **ni)
                except TypeError:
                    return model(**ni)
            except Exception:
                state['neg'] = True
        model.use_negative = state['neg']
    try:
        return model(mode='test', epoch=0, **ni)
    except TypeError:
        return model(**ni)


# ------------------------------- 单次重复 -------------------------------
def run_rep(rep):
    np.random.seed(11 + rep); torch.manual_seed(11 + rep)
    model = getattr(models, cfg['model']['name'])(mcfg).cuda()
    model.load_state_dict(sp, strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    if hasattr(model, 'use_negative'):
        model.use_negative = False
    P0 = int(getattr(model, 'num_props', 1))
    STATE = {'neg': None}

    C = {k: [] for k in ('base', 'CSR', 'ORACLE_c', 'ORACLE_w', 'ORACLE_box')}
    D = {k: [] for k in ('cb', 'wb', 'c1', 'cstar', 'cgt', 'din', 'gl')}
    t0 = time.time()
    with torch.no_grad():
        for bi, batch in enumerate(loader, 1):
            ni = {k: v.cuda() for k, v in batch['net_input'].items()}
            if 'words_feat' in ni:
                ni['words_feat'] = ni['words_feat'].clone()
            bsz = ni['frames_feat'].size(0)
            out = _call(model, ni, STATE)
            n = _nll_per_prop(out, bsz)
            cb = out['center'].reshape(bsz, P0).float().cpu().numpy()
            wb = out['width'].reshape(bsz, P0).float().cpu().numpy()
            j1 = np.argsort(n.cpu().numpy(), 1)[:, 0]
            r = np.arange(bsz)
            c1_ = cb[r, j1]; w1_ = wb[r, j1]
            bL = np.clip(c1_ - w1_ / 2, 0, 1); bH = np.clip(c1_ + w1_ / 2, 0, 1)

            restore, hooks = install_point_mask(model, L, WPT)
            try:
                out2 = _call(model, ni, STATE)
            finally:
                pass
            REL = -_nll_per_prop(out2, bsz).float().cpu().numpy()
            restore()
            assert REL.shape == (bsz, L), 'REL 形状异常: %s' % (REL.shape,)

            dur = np.asarray([it[1] for it in batch['raw']], dtype=np.float64)
            ts = np.asarray([it[2] for it in batch['raw']], dtype=np.float64)
            g = ts / dur[:, None]
            g0, g1 = g[:, 0], g[:, 1]
            cg = (g0 + g1) / 2
            gw = g1 - g0

            cstar, cc1, cdin, moved = csr_full(REL, c1_, w1_)
            sL = np.clip(cstar - w1_ / 2, 0, 1); sH = np.clip(cstar + w1_ / 2, 0, 1)

            C['base'].append(iou(bL, bH, g0, g1))
            C['CSR'].append(iou(sL, sH, g0, g1))
            # ORACLE_c: 真值中心 + 基础框宽（中心轴上界）
            oL = np.clip(cg - w1_ / 2, 0, 1); oH = np.clip(cg + w1_ / 2, 0, 1)
            C['ORACLE_c'].append(iou(oL, oH, g0, g1))
            # ORACLE_w: 基础中心 + 真值框宽（尺寸轴上界）
            wL = np.clip(c1_ - gw / 2, 0, 1); wH = np.clip(c1_ + gw / 2, 0, 1)
            C['ORACLE_w'].append(iou(wL, wH, g0, g1))
            # ORACLE_box: 真值框
            C['ORACLE_box'].append(iou(g0, g1, g0, g1))

            D['cb'].append(c1_); D['wb'].append(w1_); D['c1'].append(cc1)
            D['cstar'].append(cstar); D['cgt'].append(cg)
            D['din'].append(cdin); D['gl'].append(gw)
            if bi % 50 == 0:
                if bi == 50:
                    print('    [挂钩] %s' % ','.join(hooks), flush=True)
                print('    rep%d %d/%d %.1fs' % (rep, bi, len(loader), time.time() - t0), flush=True)

    return {k: np.concatenate(v) for k, v in C.items()}, {k: np.concatenate(v) for k, v in D.items()}


RES = []; DAT = None
for rep in range(K_REP):
    r_, d_ = run_rep(rep)
    RES.append(r_); DAT = d_
    print('  rep%d | base mIoU=%.4f | CSR %.4f (%+.2f) | ORACLE_c %.4f (%+.2f)' % (
        rep, r_['base'].mean(), r_['CSR'].mean(), 100 * (r_['CSR'].mean() - r_['base'].mean()),
        r_['ORACLE_c'].mean(), 100 * (r_['ORACLE_c'].mean() - r_['base'].mean())), flush=True)

J = {'tag': TAG, 'dataset': DS_NAME, 'split': os.path.basename(dargs['test_data']),
     'N': int(len(DAT['gl'])), 'L': L, 'K_REP': K_REP, 'WPT': WPT, 'ckpt': ckpt,
     'rho_geo': float(np.median(DAT['wb']) / max(np.median(DAT['gl']), 1e-9)),
     'med_wb': float(np.median(DAT['wb'])), 'med_gl': float(np.median(DAT['gl']))}


def m(v):
    return dict(mIoU=float(np.mean(v)), r5=float(np.mean(v >= 0.5)), r7=float(np.mean(v >= 0.7)))


J['rows'] = {}
print('\n' + '=' * 100, flush=True)
print('[算子表] %s  mean over %d rep' % (TAG, K_REP), flush=True)
b = {k: float(np.mean([m(r_['base'])[k] for r_ in RES])) for k in ('mIoU', 'r5', 'r7')}
for k in ('base', 'CSR', 'ORACLE_c', 'ORACLE_w', 'ORACLE_box'):
    ms = {kk: float(np.mean([m(r_[k])[kk] for r_ in RES])) for kk in ('mIoU', 'r5', 'r7')}
    print('%-12s mIoU %7.2f (%+6.2f) | R1@0.5 %7.2f (%+6.2f) | R1@0.7 %7.2f (%+6.2f)' % (
        k, 100 * ms['mIoU'], 100 * (ms['mIoU'] - b['mIoU']),
        100 * ms['r5'], 100 * (ms['r5'] - b['r5']),
        100 * ms['r7'], 100 * (ms['r7'] - b['r7'])), flush=True)
    J['rows'][k] = dict(mIoU=100 * ms['mIoU'], r5=100 * ms['r5'], r7=100 * ms['r7'],
                        dm=100 * (ms['mIoU'] - b['mIoU']),
                        d5=100 * (ms['r5'] - b['r5']), d7=100 * (ms['r7'] - b['r7']))

# ---------------- 曲线可用性 ----------------
cb, wb_ = DAT['cb'], DAT['wb']; c1, cs, cg = DAT['c1'], DAT['cstar'], DAT['cgt']
e_base = np.abs(cb - cg); e_c1 = np.abs(c1 - cg); e_cs = np.abs(cs - cg)
dirn = np.abs(cb - cg) >= DDIR
dir_acc_1 = float(np.mean(np.sign(c1 - cb)[dirn] == np.sign(cg - cb)[dirn])) if dirn.sum() else float('nan')
dir_acc_s = float(np.mean(np.sign(cs - cb)[dirn] == np.sign(cg - cb)[dirn])) if dirn.sum() else float('nan')
# 移动后是否更接近真值中心（在确实移动了的样本里）
mv = (cs != cb)
closer = float(np.mean(e_cs[mv] < e_base[mv])) if mv.sum() else float('nan')
# Δ_inside 作为「该不该移动」的判别信号
imp = []
for r_ in RES:
    imp.append(r_['CSR'] > r_['base'])
imp = np.concatenate(imp)
auc_in = auc(imp, np.tile(DAT['din'], K_REP))

J['curve'] = dict(
    mae_base=float(e_base.mean()), mae_c1=float(e_c1.mean()), mae_final=float(e_cs.mean()),
    dmae_c1=float(e_c1.mean() - e_base.mean()), dmae_final=float(e_cs.mean() - e_base.mean()),
    dir_acc_c1=dir_acc_1, dir_acc_final=dir_acc_s, move_rate=float(mv.mean()),
    closer_when_moved=closer, n_offset=int(dirn.sum()),
    auc_din=float(auc_in), frac_improved=float(imp.mean()),
)
print('\n[曲线可用性] %s' % TAG, flush=True)
print('  中心 MAE: 基础 %.4f -> 曲线质心 c1 %.4f (%+.4f) -> CSR 输出 %.4f (%+.4f)' % (
    e_base.mean(), e_c1.mean(), e_c1.mean() - e_base.mean(), e_cs.mean(), e_cs.mean() - e_base.mean()), flush=True)
print('  方向准确率 (|Δc_b|>=%.2f, n=%d): c1 %.4f | 最终 %.4f' % (DDIR, dirn.sum(), dir_acc_1, dir_acc_s), flush=True)
print('  移动率 %.4f | 移动后更接近真值中心的比例 %.4f | Δ_inside AUC %.4f | 被改善样本占比 %.4f' % (
    mv.mean(), closer, auc_in, imp.mean()), flush=True)

with open(OUT, 'w') as f:
    json.dump(J, f, ensure_ascii=False, indent=1)
print('\nWROTE %s' % OUT, flush=True)
print('DONE', flush=True)
