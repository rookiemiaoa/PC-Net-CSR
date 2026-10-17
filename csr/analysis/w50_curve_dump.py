# -*- coding: utf-8 -*-
"""w50_curve_dump.py —— 逐样本相关性曲线转储（服务器端，只做前向）

背景（本脚本要解决的根因）
================================================================================
CSR 在 PC-Net / CNM 上为正，在 CPL / QMN / CCR 上为负。已定位到两条根因线索：

 RC-1【读出上下文不足】逐位置曲线是用 L=50 个**固定宽度 WPT** 的点掩码读出的
      （w32: install_point_mask(model, L, WPT)，WPT 默认 0.10）。已有证据：
      CPL 上把 WPT 从 0.15 抬到 0.50，CSR 的伤害由 ΔmIoU −3.76 缩到 −1.52，
      短桶 −2.49→−0.79，长桶 −7.11→−3.42，R1@0.7 由 −1.56 转 +0.29。
      但 0.10 / 0.15 / 0.20 / 0.30 / 0.50 的刻度太粗，且 0.10 那次是 N=3096、
      其余是 N=1024 的子集，跨点不可比。**本脚本在同一子集上把 WPT 细扫。**

 RC-2【死区标定】closer_when_moved 完美分开符号：
      pcnet 0.629 / cnm 0.664（正） vs cpl 0.317 / qmn 0.336 / ccr 0.358（负）。
      ⇒ 同一个无量纲死区 ρ=0.10 在不同骨干上筛出的是不同置信度的子集。

本脚本的职责边界（刻意的设计）
================================================================================
服务器**只做前向**，把逐样本原始量全部落盘；
所有标定搜索（死区 ρ、步长 α、分位 q、窗口 β）在本地 CPU 上做。
理由：前向是 GPU 瓶颈且不可复算；标定是纯数组运算，本地零成本、可反复扫。
⇒ 这样一次 GPU 投入可以支撑任意多次标定搜索，避免反复占用卡。

输出（npz）
  rel : (K_REP, NW, N, L) float32   —— 第 wi 个宽度的逐位置曲线
  cb  : (K_REP, N)                  —— 基础框中心（未挂钩前向，top-1 提案）
  wb  : (K_REP, N)                  —— 基础框宽度
  gs, ge : (K_REP, N)               —— 真值起止（已按视频时长归一化）
  wpts: (NW,)                       —— 各宽度值
⚠️ 每个 rep 的 cb/wb/rel 互不相同（前向含随机性，K_REP=3 就是为此）。
   下游所有度量必须**逐 rep 自洽**，混用会把锚点漂掉（rc_q2check.py 实测过）。

内建锚点校验
  WPT=0.10 且 N 与 w32_<tag>.json 一致时，base/CSR 三列必须复现 w32 的结果；
  不一致则打印 DIFF（绝不静默通过）。

用法
  python w50_curve_dump.py <repo_root> <config> <ckpt> <tag>
环境变量
  BS=16 K_REP=3 LIMIT=0
  WPTS="0.10,0.25,0.40,0.55"
  TEST_DATA=<json>        覆盖 config 里的 test_data（用于换切分）
  OUT_NPZ=<repo>/results/curve_availability/w50_<tag>.npz
  OUT=<repo>/results/curve_availability/w50_<tag>.json
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
TAG = sys.argv[4] if len(sys.argv) > 4 else 'x'
sys.path.insert(0, REPO)
os.chdir(REPO)

import models                          # noqa: E402
from utils import load_json            # noqa: E402
import datasets as da                  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

# w32_<tag>.json 里的 (base mIoU, CSR mIoU) 作为锚点（WPT=0.10, N=3096, BS 同款）
ANCHOR = {
    'pcnet': (47.132, 48.178),
    'cpl':   (46.468, 42.712),
    'qmn':   (46.316, 42.783),
    'ccr':   (46.320, 43.137),
}


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
WPTS = [float(x) for x in os.environ.get('WPTS', '0.10,0.25,0.40,0.55').split(',')]
NW = len(WPTS)
THR = float(os.environ.get('THR', 0.35))
OUT_NPZ = os.environ.get('OUT_NPZ') or os.path.join(REPO, 'results', 'curve_availability', 'w50_%s.npz' % TAG)
OUT = os.environ.get('OUT') or os.path.join(REPO, 'results', 'curve_availability', 'w50_%s.json' % TAG)
loader = DataLoader(ds, batch_size=BS, shuffle=False, collate_fn=COLLATE, num_workers=4)

print('[GPU] %s' % torch.cuda.get_device_name(0), flush=True)
print('[BASE] %s model=%s split=%s N=%d L=%d BS=%d K_REP=%d WPTS=%s' % (
    TAG, cfg['model']['name'], os.path.basename(dargs['test_data']), len(ds), L, BS, K_REP, WPTS), flush=True)
_t0 = time.time()


# ----------------------------------------------------------- 算子（与 w32 逐行一致）
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


def csr_center(z_raw, cb, wb, beta=0.20, rho=0.10, alpha=1.0, dmin=0.01):
    zs = smooth_curve(zscore(z_raw), 7)
    c1 = est_centroid(zs, cb, beta)
    d = c1 - cb
    d = np.where(np.abs(d) < rho * wb, 0.0, d) * alpha
    c2 = np.clip(cb + d, 0, 1)
    if dmin is not None:
        din = inside_score(zs, c2 - wb / 2, c2 + wb / 2) - inside_score(zs, cb - wb / 2, cb + wb / 2)
        c2 = np.where(din >= dmin, c2, cb)
    return c2


def box_ious(c, wb, gs, ge):
    return iou(np.clip(c - wb / 2, 0, 1), np.clip(c + wb / 2, 0, 1), gs, ge)


# ----------------------------------------------------------- 点掩码桩（与 w32 一致）
class PointGaussStub(nn.Module):
    def __init__(self, n_props, width):
        super().__init__()
        self.n_props = n_props
        self.width = width
        self.c = None

    def forward(self, h):
        bsz = h.size(0)
        if self.c is None or self.c.size(0) != self.n_props:
            self.c = torch.tensor(CG, dtype=torch.float32, device=h.device)
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


def install_point_mask(model, n_props, width):
    """与 w32 逐行一致的挂钩：**一次只装一个宽度**，保证与已有协议可比。"""
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
    saved['num_props'] = model.num_props
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


# ----------------------------------------------------------- 单次重复
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

    RELW = [[] for _ in range(NW)]
    CB, WB, GS, GE, GL = [], [], [], [], []
    t0 = time.time()
    with torch.no_grad():
        for bi, batch in enumerate(loader, 1):
            ni = {k: v.cuda() for k, v in batch['net_input'].items()}
            if 'words_feat' in ni:
                ni['words_feat'] = ni['words_feat'].clone()
            bsz = ni['frames_feat'].size(0)

            # (1) 基础前向：未挂钩，取 top-1 提案的中心/宽度
            out = _call(model, ni, STATE)
            n = _nll_per_prop(out, bsz)
            cb = out['center'].reshape(bsz, P0).float().cpu().numpy()
            wb = out['width'].reshape(bsz, P0).float().cpu().numpy()
            j1 = np.argsort(n.cpu().numpy(), 1)[:, 0]
            r = np.arange(bsz)
            c1_, w1_ = cb[r, j1], wb[r, j1]

            # (2) 每个宽度各做一次读出前向（一次只装一个宽度 ⇒ 与既有协议同构）
            for wi, wp in enumerate(WPTS):
                restore, hooks = install_point_mask(model, L, wp)
                try:
                    out2 = _call(model, ni, STATE)
                    rel = -_nll_per_prop(out2, bsz).float().cpu().numpy()
                finally:
                    restore()
                assert rel.shape == (bsz, L), 'REL 形状异常 %s' % (rel.shape,)
                RELW[wi].append(rel)

            dur = np.asarray([it[1] for it in batch['raw']], dtype=np.float64)
            ts = np.asarray([it[2] for it in batch['raw']], dtype=np.float64)
            g = ts / dur[:, None]
            CB.append(c1_); WB.append(w1_)
            GS.append(g[:, 0]); GE.append(g[:, 1]); GL.append(g[:, 1] - g[:, 0])
            if bi % 25 == 0:
                if bi == 25:
                    print('    [挂钩] %s' % ','.join(hooks), flush=True)
                el = time.time() - t0
                print('    rep%d %d/%d  %.1fs  (eta %.1fs)' % (rep, bi, len(loader), el,
                      el / bi * (len(loader) - bi)), flush=True)

    rel = np.stack([np.concatenate(RELW[wi], 0) for wi in range(NW)], 0)   # (NW, N, L)
    return rel, np.concatenate(CB), np.concatenate(WB), \
        np.concatenate(GS), np.concatenate(GE), np.concatenate(GL)


RL, CBL, WBL, GSL, GEL, GLL = [], [], [], [], [], []
for rep in range(K_REP):
    tR = time.time()
    a, b, c, d, e, f = run_rep(rep)
    RL.append(a); CBL.append(b); WBL.append(c); GSL.append(d); GEL.append(e); GLL.append(f)
    print('  rep%d done  %.1fs' % (rep, time.time() - tR), flush=True)

rel = np.stack(RL, 0)          # (K_REP, NW, N, L)
cb = np.stack(CBL, 0); wb = np.stack(WBL, 0)
gs = np.stack(GSL, 0); ge = np.stack(GEL, 0); gl = np.stack(GLL, 0)

print('\n[N] N=%d K_REP=%d  rep 间 cb 互异=%s rel 互异=%s | med_wb=%.4f 原生宽度 p10/p50/p90=%.3f/%.3f/%.3f'
      % (cb.shape[1], K_REP,
         not all(np.array_equal(cb[0], cb[k]) for k in range(K_REP)),
         not all(np.array_equal(rel[0], rel[k]) for k in range(K_REP)),
         float(np.mean([np.median(w) for w in wb])),
         *np.percentile(wb.mean(0), [10, 50, 90])), flush=True)

np.savez_compressed(OUT_NPZ, rel=rel.astype(np.float32), cb=cb.astype(np.float32),
                    wb=wb.astype(np.float32), gs=gs.astype(np.float32),
                    ge=ge.astype(np.float32), gl=gl.astype(np.float32),
                    wpts=np.array(WPTS))
print('WROTE %s' % OUT_NPZ, flush=True)


# ----------------------------------------------------------- 就地汇总（逐 rep 自洽）
def eval_reps(fn, wi):
    m, r5, r7 = [], [], []
    for k in range(K_REP):
        c = fn(rel[k, wi], cb[k], wb[k])
        v = box_ious(c, wb[k], gs[k], ge[k])
        m.append(v.mean()); r5.append((v >= 0.5).mean()); r7.append((v >= 0.7).mean())
    return 100 * float(np.mean(m)), 100 * float(np.mean(r5)), 100 * float(np.mean(r7))


def diag_reps(wi, rho=0.10):
    da_, mr, cl, dm_, ratio = [], [], [], [], []
    for k in range(K_REP):
        cgt = (gs[k] + ge[k]) / 2
        off = np.abs(cb[k] - cgt) >= 0.08
        z = smooth_curve(zscore(rel[k, wi]), 7)
        c1 = est_centroid(z, cb[k], 0.20)
        dl = c1 - cb[k]
        mv = np.abs(dl) >= rho * wb[k]
        da_.append(float(np.mean(np.sign(dl[off]) == np.sign((cgt - cb[k])[off]))))
        mr.append(float(mv.mean()))
        cl.append(float(np.mean(np.abs(c1[mv] - cgt[mv]) < np.abs(cb[k][mv] - cgt[mv]))) if mv.sum() else np.nan)
        dm_.append(float(np.mean(np.abs(c1 - cgt)) - np.mean(np.abs(cb[k] - cgt))))
        ratio.append(np.abs(dl) / np.maximum(wb[k], 1e-9))
    return float(np.mean(da_)), float(np.mean(mr)), float(np.nanmean(cl)), float(np.mean(dm_)), np.concatenate(ratio)


BASE = eval_reps(lambda r, c, w: c, 0)
print('\n' + '=' * 108)
print('[WPT 扫描] %s  同一切分、同一批样本、同一随机种子，只换点掩码宽度' % TAG)
print('%8s %9s %10s %10s %8s | %9s %9s %9s | %9s %9s %9s'
      % ('WPT', 'dir_acc', 'closer', 'move_rate', 'maeΔ', 'base_mIoU', 'base_r5', 'base_r7', 'CSR_mIoU', 'CSR_r5', 'CSR_r7'))
print('%8s %9s %10s %10s %8s | %9.2f %9.2f %9.2f | %9s %9s %9s'
      % ('base', '-', '-', '-', '-', BASE[0], BASE[1], BASE[2], '-', '-', '-'))
JQ = {}
for wi, wp in enumerate(WPTS):
    dA, mR, cL, dM, _ = diag_reps(wi)
    cm, cr5, cr7 = eval_reps(lambda r, c, w: csr_center(r, c, w, 0.20, 0.10, 1.0, 0.01), wi)
    JQ['w%.3f' % wp] = dict(wpt=wp, dir_acc=dA, closer=cL, move_rate=mR, dmae=dM,
                            csr_mIoU=cm, csr_r5=cr5, csr_r7=cr7, dm=cm - BASE[0])
    print('%8.3f %9.4f %10.4f %10.4f %+8.4f | %9.2f %9.2f %9.2f | %9.2f %9.2f %9.2f  ΔmIoU=%+.3f'
          % (wp, dA, cL, mR, dM, BASE[0], BASE[1], BASE[2], cm, cr5, cr7, cm - BASE[0]))

anc = ANCHOR.get(TAG)
if anc is not None and abs(WPTS[0] - 0.10) < 1e-9:
    g0 = JQ['w0.100']
    db = abs(BASE[0] - anc[0]); dc = abs(g0['csr_mIoU'] - anc[1])
    print('\n[锚点] WPT=0.10 应与 w32_%s.json 一致：base %.3f(实测 %.3f, 差 %.4f) | CSR %.3f(实测 %.3f, 差 %.4f)  %s'
          % (TAG, anc[0], BASE[0], db, anc[1], g0['csr_mIoU'], dc,
             'OK' if (db < 0.02 and dc < 0.02) else 'DIFF ⚠'))
    if abs(LIMIT) > 0:
        print('       ⚠ 本次 LIMIT=%d（子集），与全量锚点不可比，忽略上面的 DIFF' % LIMIT)

json.dump(dict(tag=TAG, repo=REPO, ckpt=ckpt, split=os.path.basename(dargs['test_data']),
               N=int(cb.shape[1]), L=L, K_REP=K_REP, BS=BS, wpts=WPTS,
               med_wb=float(np.mean([np.median(w) for w in wb])),
               base=dict(mIoU=BASE[0], r5=BASE[1], r7=BASE[2]), wpt_scan=JQ,
               npz=OUT_NPZ, secs=time.time() - _t0),
          open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('\nWROTE %s' % OUT)
print('DONE  total %.1f min' % ((time.time() - _t0) / 60))
