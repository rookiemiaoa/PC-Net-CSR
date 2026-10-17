# -*- coding: utf-8 -*-
"""w52_width_official.py —— 忠实克隆 csr/eval/eval_csr.py 的**读出宽度扫描**

为什么需要这个脚本（与 w50 的区别）
================================================================================
官方 CSR 评估路径里，逐点相关性曲线来自把 `proposal_generator` 换成 L 个固定点掩码：

    _c_pt = torch.linspace(0.5/L, 1-0.5/L, L)
    _w_pt = torch.full((L,), max(2.0/L, 0.02))      # L=50 ⇒ **0.04**

也就是说：**官方读出宽度 = 0.04**（归一时长），而模型自己预测的框宽中位数是
**0.351** —— 读出窗比目标尺度**窄 8.8 倍**。这个 0.04 从未被扫描过。

w50 扫的是另一个挂钩（fc_gauss 桩宽），pg 固定 0.04，因此它测的不是官方那条轴。
本脚本严格按官方路径：**只替换 proposal_generator，并把 `_w_pt` 作为自变量**。

锚点（硬校验）
  WPT=0.04 时，base 必须复现 s1/eval_csr 实测的 47.20/55.22/26.58，
  CSR(默认五标量) 必须复现 48.33/55.89/28.67。差 >0.02 直接标 DIFF。

用法: python w52_width_official.py <repo_root> <config> <ckpt> <tag>
环境变量: BS=16 K_REP=3 LIMIT=0 WPTS="0.02,0.04,0.08,0.15,0.35"
          OUT_NPZ=.../w52_<tag>.npz  OUT=.../w52_<tag>.json
"""
import json
import os
import pickle
import sys
import time

import numpy as np
import torch

REPO = sys.argv[1]
TAG = sys.argv[4] if len(sys.argv) > 4 else 'x'
sys.path.insert(0, REPO)
os.chdir(REPO)

from csr.paths import DATA_ROOT            # noqa: E402
from csr import ops                        # noqa: E402
from utils import load_json                # noqa: E402
import datasets as da                      # noqa: E402
from models import PCNet                   # noqa: E402
from models.loss import cal_nll_loss       # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

# 官方日志 base（CharadesSTA）与 s1 实测 base/CSR —— w52 的锚点
REF_LOG = dict(mIoU=0.4712, r1_05=0.5484, r1_07=0.2668)
REF_S1 = dict(base=(47.20, 55.22, 26.58), csr=(48.33, 55.89, 28.67))

cfg = load_json(sys.argv[2])
ckpt = sys.argv[3]
dargs = cfg['dataset']
DS_NAME = dargs['dataset']
if os.environ.get('TEST_DATA'):
    dargs['test_data'] = os.environ['TEST_DATA']
vocab = pickle.load(open(dargs['vocab_path'], 'rb'))
try:
    ds = getattr(da, DS_NAME)(data_path=dargs['test_data'], vocab=vocab, args=dargs, split='test')
except TypeError:
    ds = getattr(da, DS_NAME)(data_path=dargs['test_data'], vocab=vocab, args=dargs)
VOCAB_SIZE = ds.vocab_size
COLLATE = ds.collate_data

FP = os.environ.get('FEAT_PATH', dargs['feature_path'])
if not os.path.isabs(FP) and not os.path.exists(FP):
    for cand in (os.path.join(DATA_ROOT, os.path.basename(FP)), os.path.join(REPO, os.path.basename(FP))):
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
CG = (np.arange(L) + 0.5) / L
GRID = np.linspace(0.0, 1.0, L)
BS = int(os.environ.get('BS', 16))
K_REP = int(os.environ.get('K_REP', 3))
WPTS = [float(x) for x in os.environ.get('WPTS', '0.02,0.04,0.08,0.15,0.35').split(',')]
NW = len(WPTS)
THR = float(os.environ.get('THR', 0.35))
OUT_NPZ = os.environ.get('OUT_NPZ') or os.path.join(REPO, 'results', 'curve_availability', 'w52_%s.npz' % TAG)
OUT = os.environ.get('OUT') or os.path.join(REPO, 'results', 'curve_availability', 'w52_%s.json' % TAG)
loader = DataLoader(ds, batch_size=BS, shuffle=False, collate_fn=COLLATE, num_workers=8)
_t0 = time.time()

print('[GPU] %s' % torch.cuda.get_device_name(0), flush=True)
print('[BASE] %s model=%s split=%s N=%d num_props=%d L=%d BS=%d K_REP=%d WPTS=%s' % (
    TAG, DS_NAME, os.path.basename(dargs['test_data']), len(ds), P, L, BS, K_REP, WPTS), flush=True)
print('CSR(默认五标量): smooth=7 beta=0.20 rho=0.10 alpha=1.00 dmin=0.01', flush=True)


# ---------------------------------------------------------------- 官方算子薄封装
def iou(a0, a1, b0, b1):
    return ops.iou(a0, a1, b0, b1)


def csr_center(z, cb, wb, smooth=7, beta=0.20, rho=0.10, alpha=1.0, dmin=0.01):
    """官方流水线的**中心**输出（与 ops.refine_center 逐步等价，返回 c2 而非 box）。"""
    zs = ops.smooth_curve(ops.zscore(z), smooth)
    c1 = ops.estimate_window_center(zs, cb, beta, CG)
    c2 = ops.shift_center(cb, wb, c1, rho, alpha)
    if dmin is not None:
        din = ops.inside_score(zs, c2 - wb / 2, c2 + wb / 2, CG) \
            - ops.inside_score(zs, cb - wb / 2, cb + wb / 2, CG)
        c2 = np.where(din >= dmin, c2, cb)
    return c2


# ---------------------------------------------------------------- 单次重复（官方前向）
def run_rep(rep):
    np.random.seed(7 + rep); torch.manual_seed(7 + rep)
    model = PCNet(mcfg).cuda()
    model.load_state_dict(sp)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    _orig_pg = model.proposal_generator
    _c_pt = torch.linspace(0.5 / L, 1 - 0.5 / L, L)

    N8, C8, W8, G = [], [], [], []
    RELW = [[] for _ in range(NW)]
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
            dur = np.asarray([i[1] for i in batch['raw']], dtype=np.float64)
            ts = np.asarray([i[2] for i in batch['raw']], dtype=np.float64)
            G.append(ts / dur[:, None])

            # —— 唯一的自变量：点掩码宽度（官方默认 max(2/L,0.02)=0.04）——
            for wi, wp in enumerate(WPTS):
                _w_pt = torch.full((L,), float(wp))
                model.proposal_generator = lambda h, b, _w=_w_pt: (_c_pt.repeat(b).cuda(), _w.repeat(b).cuda())
                model.num_props = L
                o2 = model(epoch=0, **net_input)
                wm2 = o2['words_mask'].unsqueeze(1).expand(bsz, L, -1).contiguous().view(bsz * L, -1)
                wi2 = o2['words_id'].unsqueeze(1).expand(bsz, L, -1).contiguous().view(bsz * L, -1)
                n2, _ = cal_nll_loss(o2['words_logit'], wi2, wm2)
                RELW[wi].append((-n2.view(bsz, L)).cpu().numpy())
            model.proposal_generator = _orig_pg
            model.num_props = P

            if bi % 40 == 0:
                el = time.time() - t0
                print('    rep%d %d/%d %.1fs (eta %.1fs)' % (rep, bi, len(loader), el,
                      el / bi * (len(loader) - bi)), flush=True)

    G = np.concatenate(G); N8 = np.concatenate(N8); C8 = np.concatenate(C8); W8 = np.concatenate(W8)
    n = len(G); r = np.arange(n)
    GS, GE = G[:, 0], G[:, 1]; GL = GE - GS
    j1 = np.argsort(N8, axis=1)[:, 0]
    cb, wb = C8[r, j1], W8[r, j1]
    rel = np.stack([np.concatenate(RELW[wi], 0) for wi in range(NW)], 0)   # (NW, n, L)
    return rel, cb, wb, GS, GE, GL


RL, CBL, WBL, GSL, GEL, GLL = [], [], [], [], [], []
for rep in range(K_REP):
    a, b, c, d, e, f = run_rep(rep)
    RL.append(a); CBL.append(b); WBL.append(c); GSL.append(d); GEL.append(e); GLL.append(f)
    print('  rep%d done %.1fs' % (rep, time.time() - _t0), flush=True)

rel = np.stack(RL, 0); cb = np.stack(CBL, 0); wb = np.stack(WBL, 0)
gs = np.stack(GSL, 0); ge = np.stack(GEL, 0); gl = np.stack(GLL, 0)
np.savez_compressed(OUT_NPZ, rel=rel.astype(np.float32), cb=cb.astype(np.float32),
                    wb=wb.astype(np.float32), gs=gs.astype(np.float32),
                    ge=ge.astype(np.float32), gl=gl.astype(np.float32), wpts=np.array(WPTS))
print('WROTE %s' % OUT_NPZ, flush=True)
print('[N] N=%d K_REP=%d med_wb=%.4f' % (cb.shape[1], K_REP, float(np.mean([np.median(w) for w in wb]))), flush=True)


def ev(fn, wi):
    m, r5, r7 = [], [], []
    for k in range(K_REP):
        c = fn(rel[k, wi], cb[k], wb[k])
        v = iou(np.clip(c - wb[k] / 2, 0, 1), np.clip(c + wb[k] / 2, 0, 1), gs[k], ge[k])
        m.append(v.mean()); r5.append((v >= 0.5).mean()); r7.append((v >= 0.7).mean())
    return 100 * float(np.mean(m)), 100 * float(np.mean(r5)), 100 * float(np.mean(r7))


BASE = ev(lambda r, c, w: c, 0)
print('\n' + '=' * 100)
print('[读出宽度扫描 - 官方路径] %s   base mIoU=%.2f R1@0.5=%.2f R1@0.7=%.2f' % (TAG, *BASE))
print('%8s %9s %9s %9s %9s | %9s %9s %9s %9s' % ('WPT', 'dir_acc', 'closer', 'move', 'maeΔ', 'CSR_mIoU', 'CSR_r5', 'CSR_r7', 'ΔmIoU'))
JW = {}
for wi, wp in enumerate(WPTS):
    da_, cl_, mr_, dm_ = [], [], [], []
    for k in range(K_REP):
        zs = ops.smooth_curve(ops.zscore(rel[k, wi]), 7)
        c1 = ops.estimate_window_center(zs, cb[k], 0.20, CG)
        cgt = (gs[k] + ge[k]) / 2
        off = np.abs(cb[k] - cgt) >= 0.08
        dl = c1 - cb[k]
        mv = np.abs(dl) >= 0.10 * wb[k]
        da_.append(float(np.mean(np.sign(dl[off]) == np.sign((cgt - cb[k])[off]))))
        if mv.sum():
            cl_.append(float(np.mean(np.abs(c1[mv] - cgt[mv]) < np.abs(cb[k][mv] - cgt[mv]))))
        mr_.append(float(mv.mean()))
        dm_.append(float(np.mean(np.abs(c1 - cgt)) - np.mean(np.abs(cb[k] - cgt))))
    mm = ev(lambda r, c, w: csr_center(r, c, w), wi)
    JW['w%.3f' % wp] = dict(wpt=wp, dir_acc=float(np.mean(da_)), closer=float(np.mean(cl_)),
                            move=float(np.mean(mr_)), dmae=float(np.mean(dm_)),
                            mIoU=mm[0], r5=mm[1], r7=mm[2], dm=mm[0] - BASE[0])
    print('%8.3f %9.4f %9.4f %9.4f %+9.4f | %9.2f %9.2f %9.2f %+9.3f'
          % (wp, np.mean(da_), np.mean(cl_), np.mean(mr_), np.mean(dm_), mm[0], mm[1], mm[2], mm[0] - BASE[0]))

# 锚点
w04 = min(WPTS, key=lambda x: abs(x - 0.04))
c0 = JW['w%.3f' % w04]
a = REF_S1
db = max(abs(BASE[0] - a['base'][0]), abs(BASE[1] - a['base'][1]), abs(BASE[2] - a['base'][2]))
dc = max(abs(c0['mIoU'] - a['csr'][0]), abs(c0['r5'] - a['csr'][1]), abs(c0['r7'] - a['csr'][2]))
print('\n[锚点] WPT=%.3f(=官方 max(2/L,0.02))' % w04)
print('   base 期望 %.2f/%.2f/%.2f 实测 %.2f/%.2f/%.2f  最大差 %.3f  %s'
      % (*a['base'], BASE[0], BASE[1], BASE[2], db, 'OK' if db < 0.02 else 'DIFF ⚠'))
print('   CSR  期望 %.2f/%.2f/%.2f 实测 %.2f/%.2f/%.2f  最大差 %.3f  %s'
      % (*a['csr'], c0['mIoU'], c0['r5'], c0['r7'], dc, 'OK' if dc < 0.02 else 'DIFF ⚠'))
print('   官方日志 base(R1,mIoU/R1@0.5/R1@0.7) = %.4f/%.4f/%.4f' % (REF_LOG['mIoU'], REF_LOG['r1_05'], REF_LOG['r1_07']))
if LIMIT > 0:
    print('   ⚠ LIMIT=%d 为子集，锚点不可比' % LIMIT)

json.dump(dict(tag=TAG, N=int(cb.shape[1]), K_REP=K_REP, wpts=WPTS, base=BASE,
               med_wb=float(np.mean([np.median(w) for w in wb])),
               width=JW, npz=OUT_NPZ, secs=time.time() - _t0,
               anchor=dict(base_err=db, csr_err=dc)),
          open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('\nWROTE %s' % OUT)
print('DONE %.1f min' % ((time.time() - _t0) / 60))
