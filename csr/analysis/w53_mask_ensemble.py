# -*- coding: utf-8 -*-
"""w53_mask_ensemble.py —— 在**官方协议**上做两件事：定位曲线噪声 + 掩码集成读出

根因（probe2_rel.py 实证，服务器上跑的）
================================================================================
PC-Net 的 `models/pcnet.py::_mask_words` 每个样本调一次 **全局 RNG**：

    choices = np.random.choice(np.arange(1, l+1), num_masked_words, replace=False, p=p)

而 CSR 的逐位置曲线 r(t) = -nll(t) 正是「在某个随机词掩码下、用 t 处的点掩码重建」
得到的 nll。于是：

* **每多读一次曲线，全局 RNG 就前进一次 ⇒ 每次读出的曲线都不同。**
  实测：同一宽度连读两次，`maxabsdiff = 0.826`（rel 自身 mean=-3.44 / std=1.25）
  —— **噪声与信号同量级**。
* 官方协议（eval_csr.py）是「1 次基础前向 + 1 次读出前向」，两次用的是**不同随机掩码**。
  这解释了为什么弱骨干（CPL/QMN/CCR）的 dir_acc 掉到 0.45–0.51 ≈ 随机：
  不是「曲线没有定位信息」，而是**单次读出的噪声把信息淹没了**。

可执行修复（零训练、免真值、不改权重）
  掩码集成读出：对同一 batch 读 R 次（每次独立掩码），把 R 条曲线平均 → 噪声降 √R 倍。
  本脚本同时落盘 R 条**单独**曲线，用于量化噪声底噪。

⚠ 方法论红线（本轮踩过的坑）
  在**同一 batch 输入上连读多个宽度**会把「调用次序 / RNG 前进」混进「宽度」里，
  产生完全虚假的“宽度效应”（w50/w52 已实证：同宽重跑都不一致）。
  ⇒ 本脚本**一个进程只用一个宽度**，宽度只能靠多次运行来扫。

用法
  python w53_mask_ensemble.py <repo_root> <config> <ckpt> <tag>
环境变量
  MODE=official | ens         official: 精确克隆 eval_csr（1基+1读，不重播种，用于锚点）
                              ens     : 1基 + R 次独立播种的读出（掩码集成）
  R=5                         集成次数（MODE=ens）
  WPT=0.04                    点掩码宽度（官方 = max(2/L,0.02) = 0.04）
  BS=16 K_REP=3 LIMIT=0
  OUT_NPZ / OUT
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

REF_S1 = dict(base=(47.20, 55.22, 26.58), csr=(48.33, 55.89, 28.67))

cfg = load_json(sys.argv[2]); ckpt = sys.argv[3]
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
BS = int(os.environ.get('BS', 16))
K_REP = int(os.environ.get('K_REP', 3))
WPT = float(os.environ.get('WPT', max(2.0 / L, 0.02)))
R = int(os.environ.get('R', 5))
MODE = os.environ.get('MODE', 'ens')
NR = 1 if MODE == 'official' else R
THR = float(os.environ.get('THR', 0.35))
OUT_NPZ = os.environ.get('OUT_NPZ') or os.path.join(REPO, 'results', 'curve_availability', 'w53_%s.npz' % TAG)
OUT = os.environ.get('OUT') or os.path.join(REPO, 'results', 'curve_availability', 'w53_%s.json' % TAG)
loader = DataLoader(ds, batch_size=BS, shuffle=False, collate_fn=COLLATE, num_workers=8)
_t0 = time.time()

print('[GPU] %s' % torch.cuda.get_device_name(0), flush=True)
print('[BASE] %s split=%s N=%d num_props=%d L=%d BS=%d K_REP=%d MODE=%s R=%d WPT=%.4f' % (
    TAG, os.path.basename(dargs['test_data']), len(ds), P, L, BS, K_REP, MODE, NR, WPT), flush=True)


def nll_per_prop(out, bsz):
    Pu = out['words_logit'].size(0) // bsz
    wm = out['words_mask'].unsqueeze(1).expand(bsz, Pu, -1).contiguous().view(bsz * Pu, -1)
    wi = out['words_id'].unsqueeze(1).expand(bsz, Pu, -1).contiguous().view(bsz * Pu, -1)
    n, _ = cal_nll_loss(out['words_logit'], wi, wm)
    return n.view(bsz, Pu)


def run_rep(rep):
    np.random.seed(7 + rep); torch.manual_seed(7 + rep)
    model = PCNet(mcfg).cuda()
    model.load_state_dict(sp)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    _orig_pg = model.proposal_generator
    _c_pt = torch.linspace(0.5 / L, 1 - 0.5 / L, L)
    _w_pt = torch.full((L,), float(WPT))

    N8, C8, W8, G = [], [], [], []
    REL = [[] for _ in range(NR)]
    t0 = time.time()
    with torch.no_grad():
        for bi, batch in enumerate(loader, 1):
            net_input = {k: v.cuda() for k, v in batch['net_input'].items()}
            bsz = len(batch['raw'])
            if MODE == 'ens':
                # 让 base 前向的随机掩码只由 (rep, bi) 决定，与 R / WPT 无关
                # ⇒ base 在 R=1..5、各宽度之间完全一致，所有对照都是同一把尺子。
                np.random.seed(50000 + 100003 * rep + 13 * bi)
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

            for r in range(NR):
                if MODE == 'ens':
                    # 每次读出独立播种 ⇒ 独立随机词掩码；可复现
                    np.random.seed(90001 + 100003 * rep + 7919 * r)
                model.proposal_generator = lambda h, b: (_c_pt.repeat(b).cuda(), _w_pt.repeat(b).cuda())
                model.num_props = L
                o2 = model(epoch=0, **net_input)
                wm2 = o2['words_mask'].unsqueeze(1).expand(bsz, L, -1).contiguous().view(bsz * L, -1)
                wi2 = o2['words_id'].unsqueeze(1).expand(bsz, L, -1).contiguous().view(bsz * L, -1)
                n2, _ = cal_nll_loss(o2['words_logit'], wi2, wm2)
                REL[r].append((-n2.view(bsz, L)).cpu().numpy())
            model.proposal_generator = _orig_pg
            model.num_props = P

            if bi % 40 == 0:
                el = time.time() - t0
                print('    rep%d %d/%d %.1fs (eta %.1fs)' % (rep, bi, len(loader), el,
                      el / bi * (len(loader) - bi)), flush=True)

    G = np.concatenate(G); N8 = np.concatenate(N8); C8 = np.concatenate(C8); W8 = np.concatenate(W8)
    n = len(G); r = np.arange(n)
    GS, GE = G[:, 0], G[:, 1]
    j1 = np.argsort(N8, axis=1)[:, 0]
    cb, wb = C8[r, j1], W8[r, j1]
    rel = np.stack([np.concatenate(REL[i], 0) for i in range(NR)], 0)     # (NR, n, L)
    return rel, cb, wb, GS, GE, GE - GS


RL, CBL, WBL, GSL, GEL, GLL = [], [], [], [], [], []
for rep in range(K_REP):
    a, b, c, d, e, f = run_rep(rep)
    RL.append(a); CBL.append(b); WBL.append(c); GSL.append(d); GEL.append(e); GLL.append(f)
    print('  rep%d done %.1fs' % (rep, time.time() - _t0), flush=True)

rel = np.stack(RL, 0)                 # (K_REP, NR, N, L)
cb = np.stack(CBL, 0); wb = np.stack(WBL, 0)
gs = np.stack(GSL, 0); ge = np.stack(GEL, 0); gl = np.stack(GLL, 0)
np.savez_compressed(OUT_NPZ, rel=rel.astype(np.float32), cb=cb.astype(np.float32),
                    wb=wb.astype(np.float32), gs=gs.astype(np.float32),
                    ge=ge.astype(np.float32), gl=gl.astype(np.float32),
                    wpt=np.array([WPT]), mode=np.array([MODE]), nr=np.array([NR]))
print('WROTE %s' % OUT_NPZ, flush=True)
print('[N] N=%d K_REP=%d NR=%d med_wb=%.4f' % (cb.shape[1], K_REP, NR, float(np.mean([np.median(w) for w in wb]))), flush=True)

# ---------------- 噪声底噪（仅 MODE=ens） ----------------
NOISE = {}
if MODE == 'ens' and NR > 1:
    print('\n[噪声] 同一 (rep, batch) 下 R 次独立读出的曲线差异（rel 量纲 = -nll）')
    z = rel.astype(np.float64)
    sd_within = float(np.mean(z.std(0)))                     # 各样本跨 R 次读出的标准差
    sd_total = float(np.mean(z.reshape(-1, z.shape[-1]).std(0).mean()))
    sig = float(np.mean([np.abs(z[k].mean(0) - z[k].mean(1, keepdims=True)).mean() for k in range(K_REP)]))
    print('   样本级跨读出 std（噪声）= %.4f | 通道级 std = %.4f | 曲线内变程 = %.4f' % (sd_within, sd_total, sig))
    print('   ⇒ 信噪比 SNR = 曲线内变程 / 读出噪声 ≈ %.2f' % (sig / max(sd_within, 1e-9)))
    NOISE = dict(sd_within=sd_within, sd_total=sd_total, sig=sig, snr=sig / max(sd_within, 1e-9))


# ---------------- 就地评估：单次读出 vs 集成 ----------------
def iou(a0, a1, b0, b1):
    return ops.iou(a0, a1, b0, b1)


def ev(curve_list, wi_or_none):
    """curve_list: 每 rep 一个 [N,L] 曲线（已平均）。返回三列（逐 rep 自洽）。"""
    m, r5, r7 = [], [], []
    for k in range(K_REP):
        zc = curve_list[k]
        c2 = ops.refine_center(zc, cb[k], wb[k], CG, smooth=7, beta=0.20, rho=0.10, alpha=1.0, dmin=0.01)[:2]
        lo, hi = c2
        v = iou(lo, hi, gs[k], ge[k])
        m.append(v.mean()); r5.append((v >= 0.5).mean()); r7.append((v >= 0.7).mean())
    return 100 * float(np.mean(m)), 100 * float(np.mean(r5)), 100 * float(np.mean(r7))


BM, B5, B7 = [], [], []
for k in range(K_REP):
    v = iou(np.clip(cb[k] - wb[k] / 2, 0, 1), np.clip(cb[k] + wb[k] / 2, 0, 1), gs[k], ge[k])
    BM.append(v.mean()); B5.append((v >= 0.5).mean()); B7.append((v >= 0.7).mean())
BASE = (100 * float(np.mean(BM)), 100 * float(np.mean(B5)), 100 * float(np.mean(B7)))

print('\n' + '=' * 96)
print('[读出协议对照] %s   base(mIoU/R1@0.5/R1@0.7) = %.2f/%.2f/%.2f' % (TAG, *BASE))
print('%10s %10s %10s %10s %10s %10s' % ('读出方式', 'CSR_mIoU', 'CSR_r5', 'CSR_r7', 'ΔmIoU', 'Δr5'))
JC = {}
if MODE == 'official':
    mm = ev([rel[k, 0] for k in range(K_REP)], None)
    JC['official_R1'] = dict(mIoU=mm[0], r5=mm[1], r7=mm[2], dm=mm[0] - BASE[0])
    print('%10s %10.2f %10.2f %10.2f %+10.3f %+10.3f' % ('官方单读', mm[0], mm[1], mm[2], mm[0] - BASE[0], mm[1] - BASE[1]))
    a = REF_S1
    db = max(abs(BASE[0] - a['base'][0]), abs(BASE[1] - a['base'][1]), abs(BASE[2] - a['base'][2]))
    dc = max(abs(mm[0] - a['csr'][0]), abs(mm[1] - a['csr'][1]), abs(mm[2] - a['csr'][2]))
    print('   [锚点] base 期望 %.2f/%.2f/%.2f 最大差 %.3f %s | CSR 期望 %.2f/%.2f/%.2f 最大差 %.3f %s'
          % (*a['base'], db, 'OK' if db < 0.02 else 'DIFF', *a['csr'], dc, 'OK' if dc < 0.02 else 'DIFF'))
else:
    for r in range(NR):
        mm = ev([rel[k, r] for k in range(K_REP)], None)
        JC['single_r%d' % r] = dict(mIoU=mm[0], r5=mm[1], r7=mm[2], dm=mm[0] - BASE[0])
        print('%10s %10.2f %10.2f %10.2f %+10.3f %+10.3f' % ('单读 r=%d' % r, mm[0], mm[1], mm[2], mm[0] - BASE[0], mm[1] - BASE[1]))
    for Rn in range(2, NR + 1):
        mm = ev([rel[k, :Rn].mean(0) for k in range(K_REP)], None)
        JC['ens_R%d' % Rn] = dict(mIoU=mm[0], r5=mm[1], r7=mm[2], dm=mm[0] - BASE[0])
        print('%10s %10.2f %10.2f %10.2f %+10.3f %+10.3f' % ('集成 R=%d' % Rn, mm[0], mm[1], mm[2], mm[0] - BASE[0], mm[1] - BASE[1]))

json.dump(dict(tag=TAG, N=int(cb.shape[1]), K_REP=K_REP, MODE=MODE, R=NR, WPT=WPT,
               base=BASE, med_wb=float(np.mean([np.median(w) for w in wb])),
               noise=NOISE, proto=JC, npz=OUT_NPZ, secs=time.time() - _t0),
          open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
print('\nWROTE %s' % OUT)
print('DONE %.1f min' % ((time.time() - _t0) / 60))
