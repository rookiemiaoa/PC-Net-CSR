# -*- coding: utf-8 -*-
"""w62 —— 超参网格的跨划分迁移性（§4.3「900 组」那一段的唯一生成器）。

来源
----
本文件是开发期 ``scan_center_dev.py`` 的仓库内正式版本，逻辑逐行保持等价，
只把数据路径改成 ``csr.paths.CACHE_DIR`` 解析，并新增 ``--json`` 落盘。
2026-10-19 在发布缓存上重跑，得到 217 / 97 / 81 / 37.3%，与手稿逐位一致。

网格
----
``smooth`` 3 × ``beta`` 4 × ``rho`` 5 × ``alpha`` 5 × ``dmin`` 3 = **900** 组。

达标判据（``is_ok``）
---------------------
整体三项都为正、短时刻三项损失不超过 0.2/0.3/0.2 pt、长时刻 @0.5 ≥ +2.0、@0.7 ≥ +5.0。

用法
----
    python csr/analysis/w62_grid_transfer.py
    python csr/analysis/w62_grid_transfer.py --json results/grid_transfer/grid_900.json
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = _HERE
while REPO != os.path.dirname(REPO) and not os.path.exists(os.path.join(REPO, 'models', 'pcnet.py')):
    REPO = os.path.dirname(REPO)
sys.path.insert(0, REPO)

from csr.paths import CACHE_DIR  # noqa: E402

L = 50
cg = (np.arange(L) + 0.5) / L
THR = 0.35


def iou(a0, a1, b0, b1):
    inter = np.maximum(0.0, np.minimum(a1, b1) - np.maximum(a0, b0))
    union = np.maximum(a1, b1) - np.minimum(a0, b0)
    return np.where(union > 1e-9, inter / np.maximum(union, 1e-9), 0.0)


def met(i):
    return dict(mIoU=float(i.mean()), i5=float((i >= 0.5).mean()), i7=float((i >= 0.7).mean()))


def smoothZ(Z, w):
    if w <= 1:
        return Z
    pad = w // 2
    Zp = np.concatenate([np.repeat(Z[:, :1], pad, axis=1), Z,
                         np.repeat(Z[:, -1:], pad, axis=1)], axis=1)
    cs = np.concatenate([np.zeros((len(Z), 1)), np.cumsum(Zp, axis=1)], axis=1)
    return (cs[:, w:] - cs[:, :-w]) / float(w)


def inside_score(Z, lo, hi):
    cs = np.concatenate([np.zeros((len(Z), 1)), np.cumsum(Z, axis=1)], axis=1)
    li = np.clip(np.searchsorted(cg, lo), 0, L - 1)
    hii = np.clip(np.searchsorted(cg, hi, side='right') - 1, 0, L - 1)
    cnt = np.maximum(hii - li + 1, 1)
    tot = cs[np.arange(len(Z)), hii + 1] - cs[np.arange(len(Z)), li]
    return tot / cnt


def est_win_vec(Z, cb, beta):
    m = np.abs(cg[None, :] - cb[:, None]) <= beta
    P = np.where(m, np.maximum(Z, 0.0), 0.0)
    s = P.sum(1)
    c = (P @ cg) / np.maximum(s, 1e-12)
    bad = s < 1e-12
    if bad.any():
        c[bad] = cb[bad]
    return c


def eval_csr(D, smooth, beta, rho, alpha, dmin):
    Z = smoothZ(D['Z'], smooth)
    cb, wb = D['cb'], D['wb']
    c1 = est_win_vec(Z, cb, beta)
    d = c1 - cb
    d = np.where(np.abs(d) < rho * wb, 0.0, d)
    c2 = np.clip(cb + alpha * d, 0, 1)
    if dmin is not None:
        din = inside_score(Z, c2 - wb / 2, c2 + wb / 2) - inside_score(Z, D['bL'], D['bH'])
        c2 = np.where(din >= dmin, c2, cb)
    return iou(np.clip(c2 - wb / 2, 0, 1), np.clip(c2 + wb / 2, 0, 1), D['GS'], D['GE'])


def agg(DL, fn, bucket):
    vals = []
    for D in DL:
        i = fn(D)
        SH = D['GL'] <= THR
        sel = slice(None) if bucket == 'all' else (SH if bucket == 'short' else ~SH)
        vals.append(met(i[sel]))
    return {k: float(np.mean([v[k] for v in vals])) for k in ('mIoU', 'i5', 'i7')}


def table(DL, fn):
    return {b: agg(DL, fn, b) for b in ('all', 'short', 'long')}


def delta(cur, base):
    return tuple(100 * (cur[b][k] - base[b][k])
                 for b, k in (('all', 'i5'), ('all', 'mIoU'), ('all', 'i7'),
                              ('short', 'i5'), ('short', 'i7'), ('short', 'mIoU'),
                              ('long', 'i5'), ('long', 'i7')))


def score(dd):
    return (min(dd[0], 0) * 3 + min(dd[1], 0) * 3 + min(dd[2], 0) * 3
            + min(dd[3] + 0.2, 0) * 3 + min(dd[4] + 0.3, 0) * 3 + min(dd[5] + 0.2, 0) * 3
            + min(dd[6] - 2.0, 0) * 2 + min(dd[7] - 5.0, 0) * 2
            + dd[0] + dd[1] + dd[2] + 0.5 * dd[7])


def is_ok(dd):
    return (dd[1] >= 0 and dd[0] > 0 and dd[2] > 0 and dd[3] >= -0.2
            and dd[4] >= -0.3 and dd[5] >= -0.2 and dd[6] >= 2.0 and dd[7] >= 5.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', default=None)
    args = ap.parse_args()

    TRAIN = pickle.load(open(os.path.join(CACHE_DIR, 'charades_train_r1.pkl'), 'rb'))
    TEST = pickle.load(open(os.path.join(CACHE_DIR, 'charades_test_trivial_r3.pkl'), 'rb'))

    base_tr = table(TRAIN, lambda D: iou(D['bL'], D['bH'], D['GS'], D['GE']))
    base_te = table(TEST, lambda D: iou(D['bL'], D['bH'], D['GS'], D['GE']))
    print('train N=%d  test N=%d' % (TRAIN[0]['n'], TEST[0]['n']))

    GRID = []
    for smooth in (3, 5, 7):
        for beta in (0.18, 0.20, 0.22, 0.25):
            for rho in (0.10, 0.15, 0.20, 0.25, 0.30):
                for alpha in (0.7, 0.8, 0.85, 0.9, 1.0):
                    for dmin in (None, 0.0, 0.01):
                        GRID.append((smooth, beta, rho, alpha, dmin))
    print('参数网格 = %d' % len(GRID))

    res = []
    for (sm, be, rho, al, dm) in GRID:
        fn = (lambda D, sm=sm, be=be, rho=rho, al=al, dm=dm: eval_csr(D, sm, be, rho, al, dm))
        d_tr = delta(table(TRAIN, fn), base_tr)
        res.append((score(d_tr), (sm, be, rho, al, dm), d_tr))
    res.sort(key=lambda x: -x[0])

    print('\n' + '=' * 110)
    print('[A] train.json（开发集）上 top 10（按达标分）')
    print('=' * 110)
    for sc, prm, dd in res[:10]:
        print('  sm%d beta%.2f rho%.2f a%.2f dmin=%s  Δtr ALL(%+.2f,%+.2f,%+.2f) 短(%+.2f,%+.2f,%+.2f) 长(@.5%+.2f,@.7%+.2f)' % (
            prm[0], prm[1], prm[2], prm[3], prm[4], dd[1], dd[0], dd[2], dd[5], dd[3], dd[4], dd[6], dd[7]))

    print('\n' + '=' * 110)
    print('[B] 用 train 选出的 top5 参数，原样搬到 test_trivial')
    print('=' * 110)
    for sc, prm, dd_tr in res[:5]:
        fn = (lambda D, p=prm: eval_csr(D, *p))
        dd_te = delta(table(TEST, fn), base_te)
        print('  sm%d beta%.2f rho%.2f a%.2f dmin=%s  %s' % (
            prm[0], prm[1], prm[2], prm[3], prm[4], '★达标' if is_ok(dd_te) else '未达标'))
        print('        Δte ALL(%+.2f,%+.2f,%+.2f) 短(%+.2f,%+.2f,%+.2f) 长(@.5%+.2f,@.7%+.2f)' % (
            dd_te[1], dd_te[0], dd_te[2], dd_te[5], dd_te[3], dd_te[4], dd_te[6], dd_te[7]))

    print('\n' + '=' * 110)
    print('[C] 迁移性：train 与 test 各自最优参数是否一致')
    print('=' * 110)
    res_te = []
    for (sm, be, rho, al, dm) in GRID:
        fn = (lambda D, sm=sm, be=be, rho=rho, al=al, dm=dm: eval_csr(D, sm, be, rho, al, dm))
        d_te = delta(table(TEST, fn), base_te)
        res_te.append((score(d_te), (sm, be, rho, al, dm), d_te))
    res_te.sort(key=lambda x: -x[0])

    ok_tr = [r for r in res if is_ok(r[2])]
    ok_te = [r for r in res_te if is_ok(r[2])]
    print('  train 达标参数数 = %d / %d' % (len(ok_tr), len(res)))
    print('  test  达标参数数 = %d / %d' % (len(ok_te), len(res_te)))
    s_te = set(x[1] for x in ok_te)
    inter = [x for x in ok_tr if x[1] in s_te]
    pct = 100.0 * len(inter) / max(len(ok_tr), 1)
    print('  交集（train 达标 且 test 达标）= %d 组   → 占 train 达标集合的 %.1f%%' % (len(inter), pct))

    print('\n  [C2] 只用 train 信息选参的 6 种准则，各自落到 test 的结果：')
    print('       （★=test 也达标；这是"参数选择合规性"的直接检验）')
    te_map = {r[1]: r[2] for r in res_te}
    criteria = [
        ('S1 score 最大', lambda r: -r[0]),
        ('S2 短@0.7 增益最大', lambda r: -r[2][4]),
        ('S3 短 mIoU 增益最大', lambda r: -r[2][5]),
        ('S4 短@0.5 增益最大', lambda r: -r[2][3]),
        ('S5 beta 最小（最保守）', lambda r: (r[1][1],)),
        ('S6 rho 最大（最保守）', lambda r: (-r[1][2],)),
    ]
    picks = {}
    for nm, key in criteria:
        cand = sorted(ok_tr, key=key)
        if not cand:
            print('  %-22s （train 无达标参数）' % nm)
            continue
        prm = cand[0][1]
        dd = te_map[prm]
        picks[nm] = {'params': list(prm), 'test_pass': bool(is_ok(dd)),
                     'delta_test': [round(x, 4) for x in dd]}
        print('  %-22s → sm%d beta%.2f rho%.2f a%.2f dmin=%s   %s' % (
            nm, prm[0], prm[1], prm[2], prm[3], prm[4], '★达标' if is_ok(dd) else '未达标'))
        print('  %-22s   Δte ALL(%+.2f,%+.2f,%+.2f) 短(%+.2f,%+.2f,%+.2f) 长(@.5%+.2f,@.7%+.2f)' % (
            '', dd[1], dd[0], dd[2], dd[5], dd[3], dd[4], dd[6], dd[7]))

    cov = [te_map[x[1]] for x in ok_tr]
    n_ok = sum(1 for d in cov if is_ok(d))
    print('\n  [C3] train 达标集合（%d 组）中，test 也达标的比例 = %.1f%%' % (len(ok_tr), 100.0 * n_ok / len(ok_tr)))
    s7 = [d[4] for d in cov]
    print('       test 短@0.7 增益在这些参数上的分布：中位 %+.2fpt，均值 %+.2fpt，最小 %+.2fpt，最大 %+.2fpt'
          % (np.median(s7), np.mean(s7), np.min(s7), np.max(s7)))
    l7 = [d[7] for d in cov]
    print('       test 长@0.7 增益分布：            中位 %+.2fpt，均值 %+.2fpt，最小 %+.2fpt，最大 %+.2fpt'
          % (np.median(l7), np.mean(l7), np.min(l7), np.max(l7)))

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        json.dump({
            'grid_size': len(GRID),
            'grid_points': {'smooth': [3, 5, 7], 'beta': [0.18, 0.20, 0.22, 0.25],
                            'rho': [0.10, 0.15, 0.20, 0.25, 0.30],
                            'alpha': [0.7, 0.8, 0.85, 0.9, 1.0],
                            'dmin': [None, 0.0, 0.01]},
            'train_n': int(TRAIN[0]['n']), 'test_n': int(TEST[0]['n']),
            'train_pass': len(ok_tr), 'test_pass': len(ok_te),
            'intersection': len(inter), 'intersection_pct_of_train_pass': round(pct, 1),
            'selection_criteria': picks,
        }, open(args.json, 'w'), indent=2, ensure_ascii=False)
        print('wrote %s' % args.json)

    return 0


if __name__ == '__main__':
    sys.exit(main())
