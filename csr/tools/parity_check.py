# -*- coding: utf-8 -*-
"""parity_check.py —— CSR / ISR 实现的逐元素对拍（纯 CPU，不需要 GPU 与 hdf5）

用途
----
仓库里同一个算子有多份历史实现（`csr/ops.py`、`csr/eval/eval_final.py`、
`scripts/s1_seed_ops.py`、`csr/analysis/diag41_anet_oracle.py`）。本工具用
`csr/analysis/feat_cache/` 里已入库的真实曲线，把「权威实现」与「冻结参考实现」
逐元素对拍，从而在**不跑任何模型**的前提下证明：

  1) `ops.refine_center_c2` 与论文证据路径（diag41 口径）算出同一个中心；
  2) `ops.refine_center` 的 `lo/hi` 就是 `clip(c2 ∓ wb/2)`（即返回值顺序没错位）；
  3) `ops.isr_shift` 与 `eval_final.py::isr_delta`（重构前的冻结快照）给出同一个 d。

第 3 条是「把 ISR 收归 ops.py」这次重构的唯一证据来源。若 `ops.isr_shift` 尚不存在，
该项会打印 SKIP 而不是 FAIL。

用法
----
    python csr/tools/parity_check.py                  # 默认读 charades_test_trivial_r3.pkl
    python csr/tools/parity_check.py --tol 1e-12
    python csr/tools/parity_check.py --cache csr/analysis/feat_cache/charades_train_r1.pkl
    python csr/tools/parity_check.py --json out.json

退出码：0 = 全部通过；1 = 有断言失败；2 = 环境/缓存不可用。
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = _HERE
while _ROOT != os.path.dirname(_ROOT) and not os.path.exists(os.path.join(_ROOT, 'models', 'pcnet.py')):
    _ROOT = os.path.dirname(_ROOT)
sys.path.insert(0, _ROOT)

from csr import ops                                   # noqa: E402
from csr.paths import CACHE_DIR                       # noqa: E402

DEFAULT_CACHE = os.path.join(CACHE_DIR, 'charades_test_trivial_r3.pkl')
L = 50
CG = (np.arange(L) + 0.5) / L


# ---------------------------------------------------------------- 冻结参考实现
# 以下两份实现是 2026-10-17 重构前的逐字快照，**故意不复用 ops 的流水线**，
# 只用 ops 的无状态原语（zscore / smooth_curve / 质心 / 框内均值），
# 以便把「流水线接错」这类 bug 暴露出来（eval_anet_support 就是接错了）。

def _ref_csr_center(z_raw, cb, wb, smooth=7, beta=0.20, rho=0.10, alpha=1.0, dmin=0.01):
    """diag41_anet_oracle.py::csr_center 的等价实现（返回 c2，中间量不 clip）。"""
    zs = ops.smooth_curve(ops.zscore(z_raw), smooth)
    c1 = ops.estimate_window_center(zs, cb, beta, CG)
    d = c1 - np.asarray(cb)
    d = np.where(np.abs(d) < rho * np.asarray(wb), 0.0, d)
    c2 = np.clip(np.asarray(cb) + alpha * d, 0, 1)
    if dmin is not None:
        din = (ops.inside_score(zs, c2 - wb / 2, c2 + wb / 2, CG)
               - ops.inside_score(zs, np.asarray(cb) - wb / 2, np.asarray(cb) + wb / 2, CG))
        c2 = np.where(din >= dmin, c2, cb)
    return c2


def _ref_isr_delta(z_raw, cb, wb, beta0=0.20, gamma=0.5, T=2,
                   rho=0.10, alpha=1.0, dmin=0.01):
    """eval_final.py::isr_delta 的逐字快照（返回位移 d，不返回中心）。"""
    zs = ops.smooth_curve(ops.zscore(z_raw), 7)
    c = np.asarray(cb).copy()
    b = beta0
    for _ in range(T):
        c = ops.estimate_window_center(zs, c, b, CG)
        b = b * gamma
    d = c - np.asarray(cb)
    d = np.where(np.abs(d) < rho * np.asarray(wb), 0.0, d) * alpha
    c2 = np.clip(np.asarray(cb) + d, 0, 1)
    if dmin is not None:
        din = (ops.inside_score(zs, c2 - wb / 2, c2 + wb / 2, CG)
               - ops.inside_score(zs, np.asarray(cb) - wb / 2, np.asarray(cb) + wb / 2, CG))
        d = np.where(din >= dmin, d, 0.0)
    return d


# ---------------------------------------------------------------- 对拍器

def _maxdiff(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.max(np.abs(a - b))) if a.size else 0.0


def check_csr(rep, tol):
    """返回 (rows, ok)。rows = [(name, maxdiff, detail)]"""
    rel, cb, wb = rep['REL'], rep['cb'], rep['wb']
    rows = []

    ref_c2 = _ref_csr_center(rel, cb, wb)
    ref_lo = np.clip(ref_c2 - wb / 2, 0, 1)
    ref_hi = np.clip(ref_c2 + wb / 2, 0, 1)

    # 1) 权威 4 元组接口的边界必须与参考中心自洽
    lo, hi, moved, shift = ops.refine_center(rel, cb, wb, CG)
    rows.append(('refine_center lo vs ref', _maxdiff(lo, ref_lo), 'n=%d' % len(cb)))
    rows.append(('refine_center hi vs ref', _maxdiff(hi, ref_hi), 'n=%d' % len(cb)))

    ref_moved = np.abs(ref_c2 - cb) > 1e-9
    ref_shift = np.abs(ref_c2 - cb)
    rows.append(('refine_center moved', float(np.sum(moved != ref_moved)), 'mismatch 个数'))
    rows.append(('refine_center shift', _maxdiff(shift, ref_shift), 'n=%d' % len(cb)))

    # 2) 若已提供 refine_center_c2（B3 之后），中心必须逐元素相等
    if hasattr(ops, 'refine_center_c2'):
        c2, mv2, sh2 = ops.refine_center_c2(rel, cb, wb, CG)
        rows.append(('refine_center_c2 vs ref', _maxdiff(c2, ref_c2), 'n=%d' % len(cb)))
        rows.append(('refine_center_c2 moved', float(np.sum(mv2 != ref_moved)), 'mismatch 个数'))
    else:
        rows.append(('refine_center_c2 vs ref', None, 'SKIP: ops.refine_center_c2 尚不存在'))

    # 3) 历史 bug 的复现性检查：把它当反例，确认它确实不等（防止有人改回去）
    legacy_c2 = ops.refine_center(rel, cb, wb, CG)[0]          # 曾经的错误用法
    legacy_gap = _maxdiff(legacy_c2, ref_c2)
    rows.append(('legacy_[0]的偏离(应>0)', legacy_gap,
                 '若为 0 说明该缓存无边界裁剪、发现不了这个 bug'))

    ok = all((d is None) or (d <= tol) for _, d, _ in rows[:5])
    ok = ok and all((d is None) or (d <= tol) for n, d, _ in rows if n.startswith('refine_center_c2'))
    return rows, ok


def check_isr(rep, tol):
    rel, cb, wb = rep['REL'], rep['cb'], rep['wb']
    rows = []
    if not hasattr(ops, 'isr_shift'):
        rows.append(('isr_shift vs ref', None, 'SKIP: ops.isr_shift 尚不存在（B4 未落地）'))
        return rows, True, True

    ok = True
    for gamma, T in ((0.50, 2), (0.70, 3)):
        ref = _ref_isr_delta(rel, cb, wb, 0.20, gamma, T)
        got = ops.isr_shift(rel, cb, wb, CG, beta0=0.20, gamma=gamma, T=T)
        d = _maxdiff(got, ref)
        rows.append(('isr_shift (γ=%.2f,T=%d)' % (gamma, T), d,
                     'n=%d 非零位移率=%.3f' % (len(cb), float(np.mean(np.abs(ref) > 0)))))
        ok = ok and (d <= tol)
    return rows, ok, False


def main():
    ap = argparse.ArgumentParser(description='CSR / ISR 实现逐元素对拍（CPU）')
    ap.add_argument('--cache', default=DEFAULT_CACHE, help='feat_cache 的 pkl 路径')
    ap.add_argument('--tol', type=float, default=1e-12, help='允许的最大绝对差（默认 1e-12）')
    ap.add_argument('--reps', type=int, default=0, help='只检查前 N 次重复（0=全部）')
    ap.add_argument('--json', default=None, help='把结果写成 JSON')
    args = ap.parse_args()

    if not os.path.exists(args.cache):
        print('[FAIL] 找不到缓存：%s' % args.cache)
        print('       该 pkl 已入库，请确认工作目录就是仓库根目录。')
        return 2

    with open(args.cache, 'rb') as f:
        reps = pickle.load(f)
    if isinstance(reps, dict):
        reps = [reps]
    if args.reps:
        reps = reps[:args.reps]

    print('缓存  : %s' % args.cache)
    print('重复  : %d  样本/重复: %d  容差: %.1e' % (len(reps), reps[0]['n'], args.tol))
    print('=' * 78)

    report = {'cache': args.cache, 'tol': args.tol, 'reps': len(reps), 'checks': []}
    all_ok = True
    isr_ok = True
    isr_skipped = False

    for i, rep in enumerate(reps):
        rows, ok = check_csr(rep, args.tol)
        all_ok = all_ok and ok
        print('[CSR] rep %d' % i)
        for name, d, detail in rows:
            if d is None:
                print('   %-30s SKIP    %s' % (name, detail))
            else:
                flag = 'OK  ' if d <= args.tol else 'FAIL'
                if name.startswith('legacy_'):
                    flag = 'INFO'          # 期望 >0 的反例，不参与判定
                print('   %-30s %-6s max|Δ|=%.3e  %s' % (name, flag, d, detail))
            report['checks'].append({'rep': i, 'op': 'CSR', 'check': name,
                                     'maxdiff': d, 'detail': detail,
                                     'pass': (d is None) or (d <= args.tol)})
        irows, iok, iskip = check_isr(rep, args.tol)
        isr_ok = isr_ok and iok
        isr_skipped = isr_skipped or iskip
        print('[ISR] rep %d' % i)
        for name, d, detail in irows:
            if d is None:
                print('   %-32s SKIP    %s' % (name, detail))
            else:
                flag = 'OK  ' if d <= args.tol else 'FAIL'
                print('   %-32s %-6s max|Δ|=%.3e  %s' % (name, flag, d, detail))
            report['checks'].append({'rep': i, 'op': 'ISR', 'check': name,
                                     'maxdiff': d, 'detail': detail,
                                     'pass': (d is None) or (d <= args.tol)})
        print('-' * 78)

    if isr_skipped:
        print('[NOTE] ops.isr_shift 不存在，ISR 项整体 SKIP（重构完成后再跑一次即为门禁）。')

    report['ok'] = bool(all_ok and isr_ok)
    if args.json:
        with open(args.json, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print('WROTE %s' % args.json)

    print('=' * 78)
    print('RESULT: %s' % ('PASS' if report['ok'] else 'FAIL'))
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
