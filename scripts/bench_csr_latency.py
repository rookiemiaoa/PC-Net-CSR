# -*- coding: utf-8 -*-
"""复测 CSR 的 CPU 后处理耗时（论文 §3.3 / §4.6 的 0.0216 ms/查询）。

口径
----
* 论文那三个数（0.0216 / 6.22 / 8.70 ms）出自 2026-09-14 的 w31 测量，
  运行环境 RTX 2080 Ti、N=512、BS=16、L=50，原始产物已归档为
  ``results/latency/w31_{gpu_20260914.log,charades_20260914.json}``：
      GPU 前向 top-1     6.2246 ms/查询
      GPU 前向 50 点曲线  8.6978 ms/查询
      CPU 后处理 CSR      0.021623 ms/查询
* 本脚本只复测**最后一项**（纯 numpy，与显卡无关），用于确认它不随机器漂移；
  前两项是 GPU 实测，换机器必然变，不在本脚本范围内。

用法
----
    python scripts/bench_csr_latency.py --check
"""
import argparse
import json
import os
import pickle
import statistics
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(_HERE)
sys.path.insert(0, REPO)

from csr import ops                  # noqa: E402
from csr.paths import CACHE_DIR      # noqa: E402

CACHE = os.path.join(CACHE_DIR, 'charades_test_trivial_r3.pkl')
ARCHIVE = os.path.join(REPO, 'results', 'latency', 'w31_charades_20260914.json')
PAPER_MS = 0.0216
BS = 16
WARMUP = 20
REPS = 300
ROUNDS = 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--json', default=None)
    args = ap.parse_args()

    D = pickle.load(open(CACHE, 'rb'))[0]
    Z, cb, wb = D['Z'], D['cb'], D['wb']
    centers = (np.arange(Z.shape[1]) + 0.5) / Z.shape[1]

    hp = ops.DEFAULT_CSR_HYPERPARAMS
    print('CPU 后处理复测：BS=%d, warmup=%d, reps=%d, rounds=%d' % (BS, WARMUP, REPS, ROUNDS))
    print('超参：smooth=%d beta=%.2f rho=%.2f alpha=%.2f dmin=%.2f' % (
        hp['smooth'], hp['beta'], hp['rho'], hp['alpha'], hp['dmin']))

    idx = np.arange(BS)
    for _ in range(WARMUP):
        ops.refine_center(Z[idx], cb[idx], wb[idx], centers[:Z.shape[1]])
    for _ in range(WARMUP):
        ops.refine_center(Z[idx], cb[idx], wb[idx], centers[:Z.shape[1]])

    rounds = []
    for _ in range(ROUNDS):
        t0 = time.perf_counter()
        for _ in range(REPS):
            ops.refine_center(Z[idx], cb[idx], wb[idx], centers[:Z.shape[1]])
        dt = time.perf_counter() - t0
        rounds.append(dt / (REPS * BS) * 1e3)      # ms / query

    med = statistics.median(rounds)
    print('本机：%.4f ms/查询（各轮 %s）' % (med, ', '.join('%.4f' % r for r in rounds)))
    print('论文：%.4f ms/查询（2026-09-14, RTX 2080 Ti 主机）' % PAPER_MS)
    print('偏差：%+.1f%%' % (100.0 * (med - PAPER_MS) / PAPER_MS))

    arch = None
    if os.path.exists(ARCHIVE):
        arch = json.load(open(ARCHIVE))
        print('归档：base %.4f ms | dense %.4f ms | CSR-cpu %.6f ms' % (
            arch['ms_per_query_base'], arch['ms_per_query_dense'],
            arch['ms_cpu_per_query']['CSR']))

    out = {'cache': CACHE, 'batch': BS, 'reps': REPS, 'rounds': ROUNDS,
           'hyperparams': hp, 'local_ms_per_query_median': med,
           'local_ms_per_query_all_rounds': rounds,
           'paper_ms_per_query': PAPER_MS,
           'archived_2026_09_14': arch,
           'note': 'CPU 后处理与显卡无关；6.22 / 8.70 ms 是 GPU 前向实测，见归档 JSON'}
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        json.dump(out, open(args.json, 'w'), indent=2, ensure_ascii=False)
        print('wrote %s' % args.json)

    # 判据：同一数量级且偏差 < 50%（本地 CPU 与服务器 CPU 不同型号，不要求逐位）
    ok = abs(med - PAPER_MS) / PAPER_MS < 0.5
    print('结论：%s（同数量级判据 ±50%%）' % ('OK' if ok else 'DIFF'))
    return 0 if (not args.check or ok) else 1


if __name__ == '__main__':
    sys.exit(main())
