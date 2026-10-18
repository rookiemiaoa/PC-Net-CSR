# -*- coding: utf-8 -*-
"""check_weight_equiv.py —— 证明发布权重与论文权重是同一模型

问题
  论文表 3 / 表 4 的 PC-Net 行由开发机上的 ``checkpoints/CharadesSTA/official_8/``
  产出，而发布包提供的是 ``checkpoints/CharadesSTA/8_2025-05-14_13-26-12/``。
  ``official_8`` 已不在盘上，无法直接比 md5，因此"两者是否同一模型"必须用
  **行为等价**证明，而不是文件名相同。

判据
  同一套采集器（w34）在同一份数据上跑，若两个权重产生**逐位相同**的中心误差统计，
  则它们在中心轴上不可区分。要比较的量必须覆盖 CSR 全链路：
    mae_base      —— 基础中心误差（只依赖 forward 的 center/width）
    mae_c1        —— 曲线质心误差（依赖逐点相关性曲线）
    mae_final     —— 门控后输出误差（依赖完整五步）
    dir_acc_c1    —— 方向准确率
    dir_acc_final
  只要任一列不同，就说明权重或读取路径有差异。

  两个切分各比一次（train.json 与 test_trivial.json），因为表 4 的两列分别来自它们。

用法
  python scripts/check_weight_equiv.py [--tol 1e-12]
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, 'results')

# (切分名, official_8 产物, 8_2025-05-14 产物)
PAIRS = [
    ('train.json',
     os.path.join(RES, 'table4', 'w61_pcnet_TT_probe.json'),
     os.path.join(RES, 'main_table', 'pcnet_r4_train.json')),
    ('test_trivial.json',
     os.path.join(RES, 'table4', 'w61_pcnet_TT_test.json'),
     os.path.join(RES, 'curve_availability', 'w34_pcnet.json')),
]

KEYS = ('mae_base', 'mae_c1', 'mae_final', 'dir_acc_c1', 'dir_acc_final')


def load(p):
    with open(p, encoding='utf-8') as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tol', type=float, default=1e-12,
                    help='逐位判据的绝对容差（默认 1e-12，等价于要求完全相同）')
    a = ap.parse_args()

    print('权重行为等价性：official_8 vs 8_2025-05-14_13-26-12')
    print('=' * 78)
    bad = []
    for split, p_ref, p_new in PAIRS:
        if not os.path.exists(p_ref):
            print('[SKIP] %-18s 缺 %s' % (split, os.path.relpath(p_ref, ROOT)))
            bad.append('%s: 缺 official_8 产物' % split)
            continue
        if not os.path.exists(p_new):
            print('[SKIP] %-18s 缺 %s' % (split, os.path.relpath(p_new, ROOT)))
            bad.append('%s: 缺 8_2025-05-14 产物' % split)
            continue
        d_ref, d_new = load(p_ref), load(p_new)
        c_ref, c_new = d_ref['curve'], d_new['curve']
        print('\n[%s]' % split)
        print('  official_8      ckpt=%s  N=%d  K_REP=%d'
              % (os.path.basename(os.path.dirname(d_ref['ckpt'])), d_ref['N'], d_ref['K_REP']))
        print('  8_2025-05-14    ckpt=%s  N=%d  K_REP=%d'
              % (os.path.basename(os.path.dirname(d_new['ckpt'])), d_new['N'], d_new['K_REP']))
        if d_ref['N'] != d_new['N'] or d_ref['K_REP'] != d_new['K_REP']:
            bad.append('%s: N/K_REP 不一致' % split)
            print('  DIFF 采样口径不一致')
            continue
        for k in KEYS:
            x, y = float(c_ref[k]), float(c_new[k])
            ok = abs(x - y) <= a.tol
            print('  %-14s %-20.16f %-20.16f  %s'
                  % (k, x, y, 'OK' if ok else 'DIFF %.3e' % abs(x - y)))
            if not ok:
                bad.append('%s/%s %.16f != %.16f' % (split, k, x, y))

    print('\n' + '=' * 78)
    if bad:
        print('CHECK FAIL（%d 项）' % len(bad))
        for b in bad[:8]:
            print('  - ' + b)
        return 1
    print('CHECK PASS：2 切分 × 5 指标，全部逐位相同（容差 %.0e）' % a.tol)
    print('⇒ official_8 与 8_2025-05-14_13-26-12 在 CSR 全链路上不可区分，')
    print('   发布包提供的权重可替代论文所用权重。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
