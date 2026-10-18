# -*- coding: utf-8 -*-
"""render_table4.py —— 从 results/table4/*.json 渲染论文表 4

表 4 对比同一个骨干在**训练划分**与**测试划分**上的曲线方向准确率 `dir_acc`，
用来支撑「判据 a 不是拟合测试集」这一论点：两列都在 a* 同一侧分裂。

列定义
------
  训练列 = curve.dir_acc_c1，切分 = train.json / cg_train.json（N=8281），K_REP=1
  测试列 = curve.dir_acc_c1，切分 = test_trivial.json（N=3096），      K_REP=3
  两列都是 WPT=0.1、BS=16、DDIR=0.08，且都由 csr/analysis/w34_curve_usability.py 产出。

口径
----
  训练列的 K_REP=1 是**协议决定的**，不是省算力：探针只用于量骨干自身的曲线方向，
  重复取样只会改变随机掩码、不改变结论；测试列才用 K_REP=3 做噪声平均。
  两者不可互换，也不可用训练列去顶替测试列。

用法
----
    python scripts/render_table4.py            # 打印 + 写 results/tables/table4.{md,csv}
    python scripts/render_table4.py --check    # 只比对期望值（容差 2e-3）并跑断言
    python scripts/render_table4.py --stdout   # 只打印

退出码：0 = 通过；1 = 有格子或多条断言不符。
"""
import argparse
import csv
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = _HERE
while ROOT != os.path.dirname(ROOT) and not os.path.exists(os.path.join(ROOT, 'models', 'pcnet.py')):
    ROOT = os.path.dirname(ROOT)

T4 = os.path.join(ROOT, 'results', 'table4')
OUTDIR = os.path.join(ROOT, 'results', 'tables')

# 判据门限（论文 §3.4 / 图 4）
A_STAR = 0.543
# 训练列与测试列的允许偏差：同一骨干的两条切分，若差太多说明切分串了
MAX_TRAIN_TEST_GAP = 0.05
# PC-Net 相对其余三个骨干的最小领先幅度（防"擦边通过"）
MIN_LEAD = 0.15

# (显示名, 骨干 key, 论文表 4 的 (训练列, 测试列))
ROWS = [
    ('PC-Net', 'pcnet', (0.742, 0.776)),
    ('CPL',    'cpl',   (0.508, 0.510)),
    ('QMN',    'qmn',   (0.491, 0.472)),
    ('CCR',    'ccr',   (0.453, 0.447)),
]
COLS = ['dir_acc(训练)', 'dir_acc(测试)', '差']
TOL = 2e-3

# 切分与采样口径的硬约束（A1/A6 用）
SPLIT_PROBE = {'train.json', 'cg_train.json'}
SPLIT_TEST = {'test_trivial.json', 'cg_test_trivial.json'}
N_PROBE, N_TEST = 8281, 3096
KREP_PROBE, KREP_TEST = 1, 3
WPT = 0.1


def _load(fn):
    p = os.path.join(T4, fn)
    if not os.path.exists(p):
        raise SystemExit('[FAIL] 缺文件：%s' % p)
    with open(p, encoding='utf-8') as f:
        d = json.load(f)
    return d


def load_row(key):
    """返回 (train_json, test_json)。两个文件都必须存在且自洽。"""
    return _load('w61_%s_TT_probe.json' % key), _load('w61_%s_TT_test.json' % key)


def fmt(col, v):
    if col == '差':
        return '%+.4f' % v
    return '%.4f' % v


def main():
    ap = argparse.ArgumentParser(description='渲染/校验论文表 4')
    ap.add_argument('--check', action='store_true', help='只比对期望值与断言，不写文件')
    ap.add_argument('--stdout', action='store_true', help='只打印，不写文件')
    args = ap.parse_args()

    table, fails = [], []
    probes, tests, acc = {}, {}, {}

    for name, key, exp in ROWS:
        dtr, dte = load_row(key)
        tr = float(dtr['curve']['dir_acc_c1'])
        te = float(dte['curve']['dir_acc_c1'])
        probes[name], tests[name] = dtr, dte
        acc[name] = (tr, te)
        table.append((name, (tr, te, tr - te)))

        for c, v, e in zip(COLS[:2], (tr, te), exp):
            if abs(v - e) > TOL:
                fails.append('%s / %s: 实测 %.6f  论文 %.3f  (Δ=%+.6f)' % (name, c, v, e, v - e))

    # ---------- 打印 ----------
    head = '| 骨干 | ' + ' | '.join(COLS) + ' |'
    sep = '|' + '---|' + '---:|' * len(COLS)
    print(head)
    print(sep)
    for name, vals in table:
        print('| %s | %s |' % (name, ' | '.join(fmt(c, v) for c, v in zip(COLS, vals))))

    # ---------- 断言 A1–A7 ----------
    print('')

    # A1 切分文件与样本数没有串
    ok = True
    for name in probes:
        dtr, dte = probes[name], tests[name]
        if dtr.get('split') not in SPLIT_PROBE:
            ok = False
            fails.append('A1 %s 训练列切分异常：%r' % (name, dtr.get('split')))
        if dte.get('split') not in SPLIT_TEST:
            ok = False
            fails.append('A1 %s 测试列切分异常：%r' % (name, dte.get('split')))
        if dtr.get('N') != N_PROBE or dte.get('N') != N_TEST:
            ok = False
            fails.append('A1 %s 样本数异常：probe=%s test=%s（应 %d/%d）'
                         % (name, dtr.get('N'), dte.get('N'), N_PROBE, N_TEST))
    print('%s A1 切分与样本数：probe ∈ %s 且 N=%d；test ∈ %s 且 N=%d'
          % ('PASS' if ok else 'FAIL', sorted(SPLIT_PROBE), N_PROBE, sorted(SPLIT_TEST), N_TEST))

    # A2 同一骨干两列不得相距过远（否则是切分串了或模型不同）
    gaps = {n: acc[n][0] - acc[n][1] for n in acc}
    worst = max(gaps, key=lambda k: abs(gaps[k]))
    ok = abs(gaps[worst]) <= MAX_TRAIN_TEST_GAP
    if not ok:
        fails.append('A2 %s 训练/测试差 %.4f 超过 %.2f' % (worst, gaps[worst], MAX_TRAIN_TEST_GAP))
    print('%s A2 两列同源：max|train−test| = %.4f (%s) ≤ %.2f'
          % ('PASS' if ok else 'FAIL', abs(gaps[worst]), worst, MAX_TRAIN_TEST_GAP))

    # A3 训练列与测试列都必须围绕 a* 正确分裂
    for label, col in (('训练列', 0), ('测试列', 1)):
        p = acc['PC-Net'][col]
        others = [acc[n][col] for n in acc if n != 'PC-Net']
        ok = p > A_STAR and max(others) < A_STAR
        if not ok:
            fails.append('A3 %s 未在 a*=%g 处分裂：PC-Net %.4f，其余 max %.4f'
                         % (label, A_STAR, p, max(others)))
        print('%s A3 %s 阈值分裂：PC-Net %.4f > a*=%.3f > 其余 %.4f'
              % ('PASS' if ok else 'FAIL', label, p, A_STAR, max(others)))

    # A4 领先幅度不是擦边
    p = acc['PC-Net'][0]
    lead = p - max(acc[n][0] for n in acc if n != 'PC-Net')
    ok = lead >= MIN_LEAD
    if not ok:
        fails.append('A4 PC-Net 训练列领先仅 %.4f < %.2f' % (lead, MIN_LEAD))
    print('%s A4 领先幅度：PC-Net − max(其余) = %.4f ≥ %.2f'
          % ('PASS' if ok else 'FAIL', lead, MIN_LEAD))

    # A5 采样口径
    ok = True
    for name in probes:
        dtr, dte = probes[name], tests[name]
        if dtr.get('K_REP') != KREP_PROBE or dte.get('K_REP') != KREP_TEST:
            ok = False
            fails.append('A5 %s K_REP 异常：probe=%s test=%s' % (name, dtr.get('K_REP'), dte.get('K_REP')))
        if abs(float(dtr.get('WPT', -1)) - WPT) > 1e-9 or abs(float(dte.get('WPT', -1)) - WPT) > 1e-9:
            ok = False
            fails.append('A5 %s WPT 异常：probe=%s test=%s（应 %.2f）'
                         % (name, dtr.get('WPT'), dte.get('WPT'), WPT))
    print('%s A5 采样口径：K_REP=%d/%d，WPT=%.2f'
          % ('PASS' if ok else 'FAIL', KREP_PROBE, KREP_TEST, WPT))

    # A6 每个 JSON：ΔmIoU 必须等于 CSR − base（防手改）
    ok = True
    for name in probes:
        for d in (probes[name], tests[name]):
            r = d['rows']
            if abs((r['CSR']['mIoU'] - r['base']['mIoU']) - r['CSR']['dm']) > 1e-6:
                ok = False
                fails.append('A6 %s/%s: rows.CSR.dm 自洽残差超限' % (name, d.get('tag')))
    print('%s A6 自洽：8 个 JSON 的 CSR.dm == CSR.mIoU − base.mIoU' % ('PASS' if ok else 'FAIL'))

    # A7 达标骨干至少占一个：不能四行全在线下
    ok = sum(1 for n in acc if acc[n][0] > A_STAR) >= 1
    if not ok:
        fails.append('A7 训练列没有任何骨干高于 a*')
    print('%s A7 非退化：训练列至少一个骨干 > a*' % ('PASS' if ok else 'FAIL'))

    if args.check or args.stdout:
        print('')
        if fails:
            print('CHECK FAIL (%d 项)' % len(fails))
            for s in fails:
                print('  - ' + s)
            return 1
        print('CHECK PASS（容差 %g，%d 行 × %d 列；断言 A1–A7 全过）'
              % (TOL, len(table), 2))
        return 0

    os.makedirs(OUTDIR, exist_ok=True)
    md = os.path.join(OUTDIR, 'table4.md')
    cv = os.path.join(OUTDIR, 'table4.csv')
    with open(md, 'w', encoding='utf-8') as f:
        f.write('<!-- 由 scripts/render_table4.py 生成，勿手改 -->\n')
        f.write('<!-- 表 4 训练/测试划分上的曲线方向准确率；两列共用同一 a* -->\n\n')
        f.write(head + '\n' + sep + '\n')
        for name, vals in table:
            f.write('| %s | %s |\n' % (name, ' | '.join(fmt(c, v) for c, v in zip(COLS, vals))))
        f.write('\n口径：两列都是 WPT=0.1、BS=16；训练列 K_REP=1（N=8281），测试列 K_REP=3（N=3096）。\n')
        f.write('a* = %.3f。训练列不是测试列的重跑，两列不可互相顶替。\n' % A_STAR)
    with open(cv, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['backbone', 'dir_acc_train', 'dir_acc_test', 'gap'])
        for name, vals in table:
            w.writerow([name] + ['%.6f' % v for v in vals])
    print('')
    print('WROTE %s' % md)
    print('WROTE %s' % cv)
    return 0


if __name__ == '__main__':
    sys.exit(main())
