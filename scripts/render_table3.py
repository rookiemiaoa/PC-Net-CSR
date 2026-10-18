# -*- coding: utf-8 -*-
"""render_table3.py —— 从 results/curve_availability/*.json 渲染论文表 3

表 3（五个骨干上的等后处理对比，Charades-CG Test-Trivial，N=3096，K_REP=3）
的每一格都能溯源到仓库内的 5 个 JSON；但此前**没有渲染器**，读者只能手抄。
本脚本把它变成可执行的一步。

列定义
------
  ρ_geo     = 基础框宽中位数 / 目标时长中位数          -> json["rho_geo"]
  ΔMAE      = 窗内质心估计器带来的中心 MAE 变化        -> curve.dmae_c1
  dir_acc   = 骨干自身曲线的方向准确率                 -> curve.dir_acc_c1
  ORACLE_c  = 只把中心换成真值的上界（保宽）           -> rows.ORACLE_c.mIoU
  ORACLE_w  = 只把宽度换成真值的上界（保中心）         -> rows.ORACLE_w.mIoU
  ΔmIoU     = CSR − base                              -> rows.CSR.dm

用法
----
    python scripts/render_table3.py            # 打印 + 写 results/tables/table3.{md,csv}
    python scripts/render_table3.py --check    # 只与论文值比对（容差 5e-3），不写文件
    python scripts/render_table3.py --stdout   # 只打印，不落盘

退出码：0 = 通过；1 = --check 有格子不符。
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

CA = os.path.join(ROOT, 'results', 'curve_availability')
OUTDIR = os.path.join(ROOT, 'results', 'tables')

# (显示名, json 文件, 论文表 3 的期望值)
# 期望值用于 --check：ρ_geo / ΔMAE / dir_acc / ORACLE_c / ORACLE_w / ΔmIoU
ROWS = [
    ('PC-Net', 'w34_pcnet.json', (1.3465, -0.02114, 0.7760, 70.37, 46.22, 1.05)),
    ('CNM',    'w36_cnm.json',   (1.7253, -0.02290, 0.6999, 64.98, 35.52, 3.00)),
    ('CPL',    'w34_cpl.json',   (1.3380,  0.01270, 0.5096, 69.17, 43.93, -3.76)),
    ('QMN',    'w34_qmn.json',   (1.3262,  0.01281, 0.4717, 70.04, 44.59, -3.53)),
    ('CCR',    'w34_ccr.json',   (1.3306,  0.01329, 0.4467, 71.10, 45.37, -3.18)),
]
COLS = ['ρ_geo', 'ΔMAE', 'dir_acc', 'ORACLE_c', 'ORACLE_w', 'ΔmIoU']
TOL = 5e-3


def load(fn):
    p = os.path.join(CA, fn)
    if not os.path.exists(p):
        raise SystemExit('[FAIL] 缺文件：%s' % p)
    with open(p, encoding='utf-8') as f:
        d = json.load(f)
    rows = d['rows']
    vals = (
        float(d['rho_geo']),
        float(d['curve']['dmae_c1']),
        float(d['curve']['dir_acc_c1']),
        float(rows['ORACLE_c']['mIoU']),
        float(rows['ORACLE_w']['mIoU']),
        float(rows['CSR']['dm']),
    )
    # 自洽断言：ΔmIoU 必须等于 CSR − base
    gap = abs((rows['CSR']['mIoU'] - rows['base']['mIoU']) - rows['CSR']['dm'])
    meta = {'N': d.get('N'), 'K_REP': d.get('K_REP'), 'WPT': d.get('WPT'),
            'split': d.get('split'), 'tag': d.get('tag'), 'selfconsistent_gap': gap}
    return vals, meta


def fmt(col, v):
    if col in ('ORACLE_c', 'ORACLE_w', 'ΔmIoU'):
        return '%+.2f' % v if col == 'ΔmIoU' else '%.2f' % v
    if col == 'dir_acc':
        return '%.4f' % v
    return '%+.4f' % v if col == 'ΔMAE' else '%.4f' % v


def main():
    ap = argparse.ArgumentParser(description='渲染/校验论文表 3')
    ap.add_argument('--check', action='store_true', help='只比对期望值，不写文件')
    ap.add_argument('--stdout', action='store_true', help='只打印，不写文件')
    ap.add_argument('--cnm', default='w36_cnm.json',
                    help='CNM 行使用的产物文件名（重训后指向 w36_cnm.retrained.json；'
                         '此时该行与论文值的偏离记为 revised，不计 FAIL）')
    args = ap.parse_args()
    is_retrained = (args.cnm != 'w36_cnm.json')

    table, metas, fails, revised = [], [], [], []
    for name, fn, exp in ROWS:
        f_use = args.cnm if name == 'CNM' else fn
        vals, meta = load(f_use)
        metas.append((name, meta))
        table.append((name, vals))
        for c, v, e in zip(COLS, vals, exp):
            if abs(v - e) > TOL:
                msg = '%s / %s: 实测 %.6f  论文 %.6f  (Δ=%.6f)' % (name, c, v, e, v - e)
                if name == 'CNM' and is_retrained:
                    revised.append(msg)
                else:
                    fails.append(msg)

    # ---- 打印
    head = '| 骨干 | ' + ' | '.join(COLS) + ' |'
    sep = '|' + '---|' + '---:|' * len(COLS)
    print(head)
    print(sep)
    for name, vals in table:
        print('| %s | %s |' % (name, ' | '.join(fmt(c, v) for c, v in zip(COLS, vals))))

    print('')
    for name, m in metas:
        print('[meta] %-7s N=%s K_REP=%s WPT=%s split=%s  dm 自洽残差=%.2e'
              % (name, m['N'], m['K_REP'], m['WPT'], m['split'], m['selfconsistent_gap']))
        if m['selfconsistent_gap'] > 1e-6:
            fails.append('%s: rows.CSR.dm 与 (CSR−base) 不一致' % name)

    if args.check or args.stdout:
        print('')
        if revised:
            print('[revised] CNM 行来自重训（%s），与论文值的偏离不作为 FAIL：' % args.cnm)
            for s in revised:
                print('  - ' + s)
        if fails:
            print('CHECK FAIL (%d 项)' % len(fails))
            for s in fails:
                print('  - ' + s)
            return 1
        print('CHECK PASS（容差 %g，%d 行 × %d 列）' % (TOL, len(table), len(COLS)))
        if args.stdout or args.check:
            return 0

    os.makedirs(OUTDIR, exist_ok=True)
    md = os.path.join(OUTDIR, 'table3.md')
    cv = os.path.join(OUTDIR, 'table3.csv')
    with open(md, 'w', encoding='utf-8') as f:
        f.write('<!-- 由 scripts/render_table3.py 生成，勿手改 -->\n')
        f.write('<!-- 表 3 五个骨干上的等后处理对比；ORACLE_* 用了真值，只作上界 -->\n\n')
        f.write(head + '\n' + sep + '\n')
        for name, vals in table:
            f.write('| %s | %s |\n' % (name, ' | '.join(fmt(c, v) for c, v in zip(COLS, vals))))
        f.write('\n口径：同一检查点、同一条读出曲线、同一个 CSR（零调参），只有骨干不同。\n')
        f.write('ORACLE_c 只换中心、ORACLE_w 只换宽度，二者**不是** §3.1 的 +23.2/+12.2（不同评测路径）。\n')
    with open(cv, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['backbone'] + COLS)
        for name, vals in table:
            w.writerow([name] + ['%.6f' % v for v in vals])
    print('')
    print('WROTE %s' % md)
    print('WROTE %s' % cv)
    return 0


if __name__ == '__main__':
    sys.exit(main())
