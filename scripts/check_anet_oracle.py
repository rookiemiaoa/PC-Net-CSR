# -*- coding: utf-8 -*-
"""check_anet_oracle.py —— 校验 §3.1 的 ANET oracle 上界日志

论文 §3.1 引用 `+12.2`，它来自 `csr/analysis/diag41_anet_oracle.py` 的
`ORACLE_c` 行（ΔmIoU）。本脚本把「读日志、对数字」变成一条命令。

判据（与 docs/REPRODUCE_FULL.md 的容差一致）
    ΔmIoU ∈ 12.216 ± 0.05            -> PASS
    0.05 < |Δ−12.216| ≤ 0.5          -> WARN（仍可用，正文写 +12.2，README 记区间）
    0.5  < |Δ−12.216| ≤ 2.0          -> FAIL（先查特征溯源与 ckpt md5）
    > 2.0 或 base 偏离 >0.10          -> STOP

用法
----
    python scripts/check_anet_oracle.py results/oracle_bounds/diag41_anet_oracle_rerun_20261018.log

退出码：0 = PASS/WARN；1 = FAIL/STOP；2 = 日志解析失败。
"""
import argparse
import os
import re
import sys

ANET_BASE = 0.3239
ANET_DM = 12.216


def parse(path):
    txt = open(path, encoding='utf-8', errors='replace').read()
    base = re.search(r'^base\s+(\d\.\d+)±\S+\s+(\d\.\d+)±\S+\s+(\d\.\d+)±', txt, re.M)
    delta = re.search(r'^ORACLE_c\s+([+-]\d+\.\d+)\s+([+-]\d+\.\d+)\s+([+-]\d+\.\d+)\s*$', txt, re.M)
    if not base or not delta:
        return None
    out = dict(base_miou=float(base.group(3)) * 100,
               base_r5=float(base.group(1)) * 100,
               base_r7=float(base.group(2)) * 100,
               dm=float(delta.group(3)),
               d5=float(delta.group(1)),
               d7=float(delta.group(2)))
    m = re.search(r'N=(\d+)', txt)
    if m:
        out['N'] = int(m.group(1))
    m = re.search(r'K_REP=(\d+)', txt)
    if m:
        out['K_REP'] = int(m.group(1))
    m = re.search(r'ckpt=(\S+)', txt)
    if m:
        out['ckpt'] = m.group(1)
    return out


def main():
    ap = argparse.ArgumentParser(description='校验 ANET oracle 日志')
    ap.add_argument('log')
    args = ap.parse_args()
    if not os.path.exists(args.log):
        print('MISSING %s' % args.log)
        return 2
    d = parse(args.log)
    if not d:
        print('PARSE FAIL：日志里找不到 base / ORACLE_c 行')
        return 2

    print('N=%s  K_REP=%s  ckpt=%s' % (d.get('N', '?'), d.get('K_REP', '?'), d.get('ckpt', '?')))
    print('base     mIoU=%.2f  @.5=%.2f  @.7=%.2f' % (d['base_miou'], d['base_r5'], d['base_r7']))
    print('ORACLE_c ΔmIoU=%+.4f  Δ@.5=%+.4f  Δ@.7=%+.4f' % (d['dm'], d['d5'], d['d7']))

    problems = []
    if d.get('N') != 15712:
        problems.append('N=%s（应 15712）' % d.get('N'))
    if d.get('K_REP') != 3:
        problems.append('K_REP=%s（应 3）' % d.get('K_REP'))
    dev = abs(d['dm'] - ANET_DM)
    bdev = abs(d['base_miou'] - ANET_BASE * 100)
    if bdev > 10.0:
        print('STOP  base 偏离 %.2f pt（>10）——不是同一份特征，不得改论文' % bdev)
        return 1
    if dev <= 0.05:
        print('PASS  Δ=%+.2f → 正文 +12.2（偏差 %.3f）' % (d['dm'], dev))
    elif dev <= 0.5:
        print('WARN  Δ=%+.2f（偏差 %.3f）仍可写 +12.2，但需在 README 记录' % (d['dm'], dev))
    elif dev <= 2.0:
        print('FAIL  Δ=%+.2f（偏差 %.3f）先查特征溯源与 ckpt md5' % (d['dm'], dev))
    else:
        print('STOP  Δ=%+.2f（偏差 %.3f）停止改动，升级根因调查' % (d['dm'], dev))
    if problems:
        print('  另：' + '; '.join(problems))
        return 1
    return 0 if dev <= 0.5 else 1


if __name__ == '__main__':
    sys.exit(main())
