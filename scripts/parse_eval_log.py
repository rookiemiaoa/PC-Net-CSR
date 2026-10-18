# -*- coding: utf-8 -*-
"""parse_eval_log.py —— 把 eval_csr.py 的日志解析成门禁可读的 JSON

为什么要多这一层：门禁不应该手抄数字。日志是原始证据，JSON 是它的机器可读投影，
两者一起入库，`repro_gate.py --row 2` 才能既比对数值、又保留溯源。

用法
----
    python scripts/parse_eval_log.py results/main_table/eval_csr_pcnet_official.log
    python scripts/parse_eval_log.py <log> -o results/main_table/pcnet_eval_csr.json

输出字段
--------
    base / csr : [mIoU, R1@0.5, R1@0.7]（百分数）
    dm         : [ΔmIoU, ΔR1@0.5, ΔR1@0.7]（百分点）
    provenance : 日志路径、ckpt、N、K_REP、日志时间
"""
import argparse
import json
import os
import re
import sys


def parse(path):
    txt = open(path, encoding='utf-8', errors='replace').read()

    def mean_row(name):
        m = re.search(r'^%s\s+(\d+\.\d+)±\S+\s+(\d+\.\d+)±\S+\s+(\d+\.\d+)±' % re.escape(name),
                      txt, re.M)
        if not m:
            raise SystemExit('[FAIL] 找不到 %s 的 mean±std 行' % name)
        r5, r7, miou = (float(x) for x in m.groups())
        return [miou * 100, r5 * 100, r7 * 100]

    d = re.search(r'^CSR\s+([+-]\d+\.\d+)\s+([+-]\d+\.\d+)\s+([+-]\d+\.\d+)\s*$', txt, re.M)
    if not d:
        raise SystemExit('[FAIL] 找不到 CSR 的 Δ 行')
    d5, d7, dm = (float(x) for x in d.groups())

    prov = dict(log=os.path.basename(path))
    m = re.search(r'^ckpt=(.*)$', txt, re.M)
    if m:
        prov['ckpt'] = m.group(1).strip()
    m = re.search(r'N=(\d+)', txt)
    if m:
        prov['N'] = int(m.group(1))
    m = re.search(r'K_REP=(\d+)', txt)
    if m:
        prov['K_REP'] = int(m.group(1))
    m = re.search(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', txt, re.M)
    if m:
        prov['log_time'] = m.group(1)

    return dict(base=mean_row('base'), csr=mean_row('CSR'), dm=[dm, d5, d7], provenance=prov)


def main():
    ap = argparse.ArgumentParser(description='解析 eval_csr 日志为 JSON')
    ap.add_argument('log')
    ap.add_argument('-o', '--out', help='输出路径（默认与日志同目录同名 .json）')
    args = ap.parse_args()

    d = parse(args.log)
    out = args.out or os.path.splitext(args.log)[0] + '.json'
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(d, f, ensure_ascii=False, indent=1)
    print('base  mIoU=%.2f @.5=%.2f @.7=%.2f' % tuple(d['base']))
    print('CSR   mIoU=%.2f @.5=%.2f @.7=%.2f' % tuple(d['csr']))
    print('Δ     mIoU=%+.2f  @.5=%+.2f  @.7=%+.2f' % tuple(d['dm']))
    print('WROTE %s' % out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
