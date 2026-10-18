# -*- coding: utf-8 -*-
"""check_release_assets.py —— 校验发布附件与落盘位置

`checkpoints/WEIGHTS.md5` 用的是 `md5sum -c` 格式，要求文件**已经在最终路径**上；
但读者下载到的是一个平铺的目录。本脚本补上「下载目录 → 落盘路径」这一段：
按 `checkpoints/WEIGHTS.release.md5` 的附件名找到文件，校验 md5，再告诉你该放到哪。

用法
----
    python scripts/check_release_assets.py --list
    python scripts/check_release_assets.py --dir ~/Downloads/pcnet-release
    python scripts/check_release_assets.py --dir ~/Downloads/pcnet-release --copy --dest .

退出码：0 = 全部通过；1 = 有缺文件或 md5 不符。
"""
import argparse
import hashlib
import os
import shutil
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = _HERE
while ROOT != os.path.dirname(ROOT) and not os.path.exists(os.path.join(ROOT, 'models', 'pcnet.py')):
    ROOT = os.path.dirname(ROOT)

MANIFEST = os.path.join(ROOT, 'checkpoints', 'WEIGHTS.release.md5')


def parse(path=MANIFEST):
    out = []
    with open(path, encoding='utf-8') as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split('\t')
            if len(parts) != 3:
                raise SystemExit('[FAIL] %s:%d 列数不是 3：%r' % (path, ln, line))
            out.append(tuple(p.strip() for p in parts))
    return out


def md5_of(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description='校验 PC-Net-CSR 发布附件')
    ap.add_argument('--dir', help='附件下载目录（平铺）')
    ap.add_argument('--list', action='store_true', help='打印清单')
    ap.add_argument('--copy', action='store_true', help='校验通过后复制到 --dest 下的落盘路径')
    ap.add_argument('--dest', default='.', help='--copy 的目标根目录（默认当前目录）')
    args = ap.parse_args()

    rows = parse()
    if args.list or not args.dir:
        print('%-46s %-34s %s' % ('附件名', 'md5', '落盘路径'))
        for a, m, p in rows:
            print('%-46s %-34s %s' % (a, m, p))
        print('\n共 %d 个附件。' % len(rows))
        return 0

    fails = []
    for asset, want, rel in rows:
        src = os.path.join(args.dir, asset)
        if not os.path.exists(src):
            fails.append('缺文件 %s' % src)
            print('MISS  %s' % asset)
            continue
        got = md5_of(src)
        if got != want:
            fails.append('%s md5 不符：实测 %s 期望 %s' % (asset, got, want))
            print('FAIL  %s  md5=%s' % (asset, got))
            continue
        print('OK    %s' % asset)
        if args.copy:
            dst = os.path.join(args.dest, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            print('      -> %s' % dst)

    print('')
    if fails:
        print('CHECK FAIL (%d 项)' % len(fails))
        for s in fails:
            print('  - ' + s)
        return 1
    print('CHECK PASS（%d/%d 个附件 md5 全部一致）' % (len(rows), len(rows)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
