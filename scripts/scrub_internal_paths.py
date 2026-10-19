# -*- coding: utf-8 -*-
"""发布前清理：把证据文件里的开发机绝对路径换成占位符。

只替换 ``/Disks/DiskC/<user>/`` 这一层（保留其后的相对结构），因此
``/Disks/DiskC/fzc/ws_base2/CCR/...`` -> ``<DATA_ROOT>/ws_base2/CCR/...``。
用户名与内网目录布局不会出现在公开仓库里，证据的结构信息仍然完整。

用法
----
    python scripts/scrub_internal_paths.py --check     # 只报告，退出码 1 表示仍有残留
    python scripts/scrub_internal_paths.py --apply
"""
import argparse
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(_HERE)

PATTERN = re.compile(r'/Disks/DiskC/[^/\s"\']+/')
REPL = '<DATA_ROOT>/'
TEXT_EXT = ('.json', '.log', '.md', '.txt', '.py', '.sh', '.csv', '.yaml', '.yml')
SKIP_DIRS = {'.git', '__pycache__', 'feat_cache'}
SKIP_FILES = {'scrub_internal_paths.py'}      # 本文件自身含正则字面量，跳过


def walk():
    for root, dirs, files in os.walk(REPO):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for f in files:
            if f.endswith(TEXT_EXT) and f not in SKIP_FILES:
                yield os.path.join(root, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true', help='只检查（默认行为）')
    ap.add_argument('--apply', action='store_true')
    args = ap.parse_args()

    hits = []
    for p in walk():
        try:
            s = open(p, encoding='utf-8').read()
        except (UnicodeDecodeError, OSError):
            continue
        n = len(PATTERN.findall(s))
        if n:
            hits.append((p, n))
            if args.apply:
                open(p, 'w', encoding='utf-8', newline='').write(PATTERN.sub(REPL, s))

    for p, n in hits:
        print('%4d  %s' % (n, os.path.relpath(p, REPO)))
    print('---- 共 %d 个文件、%d 处' % (len(hits), sum(n for _, n in hits)))
    if args.apply:
        print('已替换为 %s' % REPL)
        return 0
    return 1 if hits else 0


if __name__ == '__main__':
    sys.exit(main())
