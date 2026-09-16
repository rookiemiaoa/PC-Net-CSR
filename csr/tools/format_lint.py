# -*- coding: utf-8 -*-
"""静态检查 % 格式串的占位符数量是否与参数数量一致。

用法: python format_lint.py [文件或目录 ...]
      默认检查当前目录下所有 diag*.py

为什么需要它: print('%-28s | %+7.2f %s' % (a, b)) 这种错在语法上完全合法,
py_compile / ast.parse 都发现不了, 只在运行到那一行时才抛
TypeError: must be real number, not str —— 往往在实验跑完几分钟后才炸。
本脚本用 AST 遍历所有 `'...' % (...)` 表达式, 静态比对两侧数量。
"""
import ast
import os
import re
import sys

# 匹配一个 % 占位符: %(name)s / %5.1f / %-28s / %+7.2f / %% (字面百分号)
PH = re.compile(r'%(?:\([^)]*\))?[-+ #0]*\d*(?:\.\d+)?[diouxXeEfFgGcrsa%]')


def count_placeholders(fmt: str):
    """返回 (占位符个数, 字面%%个数)。%% 不算占位符。"""
    n = 0
    literal = 0
    for m in PH.finditer(fmt):
        if m.group(0) == '%%':
            literal += 1
        else:
            n += 1
    return n, literal


def check_file(path):
    src = open(path, encoding='utf-8').read()
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return [('SYNTAX', e.lineno or 0, str(e))]
    problems = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Mod):
            continue
        left = node.left
        if not isinstance(left, ast.Constant) or not isinstance(left.value, str):
            continue
        fmt = left.value
        nph, _ = count_placeholders(fmt)
        if nph == 0:
            continue
        right = node.right
        if isinstance(right, ast.Tuple):
            if any(isinstance(e, ast.Starred) for e in right.elts):
                continue  # 含 *args, 数量动态, 跳过
            narg = len(right.elts)
            shapes = [type(e).__name__ for e in right.elts]
        else:
            narg = 1
            shapes = [type(right).__name__]
        if nph != narg:
            # right 是元组字面量 -> 数量可静态确定, 是真 bug
            # right 是单个表达式 -> 可能是 tuple 变量, 无法静态判定, 仅告警
            kind = 'COUNT_MISMATCH' if isinstance(right, ast.Tuple) else 'WARN_UNKNOWN'
            problems.append((
                kind,
                node.lineno,
                '占位符=%d 参数=%d  fmt=%r  args=%s' % (nph, narg, fmt[:72], shapes),
            ))
    return problems


def main():
    targets = sys.argv[1:]
    if not targets:
        targets = sorted(f for f in os.listdir('.') if f.startswith('diag') and f.endswith('.py'))
    files = []
    for t in targets:
        if os.path.isdir(t):
            files += [os.path.join(t, f) for f in sorted(os.listdir(t)) if f.endswith('.py')]
        else:
            files.append(t)
    total = 0
    for f in files:
        probs = check_file(f)
        for kind, ln, msg in probs:
            if kind == 'COUNT_MISMATCH':
                total += 1
            print('[%s] %s:%d  %s' % (kind, f, ln, msg))
    print('---')
    print('检查 %d 个文件, 发现 %d 个格式串问题' % (len(files), total))
    return 1 if total else 0


if __name__ == '__main__':
    sys.exit(main())
