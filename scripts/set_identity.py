#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把仓库从「双盲匿名模式」一键切到「实名发布模式」。

对应 docs/RELEASE_CHECKLIST.md 第 1 节的 6 项，外加论文摘要里的仓库链接。

用法::

    python scripts/set_identity.py --dry-run \\
        --name "Zhichao Feng" --email "you@example.edu.cn" --account "your-github-id"

    # 确认无误后去掉 --dry-run 真正落盘
    python scripts/set_identity.py \\
        --name "Zhichao Feng" --email "you@example.edu.cn" --account "your-github-id" \\
        --affiliation "Central China Normal University" \\
        --docx ../PC-Net-CSR_CVPR_Manuscript_CN_v13.docx

脚本只做**纯文本替换**，每处替换都断言命中次数；命中次数不符即中止并回滚
（不会写出一半）。不做 git commit —— 提交留给你自己，因为提交身份要绑定
你本机的 user.name / user.email。
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# (相对路径, 旧串, 新串模板, 期望命中次数)
# 模板里的 {name} {email} {account} {affiliation} 由命令行填入。
RULES = [
    ("LICENSE",
     "Copyright (c) 2026 Anonymous Authors",
     "Copyright (c) 2026 {name}{aff_suffix}", 1),
    ("NOTICE",
     "Copyright (c) 2026 Anonymous Authors",
     "Copyright (c) 2026 {name}{aff_suffix}", 1),
    ("CITATION.cff",
     '  - name: "Anonymous Authors"',
     '  - name: "{name}"', 1),
    ("CITATION.cff",
     'repository-code: "https://github.com/<user>/PC-Net-CSR"',
     'repository-code: "https://github.com/{account}/PC-Net-CSR"', 1),
    ("README.md",
     "<https://github.com/<user>/PC-Net-CSR/releases/tag/v1.0-repro>",
     "<https://github.com/{account}/PC-Net-CSR/releases/tag/v1.0-repro>", 1),
    ("README.md",
     "  author  = {{Anonymous Authors}},",
     "  author  = {{{name}}},", 1),
    ("README.md",
     "> 双盲期间作者与刊物字段留空；投稿 / camera-ready 时按 `docs/RELEASE_CHECKLIST.md` 恢复。\n",
     "", 1),
    ("README.md",
     "> 双盲投稿期间版权人写作 `Anonymous Authors`，提交邮箱亦已替换为匿名地址。\n"
     "> 请使用匿名镜像仓库（如 anonymous.4open.science），不要直接给出带作者姓名的 GitHub 地址；\n"
     "> camera-ready 后按 `docs/RELEASE_CHECKLIST.md` 恢复实名。\n\n",
     "", 1),
    ("README.md",
     "# ② 接上远端（把 <user> 换成你的账号）\n"
     "git remote add origin https://github.com/<user>/PC-Net-CSR.git",
     "# ② 接上远端\n"
     "git remote add origin https://github.com/{account}/PC-Net-CSR.git", 1),
]

RESIDUAL = re.compile(r"Anonymous Authors|<user>|__ACCOUNT__")
# 本脚本自身含有这些字面量（替换规则里），扫描时跳过
SKIP_REL = {"scripts/set_identity.py", "docs/RELEASE_CHECKLIST.md"}


def _fmt(tpl: str, ns: dict) -> str:
    return tpl.format(**ns)


def patch_texts(ns: dict, dry: bool, log) -> bool:
    """返回 True 表示全部规则命中次数正确。"""
    buffers: dict[str, str] = {}
    ok = True
    for rel, old, new_tpl, expect in RULES:
        path = os.path.join(ROOT, rel)
        if rel not in buffers:
            if not os.path.isfile(path):
                log("  MISS  {} 不存在".format(rel))
                ok = False
                continue
            with open(path, "r", encoding="utf-8", newline="") as fh:
                buffers[rel] = fh.read()
        text = buffers[rel]
        new = _fmt(new_tpl, ns)
        got = text.count(old)
        if got != expect:
            log("  BAD   {} 命中 {} 次，期望 {} 次 :: {}".format(
                rel, got, expect, old.splitlines()[0][:60]))
            ok = False
            continue
        buffers[rel] = text.replace(old, new)
        log("  OK    {}  x{}".format(rel, expect))

    if not ok:
        log("\n有规则未命中，**未写入任何文件**。请检查仓库当前内容是否已被部分修改。")
        return False

    for rel, text in buffers.items():
        if dry:
            continue
        with open(os.path.join(ROOT, rel), "w", encoding="utf-8", newline="") as fh:
            fh.write(text)
    return True


def patch_docx(path: str, ns: dict, dry: bool, log) -> bool:
    """把 docx 摘要里的 __ACCOUNT__ 换成真实账号（单 run 内替换，不动 rPr）。"""
    if not os.path.isfile(path):
        log("  MISS  docx 不存在：{}".format(path))
        return False
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        raw = {n: zf.read(n) for n in names}
    xml = raw["word/document.xml"].decode("utf-8")
    n = xml.count("__ACCOUNT__")
    if n == 0:
        log("  SKIP  docx 里没有 __ACCOUNT__（可能已回填过）")
        return True
    xml2 = xml.replace("__ACCOUNT__", ns["account"])
    log("  OK    docx  __ACCOUNT__ x{} -> {}".format(n, ns["account"]))
    if dry:
        return True
    tmp = path + ".tmp"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in names:
            data = xml2.encode("utf-8") if name == "word/document.xml" else raw[name]
            zf.writestr(name, data)
    shutil.move(tmp, path)
    return True


def residual_report(log) -> int:
    hits = 0
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in {".git", "__pycache__"}]
        for fn in filenames:
            if not fn.endswith((".md", ".cff", ".py", ".sh", ".txt")) and fn not in {
                    "LICENSE", "NOTICE"}:
                continue
            p = os.path.join(dirpath, fn)
            rel = os.path.relpath(p, ROOT).replace(os.sep, "/")
            if rel in SKIP_REL:
                continue
            try:
                with open(p, "r", encoding="utf-8", errors="ignore") as fh:
                    for i, line in enumerate(fh, 1):
                        if RESIDUAL.search(line):
                            hits += 1
                            log("  残留 {}:{}: {}".format(
                                os.path.relpath(p, ROOT), i, line.strip()[:80]))
            except OSError:
                pass
    return hits


def main() -> int:
    ap = argparse.ArgumentParser(description="匿名模式 -> 实名发布模式")
    ap.add_argument("--name", required=True, help="真实姓名（英文，如 Zhichao Feng）")
    ap.add_argument("--email", required=True, help="联系邮箱（写入 git 配置，不写进 LICENSE）")
    ap.add_argument("--account", required=True, help="GitHub 账号名")
    ap.add_argument("--affiliation", default="", help="单位（可选，写进 LICENSE/NOTICE 版权行）")
    ap.add_argument("--docx", default="", help="可选：论文 docx 路径，回填 __ACCOUNT__")
    ap.add_argument("--dry-run", action="store_true", help="只打印，不写盘")
    ap.add_argument("--set-git", action="store_true",
                    help="同时写入本仓库的 git user.name / user.email")
    args = ap.parse_args()

    ns = {
        "name": args.name.strip(),
        "email": args.email.strip(),
        "account": args.account.strip().lstrip("@"),
        "affiliation": args.affiliation.strip(),
        "aff_suffix": ""
        if not args.affiliation.strip()
        else ", " + args.affiliation.strip(),
    }

    lines: list[str] = []
    log = lines.append

    log("目标仓库：{}".format(ROOT))
    log("身份：{} <{}>  GitHub: {}".format(ns["name"], ns["email"], ns["account"]))
    if args.affiliation.strip():
        log("单位：{}".format(ns["affiliation"].strip()))
    log("模式：{}".format("DRY-RUN（不写盘）" if args.dry_run else "写入"))
    log("")
    log("—— 文本替换 ——")
    ok = patch_texts(ns, args.dry_run, log)

    if ok and args.docx:
        log("")
        log("—— 手稿 ——")
        ok = patch_docx(args.docx, ns, args.dry_run, log)

    if ok and not args.dry_run:
        log("")
        log("—— 残留自检 ——")
        hits = residual_report(log)
        if hits == 0:
            log("  0 条匿名 / 占位符残留")
        else:
            log("  ⚠ 仍有 {} 条，逐条人工确认（部分可能是文档说明文字，属正常）".format(hits))

    if ok and args.set_git and not args.dry_run:
        os.system('git -C "{}" config user.name "{}"'.format(ROOT, ns["name"]))
        os.system('git -C "{}" config user.email "{}"'.format(ROOT, ns["email"]))
        log("")
        log("git user.name / user.email 已写入仓库本地配置")

    print("\n".join(lines))

    if not ok:
        return 1

    print("")
    print("=" * 62)
    if args.dry_run:
        print("DRY-RUN 通过。去掉 --dry-run 重新执行即可落盘。")
    else:
        print("已改写。下一步：")
        print('  git add -A && git commit -m "chore: restore author identity for public release"')
        print("  git remote add origin https://github.com/{}/PC-Net-CSR.git".format(ns["account"]))
        print("  git push -u origin main")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    sys.exit(main())
