# 发布检查清单（匿名版 → 实名版）

本仓库的公开副本当前处于**双盲审稿模式**：版权人为 `Anonymous Authors`，
提交元数据已统一改写为匿名地址，正文与文档中不含作者单位、姓名与联系邮箱。
录用后按下列步骤恢复实名。**本文件本身不记录真实姓名**，实名信息另存于仓库外的作者私有记录。

---

## 1. camera-ready 时必须改动的位置

| # | 文件 | 位置 | 匿名版现值 | 需改为 |
|---|---|---|---|---|
| 1 | `LICENSE` | 第 3 行 | `Copyright (c) 2026 Anonymous Authors` | 真实版权人 + 单位 |
| 2 | `NOTICE` | 第 2 行 | 同上 | 同上 |
| 3 | `README.md` | §8 引用块 | `author = {{Anonymous Authors}}`、`journal = {Under Review}` | 真实作者列表与正式出处 |
| 4 | `README.md` | §9 提示 | 「双盲审稿期间版权人写作 …」 | 删除该提示段 |
| 5 | `README.md` | §10 | 远端占位 `<user>/<repo>` | 正式组织/仓库名 |
| 6 | `docs/RELEASE_CHECKLIST.md` | 本文件 | —— | 可选：保留为发布流程说明 |

## 2. 提交元数据

历史提交的作者与邮箱在匿名化时已被 `git filter-repo --mailmap` 统一改写。
恢复实名**不建议**再重写历史（会改变所有 commit hash，破坏审稿期引用）。
推荐做法：从 camera-ready 提交起，用真实 `user.name` / `user.email` 正常提交，历史保留匿名前缀。

```bash
git config user.name  "<real name>"
git config user.email "<real email>"
git commit --allow-empty -m "chore: restore author identity for camera-ready"
```

## 3. 仓库与链接

```bash
# ① 建正式远端（公开仓库）
git remote add origin https://github.com/<org>/<repo>.git
git push -u origin main
git push origin --tags

# ② 权重作为 release asset 上传（不入库）
#    checkpoints/**/model-best.pt 共 7 个，合计约 170 MB
```

- 论文摘要/脚注中的匿名链接替换为正式仓库地址。
- 若审稿期使用的是 anonymous.4open.science 镜像，录用后该镜像可下线（其内容随上游更新失效）。

## 4. 发布前自检

```bash
# 体积与残留
du -sh .git
git rev-list --objects --all | grep -c '\.pt$'      # 期望 0
git grep -n -F -e '<WS_BASE>' -e '10.102.64.'    # 期望无输出

# 许可链完整
test -f NOTICE && test -f LICENSE && head -3 models/modules/net_vlad.py

# 隐私（实名版跳过本项）
git log --format='%an <%ae>' | sort -u
```

## 5. 权限与合规

- Apache-2.0 组件（`models/modules/net_vlad.py`）保留原始许可证头，`NOTICE` 随之分发。
- fairseq 派生文件（`optimizers/`）保留 Facebook 版权头，`NOTICE` 已列明。
- 上游 PC-Net 代码的版权归原作者，不得改写其许可证声明。
