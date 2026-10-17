# 上游溯源核验报告

**对象**：`PC-Net-master/`（桌面上的这个文件夹）
**核验时间**：2026-09-13
**结论**：**是 PC-Net 官方开源代码**，且与本文实验所用的官方版本**逐字节一致**，无第三方改动。

---

## 1. 结论

| 问题 | 结论 |
|---|---|
| 是否官方开源代码？ | **是**。官方仓库为 `https://github.com/mingyao1120/PC-Net` |
| 来源是否与官方版本一致？ | **一致**。与本文全部实验所用的官方副本逐字节相同（64/64 文件 md5 相同） |
| 是否有被改动过的痕迹？ | **无**。全部源码、配置、启动脚本、数据切分、官方评测日志、官方权重均未变 |
| 本文对 backbone 的改动量 | **零**。CSR 是推理期算子，仅在运行时 monkey-patch，不落盘改官方代码 |

---

## 2. 核验依据（四条独立证据链）

### 证据 A —— 官方出处可查，且目录清单完全对应

- NeurIPS 2025 论文官方页面（`papers.neurips.cc` / OpenReview `kQAnOaayIo`）在正文中明确写明：
  > "Code is available at **https://github.com/mingyao1120/PC-Net**."
- 该 GitHub 仓库的顶层清单为
  `checkpoints/ config/ data/ datasets/ method_figure/ models/ optimizers/ runners/ scripts/ README.md train.py utils.py vocab.py`
  —— 与本地文件夹**逐项一致，不多不少**。
- 文件夹名 `PC-Net-master` 是 GitHub 下载默认分支 zip 包时自动生成的名字（`<repo>-<branch>`），
  即该文件夹是一次干净的分支导出，而非手工拼接。
- 仓库内所有文件的修改时间为 **2025-11-04**，与 GitHub 上最后一次提交（`Update README.md`, Nov 4 2025）吻合。

### 证据 B —— README 与官方仓库逐字一致

`README.md` 的 md5 为 `71300fdbb35941148f2bddd6eaa00f23`。其内容（任务定义、三处方法要点、
`54% fewer parameters`、安装依赖版本、`resume='checkpoints/CharadesSTA/8_2025-05-14_13-26-12/model-best.pt'`
这一行示例路径、引用 BibTeX 的 OpenReview 编号 `kQAnOaayIo`）与官方仓库公开内容完全对应。
其中示例 checkpoint 路径 `8_2025-05-14_13-26-12` 与本文实验所用的原权重路径**同名同源**。

### 证据 C —— 与本文实验所用的官方副本逐字节对撞

本文全部实验在一份独立部署的官方代码副本（`PC-Net-ORIG`）上进行，该副本严格复现了论文的
Charades-CG Test-Trivial 三项指标（mIoU 47.12 / R1@0.5 54.84 / R1@0.7 26.68，见 `RESULTS.md` §1）。

把本地文件夹与该副本做 **md5 逐文件对撞**：

```
IDENTICAL=64   DIFFERENT=0   MISSING=0   TOTAL=64
```

覆盖范围：全部 `.py` 源码（`models/` `models/modules/` `models/transformer/` `datasets/`
`runners/` `optimizers/` `optimizers/lr_schedulers/` + `train.py` `utils.py` `vocab.py`）、
全部 `config/*/*.json`、全部 `scripts/*.sh`、全部 `data/{charades,activitynet}/*.json` 切分、
以及 `checkpoints/` 下的官方评测日志。

三个官方权重同样逐字节相同：

| 权重 | md5 | 大小 |
|---|---|---:|
| `checkpoints/CharadesSTA/8_2025-05-14_13-26-12/model-best.pt` | `982c75b79617ccd1edd8b77926b05efe` | 22,891,349 B |
| `checkpoints/ActivityNet/8_2025-05-14_14-07-12/model-best.pt` | `a7ccf9183cb0baddb9518e57be06625f` | 29,434,581 B |
| `checkpoints/TACoS/8_2025-05-14_21-05-09/model-best.pt` | `f50e7ec692243c79f262c5c08bce8a3d` | 26,688,789 B |

⇒ 随仓库分发的权重，就是本文所有「零重训」实验所使用的那一份。

### 证据 D —— 官方 Novel 切分日志独立复核

`checkpoints/CharadesSTA/[Novel-Composition]8_2025-05-14_13-55-50.log` 与
`[Novel-Word]8_2025-05-14_13-56-06.log`（以及 ActivityNet 的两份同名日志），
与本文先前从该官方副本中导出的留档副本 **md5 完全相同**：

```
62e76db9f8a83c477d6341dd64cc757f   CharadesSTA/[Novel-Composition]8_2025-05-14_13-55-50.log
29d2c169eff0f0948b01499f485d1089   CharadesSTA/[Novel-Word]8_2025-05-14_13-56-06.log
1d50254a86e6fe5f68d8ffd44e136193   ActivityNet/[Novel-Composition]8_2025-05-14_16-59-38.log
88eb2bbcbe6ad2f59eff64616a4c16ec   ActivityNet/[Novel-Word]8_2025-05-14_17-00-03.log
```

⇒ 这两组日志正是 `RESULTS.md` §1 中 Novel 切分基准值的来源，未被篡改。

---

## 3. 上游自身存在的两处小瑕疵（非改动痕迹，请勿误判）

1. **README 引用了仓库里没有的两个脚本**。`README.md` 的 *Additional Features of Other Datasets*
   一节让读者执行 `python build_vocab.py` 与 `python convert_npy_to_hdf5.py`，但这两个文件
   **在官方仓库里就不存在**（目录清单里没有）。这是上游的文档缺口，不是本文件夹缺少文件。
   实际影响：使用官方已提供的 `data/*/glove.pkl` 与 `data/*.json` 时**不需要**这两个脚本。
2. **官方仓库未附 LICENSE 文件**。因此本文件夹也没有可继承的开源许可证文本。
   使用 backbone 与官方权重前，请自行与论文作者确认授权（见 `csr/README.md` §许可）。

---

## 4. 本次整合对官方树做了什么

**官方文件除 `README.md` 追加了一节外，一个字节都没有改**（整合脚本结尾有一道自检：
68 个官方文件在整合前后 md5 全等，自检不通过则整脚本失败）。

新增内容全部落在两个新目录下，与官方树物理隔离：

```
csr/           ← 本文新增：算子 + 评测 harness + 离线分析 + 诊断链 + 基线补丁 + 工具
docs/          ← 本文新增：方法、结果、复现、本核验报告
requirements.txt ← 本文新增（上游 README 里只有零散版本号）
```

`README.md` 中新增的内容一律以「**本仓库新增**」标注，原官方文字未删改。
逐文件改动清单见 `csr/README.md` 的「文件对照表」。
