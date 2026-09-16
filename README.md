# PC-Net-CSR —— 弱监督视频时刻检索的推理期中心重定位精化

> **一句话**：固定框宽时 IoU 对中心偏移是单峰偶函数 ⇒ 只修中心、不动宽度，即可在**零训练、
> 零参数、不使用真值**的前提下稳定提升 PC-Net 的时刻定位精度，且**不引入子群方向偏置**。

[![Python](https://img.shields.io/badge/python-3.8-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0.1-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Paper](https://img.shields.io/badge/paper-Neurocomputing%20submission-lightgrey)](#引用)

本仓库是论文 **《On the Improvability Boundary of Inference-Time Refinement for Weakly
Supervised Compositional Moment Retrieval: A Two-Axis Validity Map and Training-Free
Centre-Shift Refinement》**（Neurocomputing 投稿）的官方代码发布。它在
**PC-Net**（NeurIPS 2025, WSCMR）之上新增三个推理期精化算子及其完整评测/分析代码，
**不改动上游任何一个字节**。

---

## 目录

- [1. 这是什么](#1-这是什么)
- [2. 主要结果](#2-主要结果)
- [3. 快速开始](#3-快速开始)
- [4. 仓库结构](#4-仓库结构)
- [5. 三个算子](#5-三个算子)
- [6. 有效性判据（部署前自检）](#6-有效性判据部署前自检)
- [7. 可复现性](#7-可复现性)
- [8. 引用](#8-引用)
- [9. 许可与致谢](#9-许可与致谢)
- [10. 上传到 GitHub](#10-上传到-github)

---

## 1. 这是什么

弱监督视频时刻检索（WSCMR）只给「查询 + 视频级对齐」，模型据此回归目标时刻的起止点。
PC-Net 通过「掩码词重建」的对比学习做定位，但它的**单帧回归**在长动作上出现
**帧宽坍缩**（预测框远窄于真值），起止点因此"连接不上"。

一个自然的想法是**扩框**（本仓库的 `RBR`）。但扩框的效应符号是
`sign(真值 − 框宽)`：它**按子群产生方向偏置** —— 长动作获益、短动作受损，
实测短动作 R1@0.5 掉 **−2.98 pt**，headline 因此转负。

本仓库的核心贡献 **CSR** 换了一条路径：**只改框中心，框宽精确保留**。
在固定框宽下 IoU 对中心偏移是**单峰偶函数**，因此中心修正**不存在子群偏置**——
这是它与 RBR 的本质区别，也是它在长短两个子群上**同向非负**的原因。

> **"精确保留"的准确含义**：算子内部对宽度**不做任何缩放**
> （实测 `max|w_out − w_in| = 1.1e-16`，即机器精度）。
> 唯一的例外是中心平移后框被推出 `[0,1]` 的样本——此时按视频边界裁剪，
> 宽度会随之缩短。这是坐标本身的物理约束（框不可能超出视频），
> 不是算子引入了宽度偏置。

三个算子全部满足：**推理期 · 零训练 · 零参数 · 不访问真值**。

---

## 2. 主要结果

Charades-CG **Test-Trivial**（N = 3096，官方发布权重，K_REP = 3，`vote=False`）：

| 算子 | mIoU | ΔmIoU | ΔR1@0.5 | ΔR1@0.7 |
|---|---:|---:|---:|---:|
| base | 47.20 | — | — | — |
| **CSR**（本文核心） | **48.33** | **+1.13** | **+0.67** | **+2.09** |
| ISR(γ=0.5, T=2) | 48.36 | +1.13 | — | — |
| RBR k=0.15 | 48.18 | +0.98 | −0.97 | +1.09 |

CSR 在其余两个 Charades-CG 切分上同样为正：

| 切分 | ΔmIoU | ΔR1@0.5 | ΔR1@0.7 |
|---|---:|---:|---:|
| Novel-Comp（N=3442） | +0.75 | +0.43 | +1.98 |
| Novel-Word（N=695） | +1.18 | +0.43 | +2.11 |

**多 seed 稳健性**（5 个独立训练的种子，层级 bootstrap）：

| 切分 | ΔmIoU（mean ± std） | 正值数 |
|---|---:|---:|
| Test-Trivial | +1.37 ± 0.21 | 5 / 5 |
| Novel-Comp | +0.97 ± 0.08 | 5 / 5 |
| Novel-Word | +1.60 ± 0.12 | 5 / 5 |

15/15 个 seed × 切分组合全部为正，层级 bootstrap 全部排除 0。
而 RBR 在短动作桶上 **Test-Trivial 0/5、Novel-Comp 0/5** 为正（Novel-Word 为 2/5 且量级为 0）
—— 印证"扩框有子群偏置、改中心没有"。
CSR 在**全部 9 个（切分 × 桶）格子**上均为 5/5 为正。

> 全部数字 → [`docs/RESULTS.md`](docs/RESULTS.md)
> 复现实验 → [`docs/REPRODUCE.md`](docs/REPRODUCE.md)
> 方法细节 → [`docs/CSR_METHOD.md`](docs/CSR_METHOD.md)

---

## 3. 快速开始

### 3.1 环境

```bash
conda create -n pcnet python=3.8 -y
conda activate pcnet
pip install -r requirements.lock.txt        # 精确锁定的实测版本
# torch 2.0.1 / h5py 3.9.0 / fairseq 0.12.2 / nltk 3.8.1 / numpy 1.24.4 / scipy 1.10.1
```

宽松安装（新平台）用 `requirements.txt`。也可用容器：

```bash
docker build -t pcnet-csr .
docker run --gpus all -v /path/to/data:/data pcnet-csr smoke
```

### 3.2 数据

随仓库分发的只有**小体积**产物：

- `data/*/*.json` —— 切分与查询（Charades-CG / ActivityNet-CG / TACoS / Ego4D）
- `data/*/glove.pkl` —— 词向量
- `checkpoints/*/…/model-best.pt` —— 官方发布权重

**需要自行准备**的是特征文件 `i3d_features.hdf5`（约 3 GB，I3D 特征）。
把 `FEAT_PATH` 指向它，或放在仓库根目录：

```bash
export FEAT_PATH=/path/to/i3d_features.hdf5    # 也可设 DATA_ROOT
```

### 3.3 跑起来

```bash
# ① 复现 base 基线（应为 47.12 / 54.84 / 26.68）
python csr/eval/reproduce_base.py config/charades/main_train_test.json \
       checkpoints/CharadesSTA/8_2025-05-14_13-26-12/model-best.pt

# ② CSR 主结果
python csr/eval/eval_csr.py config/charades/main_train_test.json \
       checkpoints/CharadesSTA/8_2025-05-14_13-26-12/model-best.pt
# 环境变量: BS=16 K_REP=3 SMOOTH=7 BETA=0.20 RHO=0.10 ALPHA=1.0 DMIN=0.01 KS=0.10,0.15,0.20

# ③ 离线分析（不需要 GPU 前向）
python csr/analysis/scan_center_final.py
python csr/analysis/attrib_csr.py
```

> ⚠️ **`K_REP ≥ 3` 是硬要求**：`_mask_words` 未固定随机种子，单次评测噪声约 ±0.005；
> 低于 3 次重复无法区分 ±1 pt 量级的改进。

### 3.4 作为库调用

```python
from csr import CenterShiftRefiner, RelevanceBoundaryRefiner
from csr import ops                       # 纯 NumPy，无需 torch

csr = CenterShiftRefiner(beta=0.20, rho=0.10, alpha=1.0, dmin=0.01)
c_new = csr.refine(centers, widths, rel_curves)   # 宽度不做缩放（见 §1 注）
```

---

## 4. 仓库结构

```
PC-Net-CSR/
├── README.md                    ← 你在这里
├── LICENSE                      MIT
├── .gitignore
├── Dockerfile                   基于 pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
├── requirements.txt             宽松依赖
├── requirements.lock.txt        精确锁定（逐位复现用）
├── reproduce.sh                 smoke | charades | all 一键复现
│
├── csr/                         ★ 本文相对上游的全部新增内容
│   ├── paths.py                 统一路径解析（唯一权威定义处）
│   ├── ops.py                   纯 NumPy 算子（单一权威实现）
│   ├── csr.py                   CSR 算子（核心贡献）
│   ├── refine.py                RBR 算子（对照）
│   ├── eval/                    GPU 评测：产出论文表格数字
│   ├── analysis/                离线分析：读缓存秒级出结果
│   ├── diagnostics/             诊断链（diag32–diag39）
│   ├── baselines/               5 个外部弱监督基线的 patch 脚本
│   └── tools/                   count_params.py / format_lint.py
│
├── docs/
│   ├── REPRODUCE.md             完整复现指南
│   ├── CSR_METHOD.md            CSR 方法细节
│   ├── RESULTS.md               全部结果数字
│   └── UPSTREAM_VERIFICATION.md 上游溯源核验
│
├── results/
│   ├── README.md
│   ├── anet_cg_remeasured.json
│   └── curve_availability/      「曲线可用性」第二坐标的全部原始证据（28 文件）
│
├── models/  datasets/  runners/  optimizers/  config/  scripts/  data/  checkpoints/
│                                ← 上游 PC-Net 原样，未改动
└── train.py  utils.py  vocab.py ← 上游 PC-Net 原样
```

**对上游的改动量：零。** CSR 只在运行时 monkey-patch 临时替换
`model.proposal_generator` 与 `model.num_props`，不落盘改官方代码。
溯源核验见 [`docs/UPSTREAM_VERIFICATION.md`](docs/UPSTREAM_VERIFICATION.md)。

---

## 5. 三个算子

| 算子 | 作用轴 | 机制 | 何时可用 |
|---|---|---|---|
| **CSR** | 中心（**保宽**） | 单窗内正部质心 + 死区 + 自洽门控；框宽精确不变 | 曲线方向准确率 `a > a*` |
| **ISR** | 中心（**保宽**） | mean-shift 式迭代收缩窗中心估计；与 CSR 同族的两参数点 | 同 CSR |
| **RBR** | 边界（**扩框**） | 沿相关性曲线从 top-1 中心 flood 扩展 | 仅当 ρ_geo ≪ 1（框过窄） |

三者共享同一份纯 NumPy 实现（`csr/ops.py`），因此 CPU 也能跑通全部离线分析。

---

## 6. 有效性判据（部署前自检）

CSR **不是**普遍有效的。部署前请先量两个坐标：

### 坐标一：几何 `ρ_geo = 基础框宽中位数 / 目标时长中位数`

| ρ_geo | 判读 |
|---|---|
| 1.3 – 1.9 | 中心轴可用（Charades-CG 各切分实测 1.235 – 1.347） |
| 1.0 – 2.2 | 可用区间 |
| > 3 | **两轴皆无效**（ActivityNet-CG 3.60、TACoS 11.28） |

### 坐标二：信息 `a = 骨干重建响应曲线的方向准确率`

| a | 判读 |
|---|---|
| > ~0.70 | CSR 为正（PC-Net 0.776、CNM 0.700） |
| 0.45 – 0.51 | **同一算子显著有害**（CPL 0.510 / QMN 0.472 / CCR 0.447，−3.2 ~ −3.8 mIoU） |

在 18 个互异工作点上，**符号沿 `a` 完全单调**：所有 Δ<0 的点 `a ≤ 0.510`，
所有 Δ>0 的点 `a ≥ 0.569`，实测夹逼区间 **`a* ∈ (0.5096, 0.5691)`**。

> **两条诚实限定**：这些工作点来自 3 个骨干、彼此相关，因此该区间是
> **实测夹逼区间，不是置信区间**；且区间内只有训练中途的工作点，
> **没有骨干的收敛点落于其中**。详见 [`results/curve_availability/README.md`](results/curve_availability/README.md)。

一个有用的副产品：`a` 在训练**极早期**即可测出（CNM 第 2 个 epoch 就已 `a = 0.569 > a*`），
而 CPL/CCR 在每个被扫描的 epoch 上都停留在 0.44–0.51、从未越线。
⇒ **`a` 是骨干固有的曲线定位能力，不是训练成熟度的函数**，判据可廉价提前测出。

---

## 7. 可复现性

本仓库遵循四条纪律：

1. **base 优先** —— 先复现基线，再谈改进；
2. **`K_REP ≥ 3`** —— 阈值指标（R1@0.5）必须**逐 rep 判定后再平均**，
   否则会出现"平均后跨线"的假阳性（实测曾把 54.17 误算成 53.07）；
3. **超参只用 train 切分** —— 不碰 val/test；
4. **统一 `vote=False`** —— 不做跨特征来源的 base 比较。

复现前请跑一遍格式自检：

```bash
python csr/tools/format_lint.py
```

---

## 8. 引用

```bibtex
@article{feng2026improvability,
  title   = {On the Improvability Boundary of Inference-Time Refinement for
             Weakly Supervised Compositional Moment Retrieval:
             A Two-Axis Validity Map and Training-Free Centre-Shift Refinement},
  author  = {Feng, Zhichao and Xie, Wei and Sun, Hao},
  journal = {Neurocomputing (submitted)},
  year    = {2026},
  note    = {Corresponding author: Xie, Wei (xw@mail.ccnu.edu.cn)}
}
```

本工作建立在 **PC-Net** 之上，请一并引用：

```bibtex
@inproceedings{pcnet2025,
  title     = {PC-Net: Masked-Word Reconstruction for
               Weakly-Supervised Moment Retrieval},
  booktitle = {NeurIPS},
  year      = {2025}
}
```

---

## 9. 许可与致谢

本仓库以 **MIT** 许可发布（见 [`LICENSE`](LICENSE)）。
上游 PC-Net 代码遵循其原始许可，版权归原作者所有。
CSR / ISR / RBR 三个算子及其评测、分析代码为本文新增，按 MIT 授权。

---

## 10. 上传到 GitHub

本目录**已经是初始化好的 git 仓库**（分支 `main`，227 个受控文件，最大单文件 29 MB，
远低于 GitHub 的 100 MB 上限），只需接上远端即可推送：

```bash
cd PC-Net-master

# ① 在 GitHub 网页端建一个空仓库（不要勾选 README / .gitignore / LICENSE）
# ② 接上远端（把 <user>/<repo> 换成你的）
git remote add origin https://github.com/<user>/<repo>.git
git push -u origin main
```

想先本地确认一遍再推：

```bash
git ls-files | wc -l          # 227
git status                    # 应为 clean
git log --oneline             # 两条提交
```

推送后把仓库地址回填到论文的「Code is available at ...」处即可。

**未随仓库分发**（体积原因，见 §3.2）：`i3d_features.hdf5`（约 3 GB）、
Glove 之外的原始视频、以及各 baselines（CPL / CNM / PPS）的第三方权重。
CSV / JSON / `.pt` 等与论文数字直接对应的证据文件均已入库。

感谢 PC-Net 作者开源代码与预训练权重，使本工作的逐位复现成为可能。
