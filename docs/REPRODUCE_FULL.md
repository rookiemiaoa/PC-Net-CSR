# 完整复现手册（服务器版，全流程）

> 这份手册的目标不是「跑通一个脚本」，而是**从零把论文的 4 张表全部重算出来**。
> 想快速跑个 demo，看 [`REPRODUCE.md`](REPRODUCE.md)；想复现论文，看这里。
>
> 约定：`<REPO>` 仓库根目录，`<DATA_ROOT>` 放特征 hdf5 的目录，`<OUTDIR>` 日志输出目录。
> 全文不写任何内网地址与绝对路径，按你的环境替换即可。

---

## 0. 先说清楚：哪些能逐位复现，哪些不能

| 类别 | 内容 | 复现标准 |
|---|---|---|
| **可逐位复现**（同硬件同版本） | Charades-CG 的 base / CSR / ISR；表 3 的 30 个格子；表 4 的两列；图 3 的工作点 | 见 §8 的容差表 |
| **可区间复现** | 5 个外部基线（CPL / CCR / PPS / QMN / CNM）的重训 | ±0.5 mIoU 属正常，逐位不可能（GPU/驱动/数据顺序） |
| **不可复现，只能引用** | ActivityNet 的**绝对** base（官方日志 36.45，本机 32.39） | 只用同一份特征下的复现值做比较，不与官方值混用 |
| **仅证据，非产物** | 表 1（概念表）、架构图 `method_figure/method_pdf.png` | 无程序化生成路径 |

三条必须记住的纪律：

1. **先复现 base，再看 Δ**。base 对不上（误差 > ±0.005）时，后面所有 Δ 都不可信。
2. **`K_REP ≥ 3`**。`_mask_words` 未固定种子，单次评测噪声约 ±0.005。
3. **只比同一脚本内的 Δ**。`reproduce_base` 的 47.12 与 `eval_csr` 的 47.20 是两条评测路径，
   差 0.08 / @0.5 差 0.38，不是噪声，见 `docs/RESULTS.md` §0。

---

## 1. 环境

```bash
conda create -n pcnet python=3.8 -y
conda activate pcnet
pip install -r <REPO>/requirements.lock.txt     # 逐位复现用锁定版本
# 或 pip install -r <REPO>/requirements.txt     # 新平台试装
```

实测组合：`Python 3.8.20 / torch 2.0.1 / numpy 1.24.4 / h5py 3.9.0 / nltk 3.8.1 /
fairseq 0.12.2 / scipy 1.10.1`，CUDA 11.7，单卡 RTX 2080 Ti（22 GB）。

> 开发环境实测补充（数值，不含路径）：conda 环境约 **4.7 GB**，装在系统盘；
> 8× RTX 2080 Ti 22 GB，单进程训练约 3.5–4 GB 显存。若你的 2080 Ti 是 11 GB 版本，
> 把 `BS` 减半即可，其余不变。

额外（只有画图需要）：

```bash
pip install matplotlib
# 服务器通常没有中文字体，否则图 3 的中文标签会变成方框：
apt install fonts-noto-cjk        # 或 fonts-wqy-zenhei
rm -rf ~/.cache/matplotlib
```

首次运行前确认：

```bash
cd <REPO>
python csr/tools/check_assets.py           # 权重 / 特征 / 外部仓库 / 磁盘，一次查完
```

`check_assets.py` 的 `MISSING` 列表就是你还缺的东西，对着 §2 逐个补。

---

## 2. 资产清单

### 2.1 随仓库分发（clone 即有）

- 数据切分：`data/{charades,activitynet,tacos,ego4d}/*.json`
- 词表：`data/*/glove.pkl`
- 官方评测日志：`checkpoints/**/*.log`（`docs/RESULTS.md` §1 的来源）
- 离线分析缓存：`csr/analysis/feat_cache/*.pkl`（S0 阶段用，不需要 GPU）
- 多种子逐样本 IoU：`results/multiseed/s1/*.npz`
- 曲线可用性原始证据：`results/curve_availability/*.json`

### 2.2 在开发机磁盘上、但**不在 git 里**

- 7 个 `model-best.pt`（22–29 MB/个）。`.gitignore` 排除了 `checkpoints/**/*.pt`。
  校验：`md5sum -c checkpoints/WEIGHTS.md5`。

### 2.3 必须自备

| 资产 | 用途 | 备注 |
|---|---|---|
| `i3d_features.hdf5`（实测 3.0 GB） | Charades-CG | 放 `<DATA_ROOT>/` 并 `export FEAT_PATH=` |
| `sub_activitynet_v1-3.c3d.hdf5`（实测 3.9 GB） | ActivityNet-CG | 同上，**ANET 阶段才需要** |
| `tacos_features_c3d.hdf5`（实测 0.5 GB） | TACoS | 同上 |
| `WS_BASE`（含 CPL / QMN / PPS / CCR 克隆） | 表 2 的四个外部基线 | 只跑 S0–S3 时不需要 |
| `CNM_ROOT`（CNM 克隆 + 其 ckpt） | 表 2 的 CNM 行、§4 跨骨干 | 同上 |

---

## 3. 目录与磁盘预算

```bash
export REPO=<REPO>
export DATA_ROOT=<放 hdf5 的目录>
export FEAT_PATH=$DATA_ROOT/i3d_features.hdf5
export OUTDIR=$REPO/results/logs          # 每个脚本的 stdout 落这里
export WS_BASE=<...>  CNM_ROOT=<...>      # 只在 S3/S4 需要
```

磁盘（单盘场景）：

| 阶段 | 峰值占用 | 说明 |
|---|---:|---|
| 仓库 + 权重 + `.git` | 0.42 GB | 工作树 0.25 + 权重 0.17 |
| + Charades hdf5 | 3.5 GB | S1/S2 用 |
| + 基线重训 ckpt + 日志 | 6.5 GB | S4 用 |
| + ANET hdf5 | 11–12 GB | S6 用 |

**结论：一张 12 GB 以上的盘就能完整跑完 S0–S7。** 开发机数据盘实测可用 **576–582 GB**
（同一台共享盘在四天里从 **9.6 GB / 100% 满** 变成 576 GB，说明余量会随他人使用大幅波动，
不要把它当稳定值），因此上面的顺序不是硬约束，只是为了让人知道每一步吃掉多少。若盘确实紧张：

1. 先跑 S1/S2/S5（Charades 系），它们只用 3.5 GB；
2. S4 的基线 ckpt 跑完即可删（只留日志与 JSON）；
3. S6 的 ANET hdf5 最后放，`mv $FEAT_PATH /other/disk/` 腾出 Charades 的 3 GB 再进。

任何时刻剩余 < 1 GB 就停下来——**写满系统盘会让 conda 和 ssh 都进不去**。

---

## 4. 执行顺序（先验证、后烧 GPU）

| 阶段 | 内容 | GPU | 单卡耗时 | 3 并发 | 显存/进程 |
|---|---|---|---:|---:|---:|
| S0 | CPU 自检：对拍 + 4 个离线分析 + 表 3 + 统计 | 0 | 5 min | — | — |
| S1 | Charades：smoke → 3 切分 base/CSR → ISR → w6 | 1 | 25–35 min | — | 3–4 GB |
| S2 | TACoS | 1 | 5 min | — | 3–4 GB |
| S3 | CNM 迁移（3 切分） | 1–3 | 10 min | 5 min | 3–5 GB |
| S4 | 4 个外部基线：patch → 30 epoch → 3 切分 | 3 | 2.5–4 h | 1–1.5 h | 4–8 GB |
| S5 | 5 seed × 3 切分 + `s2_stats across` | 3 | 45 min | 20 min | 3–4 GB |
| S6 | ANET：`eval_anet_support` → `diag41` 对拍 → `anet_sens` | 3 | 40–60 min | 20–25 min | 3–4 GB |
| S7 | 图/表渲染 + 验收 | 0 | 5 min | — | — |

合计 **3–4 h（3 并发）/ 6–8 h（保守）**，不含下载特征与首次装依赖。

### S0 —— 纯 CPU 自检（不需要 hdf5）

```bash
cd $REPO && conda activate pcnet
python csr/tools/check_assets.py                 # 资产是否齐
python csr/tools/config_parity.py --published    # 训练配置是否等于论文所用（DIFF 应为 0）
python csr/tools/parity_check.py                 # 三算子实现是否可对拍（max|Δ| 应为 0）
python scripts/render_table3.py --check          # 表 3 的 30 个格子是否与论文一致
python csr/analysis/scan_center_final.py | tee $OUTDIR/s0_scan.log
python csr/analysis/attrib_csr.py        | tee $OUTDIR/s0_attrib.log
python csr/analysis/ablate_gate.py       | tee $OUTDIR/s0_ablate.log
python csr/analysis/oracle_gate.py       | tee $OUTDIR/s0_oracle.log
python scripts/s2_stats.py within results/multiseed/s1/s8_test_trivial.npz
```

或一条命令：`PY=python bash reproduce.sh check`。

**通过标准**：`parity_check` 的 `RESULT: PASS`（CSR 与 ISR 全部 `max|Δ|=0`）；
`render_table3 --check` 输出 `CHECK PASS`；四个离线分析的 base 为 47.20、CSR 为 +1.13、
短桶 +0.58、长桶 +2.72。

### S1 —— Charades-CG（主结果）

> **先看这一条，否则必然对不上数。** `train.py:46-52` 会用 argparse 的值**无条件覆盖**
> 配置里的六个字段，而 argparse 默认是 `slot_num_iteration=3`、`inter_lambda=0.13`；
> 论文用的是 `4` 与 `0.146`。只改 JSON 或不带参数直接跑，训出来的是另一个模型。
> 仓库因此提供 `config/charades/main_train_test.paper.json`（`sigma_neg=9`、`co_cl_loss=0.5`）
> 与校验工具：
>
> ```bash
> python csr/tools/config_parity.py --published   # DIFF 0 = 与产出论文权重的日志逐键一致
> python csr/tools/config_parity.py               # 裸跑发布配置：DIFF 4，正是这个陷阱
> ```
>
> 训练命令（四个覆盖值缺一不可）：
>
> ```bash
> python train.py --config-path config/charades/main_train_test.paper.json \
>     --log_dir LOG --seed 8 --slot_num_iteration 4 \
>     --co_cl_loss 0.5 --co_qua_loss 1 --num_props 8 --inter_lambda 0.146
> ```

先用 64 条样本校准速度，再放开：

```bash
CH=checkpoints/CharadesSTA/8_2025-05-14_13-26-12/model-best.pt
LIMIT=64 K_REP=1 python csr/eval/eval_csr.py config/charades/main_train_test.json $CH
# 估时：T_full ≈ T_64 × (N/64) × K_REP

# ① 官方基准（三条命令，换 config 即可）
python csr/eval/reproduce_base.py config/charades/main_train_test.json $CH
python csr/eval/reproduce_base.py config/charades/main_novel_comp.json $CH
python csr/eval/reproduce_base.py config/charades/main_novel_word.json $CH

# ② CSR 主结果（含 RBR 与分桶、显著性）
python csr/eval/eval_csr.py config/charades/main_train_test.json $CH
python csr/eval/eval_csr.py config/charades/main_novel_comp.json $CH
python csr/eval/eval_csr.py config/charades/main_novel_word.json $CH

# ③ ISR 对照（唯一产出 ISR 的入口）
python csr/eval/eval_final.py config/charades/main_train_test.json $CH

# ④ Charades 的 ORACLE_c（§3.1 的 +23.2）
python csr/analysis/w6_centre_bias.py
```

### S2 —— TACoS

```bash
python csr/eval/eval_csr.py config/tacos/main.json \
    checkpoints/TACoS/8_2025-05-14_21-05-09/model-best.pt
```

### S3 —— CNM 零调参迁移

```bash
export CNM_ROOT=<CNM 克隆目录>
# config 与 ckpt 都相对 CNM_ROOT —— 本仓库没有 config/charades/main.json
python csr/eval/eval_csr_cnm.py "$CNM_ROOT/config/charades/main.json" \
    "$CNM_ROOT/checkpoints/<cnm-run>/model-best.pt"
```

三个切分重复。若要用 CNM 官方口径单独评 base，在 CNM 仓库根目录下运行
`csr/eval/eval_cnm_base.py`（它要 import 该仓库的 `models` / `datasets`）。

### S4 —— 4 个外部基线

```bash
export WS_BASE=<含 CPL/ QMN/ pps/ CCR/ 的目录>
python csr/baselines/gen_cg_cfgs.py
python csr/baselines/patch_baselines.py
python csr/baselines/patch_qmn_ratio.py
python csr/baselines/patch_qmn_eval.py
python csr/baselines/patch_pps_wandb.py

CUDA_VISIBLE_DEVICES=0 bash csr/baselines/run/run_cpl_cg.sh
CUDA_VISIBLE_DEVICES=1 bash csr/baselines/run/run_ccr_cg.sh
CUDA_VISIBLE_DEVICES=2 bash csr/baselines/run/run_qmn_cg.sh
# 三路结束后再跑 PPS
CUDA_VISIBLE_DEVICES=0 bash csr/baselines/run/run_pps_cg.sh
```

补丁**只修兼容性 bug，不改算法**，逐条动机见 `csr/baselines/README.md`。
协议必须统一：同一 `cg_train`、同一 I3D 特征、30 epoch、**`vote=False`**、各自官方 `runner.eval()`。

### S5 —— 多种子

```bash
for S in 8 11 12 13 14; do
  python scripts/s1_seed_ops.py config/charades/main_train_test.json \
      checkpoints/CharadesSTA/${S}_*/model-best.pt s${S}_test_trivial
done
python scripts/s2_stats.py across \
  results/multiseed/s1/s{8,11,12,13,14}_test_trivial.npz
```

新 npz 会带 `meta__ckpt_md5` / `meta__git_head`（旧 npz 没有，读取方需容错）。

### S6 —— ActivityNet（放最后，磁盘最紧张）

```bash
mv $FEAT_PATH /other/disk/            # 先腾出 3 GB
export FEAT_PATH=$DATA_ROOT/sub_activitynet_v1-3.c3d.hdf5
AN=checkpoints/ActivityNet/8_2025-05-14_14-07-12/model-best.pt

python csr/eval/eval_anet_support.py config/activitynet/main_train_test.json $AN
# 与论文路径对拍：小样本上两者的 CSR 行必须一致（修好解包 bug 之后）
LIMIT=128 K_REP=1 python csr/analysis/diag41_anet_oracle.py \
    config/activitynet/main_train_test.json $AN
python csr/analysis/anet_sens.py        # 10 组单变量扫描，应 10/10 为负
```

### S7 —— 图与表

```bash
python csr/analysis/w47_traj_calibration.py       # 18 个互异点、a* 反解
python csr/analysis/w48_fig3_validity_map.py      # 图 3（需 matplotlib + CJK 字体）
python scripts/render_table3.py                   # 表 3 → results/tables/
```

---

## 5. 表 → 命令 → 预期值（追溯矩阵）

| 表 | 命令 | 预期值 | 容差 |
|---|---|---|---|
| 表 1 | — | 概念表，无程序化复现 | — |
| 表 2（PC-Net 两行） | `eval_csr.py` ×3 | 47.20→48.33（+1.13 / +0.67 / +2.09）；Novel-Comp +0.75；Novel-Word +1.18 | mIoU ±0.10 |
| 表 2（CPL/CCR/PPS/QMN） | S4 | 46.22 / 46.23 / 46.01 / 45.47（Test-Trivial mIoU） | ±0.5 |
| 表 2（CNM） | S3 | 38.43→41.43（+3.00）；NC +2.17；NW +2.84 | ±0.25 |
| 表 3 | `render_table3.py --check` | PC-Net +1.05 / CNM +3.00 / CPL −3.76 / QMN −3.53 / CCR −3.18；dir_acc 0.776/0.700/0.510/0.472/0.447 | 5e-3 |
| 表 4（测试列） | `w47` + 表 3 的 dir_acc | 0.776 / 0.510 / 0.472 / 0.447 | ±0.002 |
| 表 4（训练列） | **仓库内无生成器**（见 §9） | 0.742 / 0.508 / 0.491 / 0.453 | ±0.002 |
| 图 3 | `w47` + `w48` | 18 个互异工作点；a\* ∈ (0.5096, 0.5691)；conv_only 0.5759；all 0.5430 | 4 位小数 |

`ORACLE_c` 有两个量，**不可互换**：§3.1 的 +23.2（Charades）/ +12.2（ANET）是单骨干、
单次中心替换的诊断口径；表 3 的 +22.7~+26.5 是五骨干等后处理框架下的对比。

---

## 6. 并行与调度

- **并发 ≤ 3 个 hdf5 读进程**。再高会在 2080Ti 节点上出现 IO 抖动，总时间反而变长。
- **1 进程 / 1 卡**。22 GB 卡实测单进程 3.5–4 GB，理论上能叠 3 路，但 **DataLoader 会先打满 CPU**：
  每进程 `num_workers=16`，3 路并发 = 48 个 worker，单 epoch 从 ~1.7 min 涨到 ~4.5 min。
  ⇒ 想快就用 3 张卡各跑 1 路，不要 1 张卡跑 3 路。
- S4 用 3 张卡，S6 用另外 3 张卡，**不要与 Charades 评测同时跑**。
- 长任务用 `tmux` + `tee` 落日志，日志名与 §5 的表格一一对应。

---

## 7. 常见失败模式

| 现象 | 原因 | 处置 |
|---|---|---|
| `KeyError: 'v_xxxx'` / hdf5 打不开 | 特征与切分不匹配，或 `FEAT_PATH` 没生效 | `check_assets.py --features` 看 key 数；确认 `FEAT_PATH` 是绝对路径 |
| CUDA OOM | `BS` 太大或 K_REP 叠加 | `BS=8`，或按切分单起进程 |
| base 比论文低 4 个点（ANET） | 特征溯源不同 | 只用本次复现的 base，不与官方 36.45 混比 |
| 同一命令两次结果差 0.5pt | `K_REP=1` | 提到 3；单次噪声 ±0.005，量级放大后就是这个表现 |
| 图 3 中文变方框 | 服务器无 CJK 字体 | 装 `fonts-noto-cjk` 并清 matplotlib 缓存 |
| CNM 脚本报 `FileNotFoundError: config/...` | 路径相对 `CNM_ROOT` 而非本仓库 | 传绝对路径或用 `"$CNM_ROOT/config/..."` |
| 数值与论文差 0.2–0.4pt（RBR） | 混用了 `s1_seed_ops` 的 RBR 口径 | 正文/表 2 一律用 `eval_csr.py`（相对阈值），见 `IMPLEMENTATION_MAP.md` §2 |
| Windows 上 `git diff` 全是改动 | CRLF/LF | 用 `-c core.safecrlf=false`，或 `.gitattributes` 归一化 |

---

## 8. 验收门禁

| Gate | 判据 | 容差 |
|---|---|---|
| G0 | `parity_check` PASS；`render_table3 --check` PASS；4 个离线分析与改动前逐字节一致 | 0 |
| G1 | `reproduce_base` Test-Trivial 47.12/54.84/26.68；Novel-Comp 38.04/41.69/16.73；Novel-Word 41.06/46.91/23.60 | 每项 ±0.02 |
| G2 | `eval_csr` Test-Trivial 48.33/55.89/28.67，ΔmIoU +1.13 | ±0.10 |
| G3 | 短桶 +0.58/+0.03/+0.22；长桶 +2.72/+2.49/+7.42 | ±0.10 |
| G4 | `eval_final` ISR 48.36，Δ +1.16 | ±0.05 |
| G5 | ANET base 32.39、CSR 32.22、Δ ∈ [−0.25,−0.10]；与 `diag41` 小样本对拍 ≤1e-4 | ±0.05 |
| G6 | `anet_sens` 10/10 ΔmIoU < 0 | 全负 |
| G7 | CNM 九格全正（+3.00/+2.17/+2.84） | ±0.25 |
| G8 | 表 3 五行；表 4 两列 | 5e-3 / ±0.002 |
| G9 | `w47` 18 点、a\* 区间 | 4 位小数 |
| G10 | 仓库内不含内网地址与开发机绝对路径（检查清单见 `docs/RELEASE_CHECKLIST.md`）；`git ls-files '*.pt'` 为 0 | 0 违规 |

---

## 9. 尚未闭环的三项（务必如实转述，不要猜）

1. **表 4 的「训练」列没有入库的生成器**。它的数据（PC-Net 0.742 / CPL 0.508 / QMN 0.491 /
   CCR 0.453）来自开发机的 `w61` 批次评估产物（8 个 JSON：`{pcnet,cpl,qmn,ccr}` × `{probe,test}`），
   该批次已定位并核对存在，但**生成脚本未入库**。
   在脚本取回之前，这一列只能作为历史测量引用，不能宣称可端到端重跑。
   不要用测试列的 0.776/0.510/0.472/0.447 顶替训练列。
2. **表 3 有产物、也有采集器了，但仍有两个脚本缺失**。`results/curve_availability/*.json`
   是已入库的测量结果，`scripts/render_table3.py` 能一字不差地复现表格（30 格 `CHECK PASS`）。
   「从零重测这 30 个格子」的采集器 `csr/analysis/w34_curve_usability.py` **已入库**，
   换一个新骨干也能重测。仍缺的两个：`w32_baseline_csr.py`（其功能已被 `w34` 覆盖）与
   `w36_cnm_curve.py`（只服务 CNM，而 CNM 的代码与权重都已不可得，见 §9.1 的口径）。
3. **发布版 `train.py` 与研究版不同，二者不可互换**。本仓库的 `train.py` 是干净的发布版
   （76 行，只有上游参数）；产出论文数字的是研究分支版本（103 行，多出 `depend_center` /
   `depend_width` / `depend_width_log` / `depend_spread` / `rel_width` / `alpha_dep` / `probe`
   七个内部开关）。这些开关在论文里没有对应消融，属于内部实验设施，**故意不发布**。
   ⇒ 用发布版训练得到的是 PC-Net 基线（G1 的 47.12），CSR 是其后处理，不受影响；
   但若你想逐位复现研究版训练轨迹，需要那些开关，本仓库不提供。

这三项补上之后，本手册的 S0–S7 才是完整闭环。
