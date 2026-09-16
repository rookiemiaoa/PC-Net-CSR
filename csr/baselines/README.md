# `csr/baselines/` —— 复现 5 个外部弱监督基线

`docs/RESULTS.md` §5 的对比表需要在**统一协议**下重新训练并评测 5 个外部弱监督方法。
本目录是让它跑通所需的全部脚本。**补丁只修兼容性 bug，不改任何算法。**

---

## 1. 工作区布局

所有外部仓库克隆在同一个工作区里（默认 = PC-Net 仓库的同级目录 `ws_base/`，
可用 `WS_BASE` 覆盖）：

```
<PC-Net 同级>/
├── PC-Net-master/          ← 本仓库
└── ws_base/                ← WS_BASE
    ├── cpl/                ← CPL  (CVPR-2022)
    ├── CCR/                ← CCR  (ACM MM-2023)
    ├── pps/                ← PPS  (AAAI-2024)
    ├── QMN/                ← QMN  (CVPR-2024)
    └── logs_*/             ← 各方法的训练/评测日志
```

CNM 是另一个位置（`CNM_ROOT`，默认同级 `CNM-CG/`），见 `docs/REPRODUCE.md` §5。

---

## 2. 协议要点（为什么必须重训）

| 约定 | 值 | 原因 |
|---|---|---|
| 训练集 | `cg_train.json` | 三个 CG 测试切分与标准 Charades train **有重叠**（实测 `cg_test_trivial ∩ train.json = 1963/3096 = 63%`）⇒ **用官方发布的权重直接评测不可比**，必须重训 |
| 特征 | I3D（与 PC-Net / CNM 同） | 保证特征侧一致 |
| epoch | 30 | 与各方法官方设置一致 |
| 评测口径 | 统一 **`vote=False`** | 见下方 §4 |
| IoU 计算 | 各方法自己的官方 `runner.eval()` | 不重写指标实现，避免自造口径 |

---

## 3. 兼容性补丁

| 脚本 | 修什么 | 为什么 |
|---|---|---|
| `patch_baselines.py` | QMN 的 config 补 `ratio` / `margin_dis` / loss 键；CCR 的 `ivc_loss` 给 `gpu_idx` 默认值 0 | QMN 的模型 `__init__` 直接读这些键，缺了会 `KeyError`；CCR 作者把 `gpu_idx` 声明成必需位置参数但 runner 没传（签名笔误），单卡映射后 device 即 0 |
| `patch_qmn_ratio.py` | QMN `ratio`: **0 → 1** | ★ 关键 bug，见 §4 |
| `patch_qmn_eval.py` | QMN `main_runner.py` 的非 vote 分支补 `res = top_1_metric(selected_props[:, 0], gt)` | 该分支结尾只 `predictions_list.extend(...)` 却从未给 `res` 赋值，随后的 `for key, v in res.items()` 会抛 `UnboundLocalError`（CPL 谱系里有这一行，QMN 漏了） |
| `patch_pps_wandb.py` | PPS 的 `import wandb` 改为 try/except 可选 | 服务器上没有 wandb，硬导入会直接崩 |
| `gen_cg_cfgs.py` | 为 QMN / PPS / CCR 生成 CG 切分协议的 train/eval config | 让三者与 CPL / CNM / PC-Net 的数据契约一致 |

另有 `cpl_prep.py`：把 CPL 官方 `config/charades/main.json` 克隆出 `cg_main.json` +
三份 `eval_<split>.json`（CPL 官方没提供 CG 协议配置）。

> **一个配置陷阱**：CCR 作者原 `main.json` 里 `model.name` 误写为 `"CPL"`（CCR 派生自 CPL 未改名），
> 而 `models/__init__.py` 只导出 `CCR` ⇒ 必须改成 `"CCR"` 才能 `getattr` 到类。
> `gen_cg_cfgs.py` 已经处理。

---

## 4. ★ QMN 的 `ratio` bug（最容易被误判成"方法不行"）

- QMN 的 `train.py` 里 `--ratio` 的默认值是 **1**（提案融合系数），`--margin-dis` 默认 0。
- 早期补丁把 `ratio` 误设成 **0**，结果：**提案融合失效，mIoU 塌到约 0.21（≈随机）**。
- 症状极具误导性——看起来像"QMN 这个方法本身很差"，而不是"配置写错了"。
- 修复：`ratio = 1`（见 `patch_qmn_ratio.py`），并在 `patch_baselines.py` 的注释里写明
  正确默认值是 1 而非 0。

**`vote` 口径同理**：CCR / QMN 的原文评测走 `--vote` 投票策略，而 PC-Net 官方评测、
CPL、PPS 都是非 vote。若混用会与其余方法不可比（实测差异 < 0.8pt）。
本仓库报告的全部数字统一为 `vote=False`；`run/run_ccr_cg.sh` 与 `run/run_qmn_cg.sh`
**默认不加 `--vote`**，注释里写明了如何切回原文口径。

**PPS 还有一个坑**：PPS 只保存 `model-{epoch}.pt`，**没有 `model-best.pt`**。
驱动脚本里若用 `find -name model-best.pt` 会拿到空路径 ⇒ 评测用了空权重
（表现为 mIoU 0.2516 的垃圾值）。正确做法是用最优 epoch（`model-25.pt`，best mIoU 0.4574）
配合 `--ckpt-path` 评测。`run/run_pps_cg.sh` 已按此写。

---

## 5. 驱动脚本

```bash
export GPU=0                 # 选卡
export CONDA_SH=/path/to/miniconda3/etc/profile.d/conda.sh
export CONDA_ENV=pc-net

python csr/baselines/gen_cg_cfgs.py                 # 先生成 config
python csr/baselines/patch_baselines.py             # 再打补丁
python csr/baselines/patch_qmn_ratio.py
python csr/baselines/patch_qmn_eval.py
python csr/baselines/patch_pps_wandb.py

bash csr/baselines/run/run_cpl_cg.sh                # 每个方法一个驱动脚本
bash csr/baselines/run/run_ccr_cg.sh
bash csr/baselines/run/run_pps_cg.sh
bash csr/baselines/run/run_qmn_cg.sh
```

| 脚本 | 作用 |
|---|---|
| `run/run_cpl_cg.sh` | CPL：cg_train 重训 + 三个 CG 切分评测 |
| `run/run_cpl_eval.sh` | CPL：用官方发布权重做零样本探测（说明为什么必须重训） |
| `run/run_ccr_cg.sh` | CCR：重训 + 评测（动态查找带 tag 的 ckpt 目录） |
| `run/run_pps_cg.sh` | PPS：重训 + 评测（`--ckpt-path` + 最优 epoch） |
| `run/run_qmn_cg.sh` | QMN：重训 + 评测（不传 `--log_dir`，见下） |
| `run/run_csr_cnm.sh` | CSR-on-CNM 三切分 + WPT 稳健性扫描 |
| `run/run_eval_chain.sh` | 等 CNM 训练收敛后自动跑三切分评测（轮询日志） |
| `run/cpl_prep.py` | 生成 CPL 的 CG 协议 config |

> **QMN 的另一个坑**：QMN 的 `train.py` 在传 `--log_dir` 时会写死
> `checkpoints/{dataset}/{seed}_{ts}.log`，而该目录不存在会直接崩。
> `run_qmn_cg.sh` 因此**不传 `--log_dir`**，让 `log_filename=None`（日志走 stdout）。
>
> **并行实验的坑**：PC-Net 谱系用 `{seed}_{时间戳}` 命名 ckpt 目录，
> 同 seed 的参数扫描若在同一秒启动会**落在同一目录、`model-best.pt` 互相覆盖**。
> 各驱动脚本因此统一用 `find` 动态定位 ckpt（而不是写死目录名）。

---

## 6. 结果

见 `docs/RESULTS.md` §5。要点：统一 `vote=False` 下，
**PC-Net + CSR 在 Test-Trivial 相对最强的外部基线 CCR 领先 +2.10 mIoU / +3.08 R1@0.5**；
Novel-Comp 上 CPL 的 mIoU/@0.5 略高，但 CSR 的 R1@0.7 领先 2.00pt；
Novel-Word 上 CSR 三项均领先全部外部基线。
