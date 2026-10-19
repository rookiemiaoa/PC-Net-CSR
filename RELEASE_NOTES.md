# v1.0-repro — 2026-10-19

第一个可完整复现论文全部表格的发布版本。本版本不做方法改动，只把「代码 → 数字」这条链补齐。

## 1. 这个版本能复现什么

```bash
python scripts/repro_gate.py          # 八行总门禁，期望 ALL GREEN
```

| 行 | 内容 | 判据 | 状态 |
|---|---|---|---|
| R1 | 核心论点 | 5/5 seed Δ = +1.37 ± 0.21；CPL/QMN/CCR 为 −3.76 / −3.53 / −3.18；0.776 > a\* = 0.543 > 0.510 | PASS |
| R2 | 表 2 PC-Net 行 | 47.20 → 48.33（Δ +1.13 / +0.67 / +2.09） | PASS |
| R3 | 表 3 CPL/QMN/CCR | 18/18 格 | PASS |
| R4 | 表 3 PC-Net 行 | 权重 md5 7/7 + 行为等价 2 切分 × 5 指标逐位 + 6/6 格 | PASS |
| R5 | 表 3 CNM 行 | 从零重训 50 epoch，轨迹 600 次比对零差 | PASS |
| R6 | 表 4 训练/测试列 | 0.742/0.508/0.491/0.453 与 0.776/0.510/0.472/0.447 | PASS |
| R7 | ANET oracle | +12.2160 | PASS |
| R8 | Charades oracle | 0.4720 → 0.7040（+23.21） | PASS |

缺权重或特征时，门禁返回 `PENDING` 并打印缺哪一份，不会把「没跑」误报成 `FAIL`。

## 2. 本版新增

### 2.1 五个留证脚本（把正文数字变成可重跑）

| 脚本 | 复现的数字 | 成本 |
|---|---|---|
| `scripts/repro_estimator_table.py` | §4.3 估计器表 11 个数（MAE 6 + dirAcc 5） | CPU 2 s |
| `scripts/repro_contraction_slope.py` | §3.3 收缩斜率 1.763 / 兑现 56.7% | CPU 2 s |
| `csr/analysis/w62_grid_transfer.py` | §4.3 900 组网格 → 217 / 97 / 81 / 37.3% | CPU 2 min |
| `scripts/bench_csr_latency.py` | §3.3 CPU 后处理 0.0216 ms/查询 | CPU 10 s |
| `scripts/scrub_internal_paths.py` | 发布前检查内网路径残留 | CPU 1 s |

产物分别落在 `results/estimator_grid/`、`results/grid_transfer/`、`results/latency/`。

### 2.2 新增证据

- `results/diagnostics/diag38_20261019.log` —— 十六条免真值信号的 AUC 表
  （最强 `kratio`：校准 0.675 / 测试 0.650）
- `results/estimator_grid/` —— 估计器表与收缩斜率的 JSON + 日志
- `results/grid_transfer/` —— 900 组超参网格的完整输出
- `results/latency/` —— w31 的 GPU 时延原始产物（6.22 / 8.70 / 0.0216 ms）

### 2.3 文档

- `docs/EVIDENCE_INDEX.md` —— 每个数字 → 产物 → 命令
- `docs/DESIGN_FAMILIES.md` —— 六个设计族、323 个变体的账本
- `CITATION.cff`

## 3. 权重（Release 附件，7 个，共 170,579,859 B）

| 附件名 | 落盘路径 |
|---|---|
| `activitynet_seed8_20250514_model-best.pt` | `checkpoints/ActivityNet/8_2025-05-14_14-07-12/` |
| `tacos_seed8_20250514_model-best.pt` | `checkpoints/TACoS/8_2025-05-14_21-05-09/` |
| `charades_seed8_20250514_model-best.pt` | `checkpoints/CharadesSTA/8_2025-05-14_13-26-12/` |
| `charades_seed11_20260914_model-best.pt` | `checkpoints/CharadesSTA/11_2026-09-14_17-09-28/` |
| `charades_seed12_20260914_model-best.pt` | `checkpoints/CharadesSTA/12_2026-09-14_17-09-28/` |
| `charades_seed13_20260914_model-best.pt` | `checkpoints/CharadesSTA/13_2026-09-14_17-09-28/` |
| `charades_seed14_20260914_model-best.pt` | `checkpoints/CharadesSTA/14_2026-09-14_17-39-33/` |

```bash
python scripts/check_release_assets.py --list
python scripts/check_release_assets.py --dir ~/Downloads/pcnet-release --copy --dest .
md5sum -c checkpoints/WEIGHTS.md5
```

## 4. 快速开始

```bash
conda create -n pcnet python=3.8 -y && conda activate pcnet
pip install -r requirements.lock.txt
bash reproduce.sh check            # 纯 CPU，约 5 min，不需要权重与特征
bash reproduce.sh charades         # 需要权重 + i3d_features.hdf5
```

## 5. 兼容性说明

- 上游 `models/`、`datasets/`、`runners/`、`optimizers/`、`config/` 原样保留（零改动），
  CSR/ISR/RBR 全部是推理期算子，不落盘改上游代码。
- `data/*.json` 的切分与上游一致；`i3d_features.hdf5`（约 3 GB）需自行从上游获取。
