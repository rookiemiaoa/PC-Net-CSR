# 证据索引（每个数字 → 文件 → 命令）

本文件回答一个问题：**论文里的一个数，凭什么信？**

三条纪律：

1. 每个数字必须能指向一个**入库的产物**（`results/**`），或一条**可重跑的命令**。
2. 数字与口径绑定：同一个量在不同 harness 下不可混引（见 §7 「三种 base 口径」）。
3. 复现不了就写「不可复现」，不写估计值。

---

## 1. 三张主表

| 论文位置 | 产物 | 校验命令 |
|---|---|---|
| 表 2（7 方法 × 3 划分 × 3 指标 = 63 格） | `evidence_csr/d40_ch_*.log`、`evidence_ext/*_cg_*.log`、`results/curve_availability/w36_cnm*.json` | `python scripts/repro_gate.py --row 2`（PC-Net 行）+ `python scripts/parse_eval_log.py <log>` |
| 表 3（5 骨干 × 6 列 = 30 格） | `results/curve_availability/w3*.json`、`results/cnm_training/cg_main_2026-10-18.log` | `python scripts/render_table3.py --check`（CNM 行加 `--cnm`） |
| 表 4（两划分一致性 = 18 格） | `results/table4/w61_*_TT_{test,probe}.json`、`results/curve_availability/w34_*.json` | `python scripts/render_table4.py --check` |

总门禁：

```bash
python scripts/repro_gate.py            # R1–R8，缺权重时返回 PENDING 而不是 FAIL
python scripts/repro_gate.py --offline  # 不需要 hdf5 / GPU 的部分
```

`repro_gate.py` 的八行含义见 `docs/REPRODUCE_FULL.md` §9。

---

## 2. §3.1 诊断数字

| 数字 | 产物 | 命令 |
|---|---|---|
| 宽度中位 0.351、74% 短于它 | `results/curve_availability/w34_pcnet.json` | `python csr/analysis/w34_curve_usability.py ...` |
| Charades ORACLE +23.21 | `results/oracle_bounds/w6_charades_oracle.log` | `python csr/analysis/oracle_*.py` |
| ANET ORACLE +12.2160 | `results/oracle_bounds/anet_oracle_20261018.log` | `python csr/analysis/diag41_anet_oracle.py ...` |
| CNM 中位宽度 0.4499 | `results/curve_availability/w36_cnm*.json` | `python csr/analysis/w36_cnm_curve.py ...` |

## 3. §3.2–§3.3 机制数字（2026-10-19 补齐）

| 数字 | 产物 | 命令 | 判定 |
|---|---|---|---|
| 基线中心 MAE 0.1699 | `results/estimator_grid/estimator_grid.json` | `python scripts/repro_estimator_table.py --check` | 逐位 |
| argmax 0.1742 / dirAcc 0.684 | 同上 | 同上 | 逐位 |
| softmax τ=.25 0.1499 / 0.770 | 同上 | 同上 | 逐位 |
| 窗内质心 β=.5/.3/.2 = 0.1441/0.1443/0.1483 | 同上 | 同上 | 逐位 |
| 上述三档 dirAcc 0.769/0.778/0.769 | 同上 | 同上 | 逐位 |
| 收缩斜率 1.763（兑现 56.7%） | `results/estimator_grid/contraction_slope.json` | `python scripts/repro_contraction_slope.py --check` | 逐位（rep 0） |
| CPU 后处理 0.0216 ms/查询 | `results/latency/w31_charades_20260914.json` | `python scripts/bench_csr_latency.py --check` | 归档实测（本机复测 0.0143 ms） |
| top-1 前向 6.22 ms、50 点曲线 8.70 ms | `results/latency/w31_gpu_20260914.log` | 同上（仅归档，GPU 相关） | 归档实测 |
| 十六条免真值信号的最佳 AUC | `results/diagnostics/diag38_20261019.log` | `python csr/diagnostics/diag38_signal.py <cfg> <ckpt>` | 见该日志 §4/5 |

> 估计器表的 `dirAcc` 分母是偏移子群 `|c_b − c_g| > 0.02` 的 **2,750** 个样本；
> 主表 `dir_acc` 的分母是 `|c_b − c_g| ≥ 0.08` 的 **1,728** 个样本。
> 两个数不同源，**不可互相换算**。

## 4. §3.4 适用域

| 数字 | 产物 | 命令 |
|---|---|---|
| 18 个工作点、a\* ∈ (0.510, 0.569)、≈0.543 | `results/curve_availability/w47_traj_calibration.json` | `python csr/analysis/w47_traj_calibration.py` |

## 5. §4.1–§4.6 实验数字

| 数字 | 产物 | 命令 |
|---|---|---|
| 900 组网格 → 217 / 97 / 81 / 37.3% | `results/grid_transfer/grid_900.{log,json}` | `python csr/analysis/w62_grid_transfer.py --json ...` |
| 分桶、CI、McNemar | `results/analysis_scan_center_final.log` | `python csr/analysis/scan_center_final.py` |
| 子群 n=1068 / 1224 | `results/analysis_attrib_csr.log` | `python csr/analysis/attrib_csr.py ...` |
| 5-seed ±、p 值 | `results/multiseed/s1/*.stats.json` | `python scripts/s2_stats.py`（检验；单 seed 向量由 `s1_seed_ops.py` 导出） |
| ISR +1.416 / +1.380 | `results/curve_availability/w44_effectsize.json` | 归档产物（生成器 w44 未入发布包，数值已被 `s2_stats.py` 独立复算） |
| TACoS +0.10 / 长 +2.51 / p=3.5e-5 | `results/multiseed/s1/tacos.stats.json` | `python scripts/s2_stats.py` |
| 重训 seed 8：46.58 / 54.17 / 25.36 | `results/multiseed/s1/s8_test_trivial.npz` | `python -c "import numpy as np;b=np.load('results/multiseed/s1/s8_test_trivial.npz')['iou__base'];print(100*b.mean(),100*(b>=.5).mean(),100*(b>=.7).mean())"` |
| 全监督协议 46.51 / 54.13 / 25.16 | **无产物，已从手稿删除** | —— |

## 6. 训练可复现性

| 项 | 判据 | 产物 |
|---|---|---|
| CNM 从零重训 | 100 batch × 6 loss = 600 次比对，max\|Δ\| = 0.000e+00 | `results/cnm_training/` |
| 发布权重 = 论文权重 | 2 切分 × 5 指标，16 位小数相同 | `results/main_table/pcnet_r4_train.json` |
| 权重完整性 | 7 个文件 md5 | `checkpoints/WEIGHTS.md5`、`checkpoints/WEIGHTS.release.md5` |

## 7. 三种 base 口径（不可混引）

| 口径 | 值 | 出处 |
|---|---|---|
| 官方论文 / 官方 harness | 47.12 | PC-Net 论文；`checkpoints/CharadesSTA/8_2025-05-14_13-26-12.log` |
| 本仓库 `eval_csr`（表 2） | 47.20 | `results/main_table/pcnet_eval_csr.json` |
| 从零重训 5 seed | 46.3–46.6 | `results/multiseed/s1/*.stats.json` |

同源比较只允许在同一口径内部做差；跨口径只能比 Δ。

## 8. 已知不可复现 / 已降级

| 项 | 状态 |
|---|---|
| 全监督协议 46.51 / 54.13 / 25.16 | 无一手产物，2026-10-19 从手稿删除 |
| `/dev/shm` 上的临时缓存 | 系统会清理，不可依赖；改用页缓存 |
| `official_8` 权重 md5 | 文件已不在盘上；改以行为等价证明（2 切分 × 5 指标逐位） |
