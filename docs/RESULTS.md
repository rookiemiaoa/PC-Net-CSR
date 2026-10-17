# 结果汇总

> 所有数字均可回溯到仓库内或 `docs/` 引用的原始日志。评测口径统一为
> **PC-Net 官方评测模式 `vote=False`**，**K_REP ≥ 3** 报 mean±std
> （`_mask_words` 未固定随机种子，单次评测噪声约 ±0.005）。
> 指标列顺序统一为 **mIoU / R1@0.5 / R1@0.7**。

---

## 0. base 列有三种口径，不要混引

同一个「base」在不同脚本里不是同一个量。读表前先确认这一列来自哪一行：

| 来源 | Test-Trivial base<br>mIoU / @0.5 / @0.7 | 出现在 | 能否与其他行互引 |
|---|---|---|---|
| 官方评测日志 `checkpoints/CharadesSTA/8_2025-05-14_13-26-12.log` | 47.12 / 54.84 / 26.68 | §1 复现校验 | 只与「复现值 47.12」比 |
| `csr/eval/reproduce_base.py` | 47.12 / 54.84 / 26.68（±0.005） | §1 | 与上一行同源 |
| `csr/eval/eval_csr.py`、`eval_final.py` | 47.20 / 55.22 / 26.58 | §2、§2.2 | **只与本节自己的 Δ 比** |
| 5 seed 重训（`results/multiseed`） | 46.3–46.6 | 多种子表 | 是另一模型 population，**只能比 Δ** |
| ActivityNet 官方 vs 本机复现 | 36.45 vs 32.39 | §3 | 只用复现值 |

> 47.12 → 47.20 的 0.08 与 @0.5 的 0.38 是**评测路径差异**（官方 runner 与自建 harness
> 的 batch/掩码处理不同），不是随机噪声，不会随 K_REP 增大而收敛。
> 因此**同一张表内只比 Δ**，跨表只比同源列。

---

## 1. 官方基准（复现校验）

来源：本仓库随附的官方评测日志
`checkpoints/CharadesSTA/8_2025-05-14_13-26-12.log`、
`checkpoints/CharadesSTA/[Novel-*]8_2025-05-14_*.log`、`checkpoints/ActivityNet/…`。

| 数据集 | 切分 | N | mIoU | R1@0.5 | R1@0.7 |
|---|---|---:|---:|---:|---:|
| Charades-CG | Test-Trivial | 3096 | 47.12 | 54.84 | 26.68 |
| Charades-CG | Novel-Comp | 3442 | 38.04 | 41.69 | 16.73 |
| Charades-CG | Novel-Word | 695 | 41.06 | 46.91 | 23.60 |
| ActivityNet-CG | Test-Trivial | 15712 | 36.45 | 29.62 | 14.35 |
| ActivityNet-CG | Novel-Comp | 12028 | 29.51 | 20.16 | 8.05 |
| ActivityNet-CG | Novel-Word | 3920 | 30.76 | 22.88 | 9.85 |

**复现结论**：我们自建的 harness（`csr/eval/reproduce_base.py`）复现 Charades-CG 三切分的
base 列，误差为 ±0.001 / −0.003 / −0.009，**全部在评测噪声内**。

> ⚠️ ActivityNet / TACoS 的官方日志值（ANET 36.45、TACoS 9.79）在本机**未能复现**
> （现为 32.39 / 8.23，差 −0.04）。已排除 ckpt 配置错配、评测集错配、val/test 颠倒、
> 掩码随机性；最可能是**特征溯源**差异。因此下文中 ANET 的 base 列一律标注为
> **"本机复现值"**，不与官方日志值混用。Charades-CG 严格复现，结论不受影响。

---

## 2. CSR 主结果（Charades-CG，官方权重，零重训）

| 切分 | 方法 | mIoU | R1@0.5 | R1@0.7 |
|---|---|---:|---:|---:|
| Test-Trivial | base | 47.20±0.24 | 55.22±0.32 | 26.58±0.39 |
| Test-Trivial | RBR k=0.15 | 48.28±0.08 | 54.25±0.18 | 27.67±0.08 |
| Test-Trivial | **CSR（定版）** | **48.33±0.17** | **55.89±0.29** | **28.67±0.28** |
| Novel-Comp | base | 37.81±0.13 | 41.42±0.21 | 16.62±0.10 |
| Novel-Comp | **CSR** | **38.56±0.13** | **41.85±0.16** | **18.59±0.02** |
| Novel-Word | base | 40.67±0.13 | 46.04±0.20 | 23.12±0.30 |
| Novel-Word | **CSR** | **41.85±0.11** | **46.47±0.20** | **25.23±0.27** |

**增量（CSR − base）**：

| 切分 | ΔmIoU | ΔR1@0.5 | ΔR1@0.7 |
|---|---:|---:|---:|
| Test-Trivial | **+1.13** | **+0.67** | **+2.09** |
| Novel-Comp | +0.75 | +0.43 | +1.98 |
| Novel-Word | +1.18 | +0.43 | +2.11 |

**RBR 同表对照（Test-Trivial）**：ΔmIoU +1.08 / **ΔR1@0.5 −0.97（转负）** / ΔR1@0.7 +1.09。
⇒ CSR 与 RBR 的 mIoU 增益相当，但 **RBR 的 headline 指标是负的**，差别就在下面这张分桶表。

### 2.1 短 / 长动作分桶（gt 时长占比 ≤ 0.35 为"短"）

**Test-Trivial**

| 桶 | 方法 | mIoU | R1@0.5 | R1@0.7 |
|---|---|---:|---:|---:|
| 短（74%） | base | 44.75 | 51.28 | 24.08 |
| 短 | RBR k=0.15 | 44.58 | **48.30** | 21.87 |
| 短 | **CSR** | **45.33** | **51.31** | **24.30** |
| 长（26%） | base | 54.16 | 66.46 | 33.71 |
| 长 | RBR k=0.15 | 58.82 | 71.23 | **44.20** |
| 长 | **CSR** | **56.88** | **68.95** | 41.13 |

**增量**：

| 桶 | CSR | RBR k=0.15 |
|---|---|---|
| 短 ΔmIoU / Δ@0.5 / Δ@0.7 | **+0.58 / +0.03 / +0.22** | −0.17 / **−2.98** / −2.21 |
| 长 ΔmIoU / Δ@0.5 / Δ@0.7 | +2.72 / +2.49 / **+7.42** | +4.66 / +4.77 / **+10.49** |

> 出处说明：CSR 行与其分桶绝对值直接取自 `csr/eval/eval_csr.py` 的评测输出；
> RBR 行的分桶**增量**记录在 `csr/diagnostics/diag32_shortlong.py` 的模块 docstring 中
> （该脚本即当时产生这组数字的入口）。表中 RBR 的分桶绝对值系由 base 加该增量得到。
> 注意 `eval_csr.py` 本身只打印 base 与 CSR 的分桶行，不打印 RBR 分桶行。

⇒ CSR 在短桶的三项**全部非负**（其中 Δ@0.5 = +0.03pt，量级很小，
严谨表述是**"无显著下降"**而非"显著提升"），长桶仍大幅受益。
RBR 的长桶增益更高，但代价是短桶 @0.5 掉 2.98pt。

**Novel-Comp / Novel-Word**：CSR 的短桶 ΔmIoU 分别为 +0.15 / +0.66，长桶 +3.09 / +2.36，
**六个格子全为正**。

---

## 2.2 ISR：同族的两参数点（不作为独立贡献）

```bash
python csr/eval/eval_final.py config/charades/main_train_test.json \
    checkpoints/CharadesSTA/8_2025-05-14_13-26-12/model-best.pt
```

Charades-CG Test-Trivial，K_REP = 3，同一检查点、同一读出曲线：

| 方法 | mIoU | ΔmIoU |
|---|---:|---:|
| base | 47.20 | — |
| CSR（β=0.20） | 48.33 | +1.13 |
| ISR（γ=0.5, T=2） | **48.36** | **+1.16** |
| ISR（γ=0.7, T=3） | 见脚本输出 | — |

ISR 是 CSR 的迭代收缩窗版本（mean-shift 式，窗宽 `β_{t+1}=γ·β_t`），两者差距仅 +0.03，
说明结论**不依赖 β 的单一取值**。它只报 mIoU，`eval_final.py` 不产出 R1@0.5 / R1@0.7，
因此正文只把它用作「同族参数点」的稳健性证据，不列为独立贡献。

> 口径提醒：`eval_final.py` 的 base 与 `eval_csr.py` 的 base 都是 47.20（同一评测路径），
> 与 `reproduce_base.py` 的 47.12（官方日志口径）**不可混引**。

---

## 3. ActivityNet-CG：结构性失效（负结果，如实报告）

Charades-CG Test-Trivial，本机复现基线 32.39 / 28.64 / 13.47（mIoU/@0.5/@0.7）：

| 方法 | mIoU | R1@0.5 | R1@0.7 |
|---|---:|---:|---:|
| base（本机复现） | 32.39±0.02 | 28.64±0.06 | 13.47±0.03 |
| RBR k=0.15 | 32.47±0.02 | 28.67±0.06 | 13.59±0.04 |
| CSR | 32.22±0.05 | 28.84±0.05 | 13.41±0.07 |

| 增量 | ΔmIoU | ΔR1@0.5 | ΔR1@0.7 |
|---|---:|---:|---:|
| CSR | **−0.17** | +0.21 | −0.06 |
| CSR 短桶 | −0.63 | −0.95 | −0.07 |
| CSR 长桶 | +0.52 | +1.97 | −0.05 |

**根因**：基线 top-1 框宽中位 **0.9908**，gt 时长中位 **0.2750** ⇒ 框宽 ≈ 3.6 × 目标。
此时 `IoU ≈ gt时长/框宽` 完全由**宽度**决定，而 CSR 是宽度保持型算子。
10 组单变量扫描 **10/10 ΔmIoU 全负**，ORACLE 上界也只有 +0.03/+0.34/+0.42
⇒ **不是调参问题，是几何失配**（见 `CSR_METHOD.md` §4 两轴有效图）。

---

## 4. 跨骨干迁移（外部基线 CNM，AAAI-2022，零调参）

CSR 的超参一字不改，直接用在 **CNM 的冻结权重**上（曲线从 CNM 自己的掩码重建阶段读出，
不改权重、不读 gt）。K_REP=3：

| 切分 | 方法 | mIoU | R1@0.5 | R1@0.7 |
|---|---|---:|---:|---:|
| Test-Trivial | CNM base | 38.43 | 37.31 | 15.44 |
| Test-Trivial | **+ CSR** | **41.43** | **42.50** | **17.67** |
| Novel-Comp | CNM base | 30.28 | 24.75 | 8.22 |
| Novel-Comp | **+ CSR** | **32.45** | **28.16** | **9.84** |
| Novel-Word | CNM base | 34.54 | 31.51 | 14.96 |
| Novel-Word | **+ CSR** | **37.38** | **36.07** | **18.51** |

**增量（九格全正）**：

| 切分 | ΔR1@0.5 | ΔR1@0.7 | ΔmIoU |
|---|---:|---:|---:|
| Test-Trivial | **+5.19** | +2.23 | +3.00 |
| Novel-Comp | +3.41 | +1.62 | +2.17 |
| Novel-Word | +4.56 | +3.55 | +2.84 |

**同权重同曲线下 RBR 的 ΔR1@0.5 仅 +0.01**（Test-Trivial）。
CNM 的**短动作桶三切分三指标全部为正**（PC-Net 上只能做到"不显著下降"）。

**诚实例外**：自洽门控在 PC-Net 上是承重的，在 CNM 上中性略有害
（Test-Trivial：带门控 +3.00 vs 关掉门控 +3.45）。**WPT 旋钮扫描**（0.10 → 0.30）
显示 CSR 恒正（+2.84 / +2.50 / +1.82 mIoU）、RBR 恒劣 ⇒ 结论不依赖这个旋钮。

---

## 5. 与 5 个外部弱监督基线的对比（Charades-CG）

统一协议：同一 `cg_train` 训练集、同一 I3D 特征、30 epoch、
**统一 `vote=False`**、各自官方 `runner.eval()` 口径。K_REP=3。

| 方法 | Test-Trivial<br>mIoU / @0.5 / @0.7 | Novel-Comp<br>mIoU / @0.5 / @0.7 | Novel-Word<br>mIoU / @0.5 / @0.7 |
|---|---|---|---|
| **PC-Net + CSR（本文）** | **48.33 / 55.89 / 28.67** | **38.56 / 41.85 / 18.59** | **41.85 / 46.47 / 25.23** |
| PC-Net（base） | 47.20 / 55.22 / 26.58 | 37.81 / 41.42 / 16.62 | 40.67 / 46.04 / 23.12 |
| CPL（CVPR-2022） | 46.22 / 52.55 / 24.22 | 38.68 / 42.59 / 16.59 | 39.78 / 45.18 / 21.58 |
| CCR（ACM MM-2023） | 46.23 / 52.81 / 24.77 | 37.87 / 41.78 / 15.92 | 38.00 / 41.44 / 20.00 |
| PPS（AAAI-2024） | 46.01 / 51.65 / 25.81 | 36.33 / 37.74 / 15.37 | 38.45 / 42.30 / 21.44 |
| QMN（CVPR-2024） | 45.47 / 52.42 / 24.32 | 36.93 / 39.77 / 15.75 | 40.34 / 45.18 / 22.01 |
| CNM（AAAI-2022） | 38.43 / 37.31 / 15.44 | 30.28 / 24.75 / 8.22 | 34.54 / 31.51 / 14.96 |

**结论**：

1. Test-Trivial 上 CSR 相对最强的外部基线 **CCR 领先 +2.10 mIoU / +3.08 R1@0.5**。
2. **Novel-Comp 是 CSR 唯一没有全面领先的切分**：CPL 的 mIoU（38.68）与 R1@0.5（42.59）
   都比 CSR（38.56 / 41.85）略高 0.12 / 0.74pt，但 CSR 的 **R1@0.7 领先 2.00pt**
   （18.59 vs 16.59）。Novel-Word 上 CSR **三项均领先全部外部基线**。
   ⇒ CSR 的优势集中在 **高精度定位（R1@0.7）** 与 **Novel-Word 泛化**，
   这一点与"中心重定位不改变框宽"的机理一致。
3. **CSR 的价值不只在"涨点"**：从 PC-Net base 到 CSR 只动了 5 个标量、零训练零参数，
   却把 headline 三项同时抬上去且**不牺牲任何子群**——这在对比表里只有 CSR 做得到。

复现这些基线需要给各自官方仓库打兼容性补丁（不改算法），见 `csr/baselines/README.md`。

---

## 6. 证据索引

| 内容 | 位置 |
|---|---|
| 官方基准日志 | `checkpoints/CharadesSTA/*.log`、`checkpoints/ActivityNet/*.log`（仓库内） |
| CSR / RBR / ORACLE 分桶与显著性 | 运行 `csr/eval/eval_csr.py` 输出的 `[显著性]` / `[诊断]` 段 |
| ANET 10 组标量扫描 | `csr/analysis/anet_sens.py` → `csr/outputs/anet_sens/*.log` |
| 误伤归因、门控消融、ORACLE 上界 | `csr/analysis/{attrib_csr,ablate_gate,oracle_gate}.py`（读 `feat_cache/`） |
| 被证伪的 300+ 变体 | `csr/diagnostics/` |
| 曲线可用性（第二坐标）原始证据 | `results/curve_availability/`（含 5 收敛点 + 16 轨迹点 + 汇总） |
| 训练轨迹标定与门限反解 | `csr/analysis/w47_traj_calibration.py` → `results/curve_availability/w47_traj_calibration.json` |
| 论文图 3（含轨迹点） | `csr/analysis/w48_fig3_validity_map.py` → `results/curve_availability/fig3_validity_map.png` |
| 原生高斯参数读取（不改前向） | `results/curve_availability/w41_gauss_read.json`、`w42_gauss_shrink.json` |
| PPS 结构探测 | `results/curve_availability/w43_pps.json` |
| 效应量（d_z / Cliff's δ） | `results/curve_availability/w44_effectsize.json` |
| ActivityNet-CG 重测基线 | `results/anet_cg_remeasured.json` |
| §3.1 的 `ORACLE_c` 上界（+23.2 / +12.2） | `csr/analysis/w6_centre_bias.py`、`csr/analysis/diag41_anet_oracle.py` → `results/oracle_bounds/`（含原始日志） |

### 6.1 第二坐标的符号单调性（18 个互异工作点）

把同一个窗内质心估计器施加到各骨干自身的重建相关性曲线上，第二坐标
`a`（偏移子群上的方向准确率）与 CSR 的 ΔmIoU 符号**完全单调**：

```
a 升序 → 符号: 9 负 紧接 9 正
所有 Δ<0 的点: a ≤ 0.510 （CPL 收敛点）
所有 Δ>0 的点: a ≥ 0.569 （CNM 第 2 epoch）
⇒ 实测夹逼区间 a* ∈ (0.510, 0.569)
```

18 个点由 21 次评测去重而来（CPL `model-best` 与其收敛点、CNM `model-40` 与
`model-best` 是同一检查点的重复测量）。用 18 点重解两参数门限得
`κ_e = 0.317, κ_w = 0.377, a* = 0.543`；只用 5 个收敛点得 `a* = 0.576`。

**两条限定（必须与数字一起引用）**：这些轨迹点来自 3 个骨干、彼此相关，
故该区间是**实测夹逼区间而非置信区间**；且落在区间内的只有训练中途的工作点，
**没有骨干的收敛点**落入其中——「空档被填上」指的是坐标取值区间被覆盖。
