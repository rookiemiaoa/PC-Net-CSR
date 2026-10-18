# 实现映射（谁算的、谁只是证据）

> 这份文件回答一个具体问题：**论文里的每一个数字，究竟是哪一行代码算出来的？**
> 仓库里同一个算子存在多份历史实现，若不标清楚，读者改了一处会以为改了全部。

---

## 1. 三个算子的权威实现

| 算子 | 权威位置 | 说明 |
|---|---|---|
| **CSR**（保宽，重定位中心） | `csr/ops.py::refine_center_c2`（只要中心）<br>`csr/ops.py::refine_center`（要框，返回 `(lo, hi, moved, shift)`） | 五步：z-score→平滑(7) → 窗内正部质心(β=0.20) → dead-zone(ρ=0.10) → 阻尼(α=1.00) → 自洽门控(δ=0.01)。`refine_center` 只是把 `c2` 换成 `clip(c2 ∓ wb/2)`，**返回值是 4 元组** |
| **ISR**（保宽，迭代收缩窗） | `csr/ops.py::isr_shift` | 返回**位移 d**（不是中心）；调用方再做 `c2 = clip(cb + d, 0, 1)` |
| **RBR**（扩框） | `csr/ops.py::flood_expand` | 阈值 `z > mean(z) + k*std(z)`；边界取网格中心 `centers[a]` |

⚠️ **不要用 `(lo + hi) / 2` 反推中心**：框被 `[0,1]` 裁剪后该式不成立，必须用 `refine_center_c2`。

---

## 2. 已知的「非权威」副本（保留，但数字不可互引）

| 位置 | 是什么 | 与权威实现的差异 |
|---|---|---|
| `csr/csr.py::CenterShiftRefiner` | torch 版 CSR，端到端 monkey-patch `proposal_generator` 读曲线 | 数学一致，但**只用于库调用/说明**；仓库内没有任何评测脚本调用它 |
| `csr/refine.py::RelevanceBoundaryRefiner` | torch 版 RBR | 与 `ops.flood_expand` 数学一致 |
| `scripts/s1_seed_ops.py::csr_center` | 多种子脚本的本地 CSR | 与 `ops.refine_center_c2` 数学一致，但**未收归**（改这里不影响正文数字） |
| `scripts/s1_seed_ops.py::flood_expand` | 多种子脚本的本地 RBR | **口径不同**：绝对阈值 `z >= k` + bin 边界 ⇒ 与表 2 的 RBR 列不可互引 |
| `csr/analysis/diag41_anet_oracle.py::csr_center` | ANET oracle 诊断脚本 | 数学一致；**论文 §3.1 的 ANET 上界由它产出**，因此它是「冻结证据」，不要改 |
| `csr/analysis/scan_center_final.py::csr` | 离线稳健性扫描 | 数学一致（基线框内均值用 `bL/bH` 而非 `cb±wb/2`，边界样本略有差异） |

**为什么不一并统一？** 除 RBR 的口径差异是**有意设计**外，其余副本都已被
`csr/tools/parity_check.py` 在 3×3096 条真实曲线上证明逐元素相等；改动它们不会提升
正确性，却会让「论文数字由哪版代码产出」变得不可追溯。故先标注，不做 de-dup。

---

## 3. 脚本 → 产出 → 论文位置

| 脚本 | 产出 | 进论文哪里 |
|---|---|---|
| `csr/eval/reproduce_base.py` | 官方日志口径的 base 复现值 | `docs/RESULTS.md §1`，作为「环境自检」 |
| `csr/eval/eval_csr.py` | base / RBR×3 / CSR + 短长分桶 + 显著性 | **表 2 的 PC-Net 两行**、§4 分桶、Δ 与检验 |
| `csr/eval/eval_final.py` | base / CSR / ISR(γ.5,T2) / ISR(γ.7,T3) | **§3.4 的 ISR 对照**（只报 mIoU） |
| `csr/eval/eval_anet_support.py` | ANET 的 base / CSR / ORACLE_* / 宽度自适应族 | **§3.1 的 ANET 负结果**（CSR 行） |
| `csr/analysis/diag41_anet_oracle.py` | ANET 的 ORACLE_c / ORACLE_w（正确实现） | **§3.1 的 `+12.2`**（原始日志 `results/oracle_bounds/`） |
| `csr/analysis/w6_centre_bias.py` | Charades 的 ORACLE_c（+23.2） | **§3.1 的 `+23.2`**（原始日志 `results/oracle_bounds/`） |
| `csr/eval/eval_csr_cnm.py` | CNM 上零调参迁移（三切分） | **表 2 的 CNM 行**、§4 跨骨干 |
| `csr/eval/eval_cnm_base.py` | CNM 官方口径 base | 同上（base 列） |
| `csr/baselines/*` + `csr/baselines/run/*.sh` | CPL / CCR / PPS / QMN 的重训与评测 | **表 2 的四个外部基线行** |
| `scripts/s1_seed_ops.py` + `scripts/s2_stats.py` | 5 seed × 3 切分的逐样本 IoU 与统计检验 | **§4.13 多种子**、§4.7 显著性 |
| `scripts/render_table3.py` | 五骨干等后处理对比表 | **表 3**（30 个格子，`--check` 与论文一致） |
| `csr/analysis/w47_traj_calibration.py` | 18 个互异工作点、`a*` 反解 | **表 4 的 a 坐标、图 3 的门限** |
| `csr/analysis/w48_fig3_validity_map.py` | 有效性地图 PNG | **图 3** |
| `csr/analysis/{scan_center_final,attrib_csr,ablate_gate,oracle_gate}.py` | 稳健性 / 误伤归因 / 门控消融 / ORACLE 门控 | 正文的消融段（读 `feat_cache/`，无需 GPU） |
| `csr/tools/config_parity.py` | 训练配置与「产出论文权重的日志」逐键对账（44 键） | 不产出数字；**S1 之前必跑**，否则会用错超参（见下） |

**训练超参陷阱（会让复现者拿到完全不同的模型）**：`train.py:46-52` 在读完 JSON 后
**无条件用 argparse 覆盖** `num_iteration / num_props / co_cl_loss / co_qua_loss /
inter_lambda`，其默认值为 `3 / 8 / 1.0 / 1.0 / 0.13`，而论文用的是
`4 / 8 / 0.5 / 1.0 / 0.146`，配置里 `sigma_neg` 也要从发布默认的 `4.5` 改成 `9`。
`config/charades/main_train_test.paper.json` + `--published` 校验即是这一口径的固化：
`python csr/tools/config_parity.py --published` 应输出 `DIFF: 0`，而不带 `--published`
会如实报出 4 处差异。

---

## 4. 缺口（截至 2026-10-17，已做一次全盘检索）

| 论文位置 | 现状 | 说明 |
|---|---|---|
| **表 4 的「训练」列** | 数据**已定位**：开发机 `w61` 批次共 8 个 JSON（`{pcnet,cpl,qmn,ccr}` × `{probe,test}`）；**生成脚本未入库** | 该列（PC-Net 0.742 / CPL 0.508 / QMN 0.491 / CCR 0.453）目前只能作为历史测量引用；脚本取回后才能端到端复现 |
| **表 3 的生成器** | `w34_curve_usability.py` **已入库**（`csr/analysis/`，表 3 的唯一采集器） | 渲染器 `scripts/render_table3.py` 能逐格复现（`--check` 30/30 PASS）；采集器入库后，「在新骨干上重测这 30 个格子」才成为可能。`w32_baseline_csr.py`、`w36_cnm_curve.py` 仍未见（前者被 w34 覆盖，后者只服务已不可得的 CNM） |
| **训练脚本的研究分支** | 本仓库 `train.py` 是 76 行发布版；产出论文数字的是 103 行研究版（多 7 个内部开关） | 见 §5。发布版足以复现 base 与全部 CSR 结果，但复现不出研究版的训练轨迹 |
| **图 3 以外的图** | 架构图 `method_figure/method_pdf.png` 非程序化生成 | 源文件不随仓库分发 |

---

## 5. 发布版 `train.py` 与研究版的关系（重要）

| | 本仓库 `train.py` | 研究分支 `train.py` |
|---|---|---|
| 行数 | 76 | 103 |
| 参数 | 上游参数 + `seed/slot_num_iteration/co_cl_loss/co_qua_loss/num_props/inter_lambda` | 再多 7 个：`depend_center`、`depend_width`、`depend_width_log`、`depend_spread`、`rel_width`、`alpha_dep`、`probe` |
| 用途 | 给复现者一个干净起点 | 内部提案几何实验（探针、依赖项加权），产出论文的训练列 |

**结论**：CSR 是**推理期后处理**，不进入 `forward`，所以用发布版训出的 PC-Net 一样能跑 CSR，
表 2/表 3 的 Δ 不受影响；受影响的只有「逐位重放研究版训练轨迹」这一件事。
那 7 个开关在论文里没有对应消融，属于内部设施，**故意不发布**——不要为了对齐数字去猜它们的语义。
