# `csr/analysis/` —— 离线分析

这些脚本**不跑模型**，直接读 `feat_cache/` 里的逐样本中间量，秒级出结果。
主流程（`csr/eval/`）不依赖本目录，`feat_cache/*.pkl` 也可以直接删掉。

---

## 1. 特征缓存

**文件名约定**：`<数据集>_<切分>_r<K_REP>.pkl`
→ `charades_test_trivial_r3.pkl` = 3 次重复、每次 n = 3096；
  `charades_train_r1.pkl` = 1 次、n = 8281（开发集校准用）。

**结构**：`list`，长度 = 重复次数；每个元素是一个 dict：

| 键 | 形状 | 含义 |
|---|---|---|
| `GS`, `GE` | (n,) | gt 起止（归一化到 [0,1]） |
| `GL` | (n,) | gt 时长 = `GE − GS` |
| `cb`, `wb` | (n,) | 基线 **top-1 框**的中心 / 宽度 |
| `bL`, `bH` | (n,) | 基线 top-1 框的左 / 右边界 |
| `bli`, `bhi` | (n,) int | 基线框映射到 L=50 网格上的左右下标（闭区间） |
| `REL` | (n, 50) | 逐点提案的原始相关性 `r(t) = −nll(t)`，L = `max_num_frames // 4` = 50 |
| `Z` | (n, 50) | `REL` 的**逐样本 z-score**。⚠️ **未平滑** —— 平滑窗口由使用方按需施加 |
| `s15`, `e15` | (n,) | RBR（k=0.15）flood 出的左 / 右边界，作对照用 |
| `N1`, `N2` | (n,) | 前两名提案的 nll（`margin = N2 − N1`） |
| `c1`, `c2` | (n,) | 前两名提案的中心（`c1` 即基线 top-1 的中心） |
| `n` | int | 样本数 |

> 字段定义以产生缓存的代码为准：`csr/diagnostics/diag38_signal.py` 中
> `collect()` 末尾的 `out.append(dict(...))`（"v2 缓存"，额外保存 `REL` / `N1` / `N2`）。
> 已实测校验：`Z` 与 `(REL − REL.mean(1)) / (REL.std(1) + 1e-8)` 的**逐元素最大差为 0**
> ⇒ `Z` 确实是未平滑的 z-score，使用方可以自由换平滑窗口。
>
> **缓存不需要重新生成**即可跑本目录的脚本。若要重建，跑 `csr/diagnostics/diag38_signal.py`
> （`CACHE_DIR` 默认就指向本目录）。

---

## 2. 脚本

| 脚本 | 输出 | 对应结论 |
|---|---|---|
| `scan_center_final.py` | 逐 rep 指标与 mean±std、定版稳健性扫描、**配对 bootstrap（B=2000）+ McNemar** | `RESULTS.md` §2 的显著性 |
| `attrib_csr.py` | 按 `|c_b − c_gt|` 分箱的 n / 移动率 / ΔmIoU / Δ@0.5 / Δ@0.7 | 误伤归因：短样本伤害来自"中心本来就准"的那部分 |
| `ablate_gate.py` | 关掉自洽门控前后的对比 | 门控在 PC-Net 上是承重的 |
| `oracle_gate.py` | ORACLE 门控上界 + `ORACLE_c`（只换 gt 中心、保持框宽） | 证明"存在解"，缺口是逐样本选择 |
| `anet_sens.py` | 10 组单变量扫描的汇总表（会调用 `csr/eval/eval_csr.py`） | `RESULTS.md` §3：10/10 全负 |

```bash
python csr/analysis/scan_center_final.py
python csr/analysis/attrib_csr.py
python csr/analysis/ablate_gate.py
python csr/analysis/oracle_gate.py
python csr/analysis/anet_sens.py            # 需要 GPU + ANET 权重与特征
```

⚠️ `oracle_gate.py` **使用 gt**，只能当作**上界**看，其结果不能当作方法性能。
其余脚本均为零 gt 推理。

---

## 3. 缓存路径

默认 `<仓库根>/csr/analysis/feat_cache`，可用环境变量覆盖：

```bash
export CACHE_DIR=/path/to/feat_cache          # 或 FEAT_CACHE_DIR
```

## 曲线可用性（第二轮）

| 脚本 | 作用 | 数据 |
|---|---|---|
| `w46_prop2_proof_check.py` | 命题 1 闭式 κ(h)=1−m(h)/h 的数值核对；命题 2 两参数最小二乘反解 | 自算（无外部输入） |
| `w47_traj_calibration.py` | 5 收敛点 + 16 轨迹点 → 18 个互异工作点；符号单调性、实测夹逼区间、门限重解 | `results/curve_availability/w34_*.json`、`w36_cnm.json`、`w45_*.json` |
| `w48_fig3_validity_map.py` | 论文图 3：第一坐标 ρ_geo 与第二坐标 a 的双坐标图（含轨迹点） | 同上（数值内嵌于脚本 docstring 说明） |

`w47` 的默认数据目录是包内 `results/curve_availability/`，可用
`python w47_traj_calibration.py <dir>` 或环境变量 `CSR_CURVE_DIR` 覆盖。
