# multiseed —— 多种子训练与正式显著性检验的原始证据

本目录是论文 **§4.13「多种子稳健性」** 与 **§4.7「统计显著性检验」** 的全部原始证据。

回答一个问题：

> CSR 那 +1.13 mIoU 的改进，是**真实效应**，还是**单次评测的噪声 / 单一种子的运气**？

---

## 1. 结论

**是真实效应。** 5 个独立训练的种子 × 3 个 Charades-CG 切分 = **15/15 全部为正**，
层级 bootstrap 的 95% CI 全部排除 0：

| 切分 | ΔmIoU（mean ± std，5 seed） | 正值数 | 层级 bootstrap 95% CI |
|---|---:|---:|---|
| Test-Trivial | +1.37 ± 0.21 | 5 / 5 | [1.15, 1.57]，排除 0 |
| Novel-Comp | +0.97 ± 0.08 | 5 / 5 | [0.83, 1.11]，排除 0 |
| Novel-Word | +1.60 ± 0.12 | 5 / 5 | [1.30, 1.89]，排除 0 |

### 对照：RBR 的子群方向偏置

同一批种子、同一口径下，RBR（k=0.15）的表现**依赖子群**：

| 切分 | RBR 短桶 ΔmIoU | 正值种子数 |
|---|---:|---:|
| Test-Trivial | **−1.02** | **0 / 5** |
| Novel-Comp | **−0.77** | **0 / 5** |
| Novel-Word | −0.00 | 2 / 5（量级为 0） |

而 CSR 在**全部 9 个（切分 × 桶）格子**上均为 5/5 为正。
这正是"扩框有子群方向偏置、改中心没有"的直接证据
（RBR 的长桶增益更高：+8.81 / +6.69 / +7.47，但代价由短桶支付）。

---

## 2. 目录结构

```
results/multiseed/
├── README.md                    ← 你在这里
├── s1/                          逐样本导出与检验结果（48 个文件）
│   ├── s8_test_trivial.npz      种子 8 的逐样本 IoU 向量
│   ├── s8_test_trivial.stats.json
│   ├── s11_*.npz / s12_*.npz / s13_*.npz / s14_*.npz
│   ├── 5seed_test_trivial.stats.json   ← 5 种子聚合检验（正文引用）
│   ├── 5seed_novel_comp.stats.json
│   ├── 5seed_novel_word.stats.json
│   ├── anet_*_seed{8,11,12}.npz       ← ActivityNet 三种子
│   └── tacos.npz / tacos.stats.json
└── multiseed/
    ├── seed11.log               4 个种子的完整训练日志
    ├── seed12.log
    ├── seed13.log
    └── seed14.log
```

对应的 checkpoint 位于 `checkpoints/CharadesSTA/{8,11,12,13,14}_*/model-best.pt`。

---

## 3. 复现命令

```bash
# ① 逐样本导出（每个种子跑一次；会写 s1_<TAG>.npz）
python scripts/s1_seed_ops.py config/charades/main_train_test.json \
       checkpoints/CharadesSTA/8_2025-05-14_13-26-12/model-best.pt s8_test_trivial
# 环境变量: BS=16 K_REP=3 OUT_NPZ=<输出路径>

# ② 单种子内检验（随机化 / 配对 t / Wilcoxon / McNemar / Holm / TOST）
python scripts/s2_stats.py within results/multiseed/s1/s8_test_trivial.npz

# ③ 跨种子聚合（层级 bootstrap：先重采样 seed、再在 seed 内重采样样本）
python scripts/s2_stats.py across \
    results/multiseed/s1/s8_test_trivial.npz \
    results/multiseed/s1/s11_test_trivial.npz \
    results/multiseed/s1/s12_test_trivial.npz \
    results/multiseed/s1/s13_test_trivial.npz \
    results/multiseed/s1/s14_test_trivial.npz \
    --json out/5seed.json
```

---

## 4. 检验方法选择依据（近五年文献）

`scripts/s2_stats.py` 的方法组合不是随意选的：

1. **配对随机化检验**（主检验）—— Ihemelandu & Ekstrand, SIGIR 2023
   *Inference at Scale* 实测：N 达数千时 Wilcoxon / Sign 的 Type-I 错误率显著膨胀，
   而 randomization / bootstrap / t 检验与名义 α 一致。故本工作以随机化检验为主，
   **Wilcoxon 仅作稳健性参照并明示其在大样本下的缺陷**。
2. **配对 bootstrap 95% CI**（percentile）—— Δ 的区间估计。
3. **McNemar 精确检验**（二项精确，非 χ² 近似）—— 阈值化指标 R1@0.5 / R1@0.7。
4. **等价检验 TOST + 无差异区间** —— Sakai, SIGIR 2025
   *My System Is As Effective As Yours*：**不显著 ≠ 无差异**，
   要主张"两轴皆无效"必须做等价检验并预先声明无差异区间（本工作设 `--iz 0.005`）。
5. **效应量**（Cohen's d_z、Cliff's δ、MDE）—— 大样本下几乎所有检验都会显著，
   此时应以效应量判定实践意义。
6. **Holm–Bonferroni** —— 控制同一族内的多重比较。
7. **层级 bootstrap** —— 让"种子方差"与"评测方差"都被计入 Δ 的置信区间。

---

## 5. 一条必须遵守的纪律：阈值指标要逐 rep 判定后平均

`_mask_words` 未固定随机种子，因此阈值指标（R1@0.5）**必须逐 rep 分别判定是否命中，
再对 0/1 结果取平均**；若先把分数跨 rep 平均再判定，会出现"平均后跨线"的假阳性。

本工作早期曾因此把 Test-Trivial 的 R1@0.5 误算为 **54.17**（正确值 **53.07**），
并据此错误地宣称"CSR 在 @0.5 上不显著"。该结论**已撤销**。
`scripts/s2_stats.py` 已按逐 rep 判定实现。

---

## 6. 三条诚实限定

1. 5 个种子全部是 **Charades-CG** 切分；ActivityNet 只有 3 个种子
   （`anet_*_seed{8,11,12}.npz`），且其 base 列为**本机复现值**（见 `docs/RESULTS.md`）。
2. 种子的随机性只覆盖**训练初始化与数据顺序**，不覆盖评测的掩码随机性
   （后者通过 `K_REP ≥ 3` 与层级 bootstrap 吸收）。
3. **这 5 个种子的 base（0.463–0.466）低于官方权重的 0.4720，是正常的**：
   它们是 5 次**重新训练**得到的检查点，不是同一个模型重复评测 5 次；
   base 的差异来自训练随机性，因此本目录的表**只能读 Δ，不能读绝对值**，
   也不要把这里的 0.4658 与 §1 的 47.12 / §2 的 47.20 相减。

   > 溯源缺口（2026-10-17 发现）：早期 npz 只存了 `meta__N` / `meta__K_REP`，
   > **没记检查点路径与 md5**，所以无法逐条反查每个 seed 对应哪个 `model-best.pt`。
   > `scripts/s1_seed_ops.py` 已补 `[PROV]` 日志行与
   > `meta__ckpt / meta__ckpt_md5 / meta__git_head / meta__split` 四项，
   > 重跑一次即可把这些字段写进新 npz；旧 npz 保持原样、不做推测性补写。
