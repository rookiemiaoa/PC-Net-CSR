# 设计族账本（「六个设计族、逾三百种配置」的依据）

论文 §3.2 的这句话不是形容词，是可数的。本文件把每个族的**脚本、网格常量、变体数**列出来，
任何人可以照着数一遍。

口径：一个「变体」= 一组不重复的算子超参（阈值 × 生长长度 × 方向策略），
在同一 harness、同一权重、同一 Test-Trivial 划分上评测一次。

---

## 1. 六个族

| 族 | 机制 | 脚本 | 网格常量 | 变体数 |
|---|---|---|---|---|
| **U**（union flood） | 从框中心按宽松阈值 flood 到最远，取 s15/e15 与框的并集 | `csr/diagnostics/diag37_symctr.py` | `U_k0.15` | 1 |
| **AGG**（aggregate gate） | RBR 的 flood 扩展量 + k/b/t 三参数联合扫描 | `csr/diagnostics/diag32_shortlong.py` | k∈{0.15,0.50,1.00…}、b∈{0.05,0.10,0.15,0.25}、t 同 | 31 |
| **EGS**（edge-grow） | 从框边界**外侧**往外长，z 不低于 `a·max(z)` 才继续 | `csr/diagnostics/diag35_edgegrow.py` | `A_GRID` 9 × `G_GRID` 5 × `R_GRID` 3 | 135 |
| **SGG**（sym-grow gate） | 用 EGS 的布尔平台判定当门控，开关 RBR 的扩展量 | `csr/diagnostics/diag36_gate.py` | `A_GRID` 8 × `M_GRID` 4 | 32 |
| **CTR**（contrast gate） | 尺度无关的对比度 `ρ = mean(z_out)/mean(z_in)` 逐侧放行 | `csr/diagnostics/diag37_symctr.py` | `THR_GRID` 4 × `M_GRID` 2 × `BETA_GRID` 5 | 40 |
| **SYM**（symmetric grow） | 两侧取**较小**的扩展量，保证对称生长 | `csr/diagnostics/diag37_symctr.py` | SYM2 4×2=8，SYM 4×2×5=40 | 48 |

合计 **287**。再加 `diag39_kratio.py` 的两信号（kratio / margin）× 9 个分位阈值 × 正反方向
= **36** 组零真值规则，总计 **323**。

> 323 ≥ 300，所以「逾三百种配置」成立。
> 未随包发布的早期扫描（d33/d34 系列）没有计入，因此这是一个**下界**。

## 2. 为什么它们都失败（同一条判据）

统一判据：**短时刻（gt ≤ 0.35）三项指标不退化，且长时刻 @0.5 ≥ +2.0、@0.7 ≥ +5.0**。

`diag38_signal.py` 的 AUC 表给出了失败的结构性原因：16 个免真值信号里，
只有 `kratio`（flood 核宽 / 框宽）与 `edgeclose` 明显高于 0.5，
最强的 `kratio` 也只有 0.675（校准集）/ 0.650（测试集）。
即「这一侧该不该扩」在当前特征下**几乎不可分**，而不是阈值没调好。

产物：`results/diagnostics/diag38_20261019.log`。

## 3. 复算

```bash
python csr/diagnostics/diag35_edgegrow.py <cfg> <ckpt>   # 135
python csr/diagnostics/diag36_gate.py      <cfg> <ckpt>   #  32
python csr/diagnostics/diag37_symctr.py    <cfg> <ckpt>   #  89 + 1
python csr/diagnostics/diag32_shortlong.py <cfg> <ckpt>   #  31
python csr/diagnostics/diag39_kratio.py    <cfg> <ckpt>   #  36
```

每个脚本都会在运行时打印 `参数网格 = N`，可直接核对上表。
