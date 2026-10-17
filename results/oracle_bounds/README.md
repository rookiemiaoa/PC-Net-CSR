# ORACLE_c 上界：论文 §3.1「余量在位置、不在范围」的唯一依据

本目录存放论文第 3.1 节两个 oracle 数字的**原始运行日志**。
正文写法：`Charades-CG +23.2`、`ActivityNet-CG +12.2`（单位为 mIoU 点）。

| 数据集 | 脚本 | 日志 | N | K_REP |
|---|---|---|---:|---:|
| Charades-CG Test-Trivial | `csr/analysis/w6_centre_bias.py` | `w6_charades_oracle.log` | 3,096 | 3 |
| ActivityNet-CG Test-Trivial | `csr/analysis/diag41_anet_oracle.py` | `diag41_anet_oracle.log` | 15,712 | 3 |

`ORACLE_c` = **只把预测框中心替换为真值中心、保留预测框宽**。它不是方法，
只在分析中使用真值，用于把「中心误差」与「宽度误差」分离。

---

## 1. Charades-CG：+23.2（日志原值 +23.21）

`w6_charades_oracle.log` 第 38–39 行：

```
base: ALL mIoU=0.4720@.5=0.5522@.7=0.2658 | SHORT 0.4475/0.5128/0.2408 | LONG 0.5416/0.6646/0.3371
  完美中心(框宽不变)   mIoU=0.7040 @.5=0.8579 @.7=0.5388 | ΔmIoU +23.21
```

- ΔmIoU = 0.7040 − 0.4720 = **+0.2321** ⇒ **+23.21 点**
- 同批日志另给两个对照，说明「知道方向但不知道幅度」远不够：
  方向（估计）+ 全局平均幅度 **−1.92**；方向（估计）+ 逐样本最优幅度 **+7.15**。
- ⚠️ 旧稿曾把该值写成 `+23.12`（**数字颠倒的笔误**），v10 起统一按两位有效数字写作 `+23.2`。

## 2. ActivityNet-CG：+12.2（日志原值 +12.22）

`diag41_anet_oracle.log` 第 45–50 行：

```
                         R1@0.5         R1@0.7           mIoU
base               0.2864±0.0006   0.1347±0.0003   0.3239±0.0002
ORACLE_c           0.3886±0.0003   0.1945±0.0004   0.4461±0.0002
ORACLE_w           0.3057±0.0005   0.1191±0.0009   0.2670±0.0003
ORACLE_cw          1.0000±0.0000   1.0000±0.0000   1.0000±0.0000
```

- ΔmIoU = 0.4461 − 0.3239 = **+0.1222** ⇒ **+12.22 点**
- R1@0.5 **+10.22 点**、R1@0.7 **+5.98 点**
- 同一日志的宽度侧上界 `ORACLE_w` 为 **−5.70** mIoU ⇒ 宽度轴在 ActivityNet-CG 上是负的。
- ⚠️ 早期（`diag40` 前后）曾记 `+12.21`，`diag41` 以 K_REP=3 正式复算后为 `+12.22`；
  两者差 0.01 属同批次舍入，正文按两位有效数字写作 `+12.2`，与两条记录均相容。

## 3. 复现

```bash
# Charades-CG（需 feat_cache/charades_test_trivial_r3.pkl）
python csr/analysis/w6_centre_bias.py

# ActivityNet-CG（需 c3d 特征 + 官方 ActivityNet 检查点）
python csr/analysis/diag41_anet_oracle.py config/activitynet/main_train_test.json \
       checkpoints/ActivityNet/8_2025-05-14_14-07-12/model-best.pt
```

脚本自动向上定位仓库根目录；特征路径可用 `FEAT_PATH` 指定，
另可用 `FEAT_DIR` / `LM_ROOT` 覆盖默认的查找根目录。

## 4. 与表 3 `ORACLE_c` 的区别

表 3 的 `ORACLE_c`（五个骨干逐行，+22.71 ~ +26.55）是**等后处理框架下**、
同一条读出曲线、同检查点的对比；本目录的 `+23.21` 是 Charades-CG 单骨干
（官方 PC-Net 检查点）的绝对上界。两者**不是同一个量**，不可互相替换引用。
