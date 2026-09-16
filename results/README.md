# results —— 论文中「重测值」与原始证据

本目录只放两类东西：**论文正文引用但取值与已发表数字不同**的复现结果，
以及**判据类主张的原始 JSON**。每个子目录都有自己的 README。

## `anet_cg_remeasured.json`

ActivityNet-CG 三个切分的**重测基线**。论文正文与已发表值存在 3–4 mIoU 的差距，
该文件是这一差距的唯一依据：逐 seed（8 / 11 / 12）的 base mIoU 为
0.3239 / 0.3089 / 0.3043，聚合后作为本文对比所用的基线。
任何引用 ANET 绝对数值的地方都必须落到这个文件，而不是已发表值。

## `curve_availability/`

「曲线可用性」第二坐标（论文 §3.6 / §4.10、附录 B.2–B.3）的全部原始证据：

- 5 个骨干的收敛工作点（`w34_*.json`、`w36_cnm.json`）
- 3 个骨干沿训练 epoch 的轨迹点（`w45_*.json`，16 次）
- 去重后的 **18 个互异工作点**汇总与门限反解（`w47_traj_calibration.json`）
- 论文图 3 的生成图与脚本
- 第二条读取路径（原生高斯参数，不替换任何模块）与 PPS 结构探测

核心结论：沿第二坐标 `a`（曲线方向准确率）符号**完全单调** ——
`ΔmIoU < 0` 的点其 `a ≤ 0.510`，`ΔmIoU > 0` 的点其 `a ≥ 0.569`。

详细说明见 `curve_availability/README.md`。

## 复现

```bash
# 门限反解（需先有 curve_availability/ 下的 json）
python csr/analysis/w47_traj_calibration.py

# 重画论文图 3
python csr/analysis/w48_fig3_validity_map.py

# 命题 1 / 命题 2 的数值核对（闭式 κ 与两参数反解）
python csr/analysis/w46_prop2_proof_check.py
```
