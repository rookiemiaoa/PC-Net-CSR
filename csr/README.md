# `csr/` —— PC-Net + CSR：推理期中心重定位精修

本目录是本文相对**官方 PC-Net 仓库**的全部新增内容。

> **对 backbone 的改动量：零。** 官方 `models/`、`datasets/`、`runners/`、`optimizers/`、
> `train.py` 等一个字节都没有改；CSR 只在运行时 monkey-patch 临时替换
> `model.proposal_generator` 与 `model.num_props`，不落盘改官方代码。
> 溯源核验见 `docs/UPSTREAM_VERIFICATION.md`。

---

## 1. 一句话

- **问题**：PC-Net 的弱监督框在长动作上"起止点连接失败"；而直接扩框（RBR）的效应符号是
  `sign(gt − 框宽)` ⇒ 存在**子群方向偏置**，短动作 R1@0.5 掉 2.98pt，headline 转负。
- **解法**：固定框宽时 IoU 对中心偏移是**单峰偶函数** ⇒ 中心修正**无子群偏置**。
  CSR 沿模型自己的重建相关性曲线重定位框的**中心**，**宽度保持不变**。
- **结果**（Charades-CG Test-Trivial，官方权重，K_REP=3）：
  mIoU 47.20 → **48.33**、R1@0.5 55.22 → **55.89**、R1@0.7 26.58 → **28.67**；
  短/长/整体三个子群**同向非负**；零调参搬到 CNM 后**九格全正**。

方法细节 → `docs/CSR_METHOD.md`；全部数字 → `docs/RESULTS.md`；跑起来 → `docs/REPRODUCE.md`。

---

## 2. 目录与文件对照表

> eval 脚本采用**语义化命名**；历史编号保留在各脚本 docstring 首行，
> 对应的映射见下方 §2.1。

### 2.1 历史编号 ↔ 语义化命名

部分脚本的 docstring 首行仍保留最初的实验编号（如 `w11_eval_final.py`），
按编号检索论文/实验日志时请对照下表：

| docstring 首行的历史名 | 现文件名 |
|---|---|
| `w11_eval_final.py` | `csr/eval/eval_final.py` |
| `w30_main_ops.py` | 见独立发布包 `PC-Net-CSR-release/eval_all_ops.py` |
| `w46_prop2_proof_check.py` | `csr/analysis/w46_prop2_proof_check.py`（名未变） |
| `w47_traj_analysis.py` | `csr/analysis/w47_traj_calibration.py` |
| `w48_fig3_b.py` | `csr/analysis/w48_fig3_validity_map.py` |

```
csr/
├── __init__.py                  惰性导出 CenterShiftRefiner / RelevanceBoundaryRefiner；直接导出 ops
├── paths.py                     统一的路径解析（本目录是唯一定义处）
├── ops.py                       ★ 纯 NumPy 算子（iou/平滑/质心/中心重定位/flood），单一权威实现
├── csr.py                       ★ CSR 算子：CenterShiftRefiner（核心贡献）
├── refine.py                    ★ RBR 算子：RelevanceBoundaryRefiner（对照）
├── eval/                        需要 GPU + 权重，产出论文表格里的数字
│   ├── reproduce_base.py        基线复现 harness（历史名 diag30_gen.py）
│   ├── eval_csr.py              CSR 主验证（历史名 diag40_gen.py）
│   ├── eval_anet_support.py     ANET-CG 根因诊断（历史名 diag41_gen.py）
│   ├── eval_csr_cnm.py          跨骨干迁移：CSR 零调参搬到 CNM（历史名 csr_cnm.py）
│   └── eval_cnm_base.py         CNM 官方口径评测包装（历史名 eval_cnm.py）
├── analysis/                    不跑模型，读 feat_cache 秒级出结果
│   ├── attrib_csr.py            逐样本误伤归因（按 |c_b − c_gt| 分箱）
│   ├── ablate_gate.py           门控消融（关掉自洽门控）
│   ├── oracle_gate.py           ORACLE 门控上界（用 gt，仅作上界）
│   ├── scan_center_final.py     定版配置的稳健性扫描 + 配对显著性检验
│   ├── anet_sens.py             ANET 10 组单变量扫描的自动化驱动
│   ├── feat_cache/*.pkl         可选派生件：逐样本相关性曲线缓存（字段见同目录 README）
│   └── README.md                缓存字段说明
├── diagnostics/                 记录被证伪的 300+ 变体（负结果，见 docs/CSR_METHOD.md §6）
│   ├── diag32_shortlong.py      短/长分桶口径的起点（纠正"只看 mean IoU"的误判）
│   ├── diag35_edgegrow.py       边界外推 Edge-Grow
│   ├── diag36_gate.py           Edge-Grow 判定当门控
│   ├── diag37_symctr.py         对称生长 + 对比度门控
│   ├── diag38_signal.py         ★ 逐侧决策的信号筛选（AUC 诊断，决定性证据）
│   ├── diag39_kratio.py         kratio / margin 两信号
│   └── README.md                各条路线的结论摘要
├── baselines/                   复现 5 个外部弱监督基线所需的兼容性补丁
│   ├── gen_cg_cfgs.py           生成 QMN / PPS / CCR 的 CG 切分配置
│   ├── patch_baselines.py       QMN config 键 + CCR ivc_loss 的 gpu_idx
│   ├── patch_qmn_ratio.py       QMN ratio: 0 → 1（关键 bug）
│   ├── patch_qmn_eval.py        QMN 非 vote 分支的 UnboundLocalError
│   ├── patch_pps_wandb.py       PPS 的 wandb 硬 import 改为可选
│   └── README.md                逐条说明「修什么 / 为什么」
└── tools/
    ├── format_lint.py           AST 静态检查 '%' 占位符数与参数数是否一致
    └── count_params.py          逐数据集重数 PC-Net 参数量（实例化 + parameters() 求和）
```

---

## 3. 快速开始

### 3.1 CSR 算子（独立可用）

```python
import torch
from models import PCNet            # 官方 backbone（零改动）
from csr import CenterShiftRefiner

model = PCNet(cfg['model']['config']).cuda()
model.load_state_dict(torch.load('checkpoints/CharadesSTA/8_2025-05-14_13-26-12/model-best.pt',
                                 map_location='cuda')['model_parameters'])

csr = CenterShiftRefiner(model, num_point=50)   # 默认即 train.json 校准值
out = csr(net_input)                # net_input 与 PC-Net forward 的输入一致
out['boxes_top1']                   # [bsz, 2] 精修后 top-1 框 (start, end)，用于 R@1
out['boxes_pool5']                  # [bsz, 5, 2] CSR 框顶替原 top-1 + 原 rank2..5，用于 R@5
```

### 3.2 RBR 算子（对照）

```python
from csr import RelevanceBoundaryRefiner
rbr = RelevanceBoundaryRefiner(model, num_point=50, k=0.15)
```

### 3.3 完整验证

```bash
python csr/eval/eval_csr.py config/charades/main_train_test.json \
    checkpoints/CharadesSTA/8_2025-05-14_13-26-12/model-best.pt
```

---

## 4. 默认超参

| 超参 | 值 | 含义 |
|---|---:|---|
| `smooth` | 7 | 相关性曲线平滑窗口 |
| `beta` | 0.20 | 中心估计搜索窗口半径（归一化时长单位） |
| `rho` | 0.10 | dead-zone 阈值 = ρ × 框宽 |
| `alpha` | 1.00 | 位移阻尼 |
| `dmin` | 0.01 | 自洽门控阈值（`None` 关闭） |
| `num_point` | 50 | 逐点提案数（= `max_num_frames // 4`） |

在 `data/charades/train.json`（开发集）上按"短样本优先"准则从 900 组网格中选出，
测试集原样使用；train 达标 217/900、test 达标 97/900、交集 81 组。
**推理期不使用任何 gt。**

---

## 5. 结果速查

| Charades-CG Test-Trivial | ΔR1@0.5 | ΔR1@0.7 | ΔmIoU |
|---|---:|---:|---:|
| RBR k=0.15（全量） | **−0.97** | +1.09 | +1.08 |
| **CSR（定版，全量）** | **+0.67** | **+2.09** | **+1.13** |
| RBR k=0.15 短桶 | **−2.98** | −2.21 | −0.17 |
| **CSR 短桶** | **+0.03** | **+0.22** | **+0.58** |
| RBR k=0.15 长桶 | +4.77 | **+10.49** | +4.66 |
| **CSR 长桶** | +2.49 | +7.42 | **+2.72** |

**Novel-Comp**：ΔmIoU +0.75 / Δ@0.5 +0.43 / Δ@0.7 +1.98
**Novel-Word**：ΔmIoU +1.18 / Δ@0.5 +0.43 / Δ@0.7 +2.11

**跨骨干 CNM（零调参，九格全正）**：Test-Trivial +5.19/+2.23/+3.00、
Novel-Comp +3.41/+1.62/+2.17、Novel-Word +4.56/+3.55/+2.84（顺序 Δ@0.5/Δ@0.7/ΔmIoU）。

**ActivityNet-CG（负结果）**：ΔmIoU **−0.17**；10 组扫描 10/10 全负 ⇒ 几何失配。

**适用条件**：CSR 有效的必要条件是**基线框宽与 gt 时长可比**（Charades 1.35 → 全正；
ANET 3.60 → 无提升）。详见 `docs/CSR_METHOD.md` §4 两轴有效图。

---

## 6. 路径约定

本目录所有脚本都不含机器专有路径。默认行为与覆盖变量见 `csr/paths.py`：

| 变量 | 默认 | 用途 |
|---|---|---|
| `PCNET_ROOT` | 自动定位（向上找 `models/pcnet.py` + `train.py`） | 仓库根 |
| `DATA_ROOT` | = `PCNET_ROOT` | 大特征文件所在目录 |
| `FEAT_PATH` | 取配置里的 `feature_path` | 覆盖特征文件路径 |
| `WS_BASE` | `<仓库同级>/ws_base` | 外部基线工作区（QMN/pps/CCR） |
| `CNM_ROOT` | `<仓库同级>/CNM-CG` | CNM 克隆目录（跨骨干实验） |
| `CACHE_DIR` | `<仓库>/csr/analysis/feat_cache` | 离线分析的特征缓存 |
| `CSR_OUTPUT_DIR` | `<仓库>/csr/outputs` | 扫描结果落盘目录 |

兼容旧名：`LM_ROOT` 等价于 `PCNET_ROOT`。

---

## 7. 复现纪律（verify-then-report）

1. **先复现基线**：`reproduce_base.py` 的 base 列必须与官方日志对到 ±0.005 以内，再看增量。
2. **`K_REP ≥ 3`**：`_mask_words` 未固定种子，单次评测噪声约 ±0.005。
3. **超参只在 `train.json` 上选**，测试集三个切分原样使用。
4. **口径统一 `vote=False`**（PC-Net 官方评测模式），IoU 用各方法官方 `runner.eval()`。
5. **推送脚本前先跑** `python csr/tools/format_lint.py <文件>`。
6. **报参数量前先重数**：`python csr/tools/count_params.py`。PC-Net 的参数量**逐数据集不同**
   （`frame_fc` / `pred_vec` 随 `frame_dim`、`fc_comp` 随 `vocab_size`），实测
   Charades-CG **5.71 M**、ActivityNet-CG **7.34 M**、TACoS **6.66 M**、Ego4D **6.42 M**。
   后两者与原文一致；原文的 3.34 M / 4.97 M 各比实测低 **2,372,352**（= `DualTransformer.decoder1`
   = `.decoder2`，两个独立 3 层栈）⇒ 属**计数口径差异**，不是模型差异。**不要跨数据集、跨论文比参数量。**

---

## 8. 引用与许可

若使用本目录代码，请同时引用 PC-Net（backbone）与本文工作（CSR / RBR 算子）：

```
@inproceedings{zhou2025pcnet,
  title={{PC}-Net: Weakly Supervised Compositional Moment Retrieval via Proposal-Centric Network},
  author={Mingyao Zhou and Hao Sun and Wei Xie and Ming Dong and Chengji Wang and Mang Ye},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year={2025},
  url={https://openreview.net/forum?id=kQAnOaayIo}
}
```

**许可**：backbone 代码与官方权重（`models/`、`checkpoints/` 等）版权归其原作者所有，
**上游仓库未附 LICENSE 文件**，使用前请与作者确认授权。本目录新增的算子与 harness 源码
随本工作一并提供；CNM 等其他外部仓库同样遵循各自官方许可。
