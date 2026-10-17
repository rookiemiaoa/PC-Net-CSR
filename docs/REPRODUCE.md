# 复现指南

> 所有命令都假定**当前工作目录 = 本仓库根目录**（`csr.paths` 会自动定位仓库根，
> 也可用 `PCNET_ROOT` 显式指定）。

---

## 0. 环境

```bash
conda create -n pcnet python=3.8 -y
conda activate pcnet
pip install -r requirements.lock.txt     # 精确锁定（推荐，用于逐位复现）
# 或：pip install -r requirements.txt    # 宽松下限（用于新平台试装）
```

实测环境：`Python 3.8.20 / torch 2.0.1 / numpy 1.24.4 / h5py 3.9.0 /
nltk 3.8.1 / fairseq 0.12.2 / scipy 1.10.1`（CUDA 11.7，RTX 2080 Ti）。
本仓库新增的算子本身只依赖 `torch` + `numpy`；统计检验面板另需 `scipy`。

---

## 0.1 容器化一键复现（推荐路径）

仓库自带 `Dockerfile`（基镜像 `pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime`，
与锁版本精确对齐）与 `reproduce.sh`：

```bash
# 构建（依赖按 requirements.lock.txt 精确安装）
docker build -t pcnet-csr:1.0 .

# 冒烟：只跑 Novel-Word（695 条，最快；`eval_csr` 口径 base 应 ≈ 40.67，
#       官方日志口径为 41.06，两者差 0.39 属评测路径差异，不是噪声）
docker run --gpus all -v /path/to/i3d_features.hdf5:/data/i3d_features.hdf5 \
    -e FEAT_PATH=/data/i3d_features.hdf5 pcnet-csr:1.0 bash reproduce.sh smoke

# 全部：Charades 三切分 + ActivityNet + TACoS + 离线分析
docker run --gpus all -v /path/to/i3d_features.hdf5:/data/i3d_features.hdf5 \
    -e FEAT_PATH=/data/i3d_features.hdf5 pcnet-csr:1.0

# 只用 Charades 三切分（跳过 12k 样本的 ActivityNet）
docker run --gpus all -v /path/to/i3d_features.hdf5:/data/i3d_features.hdf5 \
    -e FEAT_PATH=/data/i3d_features.hdf5 pcnet-csr:1.0 bash reproduce.sh charades
```

`reproduce.sh` 依次跑「官方基准 → CSR 主结果 → ActivityNet 负结果 → TACoS →
离线分析」，每个脚本的完整 stdout 落在 `results/*.log`，可直接与 `RESULTS.md`
逐行对照。环境变量：`K_REP`（默认 3）、`BS`（默认 16）、`FEAT_PATH`、`OUTDIR`。

> 无 GPU 时同样可跑：`PART 4` 的离线分析只需 CPU（复用 `csr/analysis/feat_cache/`），
> 秒级完成，覆盖稳健性、逐样本归因、门控消融与 ORACLE 上界四张表。

---

## 1. 数据准备

**随仓库已分发**（无需额外操作）：

- 数据切分：`data/charades/{train,test_trivial,novel_comp,novel_word}.json`、
  `data/activitynet/{train,val,test_trivial,novel_comp,novel_word}.json`
- 词表：`data/{charades,activitynet,tacos,ego4d}/glove.pkl`
- 官方评测日志：`checkpoints/**/*.log`（`RESULTS.md` §1 的来源）
- **官方权重：不在 git 内**。开发机盘上有 7 个 `model-best.pt`，但 `.gitignore` 排除了
  `checkpoints/**/*.pt`；公开版需从发布附件下载，并用 `checkpoints/WEIGHTS.md5` 校验后
  落到 `README.md §3.2` 的同名路径。

**需要自行准备**：视频特征 `i3d_features.hdf5`（Charades，约 3 GB）等。
配置里写的是裸文件名（`config/charades/main_train_test.json` 中 `"feature_path": "i3d_features.hdf5"`），
按下面任一方式放好即可：

```bash
# 方式 A：放到仓库根目录（与官方 README 的约定一致）
cp /path/to/i3d_features.hdf5 .

# 方式 B：放到别处，用环境变量指过去
export FEAT_PATH=/data/features/i3d_features.hdf5
```

> harness 的查找顺序：`FEAT_PATH` 环境变量 → 配置里的 `feature_path` →
> `DATA_ROOT/` 与仓库根目录（`DATA_ROOT` 见 `csr/paths.py`）。

---

## 2. 复现官方基准（先做这一步，误差 ≤ ±0.005 才看后面的增量）

```bash
python csr/eval/reproduce_base.py config/charades/main_train_test.json \
    checkpoints/CharadesSTA/8_2025-05-14_13-26-12/model-best.pt
```

对照 `RESULTS.md` §1：Charades-CG Test-Trivial 应为 **47.12 / 54.84 / 26.68**（mIoU/@0.5/@0.7）。
三个切分依次换 config 即可（`main_novel_comp.json` / `main_novel_word.json`）。

---

## 3. CSR 主实验（`RESULTS.md` §2）

```bash
python csr/eval/eval_csr.py config/charades/main_train_test.json \
    checkpoints/CharadesSTA/8_2025-05-14_13-26-12/model-best.pt
```

输出包含 `base` / `RBR k∈{0.10,0.15,0.20}` / `CSR` 三组，以及

- `[short/long 分桶]`（`RESULTS.md` §2.1）
- `[显著性]`（配对检验）
- `[诊断]`（移动率、平均位移）

可调环境变量：

```
BS=16  K_REP=3  LIMIT=0  FEAT_PATH=...  THR=0.35
SMOOTH=7  BETA=0.20  RHO=0.10  ALPHA=1.0  DMIN=0.01
KS=0.10,0.15,0.20          # RBR 的 k 扫描点
```

> `K_REP ≥ 3` 是硬要求：`_mask_words` 未固定随机种子，单次评测噪声约 ±0.005。
> 跑 Novel-Word（695 条）很快，跑 Test-Trivial 建议单切分单独起进程，
> 避免长命令被 SSH 通道超时打断。

---

## 4. ActivityNet-CG 负结果（`RESULTS.md` §3）

```bash
python csr/eval/eval_anet_support.py config/activitynet/main_train_test.json \
    checkpoints/ActivityNet/8_2025-05-14_14-07-12/model-best.pt

# 10 组单变量扫描（自动化驱动，逐个跑上面的脚本）
python csr/analysis/anet_sens.py
```

⚠️ ActivityNet-CG 的**官方日志值（36.45）在本机复现不出来**（本机 32.39），
原因最可能是特征溯源差异（见 `RESULTS.md` §1 的警告）。
比较 baseline 时请用**同一份特征下的复现值**，不要与官方日志值混用。

---

## 5. 跨骨干迁移（`RESULTS.md` §4）

需要一个 CNM（AAAI-2022）的克隆目录：

```bash
# 默认约定：与 PC-Net 仓库同级，目录名 CNM-CG/
ls ../CNM-CG
# 或显式指定
export CNM_ROOT=/path/to/CNM-CG

python csr/eval/eval_csr_cnm.py config/charades/main.json checkpoints/<cnm-ckpt>.pt
```

CSR 的超参**一字不改**（零调参）。曲线从 CNM 自己的掩码重建阶段读出。

> ⚠️ **`config` 与 `ckpt` 的相对路径是相对 `CNM_ROOT` 的，不是相对本仓库**。
> `eval_csr_cnm.py` 会先 `chdir(CNM_ROOT)`（见 `csr/paths.py`），而本仓库**没有**
> `config/charades/main.json`。正确写法是 `"$CNM_ROOT/config/charades/main.json"` 与
> `"$CNM_ROOT/checkpoints/.../model-best.pt"`，或在该脚本所在目录下用绝对路径传入。

若要用 CNM 官方口径单独评测，请在 CNM 仓库根目录下运行
`csr/eval/eval_cnm_base.py`（它需要 `import` 该仓库的 `models` / `datasets`）。

---

## 6. 复现 5 个外部弱监督基线（`RESULTS.md` §5）

```bash
export WS_BASE=../ws_base          # 默认与 PC-Net 同级；内含 QMN/ pps/ CCR/ 的克隆

python csr/baselines/gen_cg_cfgs.py          # 1) 生成 CG 切分协议下的 config
python csr/baselines/patch_baselines.py      # 2) 兼容性补丁（QMN config 键 + CCR gpu_idx）
python csr/baselines/patch_qmn_ratio.py      # 3) QMN ratio: 0 -> 1
python csr/baselines/patch_qmn_eval.py       # 4) QMN 非 vote 分支的 UnboundLocalError
python csr/baselines/patch_pps_wandb.py      # 5) PPS 的 wandb 改为可选导入
```

补丁**只修兼容性 bug，不改算法**，逐条动机见 `csr/baselines/README.md`。
统一协议：同一 `cg_train`、同一 I3D 特征、30 epoch、**`vote=False`**、各自官方 `runner.eval()`。

> ⚠️ **`vote` 口径必须统一**。PC-Net 官方评测用 `vote=False`，CPL 亦为非 vote，
> PPS 无 vote 机制；CCR / QMN 若用 `--vote` 会与其余方法不可比（实测差异 < 0.8pt）。
> 本仓库报告的全部数字均为 `vote=False`。

---

## 7. 离线分析（不跑模型，秒级）

```bash
ls csr/analysis/feat_cache/          # charades_test_trivial_r3.pkl / charades_train_r1.pkl
python csr/analysis/scan_center_final.py       # 稳健性 + 配对显著性
python csr/analysis/attrib_csr.py              # 逐样本误伤归因
python csr/analysis/ablate_gate.py             # 关掉自洽门控的消融
python csr/analysis/oracle_gate.py             # ORACLE 门控上界（用 gt，仅作上界）
```

缓存的字段含义见 `csr/analysis/README.md`。
这两个 `.pkl` 是**可选派生件**，删掉不影响主流程。

---

## 8. 诊断链（被证伪的路线，`CSR_METHOD.md` §6）

```bash
python csr/diagnostics/diag32_shortlong.py --help    # 短/长分桶口径的起点
python csr/diagnostics/diag35_edgegrow.py
python csr/diagnostics/diag36_gate.py
python csr/diagnostics/diag37_symctr.py
python csr/diagnostics/diag38_signal.py              # AUC 信号筛选（决定性诊断）
python csr/diagnostics/diag39_kratio.py
```

这些脚本用 `CACHE_DIR`（默认 `csr/analysis/feat_cache`）缓存逐样本相关性曲线，
首次运行会重新前向一遍模型，之后秒级迭代。

---

## 9. 复现纪律（务必遵守，否则容易得出错误结论）

1. **先复现 base，再看增量**。base 列对不上官方日志（误差 > ±0.005）时，
   后面的所有 Δ 都不可信。
2. **`K_REP ≥ 3`**，报 mean±std。单次评测的 ±0.005 噪声足以吞掉 0.3pt 级别的差异。
3. **超参只在 `data/charades/train.json` 上选**，测试集三个切分原样使用。推理期不使用 gt
   （`oracle_gate.py` 用 gt 但只用来算**上界**，其结果不能当作方法性能）。
4. **口径统一**：`vote=False`；IoU 用各方法官方 `runner.eval()`。
5. **不同特征来源的 base 不可混比**（ActivityNet 的教训）。
6. 推送脚本到远程机器前，先跑 `python csr/tools/format_lint.py <文件>`：
   它用 AST 静态比对 `'...' % (...)` 的占位符数与参数个数。
   `ast.parse` 查不出这类错误——它只在执行到那一行时才抛
   `TypeError: must be real number, not str`。

## 10. 复现「曲线可用性」判据

```bash
# 1) 反解门限并输出 18 个互异工作点的符号序列（读 results/curve_availability/）
python csr/analysis/w47_traj_calibration.py
#    预期末四行：18 个互异工作点 / a* ∈ (0.5096, 0.5691) /
#                conv_only a*=0.5759 / all a*=0.5430

# 2) 重画论文图 3（含训练轨迹点）—— 需额外装 matplotlib
pip install matplotlib            # 不进主路径依赖，仅此脚本需要
python csr/analysis/w48_fig3_validity_map.py
#    输出 csr/analysis/figures/fig3_validity_map.png（可用 CSR_FIG_DIR 覆盖）

# 3) 命题 1 的闭式收缩率与命题 2 的两参数反解数值核对
python csr/analysis/w46_prop2_proof_check.py
#    输出 csr/analysis/figures/w46_proof.json（可用 CSR_OUTPUT_DIR 覆盖）
```

> **w48 的中文字体**：脚本按
> `Microsoft YaHei → SimHei → Noto Sans CJK → WenQuanYi → DejaVu Sans` 回退。
> Linux 服务器通常**没有** YaHei/SimHei，需先装开源 CJK 字体，否则中文标签会渲染成空白方框：
>
> ```bash
> sudo apt install fonts-noto-cjk        # 或 fonts-wqy-zenhei
> rm -rf ~/.cache/matplotlib             # 清字体缓存
> ```
>
> 脚本在检测不到任何 CJK 字体时会打印告警（不会静默出方框）。

判定规则：**沿第二坐标 `a` 升序，CSR 的 ΔmIoU 符号必须单调**。
若某个新骨干的 `a` 落在 (0.510, 0.569) 内，该骨干的符号即为该区间定位的
新证据——这正是当前唯一仍待更多**独立收敛骨干**填充的窗口。
