#!/usr/bin/env bash
# =============================================================================
# reproduce.sh —— 一键复现 PC-Net-CSR 的全部已发表数值
# -----------------------------------------------------------------------------
# 用法：
#   bash reproduce.sh                 # 全部（Charades 三切分 + ActivityNet + TACoS）
#   bash reproduce.sh charades        # 只跑 Charades 三切分
#   bash reproduce.sh smoke           # 冒烟：只跑 Novel-Word（695 条，最快）
#
# 环境变量（均有默认值）：
#   FEAT_PATH   视频特征 hdf5 路径（Charades 的 i3d_features.hdf5）
#   ANET_FEAT   ActivityNet 特征（默认读 config 内的裸文件名）
#   K_REP       评测重复次数，默认 3（硬要求 ≥3，见 docs/REPRODUCE.md §9）
#   BS          batch size，默认 16
#   OUTDIR      结果输出目录，默认 results/
#
# 产出：results/*.log —— 每个脚本的完整 stdout，可直接与 docs/RESULTS.md 对照。
# =============================================================================
set -u

MODE="${1:-all}"
K_REP="${K_REP:-3}"
BS="${BS:-16}"
OUTDIR="${OUTDIR:-results}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT" || exit 1

export PCNET_ROOT="$ROOT"
export K_REP BS
[ -n "${FEAT_PATH:-}" ] && export FEAT_PATH

PY="${PY:-python}"
mkdir -p "$OUTDIR"

CH_CKPT="checkpoints/CharadesSTA/8_2025-05-14_13-26-12/model-best.pt"
AN_CKPT="checkpoints/ActivityNet/8_2025-05-14_14-07-12/model-best.pt"
TA_CKPT="checkpoints/TACoS/8_2025-05-14_21-05-09/model-best.pt"

say() { echo ""; echo "=== $* ==="; }

run() {  # run <logname> <cmd...>
  local name="$1"; shift
  say "$name"
  echo "\$ $*"
  ( "$@" ) 2>&1 | tee "$OUTDIR/$name.log" | tail -n 40
  return "${PIPESTATUS[0]}"
}

# ---------------------------------------------------------------- 自检（纯 CPU）
if [ "$MODE" = "check" ]; then
  say "CHECK / 代码与资产自检（不需要 GPU、不需要 hdf5，约 5 分钟）"
  run "check_assets" "$PY" csr/tools/check_assets.py || true
  run "config"       "$PY" csr/tools/config_parity.py --published || exit 1
  run "parity"       "$PY" csr/tools/parity_check.py  || exit 1
  run "table3"       "$PY" scripts/render_table3.py --check || exit 1
  for S in scan_center_final attrib_csr ablate_gate oracle_gate; do
    run "analysis_$S" "$PY" "csr/analysis/$S.py" || exit 1
  done
  run "stats_s8" "$PY" scripts/s2_stats.py within results/multiseed/s1/s8_test_trivial.npz || exit 1
  echo ""
  echo "[OK] 自检通过：算子实现可对拍、表 3 与论文一致、离线分析可重跑。"
  exit 0
fi

# ---------------------------------------------------------------- 冒烟
if [ "$MODE" = "smoke" ]; then
  say "SMOKE: Charades Novel-Word (695) base + CSR"
  K_REP=1 run smoke_novel_word \
    "$PY" csr/eval/eval_csr.py config/charades/main_novel_word.json "$CH_CKPT"
  echo ""
  echo "[OK] 冒烟完成。若 base mIoU ≈ 40.67（Novel-Word，eval_csr 口径），环境可用。"
  echo "     注：官方日志口径为 41.06，两者差 0.39，属评测路径差异。"
  exit 0
fi

# ---------------------------------------------------------------- Charades
say "PART 1 / Charades-CG 三切分：官方基准 + CSR"
for SPLIT in train_test novel_comp novel_word; do
  cfg="config/charades/main_${SPLIT}.json"
  tag="charades_${SPLIT}"
  run "base_${tag}" "$PY" csr/eval/reproduce_base.py "$cfg" "$CH_CKPT"
  run "csr_${tag}"  "$PY" csr/eval/eval_csr.py     "$cfg" "$CH_CKPT"
done

echo ""
echo "[对照] docs/RESULTS.md §1–§2（Charades-CG，K_REP=3）"
echo "  Test-Trivial base 47.12 / 54.84 / 26.68 ；CSR 48.33 / 55.89 / 28.67"
echo "  Novel-Comp   base 38.04 / 41.69 / 16.73"

# ---------------------------------------------------------------- ISR
# eval_final.py 是 ISR 数字的唯一产出入口（此前不在本脚本内，导致文档可复现性缺口）
say "PART 1b / ISR 对照（eval_final.py）"
run "isr_charades_train_test" "$PY" csr/eval/eval_final.py \
    config/charades/main_train_test.json "$CH_CKPT"
echo "  ISR(γ=0.5,T=2) 应为 48.36（base 47.20 ⇒ Δ +1.16）"

# ---------------------------------------------------------------- ActivityNet
if [ "$MODE" = "all" ]; then
  say "PART 2 / ActivityNet-CG 负结果（12k+ 样本，较慢）"
  run "anet_base" "$PY" csr/eval/eval_anet_support.py config/activitynet/main_train_test.json "$AN_CKPT"
  run "anet_sens" "$PY" csr/analysis/anet_sens.py

  say "PART 3 / TACoS（两轴皆无效的第四个数据集）"
  run "tacos" "$PY" csr/eval/eval_csr.py config/tacos/main.json "$TA_CKPT"
fi

# ---------------------------------------------------------------- 离线分析（秒级）
say "PART 4 / 离线分析（无需 GPU，复用 feat_cache）"
for S in scan_center_final attrib_csr ablate_gate oracle_gate; do
  run "analysis_$S" "$PY" "csr/analysis/$S.py"
done

echo ""
echo "=============================================================="
echo " 全部完成。结果在 $OUTDIR/"
echo " 逐项对照：docs/RESULTS.md（正结果与负结果同等篇幅）"
echo "=============================================================="
