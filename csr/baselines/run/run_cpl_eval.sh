#!/bin/bash
# CPL evaluation driver on Charades-STA + Charades-CG splits.
# Stage 1: reproduce the official Charades-STA number with the released checkpoint.
# Stage 2: evaluate the same released checkpoint on the three CG test splits.
# ---- 路径解析（开源版：不依赖任何机器专有路径，全部可用环境变量覆盖）----
#   WS_BASE    外部基线工作区（内含 cpl/ CCR/ pps/ QMN/），默认 = PC-Net 仓库的同级 ws_base/
#   CNM_ROOT   CNM 仓库克隆目录，              默认 = PC-Net 仓库的同级 CNM-CG/
#   CONDA_SH   conda 的 profile 脚本；若已在目标环境里则无需设置
#   CONDA_ENV  conda 环境名，默认不切换
#   GPU        使用的显卡号
set -u
_HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$_HERE/../../.." && pwd)
WS_BASE=${WS_BASE:-$(dirname "$REPO")/ws_base}
CNM_ROOT=${CNM_ROOT:-$(dirname "$REPO")/CNM-CG}
[ -n "${CONDA_SH:-}" ] && source "$CONDA_SH"
[ -n "${CONDA_ENV:-}" ] && conda activate "$CONDA_ENV"
cd "$WS_BASE/cpl"
export CUDA_VISIBLE_DEVICES=${GPU:-2}
OUT="$WS_BASE/logs_cpl"
mkdir -p "$OUT"
CKPT=checkpoints/charades/model-best.pt

echo "START $(date '+%F %T')"
echo "ckpt=$CKPT"

echo "=== std_test ==="
python -u train.py --config-path config/charades/main.json --resume "$CKPT" --eval > "$OUT/cpl_std_test.log" 2>&1
echo "exit=$? $(date '+%T')"

for s in test_trivial novel_comp novel_word; do
  echo "=== $s ==="
  python -u train.py --config-path "config/charades/eval_$s.json" --resume "$CKPT" --eval > "$OUT/cpl_$s.log" 2>&1
  echo "exit=$? $(date '+%T')"
done

echo "CPL_STAGE1_ALL_DONE $(date '+%F %T')"
