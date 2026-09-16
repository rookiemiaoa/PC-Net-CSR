#!/bin/bash
# run_eval_chain.sh —— 等 CNM 训练收敛后，自动在三套 CG 切分上跑 WSCMR 口径评测
# 由 setsid 后台启动，全部输出写入 logs/eval_chain.log，末尾打印 CHAIN_DONE
_HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$_HERE/../../.." && pwd)
CNM_ROOT=${CNM_ROOT:-$(dirname "$REPO")/CNM-CG}
[ -n "${CONDA_SH:-}" ] && source "$CONDA_SH"
[ -n "${CONDA_ENV:-}" ] && conda activate "$CONDA_ENV"
cd "$CNM_ROOT"
export NLTK_DATA=${NLTK_DATA:-$(dirname "$REPO")/nltk_data}
export CUDA_VISIBLE_DEVICES=${GPU:-0}
LOG=$CNM_ROOT/logs/eval_chain.log
TLOG=$CNM_ROOT/logs/cg_main.log
CKPT=checkpoints/charades/cg_main/cg_main/model-best.pt

{
  echo "=== [$(date '+%F %T')] waiting for CNM training to finish ==="
  while true; do
    n=$(grep -c 'Start Epoch' "$TLOG")
    done_marker=$(grep -c 'Best results:' "$TLOG")
    if [ "$done_marker" -ge 1 ] || [ "$n" -ge 50 ]; then
      echo "[$(date '+%F %T')] epochs_started=$n  final_marker=$done_marker  -> proceeding"
      break
    fi
    sleep 45
  done
  sleep 20
  echo "=== [$(date '+%F %T')] training finished; checkpoint: ==="
  ls -l "$CKPT"
  for tag in test_trivial novel_comp novel_word; do
    case $tag in
      test_trivial) cfg=config/charades/cg_main.json ;;
      novel_comp)   cfg=config/charades/cg_novel_comp.json ;;
      novel_word)   cfg=config/charades/cg_novel_word.json ;;
    esac
    echo ""
    echo "######## SPLIT=$tag  CFG=$cfg ########"
    python -u "$REPO/csr/eval/eval_cnm_base.py" "$cfg" "$CKPT"
  done
  echo ""
  echo "=== [$(date '+%F %T')] all evals finished ==="
  echo "CHAIN_DONE"
} > "$LOG" 2>&1
