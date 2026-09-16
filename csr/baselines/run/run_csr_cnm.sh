#!/bin/bash
# run_csr_cnm.sh —— CSR-on-CNM 三切分正式跑（K_REP=3，同一权重）+ 尖峰宽度稳健性检查
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
cd "$CNM_ROOT"
export NLTK_DATA=${NLTK_DATA:-$(dirname "$REPO")/nltk_data}
export CUDA_VISIBLE_DEVICES=${GPU:-0}
CK=checkpoints/charades/cg_main/cg_main/model-best.pt
mkdir -p logs_csr
echo "START $(date '+%F %T')" > logs_csr/driver.log
echo "ckpt=$CK" >> logs_csr/driver.log

run () {
  name=$1; cfg=$2; shift 2
  if [ ! -f "$cfg" ]; then echo "MISSING $cfg" >> logs_csr/driver.log; return; fi
  echo "=== $name  cfg=$cfg  $*  $(date '+%T') ===" >> logs_csr/driver.log
  env "$@" K_REP=3 BS=32 python -u "$REPO/csr/eval/eval_csr_cnm.py" "$cfg" "$CK" > "logs_csr/csr_cnm_${name}.log" 2>&1
  echo "exit=$?  $(date '+%T')" >> logs_csr/driver.log
}

run test_trivial config/charades/cg_main.json
run novel_comp   config/charades/cg_novel_comp.json
run novel_word   config/charades/cg_novel_word.json

# 尖峰宽度稳健性：WPT 是唯一一个由我们选定的「曲线提取」超参，需证明结论不依赖它
for w in 0.15 0.20 0.30; do
  echo "=== wpt_${w}  cfg=cg_main.json  WPT=$w  $(date '+%T') ===" >> logs_csr/driver.log
  WPT=$w K_REP=1 BS=32 python -u "$REPO/csr/eval/eval_csr_cnm.py" config/charades/cg_main.json "$CK" \
      > "logs_csr/csr_cnm_wpt_${w}.log" 2>&1
  echo "exit=$?  $(date '+%T')" >> logs_csr/driver.log
done

echo "CSR_CNM_ALL_DONE  $(date '+%T')" >> logs_csr/driver.log
