#!/bin/bash
# QMN 公平协议：cg_train 重训 + 3 个 CG 切分评测
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
cd "$WS_BASE/QMN"
export CUDA_VISIBLE_DEVICES=${GPU:-0}
OUT="$WS_BASE/logs_qmn"
mkdir -p "$OUT"

echo "TRAIN_START $(date '+%F %T')"
# 注意：QMN 的 train.py 在传 --log_dir 时会写死 checkpoints/{dataset}/{seed}_{ts}.log，
# 该目录不存在会崩；这里不传 --log_dir，让 log_filename=None（日志走 stdout）。
mkdir -p checkpoints/cg_main/qmn
python -u train.py --config-path config/charades/cg_main.json --tag cgmain > "$OUT/qmn_train_cgmain.log" 2>&1
echo "train_exit=$? $(date '+%T')"

# QMN 的 model_saved_path 追加了 seed_时间戳（见 train.py L78），需动态找 model-best.pt
CK=$(find checkpoints/cg_main/qmn -name "model-best.pt" 2>/dev/null | head -1)
if [ -z "$CK" ]; then
  CK=$(find checkpoints/cg_main -name "model-best.pt" 2>/dev/null | head -1)
fi
echo "CK=$CK"
ls -la "$CK" 2>&1

for s in test_trivial novel_comp novel_word; do
  echo "=== eval $s ==="
  # 口径统一为 vote=False（见 run_ccr_cg.sh 的同款说明）。
  # QMN 的 --vote 是 type=bool 需传值，故若要开启应写 --vote True。
  python -u train.py --config-path "config/charades/eval_$s.json" --resume "$CK" --eval > "$OUT/qmn_cg_$s.log" 2>&1
  echo "exit=$? $(date '+%T')"
done
echo "QMN_CG_ALL_DONE $(date '+%F %T')"
