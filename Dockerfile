# =============================================================================
# PC-Net-CSR —— 一键可复现环境
# -----------------------------------------------------------------------------
# 基镜像选 pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime：
#   与 requirements.lock.txt 的 torch==2.0.1 精确对齐，且自带 CUDA/cuDNN，
#   避免在容器内再装一份 CUDA 工具链（复现只需 runtime，不需要 nvcc）。
# 构建：  docker build -t pcnet-csr:1.0 .
# 运行：  docker run --gpus all -v /path/to/i3d_features.hdf5:/data/i3d_features.hdf5 \
#             -e FEAT_PATH=/data/i3d_features.hdf5 pcnet-csr:1.0
# =============================================================================
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    NLTK_DATA=/opt/nltk_data \
    PCNET_ROOT=/workspace/PC-Net-CSR

# 仅装运行期需要的系统库（h5py 的 HDF5 后端、fairseq 的音频解码占位）
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libhdf5-dev \
        libsndfile1 \
        git \
    && rm -rf /var/lib/apt/lists/*

# 依赖：精确锁定版本，先装依赖再拷代码以利用层缓存
COPY requirements.lock.txt /tmp/requirements.lock.txt
RUN pip install --no-cache-dir -r /tmp/requirements.lock.txt

# NLTK 语料（词表构建用）
RUN python -m nltk.downloader -d /opt/nltk_data wordnet omw-1.4 || true

WORKDIR /workspace/PC-Net-CSR
COPY . /workspace/PC-Net-CSR

# 默认：跑一键复现（官方基准 → CSR 主结果 → 负结果 → 跨骨干）
CMD ["bash", "reproduce.sh"]
