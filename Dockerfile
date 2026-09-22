# CUDA 12.8 開発環境対応ベースイメージを使用
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

# 環境変数の設定
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 作業ディレクトリの設定
WORKDIR /app

# システム依存パッケージのインストール (安定のPython 3.10に統一)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    git \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip

# pipの更新
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# RTX 5060 Ti (sm_120) 対応: PyTorch Nightly (cu128) のインストール
RUN pip install --no-cache-dir --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cu128

# requirements.txtが存在する場合は自動インストール
COPY requirements.txt* /app/
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi

# 追加でよく使うライブラリ
RUN pip install --no-cache-dir \
    "numpy<2" \
    pandas \
    matplotlib \
    scikit-learn \
    jupyterlab \
    opencv-python

# コンテナ起動時にJupyter Labを立ち上げる設定
#CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]
