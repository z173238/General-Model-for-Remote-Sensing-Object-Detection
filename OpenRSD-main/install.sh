#!/bin/bash
# ============================================================
# OpenRSD ICCV 2025 — 一键安装脚本 (Git Bash / Linux)
# ============================================================
set -e

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

echo "============================================"
echo " OpenRSD 安装脚本"
echo " 工作目录: $ROOT_DIR"
echo "============================================"

# ==================== Step 0: 环境检测 ====================
echo ""
echo "[0/5] 检测环境..."

# Python
PYTHON_VER=$(python --version 2>&1 | awk '{print $2}')
echo "  Python: $PYTHON_VER"

# CUDA
if command -v nvidia-smi &>/dev/null; then
    CUDA_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
    echo "  CUDA Driver: $CUDA_VER"
    echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
else
    echo "  [WARN] nvidia-smi 不可用，确认 GPU 驱动已安装"
fi

# ==================== Step 1: Conda 环境 ====================
echo ""
echo "[1/5] 创建 conda 环境 (openrsd)..."

ENV_NAME="openrsd"

if conda env list 2>/dev/null | grep -q "^${ENV_NAME} "; then
    echo "  [SKIP] conda 环境 ${ENV_NAME} 已存在"
    echo "  如需重建: conda env remove -n ${ENV_NAME} -y"
else
    # 尝试用 environment.yml，如果失败则手动创建
    if conda env create -f environment.yml -n "${ENV_NAME}" 2>&1; then
        echo "  [OK] 从 environment.yml 创建成功"
    else
        echo "  [WARN] environment.yml 创建失败(可能是Linux专用)，手动创建..."
        conda create -n "${ENV_NAME}" python=3.10 -y
        echo "  [OK] 手动创建 conda 环境成功"
    fi
fi

# 激活 conda 环境的方式（Git Bash 下不能直接 source activate）
CONDA_BASE=$(conda info --base 2>/dev/null)
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate "${ENV_NAME}"
elif [ -f "$CONDA_BASE/Scripts/activate" ]; then
    source "$CONDA_BASE/Scripts/activate" "${ENV_NAME}"
fi

# ==================== Step 2: PyTorch ====================
echo ""
echo "[2/5] 安装 PyTorch..."

# 检测 CUDA 版本决定装什么 PyTorch
# 默认装 CUDA 12.1 版本，兼容性好
TORCH_INSTALLED=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "")

if [ -n "$TORCH_INSTALLED" ]; then
    echo "  [SKIP] PyTorch 已安装: $TORCH_INSTALLED"
else
    echo "  安装 PyTorch + torchvision + torchaudio (CUDA 12.1)..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    echo "  [OK] PyTorch 安装完成"
fi

# ==================== Step 3: mmcv ====================
echo ""
echo "[3/5] 安装 mmcv (从本地 mmcv-main.zip)..."

MMCV_INSTALLED=$(python -c "import mmcv; print(mmcv.__version__)" 2>/dev/null || echo "")

if [ -n "$MMCV_INSTALLED" ]; then
    echo "  [SKIP] mmcv 已安装: $MMCV_INSTALLED"
else
    if [ -f "mmcv-main.zip" ]; then
        echo "  解压 mmcv-main.zip..."
        unzip -o mmcv-main.zip -d mmcv-extracted
        MMCV_DIR=$(ls -d mmcv-extracted/mmcv-main* 2>/dev/null | head -1 || echo "mmcv-extracted/mmcv-main")
        if [ -d "$MMCV_DIR" ]; then
            echo "  安装 mmcv from $MMCV_DIR..."
            cd "$MMCV_DIR"
            MMCV_WITH_OPS=1 pip install -e . --no-build-isolation 2>&1 || \
                pip install -e . --no-build-isolation 2>&1 || \
                python setup.py install 2>&1
            cd "$ROOT_DIR"
            echo "  [OK] mmcv 安装完成"
        else
            echo "  [ERR] 找不到 mmcv 源码目录"
        fi
    else
        echo "  [WARN] mmcv-main.zip 不存在，尝试在线安装 mmcv..."
        pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html
    fi
fi

# ==================== Step 4: 项目依赖 ====================
echo ""
echo "[4/5] 安装项目依赖..."

# 安装核心依赖
pip install -r requirements/optional.txt 2>&1 || true

# 关键依赖清单（按 README 和代码推断）
echo "  安装关键依赖..."
pip install \
    addict \
    albumentations \
    dill \
    einops \
    ftfy \
    huggingface-hub \
    mmengine \
    numpy \
    opencv-python \
    opencv-contrib-python \
    pandas \
    pillow \
    prettytable \
    pycocotools \
    pyyaml \
    regex \
    requests \
    rich \
    scikit-image \
    scipy \
    seaborn \
    shapely \
    terminaltables \
    tifffile \
    timm \
    tqdm \
    transformers \
    yapf \
    2>&1

echo "  [OK] 依赖安装完成"

# ==================== Step 5: 安装 OpenRSD ====================
echo ""
echo "[5/5] 安装 OpenRSD..."

pip install -v -e . 2>&1 || python setup.py develop 2>&1

echo "  [OK] OpenRSD 安装完成"

# ==================== 验证 ====================
echo ""
echo "============================================"
echo " 安装完成 — 验证环境"
echo "============================================"
python -c "
import torch
print(f'  PyTorch:  {torch.__version__}')
print(f'  CUDA OK:  {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU:      {torch.cuda.get_device_name(0)}')
" 2>&1 || echo "  [WARN] PyTorch 验证失败"

python -c "import mmcv; print(f'  mmcv:     {mmcv.__version__}')" 2>&1 || echo "  [WARN] mmcv 验证失败"

python -c "import mmdet; print(f'  mmdet:    {mmdet.__version__}')" 2>&1 || echo "  [WARN] mmdet 验证失败"

python -c "import mmrotate; print(f'  mmrotate: {mmrotate.__version__}')" 2>&1 || echo "  [WARN] mmrotate 验证失败"

echo ""
echo "============================================"
echo " 下一步:"
echo " 1. 下载模型权重: 百度网盘"
echo "    https://pan.baidu.com/s/1QWWZOfrjAWhEbk1eQASXVQ?pwd=usnc"
echo "    或轻量版:"
echo "    https://pan.baidu.com/s/1qJJb6NW5DW1hpXpxWOdrOg?pwd=cwy9"
echo ""
echo " 2. 下载数据集放到 ./data/ 目录"
echo ""
echo " 3. 快速测试推理:"
echo "    python SimpleRun/step1_inference.py"
echo ""
echo " 4. 批量训练:"
echo "    cd EXP_CONFIG"
echo "    python multi_train_any_gpu.py -k A08 -c 10"
echo "============================================"
