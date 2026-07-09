@echo off
REM ============================================================
REM OpenRSD ICCV 2025 — 一键安装脚本 (Windows CMD)
REM ============================================================
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo ============================================
echo  OpenRSD 安装脚本 (Windows)
echo  工作目录: %CD%
echo ============================================

REM === Step 0: 环境检测 ===
echo.
echo [0/5] 检测环境...
python --version
if %errorlevel% neq 0 (
    echo [ERR] Python 未找到！请先安装 Python 3.8+
    pause
    exit /b 1
)

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv 2>nul
if %errorlevel% neq 0 (
    echo [WARN] nvidia-smi 不可用，请确认 GPU 驱动已安装
)

REM === Step 1: 确认 conda ===
echo.
echo [1/5] 检查 conda...
where conda >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARN] conda 未找到，将使用系统 Python
    goto :skip_conda
)

REM 检查环境是否存在
conda env list 2>nul | findstr "openrsd" >nul
if %errorlevel% equ 0 (
    echo [SKIP] conda 环境 openrsd 已存在
) else (
    echo 创建 conda 环境 openrsd (python=3.10)...
    conda create -n openrsd python=3.10 -y
)

echo 激活 conda 环境...
call conda activate openrsd 2>nul || (
    echo [WARN] 无法自动激活 conda，请手动执行: conda activate openrsd
)
:skip_conda

REM === Step 2: PyTorch ===
echo.
echo [2/5] 安装 PyTorch...
python -c "import torch; print(torch.__version__)" 2>nul
if %errorlevel% equ 0 (
    echo [SKIP] PyTorch 已安装
) else (
    echo 安装 PyTorch + CUDA 12.1...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
)

REM === Step 3: mmcv ===
echo.
echo [3/5] 安装 mmcv...
python -c "import mmcv; print(mmcv.__version__)" 2>nul
if %errorlevel% equ 0 (
    echo [SKIP] mmcv 已安装
) else (
    if exist "mmcv-main.zip" (
        echo 解压 mmcv-main.zip...
        tar -xf mmcv-main.zip -C mmcv-extracted 2>nul || (
            powershell -Command "Expand-Archive -Path mmcv-main.zip -DestinationPath mmcv-extracted -Force"
        )
        echo 安装 mmcv...
        cd mmcv-extracted\mmcv-main
        pip install -e . --no-build-isolation 2>nul || python setup.py install
        cd ..\..
    ) else (
        echo [WARN] mmcv-main.zip 不存在，尝试在线安装...
        pip install mmcv==2.1.0
    )
)

REM === Step 4: 项目依赖 ===
echo.
echo [4/5] 安装项目依赖...
pip install addict albumentations dill einops ftfy huggingface-hub mmengine numpy opencv-python opencv-contrib-python pandas pillow pycocotools pyyaml regex requests rich scikit-image scipy seaborn shapely terminaltables tifffile timm tqdm transformers yapf

pip install -r requirements\optional.txt 2>nul

REM === Step 5: 安装 OpenRSD ===
echo.
echo [5/5] 安装 OpenRSD...
pip install -v -e . 2>nul || python setup.py develop

REM === 验证 ===
echo.
echo ============================================
echo  安装完成 — 验证环境
echo ============================================
python -c "import torch; print('  PyTorch:', torch.__version__); print('  CUDA OK:', torch.cuda.is_available());" 2>nul && (
    python -c "import torch; print('  GPU:', torch.cuda.get_device_name(0))" 2>nul
)

python -c "import mmcv; print('  mmcv:', mmcv.__version__)" 2>nul || echo   [WARN] mmcv 未成功
python -c "import mmdet; print('  mmdet:', mmdet.__version__)" 2>nul || echo   [WARN] mmdet 未成功
python -c "import mmrotate; print('  mmrotate:', mmrotate.__version__)" 2>nul || echo   [WARN] mmrotate 未成功

echo.
echo ============================================
echo  安装完毕！
echo ============================================
echo  下一步：
echo   1. 下载权重: 百度网盘 (见 README)
echo   2. 准备数据集到 .\data\ 目录
echo   3. python SimpleRun/step1_inference.py
echo ============================================
pause
