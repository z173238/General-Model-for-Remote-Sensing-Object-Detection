# 遥感通用检测引擎

基于 SM3Det (AAAI 2026 Oral) 的统一遥感目标检测引擎，一个模型同时处理光学和 SAR 图像。

## 性能

| 任务 | 模型 | mAP/AP | FPS |
|------|------|--------|-----|
| 光学 OBB | SM3Det rgb head | **mAP 0.569** (DOTA) | 35 |
| SAR HBB | SM3Det sar head | **AP 0.817** (SARDet ship) | 49 |
| 光学 Mask | SAM ViT-B (零样本) | — | ~12 |

## 快速开始

```python
from engine import UnifiedRSDetectionEngine
engine = UnifiedRSDetectionEngine()
result = engine.detect(image)  # 自动判断光学/SAR
```

## 依赖安装

### 1. SM3Det (mmdet 2.x 环境)
```bash
git clone https://github.com/zcablii/SM3Det.git
conda create -n rs_detection python=3.10 -y
conda activate rs_detection
# PyTorch 2.12 (RTX 5090) 或 2.0+ (其他GPU)
conda install pytorch=2.12.0=gpu_cuda129 -c main
pip install pybind11 timm einops
# mmcv-full 源码编译
CPLUS_INCLUDE_PATH="$(python3 -c 'import pybind11; print(pybind11.get_include())')" \
  pip install "mmcv-full==1.7.2" --no-build-isolation --no-deps
pip install "mmdet>=2.25.1,<3.0.0"
pip install -e ./SM3Det/
```

### 2. DIOR 微调 (mmdet 3.x 环境, 可选)
```bash
git clone --depth 1 --branch dev-1.x https://github.com/open-mmlab/mmrotate.git mmrotate-1.x
pip install mmcv mmdet mmrotate
pip install -e ./mmrotate-1.x/
```

### 3. SAM 分割 (可选)
```bash
pip install segment-anything
# 自动下载 SAM ViT-B 权重
```

## 数据集

| 数据集 | 链接 | 用途 |
|------|------|------|
| SOI-Det (DOTA+SARDet+DroneVehicle) | [Kaggle](https://www.kaggle.com/datasets/greatbird/soi-det) | SM3Det 训练/评测 |
| DIOR | [官网](https://gcheng-nwpu.github.io/) | 光学检测评测 |
| SSDD | [Kaggle](https://www.kaggle.com/datasets/bitsandlayers/sar-ship-detection-dataset) | SAR 舰船评测 |

## 模型权重

| 模型 | 链接 | 用途 |
|------|------|------|
| SM3Det ConvNeXt-tiny | [Kaggle](https://www.kaggle.com/models/greatbird/sm3det) | 光学 OBB + SAR HBB |
| SAM ViT-B | [官方](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) | 零样本分割 |
| DIOR fine-tuned | `work_dirs/dior_finetune/epoch_1.pth` | 光学 OBB (DIOR 20类) |

## 项目文件

- `engine.py` — 统一推理引擎 (260行)
- `调研报告_遥感通用检测引擎.md` — 完整调研报告 (v1-v19)
- `training_configs/` — DIOR/DOTA 训练脚本和配置

## 参考

- SM3Det: [arXiv:2412.20665](https://arxiv.org/abs/2412.20665) (AAAI 2026 Oral)
- SAM: [arXiv:2304.02643](https://arxiv.org/abs/2304.02643)
