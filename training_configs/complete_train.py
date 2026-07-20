"""DIOR 完整训练: 从 epoch_1 权重继续, 3 epochs"""
import torch, mmengine, os
torch.serialization.add_safe_globals([mmengine.logging.HistoryBuffer])
_orig_load = torch.load
def _p(f, **kw):
    kw['weights_only'] = False
    return _orig_load(f, **kw)
torch.load = _p

from mmengine.runner import Runner
from mmengine import Config

cfg = Config.fromfile(os.path.join(os.path.dirname(__file__), 'dior_finetune2.py'))
cfg.launcher = 'none'

# 数据加载: num_workers=0 避免 multiprocessing 问题
cfg.train_dataloader.num_workers = 0
cfg.train_dataloader.persistent_workers = False
cfg.train_dataloader.batch_size = 4
cfg.val_dataloader.num_workers = 0
cfg.val_dataloader.persistent_workers = False
cfg.val_dataloader.batch_size = 4
cfg.test_dataloader.num_workers = 0
cfg.test_dataloader.persistent_workers = False
cfg.test_dataloader.batch_size = 4

# 训练: 3 epochs, 每 epoch 做 validation
cfg.max_epochs = 3
cfg.train_cfg.val_interval = 1
cfg.default_hooks.checkpoint.interval = 1
cfg.default_hooks.checkpoint.max_keep_ckpts = 3

# 从现有 DIOR 权重继续 (非 resume, 避免 EMA 冲突)
ckpt = "/home/ubuntu/workspace/General Model for Remote Sensing Object Detection/work_dirs/dior_finetune/epoch_1.pth"
if os.path.exists(ckpt):
    cfg.load_from = ckpt
    print(f"Loading from: {ckpt}")
cfg.resume = False

# 禁用 EMA
cfg.custom_hooks = [dict(type='mmdet.NumClassCheckHook')]

# 减小图片加速
cfg.train_pipeline[3]['scale'] = (800, 800)
cfg.test_pipeline[1]['scale'] = (800, 800)

print(f"DIOR Training: {cfg.max_epochs} epochs, batch=4, 800x800, 20 classes")
print(f"Work dir: {cfg.work_dir}")

runner = Runner.from_cfg(cfg)
runner.train()
print("DONE")
