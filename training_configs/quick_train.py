import torch, mmengine
torch.serialization.add_safe_globals([mmengine.logging.HistoryBuffer])
_orig_load = torch.load
def _p(f, map_location=None, weights_only=False, **kw):
    return _orig_load(f, map_location=map_location, weights_only=False, **kw)
torch.load = _p

from mmengine.runner import Runner
from mmengine import Config

cfg = Config.fromfile('/tmp/dior_finetune2.py')
cfg.launcher = 'none'
cfg.train_dataloader.num_workers = 0
cfg.train_dataloader.persistent_workers = False
cfg.train_dataloader.batch_size = 4
cfg.val_dataloader.num_workers = 0
cfg.val_dataloader.persistent_workers = False
cfg.val_dataloader.batch_size = 4
cfg.test_dataloader.num_workers = 0
cfg.test_dataloader.persistent_workers = False
cfg.test_dataloader.batch_size = 4
cfg.max_epochs = 1
cfg.train_cfg.val_interval = 1
cfg.default_hooks.checkpoint.interval = 1

print('DIOR 1-epoch quick train')
runner = Runner.from_cfg(cfg)
runner.train()
print('DONE')
