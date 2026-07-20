"""SM3Det → DIOR fine-tuning: 1 epoch at a time"""
import torch, sys, os
sys.path.insert(0, '/home/ubuntu/workspace/General Model for Remote Sensing Object Detection/SM3Det')
import builtins
builtins._orig_load = torch.load
def fl(f, **kw): kw['weights_only'] = False; return builtins._orig_load(f, **kw)
torch.load = fl

from mmcv import Config
cfg = Config.fromfile(os.path.join(os.path.dirname(__file__), 'sm3det_dior_finetune.py'))

# Build model
from mmrotate.models import build_detector
model = build_detector(cfg.model)

# Load SM3Det weights, map keys
ckpt = torch.load(cfg.load_from, map_location='cpu')
tri_sd = ckpt['state_dict']

mapped_sd = {}
for k, v in tri_sd.items():
    if k.startswith('backbone.') or k.startswith('neck.'):
        mapped_sd[k] = v
    elif k.startswith('rgb_rpn_head.'):
        mapped_sd[k.replace('rgb_rpn_head.', 'rpn_head.')] = v
    elif k.startswith('rgb_roi_head.'):
        new_k = k.replace('rgb_roi_head.', 'roi_head.')
        # Skip cls head (27→21 class mismatch, will be random init)
        if 'fc_cls' not in new_k:
            mapped_sd[new_k] = v

missing, unexpected = model.load_state_dict(mapped_sd, strict=False)
print(f'Loaded SM3Det weights: {len(mapped_sd)} keys')
print(f'  Missing (random init): {len(missing)} (cls head OK)')
print(f'  Unexpected (SAR/IFR): {len(unexpected)}')

# Freeze backbone (optional, for faster training)
for name, param in model.backbone.named_parameters():
    param.requires_grad = False
print('Backbone frozen')

model.cuda()
import mmdet.apis
# Save the prepared model for training
torch.save({'state_dict': model.state_dict(), 'model_cfg': cfg.model},
           '/home/ubuntu/workspace/General Model for Remote Sensing Object Detection/work_dirs/sm3det_dior_init.pth')
print(f'Model ready: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params')
print(f'Train: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.1f}M trainable')
print('\nRun training:')
print(f'  cd /home/ubuntu/workspace/General\\ Model\\ for\\ Remote\\ Sensing\\ Object\\ Detection/SM3Det && TORCH_FORCE_WEIGHTS_ONLY_LOAD=0 python3 tools/train.py /home/ubuntu/workspace/General\\ Model\\ for\\ Remote\\ Sensing\\ Object\\ Detection/training_configs/sm3det_dior_finetune.py --launcher none --no-validate')
