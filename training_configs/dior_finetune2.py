_base_ = '/home/ubuntu/workspace/General Model for Remote Sensing Object Detection/mmrotate-1.x/configs/rotated_rtmdet/rotated_rtmdet_tiny-3x-dota.py'

# Override for DIOR
num_classes = 20
data_root = '/home/ubuntu/dataset/DIOR_dota_format/'
work_dir = '/home/ubuntu/workspace/General Model for Remote Sensing Object Detection/work_dirs/dior_finetune/'
load_from = '/home/ubuntu/.cache/torch/hub/checkpoints/rotated_rtmdet_tiny-3x-dota-9d821076.pth'
max_epochs = 6
batch_size = 8
base_lr = 0.0001

metainfo = dict(
    classes=('ship','vehicle','tenniscourt','storagetank','baseballfield',
             'windmill','harbor','airplane','bridge','overpass',
             'groundtrackfield','Expressway-Service-area','basketballcourt',
             'chimney','airport','Expressway-toll-station','stadium','dam',
             'golffield','trainstation'))

model = dict(
    bbox_head=dict(num_classes=num_classes),
    test_cfg=dict(nms_pre=2000, score_thr=0.05, nms=dict(iou_threshold=0.1), max_per_img=2000))

train_dataloader = dict(
    batch_size=batch_size, num_workers=4,
    dataset=dict(
        type='DOTADataset', data_root=data_root,
        ann_file='train/annfiles/', data_prefix=dict(img_path='train/images/'),
        metainfo=metainfo))

val_dataloader = dict(
    batch_size=4, num_workers=2,
    dataset=dict(
        type='DOTADataset', data_root=data_root,
        ann_file='val/annfiles/', data_prefix=dict(img_path='val/images/'),
        metainfo=metainfo))
test_dataloader = dict(
    batch_size=4, num_workers=2,
    dataset=dict(
        type='DOTADataset', data_root=data_root,
        ann_file='val/annfiles/', data_prefix=dict(img_path='val/images/'),
        metainfo=metainfo))

val_evaluator = dict(type='DOTAMetric', metric='mAP')

train_cfg = dict(max_epochs=max_epochs, val_interval=3)
optim_wrapper = dict(optimizer=dict(lr=base_lr))
param_scheduler = [
    dict(type='LinearLR', start_factor=1.0e-5, by_epoch=False, begin=0, end=500),
    dict(type='CosineAnnealingLR', eta_min=base_lr*0.05, begin=2, end=max_epochs, by_epoch=True)]

default_hooks = dict(checkpoint=dict(max_keep_ckpts=2))
