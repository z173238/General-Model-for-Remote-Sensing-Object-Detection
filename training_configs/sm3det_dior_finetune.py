"""SM3Det DIOR fine-tuning config (mmdet 2.x)"""
num_classes = 20
data_root = '/home/ubuntu/dataset/DIOR_dota_format/'
img_scale = (800, 800)
angle_version = 'le90'
load_from = '/home/ubuntu/workspace/General Model for Remote Sensing Object Detection/work_dirs/sm3det_dior_init.pth'
work_dir = '/home/ubuntu/workspace/General Model for Remote Sensing Object Detection/work_dirs/sm3det_dior/'

# DIOR 20 classes
DIOR_CLASSES = ('airplane','airport','baseballfield','basketballcourt','bridge','chimney',
    'dam','Expressway-Service-area','Expressway-toll-station','golffield','groundtrackfield',
    'harbor','overpass','ship','stadium','storagetank','tenniscourt','trainstation','vehicle','windmill')

# ===== Model (SM3Det OrientedRCNN architecture) =====
model = dict(
    type='OrientedRCNN',
    backbone=dict(
        type='ConvNeXt_moe_MultiInput',
        MoE_Block_inds=[[],[0,2],[0,2,4,6,8],[0,2]],
        datasets=None, num_experts=8, top_k=2, arch='tiny', drop_path_rate=0.1,
        init_cfg=dict(type='Pretrained', prefix='backbone',
            checkpoint='/home/ubuntu/workspace/General Model for Remote Sensing Object Detection/SM3Det_weights/ckpt/convnext-tiny.pth')),
    neck=dict(type='FPN', in_channels=[96,192,384,768], out_channels=256, num_outs=5),
    rpn_head=dict(
        type='OrientedRPNHead', in_channels=256, feat_channels=256,
        version=angle_version,
        anchor_generator=dict(type='AnchorGenerator', scales=[8], ratios=[0.5,1.0,2.0], strides=[4,8,16,32,64]),
        bbox_coder=dict(type='MidpointOffsetCoder', angle_range=angle_version, target_means=[0,0,0,0,0,0], target_stds=[1,1,1,1,0.5,0.5]),
        loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=0.1111111111111111, loss_weight=1.0)),
    roi_head=dict(
        type='OrientedStandardRoIHead',
        bbox_roi_extractor=dict(type='RotatedSingleRoIExtractor', roi_layer=dict(type='RoIAlignRotated', out_size=7, sample_num=2, clockwise=True), out_channels=256, featmap_strides=[4,8,16,32]),
        bbox_head=dict(
            type='RotatedShared2FCBBoxHead', in_channels=256, fc_out_channels=1024, roi_feat_size=7, num_classes=num_classes,
            bbox_coder=dict(type='DeltaXYWHAOBBoxCoder', angle_range=angle_version, norm_factor=None, edge_swap=True, proj_xy=True, target_means=[0,0,0,0,0], target_stds=[0.1,0.1,0.2,0.2,0.1]),
            reg_class_agnostic=True,
            loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))),
    train_cfg=dict(
        rpn=dict(assigner=dict(type='MaxIoUAssigner', pos_iou_thr=0.7, neg_iou_thr=0.3, min_pos_iou=0.3, match_low_quality=True, ignore_iof_thr=-1),
            sampler=dict(type='RandomSampler', num=256, pos_fraction=0.5, neg_pos_ub=-1, add_gt_as_proposals=False), allowed_border=0, pos_weight=-1, debug=False),
        rpn_proposal=dict(nms_pre=2000, max_per_img=2000, nms=dict(type='nms', iou_threshold=0.8), min_bbox_size=0),
        rcnn=dict(assigner=dict(type='MaxIoUAssigner', pos_iou_thr=0.5, neg_iou_thr=0.5, min_pos_iou=0.5, match_low_quality=False, iou_calculator=dict(type='RBboxOverlaps2D'), ignore_iof_thr=-1),
            sampler=dict(type='RRandomSampler', num=512, pos_fraction=0.25, neg_pos_ub=-1, add_gt_as_proposals=True), pos_weight=-1, debug=False)),
    test_cfg=dict(
        rpn=dict(nms_pre=2000, max_per_img=2000, nms=dict(type='nms', iou_threshold=0.8), min_bbox_size=0),
        rcnn=dict(nms_pre=2000, min_bbox_size=0, score_thr=0.05, nms=dict(iou_thr=0.1), max_per_img=2000)))

# ===== Data =====
train_pipeline = [
    dict(type='LoadImageFromFile'), dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=img_scale),
    dict(type='RRandomFlip', flip_ratio=0.5, direction=['horizontal','vertical','diagonal'], version=angle_version),
    dict(type='Normalize', mean=[123.675,116.28,103.53], std=[58.395,57.12,57.375], to_rgb=True),
    dict(type='Pad', size_divisor=32), dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img','gt_bboxes','gt_labels'])]

test_pipeline = [
    dict(type='LoadImageFromFile'), dict(type='RResize', img_scale=img_scale),
    dict(type='Normalize', mean=[123.675,116.28,103.53], std=[58.395,57.12,57.375], to_rgb=True),
    dict(type='Pad', size_divisor=32), dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'], meta_keys=['filename','ori_shape','img_shape','pad_shape','scale_factor'])]

data = dict(
    samples_per_gpu=4, workers_per_gpu=0,
    train=dict(type='DOTADataset', data_root=data_root, ann_file='train/annfiles/', img_prefix='train/images/', pipeline=train_pipeline, classes=DIOR_CLASSES),
    val=dict(type='DOTADataset', data_root=data_root, ann_file='val/annfiles/', img_prefix='val/images/', pipeline=test_pipeline, classes=DIOR_CLASSES),
    test=dict(type='DOTADataset', data_root=data_root, ann_file='val/annfiles/', img_prefix='val/images/', pipeline=test_pipeline, classes=DIOR_CLASSES))

# ===== Schedule =====
optimizer = dict(type='AdamW', lr=0.00005, weight_decay=0.05)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='step', step=[4, 6], warmup='linear', warmup_iters=500, warmup_ratio=0.001)
runner = dict(type='EpochBasedRunner', max_epochs=6)
checkpoint_config = dict(interval=2)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
evaluation = dict(interval=2, metric='mAP')

log_level = 'INFO'
dist_params = dict(backend='nccl')
resume_from = None
workflow = [('train', 1)]
gpu_ids = [0]
custom_imports = dict(imports=['mmrotate.models.backbones.convnext_moe'], allow_failed_imports=False)
