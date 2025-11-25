default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
    cudnn_deterministic=False)
model = dict(
    type='EncoderDecoder',
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=2,
        size=(512, 512)),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=False,
        style='pytorch',
        contract_dilation=True,
        init_cfg=dict(
            type='Pretrained',
            checkpoint=
            'https://download.pytorch.org/models/resnet50-0676ba61.pth')),
    decode_head=dict(
        type='UPerHead',
        in_channels=[256, 512, 1024, 2048],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=3,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0,
            class_weight=[0.8, 1.0, 0.0])),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=1024,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=3,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            class_weight=[0.8, 1.0, 0.0],
            loss_weight=0.4)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole', crop_size=(512, 512), stride=(340, 340)))
dataset_type = 'UavDataset'
data_root = 'split_data'
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomResize',
        scale=(512, 512),
        ratio_range=(0.8, 1.25),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.8, 1.2),
        saturation_range=(0.8, 1.2),
        hue_delta=18),
    dict(type='PackSegInputs')
]
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='UavDataset',
        data_root='split_data',
        data_prefix=dict(img_path='train/img', seg_map_path='train/label'),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(
                type='RandomResize',
                scale=(512, 512),
                ratio_range=(0.8, 1.25),
                keep_ratio=True),
            dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
            dict(type='RandomFlip', prob=0.5, direction='horizontal'),
            dict(type='RandomFlip', prob=0.5, direction='vertical'),
            dict(
                type='PhotoMetricDistortion',
                brightness_delta=32,
                contrast_range=(0.8, 1.2),
                saturation_range=(0.8, 1.2),
                hue_delta=18),
            dict(type='PackSegInputs')
        ]))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='UavDataset',
        data_root='split_data',
        data_prefix=dict(img_path='val/img', seg_map_path='val/label'),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=(512, 512), keep_ratio=True),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs')
        ]))
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    dict(type='PackSegInputs')
]
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='UavDataset',
        data_root='split_data',
        data_prefix=dict(img_path='leftImg8bit/test'),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=(512, 512), keep_ratio=True),
            dict(type='PackSegInputs')
        ]))
test_evaluator = dict(
    type='IoUMetric', iou_metrics=['mIoU', 'mDice', 'mFscore'], ignore_index=2)
val_evaluator = dict(
    type='IoUMetric', iou_metrics=['mIoU', 'mDice', 'mFscore'], ignore_index=2)
optimizer = dict(
    type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.01)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.01),
    clip_grad=dict(max_norm=1.0, norm_type=2))
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-06, by_epoch=False, begin=0,
        end=1000),
    dict(
        type='PolyLR',
        eta_min=1e-06,
        power=0.9,
        begin=1000,
        end=40000,
        by_epoch=False)
]
train_cfg = dict(type='IterBasedTrainLoop', max_iters=12000, val_interval=500)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(
        type='LoggerHook',
        interval=50,
        log_metric_by_epoch=False,
        ignore_last=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=500,
        max_keep_ckpts=3,
        save_best='mIoU',
        rule='greater'),
    sampler_seed=dict(type='DistSamplerSeedHook'))
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    name='visualizer')
work_dir = './work_dirs/ph_77'
randomness = dict(seed=77, deterministic=False)
launcher = 'none'
resume = False
