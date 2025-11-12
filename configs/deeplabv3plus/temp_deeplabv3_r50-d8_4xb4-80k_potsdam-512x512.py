# 1. 基础配置（根据需要继承其他文件）
_base_ = './deeplabv3_r50-d8_4xb4-80k_potsdam-512x512.py'  # 继承模型基础配置

# 2. 数据集和评估器注册
custom_imports = dict(
    imports=['mmseg.datasets.uavdataset', 'mmseg.evaluation.metrics.custom_iou_metric'],
    allow_failed_imports=False
)

# 3. 数据集路径和元信息
dataset_type = 'UavDataset'
data_root = '/root/shared-nvme/mmsegmentation-1.2.2/split_data'
metainfo = dict(classes=('background', 'crack'), num_classes=2)

# 4. 插入提炼的pipeline（关键）
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomResize',
        scale=(448, 448),
        ratio_range=(0.8, 1.2),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=(448, 448), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomFlip', prob=0.3, direction='vertical'),
    dict(type='RandomRotate', prob=0.5, degree=15, seg_pad_val=0),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(448, 448), keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]
test_pipeline = val_pipeline

tta_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
    dict(
        type='TestTimeAug',
        transforms=[
            [dict(type='Resize', scale_factor=r, keep_ratio=True) for r in [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]],
            [dict(type='RandomFlip', prob=0., direction='horizontal'), dict(type='RandomFlip', prob=1., direction='horizontal')],
            [dict(type='RandomFlip', prob=0., direction='vertical'), dict(type='RandomFlip', prob=1., direction='vertical')],
            [dict(type='LoadAnnotations')],
            [dict(type='PackSegInputs')]
        ])
]

train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='train/img', seg_map_path='train/label'),
        pipeline=train_pipeline
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='val/img', seg_map_path='val/label'),
        pipeline=val_pipeline
    )
)
test_dataloader = val_dataloader

# 5. 其他配置（模型、训练策略、评估器等）
model = dict(
    data_preprocessor=dict(size=(448, 448)),
    backbone=dict(norm_cfg=dict(type='BN', requires_grad=True)),
    decode_head=dict(num_classes=2),
    auxiliary_head=dict(num_classes=2)
)

train_cfg = dict(type='IterBasedTrainLoop', max_iters=20000, val_interval=500)
val_evaluator = dict(
    type='CrackIoUMetric',
    crack_class=1,
    num_classes=2
)
test_evaluator = val_evaluator
work_dir = '/root/shared-nvme/mmsegmentation-1.2.2/work_dirs/deeplabv3_r50-d8_4xb4-80k_potsdam-512x512'
