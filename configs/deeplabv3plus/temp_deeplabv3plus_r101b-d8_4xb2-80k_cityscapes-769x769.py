# 只继承目标模型的模型结构（主干+探头），不继承其数据集/调度策略
_base_ = './deeplabv3plus_r101b-d8_4xb2-80k_cityscapes-769x769.py'

# 覆盖数据集配置（使用你的 uavdataset）
dataset_type = 'uavdataset'  # 你的数据集类型
data_root = '你的数据集路径'  # 替换为实际路径（与 uavdataset.py 一致）

# 覆盖数据加载器和 pipeline（复用你的配置）
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', scale=(448, 448), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackSegInputs')
]
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(448, 448), keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]
test_pipeline = val_pipeline

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='img_dir/train', seg_map_path='ann_dir/train'),
        pipeline=train_pipeline
    )
)
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='img_dir/val', seg_map_path='ann_dir/val'),
        pipeline=val_pipeline
    )
)
test_dataloader = val_dataloader

# 覆盖模型参数（类别数等）
model = dict(
    data_preprocessor=dict(
        size=(448, 448),
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
    ),
    decode_head=dict(num_classes=2),
    auxiliary_head=dict(num_classes=2)
)

# 覆盖训练策略（复用你的配置）
train_cfg = dict(type='IterBasedTrainLoop', max_iters=20000, val_interval=500)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# 覆盖日志和保存配置
default_hooks = dict(
    checkpoint=dict(
        interval=2500,
        by_epoch=False,
        max_keep_ckpts=2,
        save_best='mIoU'
    )
)

# 覆盖评估器
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice', 'mFscore'])
test_evaluator = val_evaluator

randomness = dict(seed=42)
work_dir = '/root/shared-nvme/mmsegmentation-1.2.2/work_dirs/deeplabv3plus_r101b-d8_4xb2-80k_cityscapes-769x769'
