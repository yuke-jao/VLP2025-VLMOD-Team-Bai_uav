# configs/deeplabv3plus/jiao.py
# 不继承任何基础配置，仅定义需要覆盖的参数
# 所有基础配置（如模型结构、优化器、数据集路径）由模型文件提供

# 数据集注册（必须保留）
custom_imports = dict(
    imports=['mmseg.datasets.uavdataset'],
    allow_failed_imports=False
)

# 1. 覆盖数据集类型和路径
dataset_type = 'UavDataset'
data_root = '/path/to/your/uavdataset'  # 你的数据集根目录

# 2. 覆盖数据预处理（输入尺寸448x448）
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

# 3. 覆盖数据加载器（batch_size等）
train_dataloader = dict(
    batch_size=4,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='img_dir/train', seg_map_path='ann_dir/train'),
        pipeline=train_pipeline
    )
)
val_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='img_dir/val', seg_map_path='ann_dir/val'),
        pipeline=val_pipeline
    )
)
test_dataloader = val_dataloader

# 4. 覆盖模型参数（类别数、输入尺寸、BN替代SyncBN）
model = dict(
    data_preprocessor=dict(size=(448, 448)),
    backbone=dict(norm_cfg=dict(type='BN', requires_grad=True)),  # 单卡用BN
    decode_head=dict(num_classes=2),
    auxiliary_head=dict(num_classes=2)
)

# 5. 覆盖训练策略（迭代次数）
train_cfg = dict(max_iters=20000, val_interval=500)

# 6. 覆盖日志和保存配置
default_hooks = dict(
    checkpoint=dict(interval=2500, save_best='mIoU', max_keep_ckpts=2),
    logger=dict(interval=100)
)

# 7. 覆盖评估器
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice', 'mFscore'])
test_evaluator = val_evaluator

# 8. 随机种子
randomness = dict(seed=42)