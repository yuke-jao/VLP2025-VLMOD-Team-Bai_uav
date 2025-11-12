# configs/_base_/datasets/uav_dataset.py

# 数据集路径
dataset_type = 'UavDataset'
data_root = 'split_data'

# 最优裁剪尺寸 - 基于你的三组数据特点
crop_size = (448, 448)  # 最适合你数据的尺寸

# 训练预处理 Pipeline
# 训练预处理 Pipeline
train_pipeline = [
    # 1. 基础加载
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    
    # 2. 尺寸标准化
    dict(
        type='RandomResize',
        scale=(448, 448),
        ratio_range=(0.8, 1.2),
        keep_ratio=True),
    
    # 3. 随机裁剪
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    
    # 4. 空间变换增强
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomFlip', prob=0.3, direction='vertical'),
    dict(type='RandomRotate', prob=0.5, degree=15, seg_pad_val=0),
    
    # 5. 色彩增强 - 修复这一行！
    dict(type='PhotoMetricDistortion'),  # 使用默认参数
    
    # 6. 打包输出
    dict(type='PackSegInputs')
]

# 验证预处理 Pipeline - 保持确定性
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(448, 448), keep_ratio=True),  # 统一到相同尺寸
    dict(type='LoadAnnotations'),  # 验证时需要标注来计算指标
    dict(type='PackSegInputs')
]

# 完整TTA后处理 - 最大化精度
img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]  # 6种尺度变换
tta_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
    dict(
        type='TestTimeAug',
        transforms=[
            [  # 第一组：多尺度变换 (6个尺度)
                dict(type='Resize', scale_factor=r, keep_ratio=True)
                for r in img_ratios
            ],
            [  # 第二组：水平翻转 (2个方向)
                dict(type='RandomFlip', prob=0., direction='horizontal'),  # 原图
                dict(type='RandomFlip', prob=1., direction='horizontal')   # 翻转图
            ],
            [  # 第三组：垂直翻转 (2个方向) - 新增！
                dict(type='RandomFlip', prob=0., direction='vertical'),    # 原图
                dict(type='RandomFlip', prob=1., direction='vertical')     # 垂直翻转
            ],
            [dict(type='LoadAnnotations')],  # 加载标注
            [dict(type='PackSegInputs')]     # 打包输出
        ])
]

# 训练 Dataloader 配置
train_dataloader = dict(
    batch_size=8,  # 448×448尺寸下可以用较大batch_size
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='train/img', 
            seg_map_path='train/label'
        ),
        pipeline=train_pipeline  # 使用优化后的训练pipeline
    )
)

# 验证 Dataloader 配置 - 使用标准测试pipeline
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='val/img', 
            seg_map_path='val/label'
        ),
        pipeline=test_pipeline  # 验证时使用标准pipeline
    )
)

# 测试 Dataloader 配置 - 
test_dataloader = dict(
     batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='val/img', 
            seg_map_path='val/label'
        ),
        pipeline=test_pipeline  # 验证时使用标准pipeline
    )
)

# 验证评估器
val_evaluator = dict(
    type='IoUMetric', 
    iou_metrics=['mIoU', 'mDice', 'mFscore']  # 多指标评估
)

# 测试评估器 - 同样使用多指标
test_evaluator = dict(
    type='IoUMetric', 
    iou_metrics=['mIoU', 'mDice', 'mFscore']
)