#!/bin/bash

export DEBIAN_FRONTEND=noninteractive

# 路径配置
ROOT_DIR=~/shared-nvme/mmsegmentation-1.2.2
CONFIG_DIR=$ROOT_DIR/configs/deeplabv3plus
WORK_DIR_ROOT=$ROOT_DIR/work_dirs

# 你的数据集路径（修改为实际路径）
DATA_ROOT="/root/shared-nvme/mmsegmentation-1.2.2/split_data"
# 类别数（背景 + 目标）
NUM_CLASSES=2
# 输入尺寸
INPUT_SIZE="(448, 448)"

cd $CONFIG_DIR || { echo "错误：无法进入配置目录 $CONFIG_DIR"; exit 1; }

# 待训练的模型（不同主干网络）
MODEL_CONFIGS=(
    "deeplabv3plus_r18-d8_4xb4-80k_potsdam-512x512.py"
    "deeplabv3plus_r18b-d8_4xb2-80k_cityscapes-512x1024.py"
    "deeplabv3plus_r50-d8_4xb4-80k_potsdam-512x512.py"
    "deeplabv3plus_r50b-d8_4xb2-80k_cityscapes-512x1024.py"
    "deeplabv3plus_r101-d8_4xb4-80k_potsdam-512x512.py"
    "deeplabv3plus_r101b-d8_4xb2-80k_cityscapes-512x1024.py"
    "deeplabv3plus_r101-d16-mg124_4xb2-80k_cityscapes-512x1024.py"
)

echo "===== 待训练模型总数：${#MODEL_CONFIGS[@]} 个 ====="
for cfg in "${MODEL_CONFIGS[@]}"; do echo "- $cfg"; done
echo "=========================="
echo ""

# 循环训练每个模型
for cfg_file in "${MODEL_CONFIGS[@]}"; do
    model_name=$(basename "$cfg_file" .py)
    work_dir="$WORK_DIR_ROOT/$model_name"
    echo "===== 开始训练：$model_name ====="
    echo "配置文件：$cfg_file"
    echo "工作目录：$work_dir"
    mkdir -p "$work_dir" || { echo "错误：无法创建工作目录"; exit 1; }

    # 生成临时配置文件（纯 Python 配置，不含 bash 命令）
    temp_cfg="temp_${model_name}.py"
    cat > "$temp_cfg" <<EOF
# 1. 基础配置（根据需要继承其他文件）
_base_ = './$cfg_file'  # 继承模型基础配置

# 2. 数据集和评估器注册
custom_imports = dict(
    imports=['mmseg.datasets.uavdataset', 'mmseg.evaluation.metrics.custom_iou_metric'],
    allow_failed_imports=False
)

# 3. 数据集路径和元信息
dataset_type = 'UavDataset'
data_root = '$DATA_ROOT'
metainfo = dict(classes=('background', 'crack'), num_classes=$NUM_CLASSES)

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
    data_preprocessor=dict(size=$INPUT_SIZE),
    backbone=dict(norm_cfg=dict(type='BN', requires_grad=True)),
    decode_head=dict(num_classes=$NUM_CLASSES),
    auxiliary_head=dict(num_classes=$NUM_CLASSES)
)

train_cfg = dict(type='IterBasedTrainLoop', max_iters=20000, val_interval=500)
val_evaluator = dict(
    type='CrackIoUMetric',
    crack_class=1,
    num_classes=$NUM_CLASSES
)
test_evaluator = val_evaluator
work_dir = '$work_dir'
EOF

    # 关键：在配置文件生成后执行 sed 命令，删除可能残留的 iou_metrics
    sed -i '/iou_metrics/d' "$temp_cfg"

    # 启动训练
    cd $ROOT_DIR || { echo "错误：无法进入根目录"; exit 1; }
    echo "训练中...（日志保存到 $work_dir/train.log）"
    python tools/train.py "$CONFIG_DIR/$temp_cfg" 2>&1 | tee "$work_dir/train.log"

    # 清理临时文件
    rm -f "$CONFIG_DIR/$temp_cfg"

    # 检查结果
    if [ -f "$work_dir/latest.pth" ]; then
        echo "===== 模型 $model_name 训练完成 ====="
    else
        echo "===== 模型 $model_name 训练失败 ====="
    fi
    echo ""
    cd $CONFIG_DIR || exit 1
done

echo "所有模型训练结束！结果在 $WORK_DIR_ROOT"