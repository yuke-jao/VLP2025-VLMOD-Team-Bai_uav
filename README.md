# VLP2025-VLMOD-Team-Bai_uav
2025 长安大学 VLP 挑战赛 VLMOD 赛道参赛代码
团队成员：焦雨可、蒋嘉杰、万家乐、momi（莫梦华）
本项目基于MMSegmentation框架，实现了面向UAV-Crack数据集的UPerFCN-ResCrack语义分割模型。该模型采用ResNet50结合UPerHead结构，在原始二分类任务基础上扩展为三分类（包含一个忽略类别），并引入空洞卷积（Dilated Convolution）以增强骨干网络的特征提取能力，从而更适用于裂缝检测任务。
突出特点：
模型结构: ResNet-50 作为骨干，搭配 UPerHead 解码器和 FCN 辅助头。
数据集: 使用 UAVCrackDataset 进行训练和评估。
空洞卷积：增强骨干网络感受野。
训练策略: 采用 AdamW 优化器，结合 Linear Warm-up 和 Poly 学习率衰减策略。
项目的文件结构如下：
.
├── configs
├── data
├── docs
├── LICENSE
├── mmseg
├── README.md
├── requirements.txt
├── results
├── tools
└── work_dirs
其中：
  configs目录下是uperfcn_res文件夹，此文件夹是存放uperfcn_res类型的模型配置的相关配置文件，文件夹之下的uperfcn_res50.py文件即为本项目的配置文件（包含了模型架构定义、数据加载与增强流程、优化器与学习率调度设置、训练策略以及运行环境配置，用于无人机图像的语义分割任务。）
data目录下
.
└── split_data
    ├── test
    ├── train
    │   ├── img
    │   └── label
    └── val
        ├── img
        └── label
  split_data即我们使用的数据集名称，它分为三个部分，训练、验证、测试。
mmseg目录下存放注册文件
.
├── apis
│   ├── inference.py
│   ├── __init__.py
│   ├── mmseg_inferencer.py 
├── datasets
│   ├── __init__.py
│   └── uavdataset.py
└── visualization
    ├── __init__.py
    ├── local_visualizer.py
  apis 文件夹：主要存放与模型训练、测试、推理等核心任务相关的接口函数。
  datasets 文件夹：用于存放数据集相关的定义与数据的注册。__init__.py负责导入注册的数据集类，uavdataset.py负责对数据集类进行注册。
  visualization 文件夹：主要负责语义分割结果的可视化功能。__init__.py负责导入注册的可视化类，local_visualizer.py对可视化类进行注册。
results目录下存放推理结果文件
.
├── gener_vis_j_2247
└── result_j_2247
  gener_vis_j_2247内部包含模型推理的300张图片的可视化，包括图片本身及模型进行的标注，保存为.jpg格式。
  result_j_2247内部包含模型推理的300张图片的二值掩码图像，保存为.png格式。
tools目录下存放工具文件
.
├── infer.py
├── test.py
└── train.py
  infer.py用于批量生成预测结果，输出掩码图像与可视化图像，其中需要设置部分参数：base_root='根路径'，config_file='模型配置文件路径'，checkpoint_file='所选模型权重路径'，img_dir='所用测试集路径'，save_mask_dir='批量保存预测掩码的目录'，save_vis_dir='批量保存可视化结果的目录'。
  test.py用于量化评估模型性能，主要输出客观指标，验证模型泛化能力。
  train.py用于进行训练。
work_dirs目录存放训练与验证结果，包括训练时的完整配置，训练得到的权重文件，每次训练的工作日志。
