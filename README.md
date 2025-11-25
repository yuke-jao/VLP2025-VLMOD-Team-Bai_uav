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
├── Untitled Folder
├── untitled.txt
└── work_dirs
其中：
configs目录下是uperfcn_res文件夹，此文件夹是存放uperfcn_res类型的模型配置的相关配置文件，文件夹之下的uperfcn_res50.py文件即为本项目的配置文件（包含了模型架构定义、数据加载与增强流程、优化器与学习率调度设置、训练策略以及运行环境配置，用于无人机图像的语义分割任务。）
data目录下




