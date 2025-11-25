1. Evaluation Data Format
After unzipping the dataset you received, you will find the following folder structure:

gtFine folder: Contains binary mask images output by your model.
Pixel value 0 represents normal road surface, 1 represents cracks.
train subfolder (inside gtFine): Includes binary images at 3 scales: X4, X8, X16, all with resolution 672×378.
leftimg8bit folder: Contains input images (original road images).
train subfolder (inside leftimg8bit): Input images at 3 scales: X4, X8, X16, resolution 672×378.
val subfolder (inside leftimg8bit): Contains 300 validation images for testing, also 672×378.
showgt.py: A visualization script to display binary masks more intuitively (not used in evaluation).
Example of Model Input Image (.jpg):



原数据存放在，在安装zip命令之后，可以使用unzip的命令行来解压文件，shared-nvme/uperfcnrescrack_uav/data/UAV-Crack-dataset.zip
我们需要将数据处理为更适合mmsegmentation框架的格式：划分数据脚本见：shared-nvme/uperfcnrescrack_uav/data/split.py
在经过数据处理之后，我们可以得到一个split_data的文件，其中目录结构为：
train/img，train/label/，val/img，val/label，test
其中是将X4, X8, X16，三个子类别中分贝取20%放入验证集,test是提交评估结果的300张测试图片
