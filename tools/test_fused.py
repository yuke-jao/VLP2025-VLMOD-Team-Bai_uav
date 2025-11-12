import argparse
import os
import os.path as osp
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from mmengine.config import Config
from mmengine.model import BaseModel
from mmseg.models import build_segmentor
from mmseg.registry import MODELS as MMSeg_MODELS
from mmengine.structures import BaseDataElement


# 模拟MMSeg的DataSample类
class MockDataSample(BaseDataElement):
    def __init__(self, img_name, img_path):
        super().__init__()
        self.set_metainfo({'img_name': img_name, 'img_path': img_path})
        self.pred_sem_seg = None


# 手动预处理
def simple_preprocess(img_path, img_scale=(480, 480)):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, img_scale, interpolation=cv2.INTER_LINEAR)
    img = img / 255.0
    img = img.transpose(2, 0, 1)
    return torch.from_numpy(img).float()


# 融合模型类
@MMSeg_MODELS.register_module()
class FusedSegmentor(BaseModel):
    def __init__(self, 
                 model1_cfg, model2_cfg, model1_ckpt, model2_ckpt,
                 weight1=0.6, weight2=0.4, threshold=0.5, **kwargs):
        kwargs.pop('data_preprocessor', None)
        super().__init__(data_preprocessor=None,** kwargs)
        
        self.model1 = build_segmentor(model1_cfg)
        self.model2 = build_segmentor(model2_cfg)
        self._load_checkpoint(self.model1, model1_ckpt)
        self._load_checkpoint(self.model2, model2_ckpt)
        self.weight1 = weight1
        self.weight2 = weight2
        self.threshold = threshold
        self.model1.eval()
        self.model2.eval()

    def _load_checkpoint(self, model, ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        return model

    def forward(self, inputs, data_samples=None, mode='predict'):
        if mode == 'predict':
            return self.predict(inputs, data_samples)
        return super().forward(inputs, data_samples, mode)

    def predict(self, inputs, data_samples=None):
        with torch.no_grad():
            out1 = self.model1(inputs, data_samples, mode='predict')
            logits1 = out1[0].seg_logits

        with torch.no_grad():
            out2 = self.model2(inputs, data_samples, mode='predict')
            logits2 = out2[0].seg_logits

        if logits1.shape[1] == 1:
            prob1 = torch.sigmoid(logits1)
        else:
            prob1 = torch.softmax(logits1, dim=1)[:, 1:2, ...]

        if logits2.shape[1] == 1:
            prob2 = torch.sigmoid(logits2)
        else:
            prob2 = torch.softmax(logits2, dim=1)[:, 1:2, ...]

        fused_prob = self.weight1 * prob1 + self.weight2 * prob2
        fused_label = (fused_prob > self.threshold).long()

        for ds in data_samples:
            ds.pred_sem_seg = fused_label[0]
        return data_samples


# 计算指标
def calculate_metrics(pred_dir, label_dir):
    pred_files = [f for f in os.listdir(pred_dir) if f.endswith(('.png', '.jpg'))]
    label_files = [f for f in os.listdir(label_dir) if f.endswith(('.png', '.jpg'))]
    common_files = list(set(pred_files) & set(label_files))
    if not common_files:
        raise ValueError("预测目录和标签目录没有共同文件")

    total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0
    for f in common_files:
        pred = cv2.imread(osp.join(pred_dir, f), cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(osp.join(label_dir, f), cv2.IMREAD_GRAYSCALE)
        pred = cv2.resize(pred, (label.shape[1], label.shape[0]), interpolation=cv2.INTER_NEAREST)

        pred = (pred > 127).astype(np.uint8)
        label = (label > 127).astype(np.uint8)

        tp = np.sum((pred == 1) & (label == 1))
        fp = np.sum((pred == 1) & (label == 0))
        fn = np.sum((pred == 0) & (label == 1))
        tn = np.sum((pred == 0) & (label == 0))

        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tn += tn

    iou = total_tp / (total_tp + total_fp + total_fn + 1e-8)
    acc = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn + 1e-8)
    precision = total_tp / (total_tp + total_fp + 1e-8)
    recall = total_tp / (total_tp + total_fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    return {
        'IoU': iou, 'Accuracy': acc, 'Dice': f1,
        'Fscore': f1, 'Precision': precision, 'Recall': recall
    }


# 数据集类（只返回图像，元信息单独处理）
class SimpleDataset(Dataset):
    def __init__(self, img_dir, img_scale=(480, 480)):
        self.img_dir = img_dir
        self.img_scale = img_scale
        self.img_names = [f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg'))]
        if not self.img_names:
            raise FileNotFoundError(f"图像目录{img_dir}中没有图片文件")

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = osp.join(self.img_dir, img_name)
        img = simple_preprocess(img_path, self.img_scale)
        # 只返回图像和元信息字典（不返回自定义对象）
        return img, img_name, img_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config1', help='模型1配置文件')
    parser.add_argument('checkpoint1', help='模型1权重')
    parser.add_argument('config2', help='模型2配置文件')
    parser.add_argument('checkpoint2', help='模型2权重')
    parser.add_argument('--weight1', type=float, default=0.6)
    parser.add_argument('--weight2', type=float, default=0.4)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--work-dir', default='work_dirs/com/final_eval')
    parser.add_argument('--img-dir', required=True, help='测试图像目录')
    parser.add_argument('--label-dir', required=True, help='测试标签目录')
    parser.add_argument('--img-scale', type=int, nargs=2, default=(480, 480))
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.work_dir, exist_ok=True)
    pred_dir = osp.join(args.work_dir, 'preds')
    os.makedirs(pred_dir, exist_ok=True)

    # 加载配置
    cfg1 = Config.fromfile(args.config1)
    cfg2 = Config.fromfile(args.config2)
    cfg1.model.pop('data_preprocessor', None)
    cfg2.model.pop('data_preprocessor', None)

    # 构建模型
    fused_model = FusedSegmentor(
        model1_cfg=cfg1.model,
        model2_cfg=cfg2.model,
        model1_ckpt=args.checkpoint1,
        model2_ckpt=args.checkpoint2,
        weight1=args.weight1,
        weight2=args.weight2,
        threshold=args.threshold
    )
    if torch.cuda.is_available():
        fused_model = fused_model.cuda()

    # 构建数据集（不使用DataLoader的自动拼接）
    dataset = SimpleDataset(img_dir=args.img_dir, img_scale=args.img_scale)
    
    # 手动迭代（避免DataLoader处理自定义对象）
    for img, img_name, img_path in dataset:
        # 构造输入格式（添加批次维度）
        inputs = img.unsqueeze(0)  # [1, C, H, W]
        data_samples = [MockDataSample(img_name=img_name, img_path=img_path)]
        
        if torch.cuda.is_available():
            inputs = inputs.cuda()
        
        # 推理
        with torch.no_grad():
            preds = fused_model(inputs, data_samples, mode='predict')
        
        # 保存结果
        pred_label = preds[0].pred_sem_seg.data.cpu().numpy()[0]
        cv2.imwrite(osp.join(pred_dir, img_name), (pred_label * 255).astype(np.uint8))

    # 计算指标
    metrics = calculate_metrics(pred_dir, args.label_dir)
    print("评估指标：")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    with open(osp.join(args.work_dir, 'metrics.txt'), 'w') as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")


if __name__ == '__main__':
    main()