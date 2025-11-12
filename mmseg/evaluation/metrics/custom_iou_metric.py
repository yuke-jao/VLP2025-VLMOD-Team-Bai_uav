import numpy as np
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger
from mmseg.registry import METRICS

@METRICS.register_module()
class CrackIoUMetric(BaseMetric):
    def __init__(self, crack_class=1, num_classes=2, **kwargs):
        if 'iou_metrics' in kwargs:
            del kwargs['iou_metrics']
        super().__init__(** kwargs)
        self.crack_class = crack_class
        self.num_classes = num_classes
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
        self.class_names = ['background', 'crack']  # 类别名称

    def process(self, data_batch, data_samples):
        for data_sample in data_samples:
            pred = data_sample['pred_sem_seg']['data'].cpu().numpy().flatten()
            true = data_sample['gt_sem_seg']['data'].cpu().numpy().flatten()
            mask = (true >= 0) & (true < self.num_classes)
            pred, true = pred[mask], true[mask]
            indices = self.num_classes * true + pred
            self.confusion_matrix += np.bincount(indices, minlength=self.num_classes**2).reshape(self.num_classes, self.num_classes)

    def compute_metrics(self, results):
        # 1. 计算所有指标
        # 类别级指标（用于类别表格）
        class_iou = []
        class_acc = []
        for i in range(self.num_classes):
            tp = self.confusion_matrix[i, i]
            total = self.confusion_matrix[i, :].sum() + self.confusion_matrix[:, i].sum() - tp
            iou = tp / (total + 1e-10) if total > 0 else 0.0
            acc = tp / (self.confusion_matrix[i, :].sum() + 1e-10) if self.confusion_matrix[i, :].sum() > 0 else 0.0
            class_iou.append(round(iou, 4))
            class_acc.append(round(acc, 4))

        # 裂缝专项指标（6个核心指标）
        tp = int(self.confusion_matrix[self.crack_class, self.crack_class])
        fp = int(self.confusion_matrix[:, self.crack_class].sum()) - tp
        fn = int(self.confusion_matrix[self.crack_class, :].sum()) - tp
        total = int(self.confusion_matrix.sum())
        all_correct = int(self.confusion_matrix.diagonal().sum())

        crack_iou = round(tp / (tp + fp + fn + 1e-10) if (tp + fp + fn) > 0 else 0.0, 4)
        precision = round(tp / (tp + fp + 1e-10) if (tp + fp) > 0 else 0.0, 4)
        recall = round(tp / (tp + fn + 1e-10) if (tp + fn) > 0 else 0.0, 4)
        f1 = round(2 * precision * recall / (precision + recall + 1e-10) if (precision + recall) > 0 else 0.0, 4)
        acc = round(all_correct / total if total > 0 else 0.0, 4)
        mIoU = round(np.mean(class_iou), 4)

        # 2. 打印类别详情表格
        logger = MMLogger.get_current_instance()
        logger.info("per class results:")
        logger.info("")
        logger.info("+------------+-------+-------+")
        logger.info("|   Class    |  IoU  |  Acc  |")
        logger.info("+------------+-------+-------+")
        for name, iou, acc_cls in zip(self.class_names, class_iou, class_acc):
            logger.info(f"| {name:10} | {iou:.2f} | {acc_cls:.2f} |")  # 保留两位小数
        logger.info("+------------+-------+-------+")
        logger.info("")  # 空行分隔

        # 3. 打印6个核心指标表格
        logger.info("overall metrics:")
        logger.info("")
        logger.info("+------------+-------+")
        logger.info("|   Metric   | Value |")
        logger.info("+------------+-------+")
        logger.info(f"| mIoU       | {mIoU:.4f} |")
        logger.info(f"| Crack IoU  | {crack_iou:.4f} |")
        logger.info(f"| Pr         | {precision:.4f} |")
        logger.info(f"| Re         | {recall:.4f} |")
        logger.info(f"| F1         | {f1:.4f} |")
        logger.info(f"| Acc        | {acc:.4f} |")
        logger.info("+------------+-------+")

        # 4. 返回指标（不影响表格展示，仅用于保存最佳模型）
        return {'mIoU': mIoU}