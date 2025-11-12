import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import os
from mmengine.config import Config
from mmseg.models import build_segmentor

# -------------------------- Your file paths --------------------------
config1_path = "work_dirs/jiao_baseline/20251103_160735/vis_data/config.py"
config2_path = "work_dirs/deeplabv3plus_r50-d8_4xb4-80k_pascal-context-59-480x480/20251106_193248/vis_data/config.py"
model1_path = "work_dirs/jiao_baseline/best_mIoU_iter_20000.pth"
model2_path = "work_dirs/deeplabv3plus_r50-d8_4xb4-80k_pascal-context-59-480x480/best_mIoU_iter_20000.pth"
img_path = "split_data/val/img/0001.png"
gt_path = "split_data/val/label/0001.png"
save_vis_path = "work_dirs/com/complement_visual.png"
# ----------------------------------------------------------------------

def load_model(config_path, checkpoint_path):
    cfg = Config.fromfile(config_path)
    if 'data_preprocessor' in cfg.model:
        del cfg.model['data_preprocessor']
    model = build_segmentor(cfg.model)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model, cfg

model1, cfg1 = load_model(config1_path, model1_path)
model2, cfg2 = load_model(config2_path, model2_path)

def preprocess_image(img_path, cfg):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    test_pipeline = cfg.test_dataloader.dataset.pipeline
    resize_cfg = next((x for x in test_pipeline if x['type'] == 'Resize'), None)
    target_size = resize_cfg['scale'] if resize_cfg else (img.shape[1], img.shape[0])
    img_resized = cv2.resize(img, target_size)
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0)
    if torch.cuda.is_available():
        img_tensor = img_tensor.cuda()
    return img_resized, img_tensor, target_size

img_resized, img_tensor, target_size = preprocess_image(img_path, cfg1)

def get_crack_prob(model, img_tensor):
    with torch.no_grad():
        output = model(img_tensor)
        logits = output[0]
        
        # Handle both single-channel (sigmoid) and two-channel (softmax) outputs
        if logits.dim() == 3:  # [1, H, W] (single channel for binary segmentation)
            prob = torch.sigmoid(logits)
            crack_prob = prob[0, :, :].cpu().numpy()  # Extract crack probability
        else:  # [1, 2, H, W] (two channels for binary segmentation)
            prob = torch.softmax(logits, dim=1)
            crack_prob = prob[0, 1, :, :].cpu().numpy()  # Extract crack probability (class 1)
        
        return crack_prob

prob1 = get_crack_prob(model1, img_tensor)
prob2 = get_crack_prob(model2, img_tensor)
fused_prob = 0.6 * prob1 + 0.4 * prob2  # Adjust weights as needed

# Load ground truth if available
gt = None
if gt_path and os.path.exists(gt_path):
    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    gt = cv2.resize(gt, target_size)

# Create output directory
os.makedirs(os.path.dirname(save_vis_path), exist_ok=True)

# Visualization
plt.figure(figsize=(20, 5))
cmap = plt.cm.RdYlBu_r  # Red = high crack probability, Blue = low

# 1. Original image
plt.subplot(1, 5, 1)
plt.imshow(img_resized)
plt.title("Original Image")
plt.axis('off')

# 2. Model 1 probability map
plt.subplot(1, 5, 2)
plt.imshow(prob1, cmap=cmap, vmin=0, vmax=1)
plt.title("Model 1 Probability")
plt.axis('off')

# 3. Model 2 probability map
plt.subplot(1, 5, 3)
plt.imshow(prob2, cmap=cmap, vmin=0, vmax=1)
plt.title("Model 2 Probability")
plt.axis('off')

# 4. Fused probability map
plt.subplot(1, 5, 4)
plt.imshow(fused_prob, cmap=cmap, vmin=0, vmax=1)
plt.title("Fused Probability")
plt.axis('off')

# 5. Ground truth
plt.subplot(1, 5, 5)
if gt is not None:
    plt.imshow(gt, cmap='gray')
    plt.title("Ground Truth (White=Crack)")
else:
    plt.text(0.5, 0.5, "No GT", ha='center', va='center')
plt.axis('off')

# Add color bar
cbar = plt.colorbar(plt.gca().get_images()[0], ax=plt.gcf().get_axes(), shrink=0.8)
cbar.set_label("Crack Probability (0=Background, 1=Crack)")

# Save result
plt.tight_layout()
plt.savefig(save_vis_path, dpi=300)
print(f"Result saved to: {save_vis_path}")
plt.show()