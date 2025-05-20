import os
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from dataset import YoloDataset

# 设置路径（你可以根据自己实际位置调整）
IMG_DIR = '../../dataset_project_iapr2025/train/'
LABEL_DIR = '../data/obj_train_data/'

# 创建 Dataset 和 DataLoader
dataset = YoloDataset(img_dir=IMG_DIR, label_dir=LABEL_DIR, img_size=(900, 600))
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# 获取一个 batch
images, labels = next(iter(dataloader))
image = images[0].permute(1, 2, 0).numpy()
label = labels[0].numpy()

# 画图显示
fig, ax = plt.subplots(1)
ax.imshow(image)

h, w = 600, 900  # 因为我们 resize 到这个尺寸

for box in label:
    cls_id, cx, cy, bw, bh = box
    x1 = (cx - bw / 2) * w
    y1 = (cy - bh / 2) * h
    box_w = bw * w
    box_h = bh * h
    rect = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor='red', facecolor='none')
    ax.add_patch(rect)
    ax.text(x1, y1 - 5, f'{int(cls_id)}', color='yellow', fontsize=10, backgroundcolor='black')

plt.title("YOLO Dataset Sample")
plt.show()
