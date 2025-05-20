import os
import torch
from torch.utils.data import DataLoader
from dataset import YoloDataset
from model import create_model
import time
from torch.optim.lr_scheduler import OneCycleLR
import numpy as np

# 配置
IMG_DIR = '../../dataset_project_iapr2025/train/'
LABEL_DIR = '../data/obj_train_data/'
VAL_IMG_DIR = '../validation/image'
VAL_LABEL_DIR = '../validation/obj_train_data/'

NUM_CLASSES = 14
BATCH_SIZE = 4
EPOCHS = 100  # 增加训练轮数
LEARNING_RATE = 1e-3  # 提高初始学习率
WEIGHT_DECAY = 1e-4
DEVICE = torch.device("cpu")
print(DEVICE)

def collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [dict(boxes=[], labels=[]) for _ in batch]
    for i, item in enumerate(batch):
        labels = item[1]
        if labels.numel() > 0:
            boxes = []
            classes = []
            for l in labels:
                cls, cx, cy, w, h = l.tolist()
                x1 = (cx - w / 2) * 900
                y1 = (cy - h / 2) * 600
                x2 = (cx + w / 2) * 900
                y2 = (cy + h / 2) * 600
                boxes.append([x1, y1, x2, y2])
                classes.append(int(cls))
            targets[i]['boxes'] = torch.tensor(boxes, dtype=torch.float32)
            targets[i]['labels'] = torch.tensor(classes, dtype=torch.int64)
    return torch.stack(images), targets

def evaluate(model, dataloader):
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(DEVICE)
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            # 在评估模式下，我们需要显式地计算损失
            loss_dict = model(images, targets)
            if isinstance(loss_dict, dict):
                losses = sum(loss for loss in loss_dict.values())
            else:
                # 如果返回的是检测结果，我们需要重新计算损失
                model.train()  # 临时切换到训练模式来计算损失
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                model.eval()  # 切换回评估模式

            total_val_loss += losses.item()

    return total_val_loss / len(dataloader)

def train():
    # 创建训练集和验证集
    torch.cuda.empty_cache()
    train_dataset = YoloDataset(IMG_DIR, LABEL_DIR, img_size=(900, 600))
    print(f"训练集大小: {len(train_dataset)}")
    if len(train_dataset) == 0:
        print(f"警告：训练集为空！")
        print(f"检查路径：")
        print(f"图像目录: {os.path.abspath(IMG_DIR)}")
        print(f"标签目录: {os.path.abspath(LABEL_DIR)}")
        return
        
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    val_dataset = YoloDataset(VAL_IMG_DIR, VAL_LABEL_DIR, img_size=(900, 600))
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model = create_model(num_classes=NUM_CLASSES)
    model = model.to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float('inf')
    model.train()

    for epoch in range(EPOCHS):
        total_loss = 0
        start = time.time()

        # 训练阶段
        model.train()
        for images, targets in train_dataloader:
            images = images.to(DEVICE)
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            total_loss += losses.item()

        # 验证阶段
        val_loss = evaluate(model, val_dataloader)

        duration = time.time() - start
        print(f"[Epoch {epoch+1:02d}] Train Loss: {total_loss/len(train_dataloader):.4f} "
              f"Val Loss: {val_loss:.4f} Time: {duration:.2f}s")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print(f"Saved new best model with validation loss: {val_loss:.4f}")
if __name__ == "__main__":
    train()
