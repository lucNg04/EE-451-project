import os
import torch
from torch.utils.data import DataLoader
from dataset import YoloDataset
from model import create_model
import time
from torch.optim.lr_scheduler import CosineAnnealingLR

# 配置
IMG_DIR = '../../dataset_project_iapr2025/train/'
LABEL_DIR = '../data/obj_train_data/'
VAL_IMG_DIR = '../validation/image'
VAL_LABEL_DIR = '../validation/obj_train_data/'

NUM_CLASSES = 14
BATCH_SIZE = 4
EPOCHS = 60
LEARNING_RATE = 5e-4  # 降低初始学习率
WEIGHT_DECAY = 1e-4   # 添加权重衰减
DEVICE = torch.device("cuda")
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
                x1 = (cx - w / 2) * 640
                y1 = (cy - h / 2) * 640
                x2 = (cx + w / 2) * 640
                y2 = (cy + h / 2) * 640
                boxes.append([x1, y1, x2, y2])
                classes.append(int(cls))
            targets[i]['boxes'] = torch.tensor(boxes, dtype=torch.float32)
            targets[i]['labels'] = torch.tensor(classes, dtype=torch.int64)
        #print("Batch labels:", classes)
    return torch.stack(images), targets
# def validate(model, val_loader, device):
#     model.eval()
#     print("\n🔍 Running validation...")
#
#     with torch.no_grad():
#         for idx, (images, _) in enumerate(val_loader):
#             images = images.to(device)
#             preds = model(images)
#             for i, pred in enumerate(preds):
#                 labels = pred["labels"].tolist()
#                 print(f"[VAL] Sample {idx}-{i} → Predicted labels: {labels}")
#     model.train()
def evaluate(model, dataloader):
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(DEVICE)
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            # 将模型移动到CPU进行评估
            model = model.cpu()
            images = images.cpu()
            targets = [{k: v.cpu() for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            if isinstance(loss_dict, dict):
                losses = sum(loss for loss in loss_dict.values())
            else:
                model.train()
                loss_dict = model(images, targets)
                losses=sum(loss for loss in loss_dict.values())
                model.eval()
            total_val_loss += losses.item()

            # 将模型移回GPU
            model = model.to(DEVICE)

    return total_val_loss / len(dataloader)



def train():
    #train
    dataset = YoloDataset(IMG_DIR, LABEL_DIR, img_size=(640, 640))
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    #validate
    val_dataset = YoloDataset(VAL_IMG_DIR, VAL_LABEL_DIR, img_size=(640, 640))
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model = create_model(num_classes=NUM_CLASSES).to(DEVICE)

    # 修改优化器配置
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999)
    )
    
    # 添加学习率调度器
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LEARNING_RATE/10)

    model.train()
    best_val_loss = float('inf')
    for epoch in range(EPOCHS):
        total_loss = 0
        start = time.time()
        model.train()
        for images, targets in dataloader:
            images = images.to(DEVICE)
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            
            # 添加梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            total_loss += losses.item()
            
        # 更新学习率
        scheduler.step()
        
        val_loss = evaluate(model, val_dataloader)
        duration = time.time() - start
        print(f"[Epoch {epoch+1:02d}] Loss: {total_loss:.4f}  Val Loss: {val_loss:.4f} Time: {duration:.2f}s LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    train()
