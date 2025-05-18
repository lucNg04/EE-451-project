import os
import torch
from torch.utils.data import DataLoader
from dataset import YoloDataset
from model import create_model
import time

# 配置
IMG_DIR = '../../dataset_project_iapr2025/train/'
LABEL_DIR = '../data/obj_train_data/'
NUM_CLASSES = 13
BATCH_SIZE = 4
EPOCHS = 30
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    return torch.stack(images), targets

def train():
    dataset = YoloDataset(IMG_DIR, LABEL_DIR, img_size=(640, 640))
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    model = create_model(num_classes=NUM_CLASSES)
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        start = time.time()

        for images, targets in dataloader:
            images = images.to(DEVICE)
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            total_loss += losses.item()

        duration = time.time() - start
        print(f"[Epoch {epoch+1:02d}] Loss: {total_loss:.4f}  Time: {duration:.2f}s")

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    train()
