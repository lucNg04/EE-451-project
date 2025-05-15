# ==============================
# src/train_detection.py
# ==============================
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from detection_dataset import DetectionDataset
from detection_model import get_detection_model

def train_detection():
    dataset = DetectionDataset("dataset_project_iapr2025/train", "choco_annotation/obj_train_data", transforms=ToTensor())
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

    model = get_detection_model(num_classes=13)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

    for epoch in range(10):
        model.train()
        for images, targets in dataloader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1} done")

    torch.save(model.state_dict(), "weights/detection_model.pth")