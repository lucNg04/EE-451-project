# ==============================
# src/detection_dataset.py
# ==============================
import os
import torch
from torch.utils.data import Dataset
from PIL import Image

class DetectionDataset(Dataset):
    def __init__(self, img_dir, label_dir, transforms=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transforms = transforms
        self.images = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))]

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        label_path = os.path.join(self.label_dir, img_name.replace('.jpg', '.txt').replace('.png', '.txt'))

        image = Image.open(img_path).convert("RGB")
        w, h = image.size
        boxes, labels = [], []

        with open(label_path) as f:
            for line in f:
                class_id, x, y, w_ratio, h_ratio = map(float, line.strip().split())
                x1 = (x - w_ratio/2) * w
                y1 = (y - h_ratio/2) * h
                x2 = (x + w_ratio/2) * w
                y2 = (y + h_ratio/2) * h
                boxes.append([x1, y1, x2, y2])
                labels.append(int(class_id))

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([idx])
        }

        if self.transforms:
            image = self.transforms(image)

        return image, target

    def __len__(self):
        return len(self.images)
