import os
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np

class YoloDataset(Dataset):
    def __init__(self, img_dir, label_dir, img_size=(640, 640), transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.transform = transform
        self.image_filenames = [
            f for f in os.listdir(self.img_dir)
            if f.endswith('.JPG') or f.endswith('.png')
        ]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.img_dir, img_name)
        label_path = os.path.join(self.label_dir, os.path.splitext(img_name)[0] + '.txt')

        # 读取图像
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.img_size)
        image = image.astype(np.float32) / 255.0
        image = torch.tensor(image).permute(2, 0, 1)

        # 读取标签
        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls, x, y, w, h = map(float, parts)

                        labels.append([cls, x, y, w, h])

        labels = torch.tensor(labels, dtype=torch.float32)

        return image, labels
