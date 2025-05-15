# ==============================
# src/classification_dataset.py
# ==============================
import pandas as pd
from torchvision import transforms
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class ClassificationDataset(Dataset):
    # def __init__(self, img_dir, csv_path, transform=None):
    #     self.img_dir = img_dir
    #     self.data = pd.read_csv(csv_path)
    #     self.transform = transform

    # def __len__(self):
    #     return len(self.data)

    # def __getitem__(self, idx):
    #     row = self.data.iloc[idx]
    #     img_path = os.path.join(self.img_dir, row['filename'])
    #     image = Image.open(img_path).convert("RGB")
    #     labels = torch.tensor(row[1:].values.astype(float), dtype=torch.float32)

    #     if self.transform:
    #         image = self.transform(image)

    #     return image, labels
        def __init__(self, csv_file, img_dir, transform=None):
            self.labels_df = pd.read_csv(csv_file)
            self.img_dir = img_dir
            self.transform = transform or T.Compose([
                T.Resize((224, 224)),
                T.ToTensor()
                ])
            self.image_ids = self.labels_df['id'].astype(str)
            self.labels = self.labels_df.drop(columns=['id']).values.astype(float)

        def __len__(self):
            return len(self.image_ids)

        def __getitem__(self, idx):
            img_id = self.image_ids.iloc[idx]
            img_path = os.path.join(self.img_dir, f"L{img_id}.JPG")
            image = Image.open(img_path).convert("RGB")
            image = self.transform(image)
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return image, label