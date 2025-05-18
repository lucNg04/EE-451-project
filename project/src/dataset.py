# dataset.py
import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class ChocoDataset(Dataset):
    def __init__(self, csv_file_or_df, img_dir, transform=None):
        if isinstance(csv_file_or_df, str):
            self.labels_df = pd.read_csv(csv_file_or_df)
        else:
            self.labels_df = csv_file_or_df.reset_index(drop=True)
        #self.labels_df = pd.read_csv(csv_file)
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
