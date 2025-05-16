# train.py
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from dataset import ChocoDataset
from model import ChocoNet
import os
from sklearn.model_selection import train_test_split
import pandas as pd


# 设置路径
##augmented
csv_path = '../../dataset_project_iapr2025/train_augmented.csv'
img_dir = '../../dataset_project_iapr2025/augmented_images'
# ##original
# csv_path = '../../dataset_project_iapr2025/train.csv'
# img_dir = '../../dataset_project_iapr2025/train'
##划分验证集和训练集
df = pd.read_csv(csv_path)
train_df,val_df = train_test_split(df,test_size=0.2,random_state=42, shuffle=True)

# 数据加载
dataset = ChocoDataset(csv_file=csv_path, img_dir=img_dir)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# 模型初始化
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
model = ChocoNet(num_classes=13).to(device)

# 损失函数 & 优化器
criterion = torch.nn.MSELoss()
optimizer = Adam(model.parameters(), lr=1e-4)

# 训练过程
epochs =30
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        preds = model(images)
        loss = criterion(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(dataloader):.4f}")

# 保存模型
os.makedirs('weights', exist_ok=True)
torch.save(model.state_dict(), 'weights/choco_model.pth')
