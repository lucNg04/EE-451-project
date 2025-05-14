# train.py
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from dataset import ChocoDataset
from model import ChocoNet
import os


# # 路径设置
# REFERENCE_DIR = "../../dataset_project_iapr2025/references"
# TRAIN_IMG_DIR = "../../dataset_project_iapr2025/train"
# CSV_PATH = "../../dataset_project_iapr2025/train.csv"
# OUTPUT_DIR = "../../dataset_cnn"
# os.makedirs(OUTPUT_DIR, exist_ok=True)
# 设置路径
csv_path = '../../dataset_project_iapr2025/train.csv'
img_dir = '../../dataset_project_iapr2025/train'

# 数据加载
dataset = ChocoDataset(csv_file=csv_path, img_dir=img_dir)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# 模型初始化
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ChocoNet(num_classes=13).to(device)

# 损失函数 & 优化器
criterion = torch.nn.MSELoss()
optimizer = Adam(model.parameters(), lr=1e-3)

# 训练过程
epochs = 150
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
