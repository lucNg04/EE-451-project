# train.py
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from dataset import ChocoDataset
# from model2 import LightResNetSE
from model_MLP import ChocoNet
# from model3 import create_model
import os
from sklearn.model_selection import train_test_split
import pandas as pd
import torchvision.transforms as T
import numpy as np

def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(pred, y_a, y_b, lam, criterion):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# 设置路径
# ##augmented
train_csv_path = '../../dataset_project_iapr2025/train_augmented.csv'
train_img_dir = '../../dataset_project_iapr2025/augmented_images'

# ##original
# train_csv_path = '../../dataset_project_iapr2025/train_.csv'
# train_img_dir = '../../dataset_project_iapr2025/train'
##划分验证集和训练集
val_img_dir = '../../dataset_project_iapr2025/validation/test_split_images'
val_csv_path = '../../dataset_project_iapr2025/validation/test_split.csv'

df_train= pd.read_csv(train_csv_path)
df_val = pd.read_csv(val_csv_path)
#train_df,val_df = train_test_split(df,test_size=0.1,random_state=42, shuffle=True)
train_transform = T.Compose([
    T.Resize((224, 224)),
    T.RandomHorizontalFlip(),
    #T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    T.RandomRotation(degrees=20),
    T.ToTensor()
])



dataset_train=ChocoDataset(df_train,train_img_dir,transform=train_transform)
dataset_val=ChocoDataset(df_val,val_img_dir)

train_loader=DataLoader(dataset_train,batch_size=8,shuffle=True)
val_loader=DataLoader(dataset_val,batch_size=8,shuffle=False)
# # 数据加载
# dataset = ChocoDataset(csv_file=csv_path, img_dir=img_dir)
# dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# 模型初始化
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
# model = ChocoNet(num_classes=13).to(device)
# model = LightResNet(num_classes=13).to(device)

model = ChocoNet(num_classes=13).to(device)


# 损失函数 & 优                                                                                                          化器
criterion = torch.nn.MSELoss()
optimizer = Adam(model.parameters(), lr=5e-4,weight_decay=5e-5)

# 训练过程
epochs =100
best_val_loss = float("inf")
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        images, labels_a, labels_b, lam = mixup_data(images, labels, alpha=0.4)
        preds = model(images)
        loss = mixup_criterion(preds, labels_a, labels_b, lam, criterion)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images)
            loss = criterion(preds, labels)
            val_loss += loss.item()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            #torch.save(model.state_dict(), 'weights/best_model.pth')
            #print(f"✅ Saved best model at Epoch {epoch + 1}")

    print(f"Epoch {epoch+1} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f}")

# 保存模型
os.makedirs('weights', exist_ok=True)
torch.save(model.state_dict(), 'weights/choco_model_MLP.pth')
