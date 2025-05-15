# ==============================
# src/train_classification.py
# ==============================
from classification_dataset import ClassificationDataset
from classification_model import get_classification_model
from torchvision import transforms
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def train_classification():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    dataset = ClassificationDataset("dataset_project_iapr2025/train", "dataset_project_iapr2025/train.csv", transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = get_classification_model(num_labels=13)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    for epoch in range(10):
        model.train()
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1} done")

    torch.save(model.state_dict(), "weights/classification_model.pth")