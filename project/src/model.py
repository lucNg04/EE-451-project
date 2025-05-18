# model.py
import torch.nn as nn
import torchvision.models as models

class ChocoNet(nn.Module):
    def __init__(self, num_classes=13):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        self.backbone.fc = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
            nn.ReLU()  # 保证非负输出
        )

    def forward(self, x):
        return self.backbone(x)