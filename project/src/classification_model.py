# ==============================
# src/classification_model.py
# ==============================
from torchvision import models
import torch.nn as nn

def get_classification_model(num_labels):
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, num_labels),
        nn.Sigmoid()
    )
    return model