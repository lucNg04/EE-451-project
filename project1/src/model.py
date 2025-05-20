import torch
from torch import nn
from torchvision.models.detection.ssd import SSD, SSDHead, DefaultBoxGenerator
from collections import OrderedDict

class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class TinyBackboneWithExtras(nn.Module):
    def __init__(self):
        super().__init__()
        # 主干部分：3层卷积
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),  # 32x(H/2)x(W/2)
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), # 64x(H/4)x(W/4)
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),# 128x(H/8)x(W/8)
            nn.ReLU(inplace=True),
            SqueezeExcitation(128)
        )
        # extra部分
        self.extra = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(128, 128, 3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                SqueezeExcitation(128)
            ),
            nn.Sequential(
                nn.Conv2d(128, 64, 3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                SqueezeExcitation(64)
            ),
            nn.Sequential(
                nn.Conv2d(64, 32, 3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                SqueezeExcitation(32)
            ),
        ])
    def forward(self, x):
        features = OrderedDict()
        x = self.features(x)
        features["0"] = x
        for i, layer in enumerate(self.extra):
            x = layer(x)
            features[str(i + 1)] = x
        return features

def create_model(num_classes=14, image_size=(300, 300)):
    backbone = TinyBackboneWithExtras()
    out_channels = [128, 128, 64, 32]
    anchor_gen = DefaultBoxGenerator(
        aspect_ratios=[[0.7, 1.0, 1.5, 2.0]] * 4,
        scales=[0.1, 0.2, 0.375, 0.55, 0.725, 0.9]
    )
    model = SSD(
        backbone=backbone,
        anchor_generator=anchor_gen,
        size=image_size,
        num_classes=num_classes,
        head=SSDHead(
            in_channels=out_channels,
            num_anchors=anchor_gen.num_anchors_per_location(),
            num_classes=num_classes
        )
    )
    return model

# 参数量统计
if __name__ == "__main__":
    model = create_model()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"num_params: {total_params/1e6:.2f}M")