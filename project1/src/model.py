# 0.71 baseline-improve?
import torch
from torch import nn
from torchvision.models.detection.ssd import SSD, SSDHead, DefaultBoxGenerator
from torchvision.models import mobilenet_v2
from collections import OrderedDict

class MobileNetV2BackboneWithExtras(nn.Module):
    def __init__(self):
        super().__init__()
        base = mobilenet_v2(weights=None)
        # 只取features部分
        self.features = nn.Sequential(*list(base.features))
        # MobileNetV2最后一层输出1280通道

        # 增强的特征提取层，添加更多卷积层来提取形状特征
        self.extra = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1280, 512, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),  # 添加额外的卷积层
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),  # 添加额外的卷积层
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),  # 添加额外的卷积层
                nn.ReLU(inplace=True)
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

def create_model(num_classes=14, image_size=(900, 600)):
    backbone = MobileNetV2BackboneWithExtras()
    out_channels = [1280, 512, 256, 256]  # 对应每个特征层的通道数

    anchor_gen = DefaultBoxGenerator(
        aspect_ratios=[[0.5, 0.75, 1.0, 1.25, 1.5, 2.0]] * 4,  # 增加更多的宽高比
        scales=[0.1, 0.15, 0.2, 0.3, 0.4, 0.6]  # 增加更多的尺度
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