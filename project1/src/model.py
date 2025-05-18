import torch
from torch import nn
from torchvision.models.detection.ssd import SSD, SSDHead, DefaultBoxGenerator

class SimpleBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            self._conv_block(3, 32, stride=2),
            self._conv_block(32, 64, stride=1),
            self._ds_block(64, 128),
            self._ds_block(128, 256),
            self._ds_block(256, 512)
        )

        self.extra = nn.ModuleList([
            self._extra_block(512, 256, 512),
            self._extra_block(512, 128, 256),
            self._extra_block(256, 128, 256),
            self._extra_block(256, 128, 256)
        ])

    def _conv_block(self, in_c, out_c, stride):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_c),
            nn.SiLU()
        )

    def _ds_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, in_c, kernel_size=3, stride=2, padding=1, groups=in_c),
            nn.Conv2d(in_c, out_c, kernel_size=1),
            nn.BatchNorm2d(out_c),
            nn.SiLU()
        )

    def _extra_block(self, in_c, mid_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, mid_c, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(mid_c, out_c, kernel_size=3, stride=2, padding=1),
            nn.SiLU()
        )

    def forward(self, x):
        features = {}
        x = self.layers(x)
        features["0"] = x
        for i, layer in enumerate(self.extra):
            x = layer(x)
            features[str(i + 1)] = x
        return features

def create_model(num_classes=13, image_size=(640, 640)):
    backbone = SimpleBackbone()
    out_channels = [512, 512, 256, 256, 256]

    anchor_gen = DefaultBoxGenerator(
        aspect_ratios=[[0.5, 1.0, 2.0]] * 5,
        scales=[0.1, 0.2, 0.35, 0.5, 0.7, 0.9]
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

if __name__ == "__main__":
    model = create_model(num_classes=13)
    total = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total / 1e6:.2f}M")
