from torch import nn
from torchvision.models.detection.ssd import SSD
from torchvision.models.detection.ssd import SSDHead, DefaultBoxGenerator
from torchvision.ops import SqueezeExcitation
import torchvision

class ChocoNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 初始卷积层
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            self._make_ds_block(64, 128, stride=1),
            self._make_ds_block(128, 256, stride=2),
            self._make_ds_block(256, 512, stride=2, use_se=True),  
            self._make_ds_block(512, 512, stride=2, use_se=True)
        )
        
        # 额外特征层
        self.extra_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
                SqueezeExcitation(512, 16)  
            ),
            nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
            ),
            nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
            ),
            nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
            )
        ])

    def _make_ds_block(self, in_ch, out_ch, stride, use_se=False):
        """深度可分离卷积块"""
        layers = [
            nn.Conv2d(in_ch, in_ch, 3, stride, 1, groups=in_ch),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(),
            nn.Conv2d(in_ch, out_ch*4, 1),  # 更大的扩展比
            nn.BatchNorm2d(out_ch*4),
            nn.ReLU(),
            nn.Conv2d(out_ch*4, out_ch, 1),
            nn.BatchNorm2d(out_ch)
        ]
        
        if use_se:
            layers.append(SqueezeExcitation(out_ch, 16))
            
        layers.append(nn.ReLU())
        
        return nn.Sequential(*layers)

    def forward(self, x):
        features = {"0": self.features(x)}
        
        for i, layer in enumerate(self.extra_layers, 1):
            features[str(i)] = layer(features[str(i-1)])
        
        return features

def create_model(num_classes=13):
    """创建兼容的SSD模型"""
    anchor_generator = DefaultBoxGenerator(
        aspect_ratios=[[0.7, 1.0, 1.5, 2.0]] * 5,  # 需要与特征层数量匹配
        scales=[0.1, 0.2, 0.375, 0.55, 0.725, 0.9],
    )
    
    # 特征通道数（必须与主干输出维度严格对应）
    backbone = ChocoNet()
    out_channels = [512, 512, 512, 256, 256]  # 对应5个特征层
    
    model = SSD(
        backbone=backbone,
        num_classes=num_classes,
        anchor_generator=anchor_generator,
        size=(2100,1400),
        head=SSDHead(
            in_channels=out_channels,
            num_anchors=anchor_generator.num_anchors_per_location(),
            num_classes=num_classes
        ),
        score_thresh=0.01,
        nms_thresh=0.45,
        detections_per_img=200 
    )
    return model

if __name__ == "__main__":
    num_classes = 13 
    model = create_model(num_classes)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"num_params: {total_params/1e6:.2f}M")