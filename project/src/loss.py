# yolo_loss.py - 简化版 YOLO Loss，用于训练小型 YOLOTiny 网络（每图支持多个对象）
import torch
import torch.nn as nn


class YOLOLoss(nn.Module):
    def __init__(self, S=13, B=3, num_classes=13, lambda_coord=5, lambda_noobj=0.5):
        super().__init__()
        self.S = S  # 网格大小
        self.B = B  # anchor 数（目前只用一个）
        self.C = num_classes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.mse = nn.MSELoss(reduction='sum')

    def forward(self, predictions, targets):
        # predictions: (B, B*(5+C), S, S)
        # targets: (B, N, 5) 每个物体：[class, cx, cy, w, h]，已归一化

        BATCH_SIZE = predictions.size(0)
        predictions = predictions.permute(0, 2, 3, 1).contiguous()  # (B, S, S, B*(5+C))
        predictions = predictions.view(BATCH_SIZE, self.S, self.S, self.B, 5 + self.C)

        obj_mask = torch.zeros_like(predictions[..., 0])
        noobj_mask = torch.ones_like(predictions[..., 0])
        loss = 0.0

        for b in range(BATCH_SIZE):
            for t in targets[b]:
                cls, cx, cy, w, h = t
                gx = int(cx.item() * self.S)
                gy = int(cy.item() * self.S)

                if gx >= self.S or gy >= self.S:
                    continue  # 忽略越界

                pred_box = predictions[b, gy, gx, 0]  # anchor 0
                pred_cls = pred_box[5:]
                true_cls = torch.zeros(self.C, device=pred_cls.device)
                true_cls[int(cls.item())] = 1.0

                loss += self.lambda_coord * self.mse(pred_box[1], cx)
                loss += self.lambda_coord * self.mse(pred_box[2], cy)
                loss += self.lambda_coord * self.mse(pred_box[3], w)
                loss += self.lambda_coord * self.mse(pred_box[4], h)
                loss += self.mse(pred_cls, true_cls)

                obj_mask[b, gy, gx, 0] = 1.0
                noobj_mask[b, gy, gx, 0] = 0.0

        pred_conf = predictions[..., 0]
        loss += self.lambda_noobj * self.mse(pred_conf * noobj_mask, torch.zeros_like(pred_conf))

        return loss / BATCH_SIZE