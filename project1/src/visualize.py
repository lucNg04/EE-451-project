import os
import cv2
import torch
import matplotlib.pyplot as plt

def visualize_prediction_vs_gt(image_path, label_path, preds, class_names, score_thresh=0.1):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]

    # === 读取 YOLO 格式 GT 标签 ===
    boxes_gt = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f.readlines():
                cls, cx, cy, bw, bh = map(float, line.strip().split())
                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)
                boxes_gt.append((x1, y1, x2, y2, int(cls)))

    # === 画 GT 红框 ===
    for (x1, y1, x2, y2, cls_id) in boxes_gt:
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(image, f"GT:{class_names[cls_id]}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # === 画预测框 蓝色 ===
    for box, cls, score in zip(preds["boxes"], preds["labels"], preds["scores"]):
        if score < score_thresh:
            continue
        x1, y1, x2, y2 = map(int, box.tolist())
        label_name = class_names[cls.item()]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 128, 255), 2)
        cv2.putText(image, f"P:{label_name} {score:.2f}", (x1, y2 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 128, 255), 1)

    # === 显示图像 ===
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.title("Prediction vs. Ground Truth")
    plt.axis("off")
    plt.show()
