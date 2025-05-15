# ==============================
# src/predict_pipeline.py
# ==============================
from PIL import Image
import torch
import pandas as pd
from torchvision.transforms import functional as F, transforms
from detection_model import get_detection_model
from classification_model import get_classification_model
import os

def run_pipeline():
    model_det = get_detection_model(num_classes=5)
    model_det.load_state_dict(torch.load("weights/detection_model.pth"))
    model_det.eval().to("cpu")

    model_cls = get_classification_model(num_labels=5)
    model_cls.load_state_dict(torch.load("weights/classification_model.pth"))
    model_cls.eval().to("cpu")

    cls_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    results = []
    for img_name in os.listdir("dataset_project_iapr2025/test"):
        if not img_name.endswith(('.jpg', '.png')): continue
        image = Image.open(os.path.join("data/test_images", img_name)).convert("RGB")
        image_tensor = F.to_tensor(image).unsqueeze(0)
        outputs = model_det(image_tensor)
        boxes = outputs[0]['boxes']

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.int().tolist()
            crop = image.crop((x1, y1, x2, y2))
            crop = cls_transform(crop).unsqueeze(0)
            preds = model_cls(crop).squeeze()

            result = {"filename": img_name, "box_id": i}
            for j in range(len(preds)):
                result[f"class_{j}"] = preds[j].item()
            results.append(result)

    df = pd.DataFrame(results)
    df.to_csv("output.csv", index=False)