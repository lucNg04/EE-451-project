import os
import torch
import pandas as pd
from torchvision.transforms import functional as F
from PIL import Image
from src.model import create_model

# # 配置
# TEST_DIR = '../dataset_project_iapr2025/test/'
# MODEL_PATH = "src/model_epoch_30.pth"
# NUM_CLASSES = 13
# IMAGE_SIZE = (640, 640)
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# 配置路径
TEST_DIR = r'../dataset_project_iapr2025/test/'
MODEL_PATH = "src/model_epoch_30.pth"
NAMES_PATH = "data/obj.names"
SAMPLE_CSV_PATH = "../dataset_project_iapr2025/sample_submission.csv"
OUTPUT_CSV_PATH = "submission.csv"
DEVICE = torch.device("cpu")


# 加载类别名称（class_id 对应顺序）
with open(NAMES_PATH, 'r', encoding='utf-8') as f:
    class_names = [line.strip() for line in f.readlines()]
class_names.insert(0, '__background__')
NUM_CLASSES = len(class_names)

# 加载 sample_submission.csv
sample_df = pd.read_csv(SAMPLE_CSV_PATH, encoding='utf-8-sig')
column_names = sample_df.columns.tolist()  # ['id', 'Jelly Wh.', ..., 'Stracciatella']
first_col_name = column_names[0]  # 'id'

# 加载模型
model = create_model(num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# 遍历每张图片
results = []
for img_id in sample_df[first_col_name]:
    filename = f"{img_id}.jpg"
    image_path = os.path.join(TEST_DIR, f"L{img_id}.jpg")

    if not os.path.exists(image_path):
        print(f"⚠️ Missing: {image_path}")
        row = [img_id] + [0] * NUM_CLASSES
        results.append(row)
        continue

    image = Image.open(image_path).convert("RGB")
    image = F.resize(image, (640, 640))
    image_tensor = F.to_tensor(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        preds = model(image_tensor)[0]  # dict with 'boxes', 'labels', 'scores'
    print(f"{filename} → Raw predicted labels:", preds["labels"].tolist())
    print(f"{filename} → Raw predicted scores:", preds["scores"].tolist())

    # 初始化统计字典
    count_dict = {name: 0 for name in column_names[1:]}  # 与 CSV 保持顺序
    for label, score in zip(preds["labels"], preds["scores"]):
        if score >= 0.5 :
            class_id = label.item()
            if 0 <= class_id < NUM_CLASSES:
                name = class_names[class_id]
                if name in count_dict:
                    count_dict[name] += 1
                else:
                    print(f"⚠️ 未匹配类别: {name}（请确保 .names 和 CSV 列一致）")


    # 按列顺序写入行
    row = [img_id] + [count_dict[col] for col in column_names[1:]]
    results.append(row)
# # 设置目标图像 ID
# img_id = "1000758"
# img_path = os.path.join(TEST_DIR, f"L{img_id}.jpg")
# label_path = os.path.join("dataset_project_iapr2025/data/obj_train_data/", f"L{img_id}.txt")
#
# image = Image.open(img_path).convert("RGB")
# image = F.resize(image, (640, 640))
# image_tensor = F.to_tensor(image).unsqueeze(0).to(DEVICE)
#
# with torch.no_grad():
#     preds = model(image_tensor)[0]
#
# from src.visualize import visualize_prediction_vs_gt
# visualize_prediction_vs_gt(img_path, label_path, preds, class_names, score_thresh=0.1)


# 写入 CSV
output_df = pd.DataFrame(results, columns=column_names)
output_df.to_csv(OUTPUT_CSV_PATH, index=False)
print(f"✅ Done. Saved to {OUTPUT_CSV_PATH}")
