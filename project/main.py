## YOUR CODE
## test
# generate_submission.py
import torch
import pandas as pd
from torch.utils.data import DataLoader
from src.model import ChocoNet
from src.dataset import ChocoDataset
import os



# 设置路径
test_img_dir = '../dataset_project_iapr2025/test/'
sample_csv = 'sample_submission.csv'  # 用于获取列名和 id
# test_img_dir = '../dataset_project_iapr2025/train/'
# sample_csv = '../dataset_project_iapr2025/train.csv'  # 用于获取列名和 id
model_path = 'src/weights/choco_model.pth'
#model_path='src/weights/best_model.pth'
output_csv = 'submission.csv'
#MLP CASE
# model_path='src/weights/choco_model_MLP.pth'
# output_csv='submission_MLP.csv'

# 读取测试 ID 列表（可以用 train.csv 的结构）
df = pd.read_csv(sample_csv)
test_ids = df['id']

# 使用原来的列名
column_names = df.columns.tolist()

# 创建只加载测试图像的数据集
class ChocoTestDataset(ChocoDataset):
    def __init__(self, csv_file, img_dir, transform=None):
        super().__init__(csv_file, img_dir, transform)
        self.labels = None  # 不读取 label

    def __getitem__(self, idx):
        img_id = self.image_ids.iloc[idx]
        img_path = os.path.join(self.img_dir, f"L{img_id}.JPG")
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, img_id

# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ChocoNet(num_classes=13).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# 加载数据
from PIL import Image
import torchvision.transforms as T

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor()
])

test_dataset = ChocoTestDataset(csv_file=sample_csv, img_dir=test_img_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# 预测
predictions = []
image_ids = []

with torch.no_grad():
    for images, ids in test_loader:
        images = images.to(device)
        outputs = model(images)
        outputs = outputs.cpu().numpy()
        outputs = outputs.round().astype(int)  # 四舍五入为整数
        predictions.extend(outputs)
        image_ids.extend(ids)

# 构造 DataFrame 保存
output_df = pd.DataFrame(predictions, columns=column_names[1:])
output_df.insert(0, 'id', image_ids)
output_df.to_csv(output_csv, index=False)

print(f"✅ Submission saved to {output_csv}")
