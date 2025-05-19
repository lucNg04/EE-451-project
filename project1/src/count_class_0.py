import os
from collections import Counter

label_dir = "../data/obj_train_data"  # 替换成你的路径
counts = Counter()

for file in os.listdir(label_dir):
    if file.endswith(".txt"):
        with open(os.path.join(label_dir, file), "r") as f:
            for line in f:
                if line.strip() == "":
                    continue
                cls_id = int(line.strip().split()[0])
                counts[cls_id] += 1

print("Class counts in training data:")
for cls_id in range(13):  # 0~12
    print(f"Class {cls_id}: {counts.get(cls_id, 0)}")
