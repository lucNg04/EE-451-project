import os

label_dir = "../validation/obj_train_data/"
for file in os.listdir(label_dir):
    if file.endswith(".txt"):
        path = os.path.join(label_dir, file)
        lines = []
        with open(path, "r") as f:
            for line in f:
                parts = line.strip().split()
                parts[0] = str(int(parts[0]) + 1)  # 类别索引偏移
                lines.append(" ".join(parts))
        with open(path, "w") as f:
            f.write("\n".join(lines))
