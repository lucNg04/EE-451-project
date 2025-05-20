import pandas as pd
import numpy as np
SUB_PATH='submission.csv'
REF_PATH='kaggle.csv'
def compute_f1_score(submission_path: str, reference_path: str):
    # 读取两个文件
    pred_df = pd.read_csv(submission_path)
    gt_df = pd.read_csv(reference_path)

    # 按照 ID 排序，并对齐列顺序
    pred_df = pred_df.sort_values("id").reset_index(drop=True)
    gt_df = gt_df.sort_values("id").reset_index(drop=True)

    # 对齐列名（确保预测列顺序和参考列顺序一致）
    cols = [col for col in gt_df.columns if col != "id"]
    pred_df = pred_df[["id"] + cols]

    assert all(pred_df["id"] == gt_df["id"]), "Mismatch in image IDs!"

    f1_total = 0
    N = len(pred_df)

    for i in range(N):
        y_true = gt_df.loc[i, cols].values.astype(int)
        y_pred = pred_df.loc[i, cols].values.astype(int)

        TP = np.sum(np.minimum(y_true, y_pred))
        FPN = np.sum(np.abs(y_true - y_pred))

        f1_i = (2 * TP) / (2 * TP + FPN) if (2 * TP + FPN) > 0 else 0
        f1_total += f1_i

    final_f1 = f1_total / N
    print(f"\n✅ Image-wise Average F1 Score: {final_f1:.4f}")
    return final_f1

# # 示例用法
# if __name__ == "__main__":
#     import argparse
#
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--submission", type=str, required=True, help="Path to your submission.csv file")
#     parser.add_argument("--reference", type=str, required=True, help="Path to the reference (kaggle ground truth) CSV")
#     args = parser.parse_args()
#
#     compute_f1_score(args.submission, args.reference)
compute_f1_score(SUB_PATH,REF_PATH)