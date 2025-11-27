import os
import shutil
import random

# 当前脚本所在目录：.../YOLOPRACTICE/data_preprocess
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 项目根目录：.../YOLOPRACTICE
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# 原始 CODA val 数据集路径（保持不动）
VAL_ROOT = os.path.join(PROJECT_ROOT, "CODAdatasets", "CODA2022-val")

# 新建的 YOLO 训练版数据集路径（只复制，不修改原数据）
OUT_ROOT = os.path.join(PROJECT_ROOT, "CODAdatasets", "CODA_yolo_train")

IMG_DIR = os.path.join(VAL_ROOT, "images")
LBL_DIR = os.path.join(VAL_ROOT, "yolo_labels_uncat")  # 你的 yolo 标签目录

SPLIT_RATIO = 0.8  # 80% 训练，20% 验证


def main():
    # 创建输出目录结构
    for split in ["train", "val"]:
        os.makedirs(os.path.join(OUT_ROOT, "images", split), exist_ok=True)
        os.makedirs(os.path.join(OUT_ROOT, "labels", split), exist_ok=True)

    # 收集所有图片文件
    images = sorted([
        f for f in os.listdir(IMG_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    random.seed(42)
    random.shuffle(images)

    n_total = len(images)
    n_train = int(n_total * SPLIT_RATIO)

    train_imgs = images[:n_train]
    val_imgs = images[n_train:]

    print(f"Total images: {n_total}")
    print(f"Train: {len(train_imgs)}, Val: {len(val_imgs)}")

    def copy_split(img_list, split):
        for img_name in img_list:
            # 复制图片
            src_img = os.path.join(IMG_DIR, img_name)
            dst_img = os.path.join(OUT_ROOT, "images", split, img_name)
            shutil.copy(src_img, dst_img)

            # 复制标签（同名 .txt）
            txt_name = os.path.splitext(img_name)[0] + ".txt"
            src_lbl = os.path.join(LBL_DIR, txt_name)
            dst_lbl = os.path.join(OUT_ROOT, "labels", split, txt_name)

            if os.path.exists(src_lbl):
                shutil.copy(src_lbl, dst_lbl)
            else:
                print(f"⚠️ Warning: label missing: {txt_name}")

    copy_split(train_imgs, "train")
    copy_split(val_imgs, "val")

    print(f"\n✅ Done! New YOLO dataset at: {OUT_ROOT}")


if __name__ == "__main__":
    main()
