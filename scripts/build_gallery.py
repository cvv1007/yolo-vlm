import os
import json
from PIL import Image
import csv

# 路径配置
ANNOT_PATH = "/Users/yxr/Desktop/AI7102/YOLOpractice/CODAdatasets/CODA2022-val/annotations.json"
IMG_DIR = "/Users/yxr/Desktop/AI7102/YOLOpractice/CODAdatasets/CODA2022-val/images"
OUT_DIR = "data_gallery/CODA2022-val/gallery"
META_CSV = "data_gallery/CODA2022-val/gallery_meta.csv"

# 读取数据
with open(ANNOT_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

images = {img["id"]: img for img in data["images"]}
categories = {c["id"]: c["name"] for c in data["categories"]}

# 创建类别文件夹（命名格式 cid_cname）
os.makedirs(OUT_DIR, exist_ok=True)
for cid, cname in categories.items():
    folder_name = f"{cid}_{cname}"
    os.makedirs(os.path.join(OUT_DIR, folder_name), exist_ok=True)

# 创建元数据 CSV（含 cid）
with open(META_CSV, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["crop_path", "class_name", "cid", "image_file", "bbox_x", "bbox_y", "bbox_w", "bbox_h"])

    # 遍历每个标注并裁剪
    for ann in data["annotations"]:
        img_info = images.get(ann["image_id"])
        if not img_info:
            continue
        img_path = os.path.join(IMG_DIR, img_info["file_name"])
        if not os.path.exists(img_path):
            continue

        img = Image.open(img_path).convert("RGB")
        x, y, w, h = ann["bbox"]
        x2 = min(img.width, x + w)
        y2 = min(img.height, y + h)
        crop = img.crop((x, y, x2, y2))

        cid = ann["category_id"]
        cname = categories.get(cid, f"cid_{cid}")
        folder_name = f"{cid}_{cname}"
        class_dir = os.path.join(OUT_DIR, folder_name)
        os.makedirs(class_dir, exist_ok=True)

        crop_name = f"{os.path.splitext(img_info['file_name'])[0]}_ann{ann['id']}.jpg"
        crop_path = os.path.join(class_dir, crop_name)
        crop.save(crop_path)

        writer.writerow([crop_path, cname, cid, img_info["file_name"], x, y, w, h])

print(f"✅ 完成！图片裁剪保存在 {OUT_DIR}")
print(f"🧾 元数据文件：{META_CSV}")

# 遍历每个类别文件夹并计数
def iter_count(GALLERY_DIR):
    os.listdir(GALLERY_DIR)
    for folder in sorted(os.listdir(GALLERY_DIR)):
        class_dir = os.path.join(GALLERY_DIR, folder)
        if os.path.isdir(class_dir):
            num_imgs = len([f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
            print(f"{folder}: {num_imgs} 张图片")
'''
total： 40835
29 categories in CODA val in total
10_motorcycle: 18 张图片
11_stroller: 24 张图片
13_cart: 85 张图片
15_construction_vehicle: 2785 张图片
17_dog: 270 张图片
19_barrier: 1477 张图片
1_pedestrian: 5343 张图片
20_bollard: 1822 张图片
22_sentry_box: 12 张图片
24_traffic_cone: 4985 张图片
25_traffic_island: 29 张图片
26_traffic_light: 72 张图片
27_traffic_sign: 445 张图片
28_debris: 94 张图片
29_suitcace: 21 张图片
2_cyclist: 2285 张图片
30_dustbin: 161 张图片
31_concrete_block: 98 张图片
32_machinery: 13 张图片
38_garbage: 68 张图片
3_car: 15470 张图片
40_plastic_bag: 32 张图片
41_stone: 88 张图片
43_misc: 787 张图片
4_truck: 2455 张图片
6_tricycle: 652 张图片
7_bus: 730 张图片
8_bicycle: 122 张图片
9_moped: 392 张图片'''