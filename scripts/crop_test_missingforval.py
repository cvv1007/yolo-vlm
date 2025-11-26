# scripts/crop_missing_from_test.py
import os, json
from PIL import Image
from pathlib import Path

# === 路径配置 ===
TEST_ANN = "/Users/yxr/Desktop/AI7102/YOLOpractice/CODAdatasets/CODA2022-test/annotations.json"
IMG_DIR = "/Users/yxr/Desktop/AI7102/YOLOpractice/CODAdatasets/CODA2022-test/images"
OUT_DIR = "data_gallery/CODA2022-test/missing_classes"

# val 中缺失的类别 ID
MISSING = {5,12,14,16,18,21,23,33,34,35,36,37,39,42}

os.makedirs(OUT_DIR, exist_ok=True)

with open(TEST_ANN, "r", encoding="utf-8") as f:
    data = json.load(f)

images = {im["id"]: im for im in data["images"]}
cats = {c["id"]: c["name"] for c in data["categories"]}

count = {cid: 0 for cid in MISSING}

for ann in data["annotations"]:
    cid = ann["category_id"]
    if cid not in MISSING:
        continue
    img_info = images.get(ann["image_id"])
    if not img_info:
        continue

    img_path = os.path.join(IMG_DIR, img_info["file_name"])
    if not os.path.exists(img_path):
        continue

    img = Image.open(img_path).convert("RGB")
    x, y, w, h = ann["bbox"]
    x1, y1 = max(0, int(x)), max(0, int(y))
    x2, y2 = min(img.width, int(x + w)), min(img.height, int(y + h))
    if x2 <= x1 or y2 <= y1:
        continue

    crop = img.crop((x1, y1, x2, y2))
    cname = cats.get(cid, f"cid_{cid}")
    save_dir = os.path.join(OUT_DIR, f"{cid}_{cname}")
    os.makedirs(save_dir, exist_ok=True)

    stem = Path(img_info["file_name"]).stem
    save_path = os.path.join(save_dir, f"{stem}_ann{ann['id']}.jpg")
    crop.save(save_path, quality=95)
    count[cid] += 1

print("✅ 完成所有缺失类裁剪：")
for cid in sorted(count):
    print(f"  {cid}_{cats.get(cid, f'cid_{cid}')}: {count[cid]} 张")
print(f"📁 保存目录: {OUT_DIR}")
