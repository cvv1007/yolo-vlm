import json, os
from pathlib import Path
from collections import defaultdict

# ==== 修改这里 ====
COCO_JSON = "/Users/yxr/Desktop/AI7102/YOLOpractice/datasets/CODA2022-val/annotations.json"
SAVE_DIR = "/Users/yxr/Desktop/AI7102/YOLOpractice/datasets/CODA2022-val/yolo_labels"
SINGLE_CLASS = False   # ← 改成 False 就保留原类别
# =================

# 读取 COCO
with open(COCO_JSON, "r", encoding="utf-8") as f:
    coco = json.load(f)

imgs = {i["id"]: i for i in coco["images"]}
cats = {c["id"]: c["name"] for c in coco["categories"]}
anns = defaultdict(list)
for a in coco["annotations"]:
    anns[a["image_id"]].append(a)

# 类别映射
if SINGLE_CLASS:
    cat2yolo = {cid: 0 for cid in cats}
else:
    cat2yolo = {cid: i for i, cid in enumerate(sorted(cats))}

# 转换
for img_id, info in imgs.items():
    w, h = info["width"], info["height"]
    label_path = Path(SAVE_DIR) / Path(info["file_name"]).with_suffix(".txt")
    label_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    for a in anns[img_id]:
        x, y, bw, bh = a["bbox"]
        cx, cy = x + bw / 2, y + bh / 2
        lines.append(f"{cat2yolo[a['category_id']]} {cx/w:.6f} {cy/h:.6f} {bw/w:.6f} {bh/h:.6f}")

    with open(label_path, "w") as f:
        f.write("\n".join(lines))
print("✅ Done! YOLO labels saved to", SAVE_DIR)
