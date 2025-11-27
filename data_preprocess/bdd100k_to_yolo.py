# convert the bdd100k label format to yolo format
'''
bdd100k/
├── 100k/
│   ├── train/
│   ├── val/
│   └── test/
│      
└── labels/
    └── 100k/
        ├── train/
        ├── val/
        └── test/
'''
import json 
from pathlib import Path
from PIL import Image
import glob

CATS = {"car":0,"bus":1,"person":2,"bike":3,"truck":4,"motor":5,"train":6,"rider":7,"traffic sign":8,"traffic light":9}
IMG_EXTS = (".jpg", ".jpeg", ".png")


def bdd_obj2yolo_line(obj, img_w, img_h):
    cat = obj.get("category")
    # print(cat)
    box = obj.get("box2d")
    if not box or not cat or cat not in CATS:
        return None
    x1,y1,x2, y2=box["x1"], box["y1"], box["x2"], box["y2"]
    cx, cy = (x1+x2)/img_w/2, (y1+y2)/img_h/2
    w, h = (x2-x1)/img_w, (y2-y1)/img_h
    return f"{CATS[cat]} {cx} {cy} {w} {h}"

def convert_one_json(json_path, img_w, img_h, out_txt_path):
    data = json.load(open(json_path, "r"))
    frames = data.get("frames", [])
    objs = frames[0].get("objects", []) if frames else []
    with open(out_txt_path, "w") as fo:
        for o in objs:
            line = bdd_obj2yolo_line(o, img_w, img_h)
            if line:
                fo.write(line+"\n")

def convert_folder(labels_dir: Path, images_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    json_files = sorted(glob.glob(str(labels_dir / "*.json")))
    for jf in json_files:
        jf = Path(jf)
        stem = jf.stem
        imgp = find_image_with_same_stem(images_dir, stem)
        if imgp is None:
            continue #无同名图片则跳过
        w, h = get_image_size(imgp)
        convert_one_json(jf, w, h, out_dir / f"{stem}.txt")

def find_image_with_same_stem(img_dir: Path, stem: str):
    for ext in IMG_EXTS:
        p = img_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None

def get_image_size(img_path: Path):
    with Image.open(img_path) as im:
        return im.size  # (w, h)
# step5_lists.py
def make_list(img_dir: Path, lab_dir: Path, out_txt: Path):
    IMG_EXTS = (".jpg",".jpeg",".png")
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    with open(out_txt, "w") as f:
        for p in sorted(img_dir.rglob("*")):
            if p.suffix.lower() in IMG_EXTS and (lab_dir / f"{p.stem}.txt").exists():
                f.write(str(p.resolve()) + "\n")

def main():
    ROOT = Path("bdd100k")
    splits = ["train", "val", "test"]
    for s in splits:
        labels_dir = ROOT / f"labels/100k/{s}"
        img_dir = ROOT / f"{'100k'}/{s}"
        out_dir = ROOT / f"yolo_labels/{s}"
        out_list = ROOT / f"yolo_labels/file_list/{s}.txt"
        print(f"\n=== Processing {s} ===")

        convert_folder(labels_dir, img_dir, out_dir)
        make_list(img_dir, out_dir, out_list)

if __name__ == "__main__":
        main()