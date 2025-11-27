import json
from copy import deepcopy
from pathlib import Path

SRC_GT = Path("/Users/yxr/Desktop/AI7102/YOLOpractice/CODAdatasets/CODA2022-test/annotations.json")
BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

OUT_COMMON  = DATA_DIR/ "gt_common.json"
OUT_AGNOSTIC = DATA_DIR/ "gt_agnostic.json"
OUT_CORNER  = DATA_DIR / "gt_agnostic_corner.json"
def add_area_iscrowd(ann):
    ann = deepcopy(ann)
    if "iscrowd" not in ann:
        ann["iscrowd"] = 0
    if "area" not in ann:
        x, y, w, h = ann["bbox"]
        ann["area"] = float(w) * float(h)
    return ann


def main():
    with open(SRC_GT, "r", encoding="utf-8") as f:
        gt = json.load(f)

    images = gt["images"]
    categories = gt["categories"]
    annotations = gt["annotations"]

    # ------------ 1) common：1~7 保留，其它全部映射成 8 --------------
    common_cat_ids = set(range(1, 8))  # 我们希望有 1~7

    # categories：保留 id 1~8（如果原来就有 cornerCase=8 就会被保留）
    common_cats = [c for c in categories if c.get("id") in common_cat_ids]

    common_anns = []
    for ann in annotations:
        new_ann = add_area_iscrowd(ann)
        cid = new_ann["category_id"]
        if 1 <= cid <= 7:
            # common 类：保持原 cid
            pass
        else:
            # 其它所有类 → 统一改成 cornerCase = 8
            new_ann["category_id"] = 8
        common_anns.append(new_ann)

    common_gt = {
        "images": images,
        "categories": common_cats,
        "annotations": common_anns,
    }

    with open(OUT_COMMON, "w", encoding="utf-8") as f:
        json.dump(common_gt, f)
    print(f"[COMMON] write {OUT_COMMON}, anns = {len(common_anns)}")

    # ------------ 2) agnostic：所有类并成 1（all_class） --------------
    agnostic_cats = [{
        "supercategory": "all",
        "id": 1,
        "name": "all_class",
    }]

    agnostic_anns = []
    for ann in annotations:
        new_ann = add_area_iscrowd(ann)
        new_ann["category_id"] = 1
        agnostic_anns.append(new_ann)

    agnostic_gt = {
        "images": images,
        "categories": agnostic_cats,
        "annotations": agnostic_anns,
    }

    with open(OUT_AGNOSTIC, "w", encoding="utf-8") as f:
        json.dump(agnostic_gt, f)
    print(f"[AGNOSTIC] write {OUT_AGNOSTIC}, anns = {len(agnostic_anns)}")

    # ------------ 3) corner：在 agnostic 的基础上，只保留 corner_case=True --------------
    corner_anns = [a for a in agnostic_anns if a.get("corner_case", False)]

    corner_gt = {
        "images": images,
        "categories": agnostic_cats,
        "annotations": corner_anns,
    }

    with open(OUT_CORNER, "w", encoding="utf-8") as f:
        json.dump(corner_gt, f)
    print(f"[CORNER] write {OUT_CORNER}, anns = {len(corner_anns)}")


if __name__ == "__main__":
    main()
