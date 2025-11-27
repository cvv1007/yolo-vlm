#!/usr/bin/env python
import sys
import os
import os.path
import copy
import json
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def load_coco_with_sanity(gt_path, preds, desc: str):
    """
    读取 GT，并保证：
      - dataset 里有 info 字段
      - annotations 存在（否则说明 GT 有问题）
    然后用 preds 调用 loadRes，返回 (cocoGt, cocoDt)。
    如果出错，打印信息并返回 (None, None)。
    """
    print(f"\n[{desc}] Loading GT from: {gt_path}")
    if not gt_path.exists():
        print(f"[ERROR] GT file not found: {gt_path}")
        return None, None

    cocoGt = COCO(str(gt_path))

    # 补 info 字段，避免 KeyError: 'info'
    if "info" not in cocoGt.dataset:
        cocoGt.dataset["info"] = {}

    if "annotations" not in cocoGt.dataset:
        print("[ERROR] GT json has no 'annotations' field.")
        return None, None

    print(f"[{desc}] Evaluating bbox!")
    try:
        cocoDt = cocoGt.loadRes(preds)
    except Exception as e:
        print(f"[ERROR] loadRes failed in {desc}: {e}")
        return None, None

    return cocoGt, cocoDt


def main():
    # ----------------- 参数检查 -----------------
    if len(sys.argv) != 3:
        print("Usage: python evaluate/evaluate.py <input_dir> <output_dir>")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    # submit_dir: 预测结果目录（你的 json 分片）
    submit_dir = input_dir

    # truth_dir: GT 目录，这里假设在 evaluate/data 下
    script_dir = Path(__file__).resolve().parent
    truth_dir = script_dir / "data"

    if not os.path.isdir(submit_dir):
        print(f"[ERROR] submit_dir '{submit_dir}' doesn't exist")
        return

    if not truth_dir.is_dir():
        print(f"[ERROR] truth_dir '{truth_dir}' doesn't exist")
        return

    os.makedirs(output_dir, exist_ok=True)

    results_shown = {}

    # ----------------- 1) 读取所有预测 json -----------------
    submission_list = sorted(os.listdir(submit_dir))
    sub_all = []
    for submission in submission_list:
        if submission.endswith(".json"):
            path = os.path.join(submit_dir, submission)
            with open(path, "r", encoding="utf-8") as f:
                try:
                    sub_part = json.load(f)
                except json.JSONDecodeError as e:
                    print(f"[WARN] skip invalid json {path}: {e}")
                    continue
            if isinstance(sub_part, list):
                sub_all += sub_part
            else:
                print(f"[WARN] {path} is not a list, skip.")

    if len(sub_all) == 0:
        print("[ERROR] No annotations (predictions) found in input_dir!")
        return

    print(f"[INFO] The submission results contain total {len(sub_all)} detections.")

    # ----------------- 2) AP-common -----------------
    print("\n========== Computing AP-common ==========")
    sub_common = [pred for pred in sub_all
                  if 1 <= int(pred.get("category_id", 0)) <= 7]

    if not sub_common:
        print("[ERROR] No predictions for common categories (1~7).")
        return

    gt_common_path = truth_dir / "gt_common.json"  # 你已有的文件名
    cocoGt, cocoDt = load_coco_with_sanity(gt_common_path, sub_common, "AP-common")
    if cocoGt is None:
        return

    img_ids = cocoGt.getImgIds()
    cocoEval = COCOeval(cocoGt, cocoDt, iouType="bbox")
    cocoEval.params.imgIds = img_ids
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    results_shown["AP-common"] = float(cocoEval.stats[0])
    print(f"[RESULT] AP-common: {results_shown['AP-common']:.4f}")

    # ----------------- 3) AP-agnostic & AR-agnostic -----------------
    print("\n========== Computing AP-agnostic & AR-agnostic ==========")
    sub_agnostic = []
    for pred in sub_all:
        p = copy.deepcopy(pred)
        p["category_id"] = 1  # 全部归为一个类
        sub_agnostic.append(p)

    gt_agnostic_path = truth_dir / "gt_agnostic.json"
    cocoGt, cocoDt = load_coco_with_sanity(
        gt_agnostic_path, sub_agnostic, "AP/AR-agnostic"
    )
    if cocoGt is None:
        return

    img_ids = cocoGt.getImgIds()
    cocoEval = COCOeval(cocoGt, cocoDt, iouType="bbox")
    cocoEval.params.imgIds = img_ids
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    results_shown["AP-agnostic"] = float(cocoEval.stats[0])
    results_shown["AR-agnostic"] = float(cocoEval.stats[8])

    print(f"[RESULT] AP-agnostic: {results_shown['AP-agnostic']:.4f}")
    print(f"[RESULT] AR-agnostic: {results_shown['AR-agnostic']:.4f}")

    # ----------------- 4) AR-agnostic-corner -----------------
    print("\n========== Computing AR-agnostic-corner ==========")
    gt_corner_path = truth_dir / "gt_agnostic_corner.json"
    cocoGt, cocoDt = load_coco_with_sanity(
        gt_corner_path, sub_agnostic, "AR-agnostic-corner"
    )
    if cocoGt is None:
        return

    img_ids = cocoGt.getImgIds()
    cocoEval = COCOeval(cocoGt, cocoDt, iouType="bbox")
    cocoEval.params.imgIds = img_ids
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    results_shown["AR-agnostic-corner"] = float(cocoEval.stats[8])
    print(f"[RESULT] AR-agnostic-corner: {results_shown['AR-agnostic-corner']:.4f}")

    # ----------------- 5) 总分 Sum -----------------
    results_shown["Sum"] = sum(results_shown.values())
    print("\n========== Final Score ==========")
    print(f"[RESULT] Sum: {results_shown['Sum']:.4f}")

    # ----------------- 6) 写 scores.txt -----------------
    out_path = os.path.join(output_dir, "scores.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"Sum: {results_shown['Sum']}\n")
        for k, v in results_shown.items():
            if k == "Sum":
                continue
            f.write(f"{k}: {v}\n")

    print(f"\n[INFO] Scores written to: {out_path}")
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
