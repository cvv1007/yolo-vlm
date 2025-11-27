#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
evaluate_corner_cat.py

只用 gt_corner_cat.json 来评估 corner case：

评估内容：
1. 有没有框到 GT（检测召回）
2. 框到了以后，预测类别的 supercategory 是否和 GT 的 supercategory 一致（大类是否正确）
3. 每个 supercategory 单独算一份表现：总 GT 数、框到多少、分类大类对多少

用法：
    python evaluate_corner_cat.py <pred_dir> <gt_corner_cat.json> <output_dir>
"""
# 把骑行者和pedestrain合起来

import json
from pathlib import Path
from collections import defaultdict
import sys


# -----------------------------
# IoU 函数（判断预测与 GT 是否匹配）
# -----------------------------
def iou_xywh(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    iw = max(0, inter_x2 - inter_x1)
    ih = max(0, inter_y2 - inter_y1)
    inter = iw * ih

    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0


# -----------------------------
# 主评估函数（核心逻辑）
# -----------------------------
def eval_corner(gt_path: Path, preds: list, iou_thresh=0.5):

    gt = json.load(open(gt_path, "r"))
    gt_cats = {c["id"]: c for c in gt["categories"]}

    # CID → supercategory（大类）
    cid2super = {cid: c.get("supercategory") for cid, c in gt_cats.items()}

    # 按 image_id 把 GT 和 Pred 聚到一起
    img2gt = defaultdict(list)
    for ann in gt["annotations"]:
        img2gt[ann["image_id"]].append(ann)

    img2pred = defaultdict(list)
    for p in preds:
        img2pred[int(p["image_id"])].append(p)

    # 统计量
    total_gt = len(gt["annotations"])   # corner GT 总数
    matched_gt = 0                      # 被任何预测框框到
    correct_super = 0                   # supercategory 对的数量

    # 分 supercategory 的统计
    per_total = defaultdict(int)
    per_matched = defaultdict(int)
    per_correct = defaultdict(int)

    # 遍历所有 GT（每个 corner 框）
    for img_id, gt_list in img2gt.items():
        preds_here = img2pred.get(img_id, [])

        for g in gt_list:
            gt_box = g["bbox"]
            gt_cid = g["category_id"]
            gt_super = cid2super.get(gt_cid)

            per_total[gt_super] += 1   # 统计这个 supercategory 的 GT 数

            # 找到 IoU 最大的预测框
            best_pred, best_iou = None, 0
            for p in preds_here:
                iou = iou_xywh(gt_box, p["bbox"])
                if iou >= iou_thresh and iou > best_iou:
                    best_iou = iou
                    best_pred = p

            if not best_pred:
                # 这个 GT 完全没被预测框到
                continue

            # 框到了
            matched_gt += 1
            per_matched[gt_super] += 1

            # 预测大类 vs GT 大类
            pred_super = cid2super.get(best_pred["category_id"])

            if pred_super == gt_super:
                correct_super += 1
                per_correct[gt_super] += 1

    # ---------- 输出整体指标 ----------
    det_recall = matched_gt / total_gt if total_gt else 0
    super_acc = correct_super / matched_gt if matched_gt else 0

    print("\n========= Corner Supercategory Evaluation =========")
    print(f"Corner GT 总数: {total_gt}")
    print(f"IoU≥{iou_thresh} 框到的 GT: {matched_gt}")
    print(f"检测召回率: {det_recall:.4f}")
    print(f"supercategory 正确率（框到的前提下）: {super_acc:.4f}")

    # ---------- 输出每个 supercategory ----------
    print("\n[按 supercategory 的表现]")
    per_stats = {}
    for sup in sorted(per_total.keys()):
        tg = per_total[sup]
        mg = per_matched.get(sup, 0)
        cg = per_correct.get(sup, 0)
        det_r = mg / tg if tg else 0
        acc = cg / mg if mg else 0

        per_stats[sup] = {
            "total_gt": tg,
            "matched_gt": mg,
            "super_correct": cg,
            "det_recall": det_r,
            "super_acc": acc
        }

        print(f"  - {sup}: GT={tg}, matched={mg}, correct={cg}, "
              f"det_recall={det_r:.4f}, super_acc={acc:.4f}")

    return {
        "det_recall": det_recall,
        "super_acc": super_acc,
        "per_super": per_stats
    }


# -----------------------------
# 读取预测目录
# -----------------------------
def load_preds(pred_dir: Path):
    all_preds = []
    for f in sorted(pred_dir.glob("*.json")):
        try:
            part = json.load(open(f, "r"))
        except:
            continue
        if isinstance(part, list):
            all_preds += part
    return all_preds


# -----------------------------
# main
# -----------------------------
def main():
    if len(sys.argv) != 4:
        print("用法：python evaluate_corner_cat.py <pred_dir> <gt_corner_cat.json> <output_dir>")
        return

    pred_dir = Path(sys.argv[1])
    gt_path = Path(sys.argv[2])
    out_dir = Path(sys.argv[3])
    out_dir.mkdir(exist_ok=True, parents=True)

    preds = load_preds(pred_dir)
    stats = eval_corner(gt_path, preds, iou_thresh=0.5)

    # 保存结果
    out = out_dir / "corner_super_scores.txt"
    with open(out, "w") as f:
        f.write(f"det_recall={stats['det_recall']}\n")
        f.write(f"super_acc={stats['super_acc']}\n\n")
        f.write("per_super:\n")
        for sup, s in stats["per_super"].items():
            f.write(f"{sup}: {s}\n")

    print(f"\n结果已保存到: {out}")


if __name__ == "__main__":
    main()
