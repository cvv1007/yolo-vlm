# pipeline/main_online.py
import argparse
import json
from pathlib import Path

from detector.yolo_crop import load_yolo, yolo_crop_one_full
from retriever.coreSearcher import load_retrieval, classify_crop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--source",
        default="/Users/yxr/Desktop/AI7102/YOLOpractice/CODAdatasets/CODA2022-test",
        help="CODA2022-test images 根目录",
    )
    ap.add_argument(
        "--crop_out",
        default="/Users/yxr/Desktop/AI7102/YOLOpractice/output/coda_tunned_yolo",
        help="YOLO 裁剪输出根目录",
    )
    ap.add_argument(
        "--index_dir",
        default="/Users/yxr/Desktop/AI7102/YOLOpractice/data_gallery/CODA2022-val/faiss_index",
        help="FAISS 索引目录",
    )
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--thresh", type=float, default=0.8)
    ap.add_argument(
        "--save_dir",
        default="submission_origin_yolo",
        help="COCO detection 结果保存目录（用于 eval 脚本），会生成多个 part json",
    )
    ap.add_argument(
        "--chunk_size",
        type=int,
        default=5000,
        help="每多少条检测结果写一个 json 文件",
    )
    args = ap.parse_args()

    src = Path(args.source)
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    images = [p for p in src.rglob("*") if p.suffix.lower() in exts]
    print(f"[PIPE] 共找到 {len(images)} 张 test 图片")

    # 1) 初始化 YOLO 和 retrieval（只初始化一次）
    yolo_model = load_yolo("yolov8x.pt") #yolov8x.p改成best.pt 以及yolo11m试一下
    ctx = load_retrieval(args.index_dir)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    preds_chunk = []   # 当前这一块的 COCO detection 结果
    part_idx = 0       # 当前写到了第几个 part 文件

    def flush_chunk():
        """把当前 preds_chunk 写成一个 json 文件并清空。"""
        nonlocal preds_chunk, part_idx
        if not preds_chunk:
            return
        out_path = save_dir / f"coda_pred_part{part_idx:03d}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(preds_chunk, f)
        print(f"[SAVE] 写出 {len(preds_chunk)} 条结果到 {out_path}")
        preds_chunk = []
        part_idx += 1

    # 2) 逐张图片处理：YOLO 检测 + 裁剪 + 检索/分类
    for img_path in images:
        print(f"\n[IMG] {img_path}")
        dets = yolo_crop_one_full(str(img_path), args.crop_out, yolo_model)

        if not dets:
            print("  [INFO] 没有检测到目标")
            continue

        # 简单约定：文件名（去掉扩展名）就是 image_id，例如 "000123.jpg" → 123
        try:
            image_id = int(img_path.stem)
        except ValueError:
            print(f"  [WARN] 无法从文件名解析 image_id: {img_path.stem}，这张图跳过")
            continue

        for det in dets:
            cp = det["crop_path"]
            bbox = det["bbox"]
            score = det["score"]

            result = classify_crop(cp, ctx, topk=args.topk, thresh=args.thresh)
            print(f"  [CROP] {cp}")
            print(
                f"    final_class={result['final_class']}, "
                f"cid={result['final_cid']}, "
                f"top1={result['top1_score']}, purity={result['purity']}, "
                f"used_qwen={result['used_qwen']}, qwen={result['qwen_word']}"
            )

            cid = result["final_cid"]
            if cid is None:
                # 连 cid 都没有，无法参与 COCO 评测，跳过
                continue

            # 一条标准 COCO detection + 你自己的分析字段
            preds_chunk.append(
                {
                    "image_id": int(image_id),
                    "category_id": int(cid),
                    "bbox": bbox,              # [x, y, w, h]
                    "score": float(score),     # 置信度

                    # 额外信息：方便之后分析 Qwen / Unknown
                    "used_qwen": bool(result["used_qwen"]),
                    "qwen_word": result["qwen_word"],
                    "is_unknown": bool(result.get("is_unknown", False)),
                }
            )

            # 如果当前 chunk 够大了，就写一个文件
            if len(preds_chunk) >= args.chunk_size:
                flush_chunk()

    # 3) 把最后残余的一块也写出去
    flush_chunk()

    print(f"\n[PIPE] 共写出 {part_idx} 个 json 文件到 {save_dir}")


if __name__ == "__main__":
    main()
