# pipeline/main_online.py
import argparse
from pathlib import Path
from detector.yolo_crop import load_yolo, yolo_crop_one
from retriever.coreSearcher import load_retrieval, classify_crop

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="/Users/yxr/Desktop/AI7102/YOLOpractice/CODAdatasets/CODA2022-test", help="CODA2022-test images 根目录")
    ap.add_argument("--crop_out", default="/Users/yxr/Desktop/AI7102/YOLOpractice/crop_out/coda_test", help="YOLO 裁剪输出根目录")
    ap.add_argument("--index_dir", default="/Users/yxr/Desktop/AI7102/YOLOpractice/data_gallery/CODA2022-val/faiss_index", help="FAISS 索引目录")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--thresh", type=float, default=0.8)
    args = ap.parse_args()

    src = Path(args.source)
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    images = [p for p in src.rglob("*") if p.suffix.lower() in exts]
    print(f"[PIPE] 共找到 {len(images)} 张 test 图片")

    # 1) 只初始化一次 YOLO 和 retrieval
    yolo_model = load_yolo("yolov8x.pt")
    ctx = load_retrieval(args.index_dir)

    # 2) 一张张处理：裁剪一张，立刻对这张图的所有 crop 做分类
    for img_path in images:
        print(f"\n[IMG] {img_path}")
        crop_paths = yolo_crop_one(str(img_path), args.crop_out, yolo_model)

        if not crop_paths:
            print("  [INFO] 没有检测到目标")
            continue

        for cp in crop_paths:
            result = classify_crop(cp, ctx, topk=args.topk, thresh=args.thresh)
            print(f"  [CROP] {cp}")
            print(f"    final_class={result['final_class']}, "
                  f"cid={result['final_cid']}, "
                  f"top1={result['top1_score']}, purity={result['purity']}, "
                  f"used_qwen={result['used_qwen']}, qwen={result['qwen_word']}")

if __name__ == "__main__":
    main()
