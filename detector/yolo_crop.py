# detector/yolo_crop.py
from pathlib import Path
from ultralytics import YOLO
import cv2

def load_yolo(model_path: str = "yolov8x.pt"):
    print("[YOLO] loading model...")
    model = YOLO(model_path)
    print("[YOLO] loaded.")
    return model

def yolo_crop_one(img_path: str, out_root: str, model) -> list[str]:
    """
    对单张图片做 YOLO 检测和裁剪。
    返回：这一张图所有 crop 的路径列表。
    """
    img_path = Path(img_path)
    out_root = Path(out_root)

    im = cv2.imread(str(img_path))
    if im is None:
        print(f"[YOLO] 读图失败: {img_path}")
        return []

    H, W = im.shape[:2]
    results = model.predict(source=str(img_path), imgsz=1024, iou=0.7,
                            device="", verbose=False)
    boxes = results[0].boxes

    img_dir = out_root / img_path.stem
    crops_dir = img_dir / "crops"
    img_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)

    drawn = im.copy()
    crop_paths = []

    if boxes and len(boxes) > 0:
        for i, b in enumerate(boxes):
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            crop = im[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            crop_path = crops_dir / f"{img_path.stem}_{i:03d}.jpg"
            cv2.imwrite(str(crop_path), crop)
            crop_paths.append(str(crop_path))

            cv2.rectangle(drawn, (x1, y1), (x2, y2), (0, 255, 0), 2)

    out_img = img_dir / f"{img_path.stem}_bbox.jpg"
    cv2.imwrite(str(out_img), drawn)

    return crop_paths
