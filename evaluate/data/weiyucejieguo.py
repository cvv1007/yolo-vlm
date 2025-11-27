import json

with open("/Users/yxr/Desktop/AI7102/YOLOpractice/CODAdatasets/CODA2022-test/annotations.json") as f:
    gt = json.load(f)

fake_pred = []
for ann in gt["annotations"]:
    fake_pred.append({
        "image_id": ann["image_id"],
        "category_id": ann["category_id"],
        "bbox": ann["bbox"],
        "score": 1.0
    })

with open("submission_origin_yolo/fake_pred.json", "w") as f:
    json.dump(fake_pred, f)
