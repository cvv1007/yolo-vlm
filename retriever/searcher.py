# -*- coding: utf-8 -*-
import os
from collections import Counter
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from utils import detect_object

import json
import argparse
from PIL import Image

import numpy as np
import torch
from torch import nn
from torchvision import models, transforms
import faiss  # pip install faiss-cpu

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"   # Apple Silicon
    return "cpu"

def build_model(device):
    # 与 build_faiss_index.py 保持一致
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Identity()
    model.eval().to(device)
    return model

def get_transform():
    # 与 build_faiss_index.py 保持一致
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])

def embed_image(path, model, device, transform):
    img = Image.open(path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model(x)                             # [1, 2048]
        feat = nn.functional.normalize(feat, p=2, dim=1)  # L2 标准化 → 余弦
    return feat.cpu().numpy().astype("float32")     # [1, 2048]

def load_mapping(mapping_jsonl_path):
    mapping = []
    with open(mapping_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            mapping.append(json.loads(line))
    return mapping

def main():
    parser = argparse.ArgumentParser(
        description="Search similar crops for ONE image using FAISS index."
    )
    parser.add_argument("--query", default="/Users/yxr/Desktop/AI7102/YOLOpractice/data_gallery/CODA2022-test/missing_classes/12_wheelchair/0269_ann4149.jpg", help="Path to the query image.")
    parser.add_argument(
        "--index_dir",
        default="/Users/yxr/Desktop/AI7102/YOLOpractice/data_gallery/CODA2022-val/faiss_index",
        help="Dir containing gallery.index and mapping.jsonl"
    )
    parser.add_argument("--topk", type=int, default=5, help="Top-K results.")
    args = parser.parse_args()

    idx_path = os.path.join(args.index_dir, "gallery.index")
    map_path = os.path.join(args.index_dir, "mapping.jsonl")

    if not os.path.exists(idx_path):
        raise FileNotFoundError(f"FAISS index not found: {idx_path}")
    if not os.path.exists(map_path):
        raise FileNotFoundError(f"mapping.jsonl not found: {map_path}")
    if not os.path.exists(args.query):
        raise FileNotFoundError(f"Query image not found: {args.query}")

    # 1) 设备 & 模型 & 预处理
    device = get_device()
    print(f"💻 Using device: {device}")
    model = build_model(device)
    transform = get_transform()

    # 2) 载入索引与映射
    print("📦 Loading FAISS index & mapping...")
    index = faiss.read_index(idx_path)
    mapping = load_mapping(map_path)

    # 3) 查询图 → 向量
    print("🔍 Embedding query image...")
    q = embed_image(args.query, model, device, transform)  # [1, 2048]

    # 4) 搜索
    print(f"🧠 Searching top-{args.topk} ...")
    sims, ids = index.search(q, args.topk)  # 由于做了L2标准化，这里的内积≈余弦相似度
    sims, ids = sims[0], ids[0]

    # 5) 打印结果
    print("\n=== Search Results ===")
    for rank, (score, idx) in enumerate(zip(sims, ids), start=1):
        if idx < 0 or idx >= len(mapping):
            continue
        item = mapping[idx]
        print(f"[{rank}] sim={score:.4f} | cid={item.get('cid')} | "
              f"class={item.get('class_name')} | path={item.get('path')}")
        
        # ==== 新的判定逻辑：先 Top-K 投票，再看置信度 ====
    THRESH = 0.80  # 相似度阈值，可以自己调

    # 1) 有效的 (score, idx)
    valid = [(float(s), int(i)) for s, i in zip(sims, ids) if 0 <= int(i) < len(mapping)]
    if not valid:
        print("\n⚠️ 未得到有效检索结果。")
        return

    # 2) Top1 的相似度（只用分数，不用它的类别）
    top1_score, _ = valid[0]

    # 3) Top-K 投票
    topk_cids = [mapping[i]["cid"] for _, i in valid]
    vote = Counter(topk_cids).most_common(1)[0]  # (cid, 票数)
    voted_cid, voted_cnt = vote
    voted_name = next(
        (mapping[i]["class_name"] for _, i in valid if mapping[i]["cid"] == voted_cid),
        str(voted_cid)
    )
    purity = voted_cnt / len(valid)  # 票王在 Top-K 中的占比

    print("\n— Top-K 投票 —")
    print(f"票王类别: {voted_name} (cid={voted_cid}), "
          f"票数={voted_cnt}/{len(valid)}, 票占比={purity:.2f}")
    print(f"Top-1 相似度: {top1_score:.4f}, 阈值: {THRESH:.2f}")

    # 4) 置信判定：
    #    只要 (Top1 相似度 < 阈值) 或 (票王占比 < 1/2)，就认为不可靠 → Ask Qwen
    if top1_score < THRESH or purity < 0.5:
        print("❓ 结论：预测不确定（Ask qwen）。")
        try:
            qwen_word = detect_object(args.query)
            print(f"🤖 Qwen 判定：{qwen_word}")
        except Exception as e:
            print(f"⚠️ Qwen 调用失败：{e}")
            print(f"回退使用投票结果：{voted_name} (cid={voted_cid})")
    else:
        # 两个条件都 ok：相似度高 + 票王占比高 → 采纳投票结果
        print("✅ 结论：预测可信，采用 Top-K 投票结果。")
        print(f"最终类别：{voted_name} (cid={voted_cid})")

    
if __name__ == "__main__":
    main()
