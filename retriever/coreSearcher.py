# -*- coding: utf-8 -*-
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
from collections import Counter

from PIL import Image
import numpy as np
import torch
from torch import nn
from torchvision import models, transforms
import faiss  # pip install faiss-cpu

from .utils import detect_object



# ===== 内部工具函数（基本照你原来的抄过来）=====

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"   # Apple Silicon
    return "cpu"

def build_model(device):
    """与 build_faiss_index.py 保持一致的 ResNet50 backbone"""
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Identity()
    model.eval().to(device)
    return model

def get_transform():
    """与 build_faiss_index.py 保持一致的预处理"""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])

def embed_image(path, model, device, transform):
    """从图片路径到 1x2048 的归一化特征向量"""
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


# ===== 对外接口函数：load_retrieval, classify_crop =====

def load_retrieval(index_dir: str):
    """
    加载检索需要的一切：
      - 设备 device
      - ResNet50 模型 model
      - 图像预处理 transform
      - FAISS 索引 index
      - mapping 列表（每个向量对应的 {path, cid, class_name}）
    返回一个 ctx 字典，后面 classify_crop 直接用。
    """
    device = get_device()
    print(f"[RET] Using device: {device}")
    model = build_model(device)
    transform = get_transform()

    idx_path = os.path.join(index_dir, "gallery.index")
    map_path = os.path.join(index_dir, "mapping.jsonl")

    if not os.path.exists(idx_path):
        raise FileNotFoundError(f"FAISS index not found: {idx_path}")
    if not os.path.exists(map_path):
        raise FileNotFoundError(f"mapping.jsonl not found: {map_path}")

    index = faiss.read_index(idx_path)
    mapping = load_mapping(map_path)

    ctx = {
        "device": device,
        "model": model,
        "transform": transform,
        "index": index,
        "mapping": mapping,
    }
    return ctx


def classify_crop(crop_path: str, ctx, topk: int = 5, thresh: float = 0.80):
    """
    对一张裁剪图做：
      1) ResNet50 + FAISS 检索 Top-K
      2) Top-K 投票，计算票王类别和占比 purity
      3) 看 top1_score 和 purity：
         如果 top1_score < thresh 或 purity < 0.5 → 认为不可靠，Ask Qwen
         否则 → 采用投票结果

    返回一个 dict，不直接打印，方便 pipeline 或 CLI 使用：
      {
        "final_class": str 或 None,
        "final_cid": int 或 None,
        "top1_score": float 或 None,
        "purity": float 或 None,
        "used_qwen": bool,
        "qwen_word": str 或 None,
        "voted_name": str,
        "voted_cid": int,
        "sims": [..topk..],
        "ids":  [..topk..],
      }
    """
    if not os.path.exists(crop_path):
        raise FileNotFoundError(f"crop image not found: {crop_path}")

    device = ctx["device"]
    model = ctx["model"]
    transform = ctx["transform"]
    index = ctx["index"]
    mapping = ctx["mapping"]

    # 1) 嵌入
    q = embed_image(crop_path, model, device, transform)  # [1, 2048]

    # 2) 搜索
    sims, ids = index.search(q, topk)
    sims, ids = sims[0], ids[0]

    # 有效 (score, idx)
    valid = [(float(s), int(i)) for s, i in zip(sims, ids) if 0 <= int(i) < len(mapping)]
    if not valid:
        # 索引没返回有效结果，直接问 Qwen
        qwen_word = detect_object(crop_path)
        return {
            "final_class": None,
            "final_cid": None,
            "top1_score": None,
            "purity": None,
            "used_qwen": True,
            "qwen_word": qwen_word,
            "voted_name": None,
            "voted_cid": None,
            "sims": sims.tolist(),
            "ids": ids.tolist(),
        }

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

    # 4) 置信判定：
    #    只要 (Top1 相似度 < 阈值) 或 (票王占比 < 1/2)，就认为不可靠 → Ask Qwen
    if top1_score < thresh or purity < 0.5:
        try:
            qwen_word = detect_object(crop_path)
        except Exception as e:
            qwen_word = f"Qwen_error:{e}"
        final_class = voted_name   # 也可以改成用 qwen_word，看你实验怎么设计
        final_cid = voted_cid
        used_qwen = True
    else:
        final_class = voted_name
        final_cid = voted_cid
        qwen_word = None
        used_qwen = False

    return {
        "final_class": final_class,
        "final_cid": final_cid,
        "top1_score": top1_score,
        "purity": purity,
        "used_qwen": used_qwen,
        "qwen_word": qwen_word,
        "voted_name": voted_name,
        "voted_cid": voted_cid,
        "sims": sims.tolist(),
        "ids": ids.tolist(),
    }
