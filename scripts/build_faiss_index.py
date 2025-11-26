import os
import json
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
from torch import nn
from torchvision import models, transforms
import faiss                   # 在 Mac 上用 pip install faiss-cpu

# === 1. 基本路径配置 ===
META_CSV = "data_gallery/CODA2022-val/gallery_meta.csv"   # 之前生成的元数据
SAVE_DIR = "data_gallery/CODA2022-val/faiss_index"
os.makedirs(SAVE_DIR, exist_ok=True)

# === 2. 读取 CSV ===
print("📂 Loading gallery metadata...")
df = pd.read_csv(META_CSV)
paths = df["crop_path"].tolist()
class_names = df["class_name"].tolist()
cids = df["cid"].tolist()

# === 3. 定义图像预处理和模型 ===
# Mac 一般没 GPU，自动切换到 CPU
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# 预训练 ResNet50，并去掉最后分类层
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
model.fc = nn.Identity()
model = model.to(device).eval()

# 图像预处理（要和 ResNet50 的输入保持一致）
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

# === 4. 定义辅助函数 ===
def get_embedding(img_path):
    """读取图片 -> 提取2048维特征"""
    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model(x)
        feat = nn.functional.normalize(feat, p=2, dim=1)  # L2标准化
    return feat.cpu().numpy()[0]  # 取出 numpy 向量

# === 5. 批量提取特征 ===
print("🔍 Extracting features...")
embeddings = []
meta_info = []

for i, path in enumerate(tqdm(paths)):
    if not os.path.exists(path):
        continue
    emb = get_embedding(path)
    embeddings.append(emb)
    meta_info.append({
        "path": path,
        "class_name": class_names[i],
        "cid": int(cids[i]),
    })

embeddings = np.array(embeddings).astype("float32")

# === 6. 建立 FAISS 索引 ===
print("🧠 Building FAISS index...")
d = embeddings.shape[1]  # 特征维度（2048）
index = faiss.IndexFlatIP(d)  # 内积索引（等价于余弦相似度）
index.add(embeddings)

# === 7. 保存结果 ===
faiss.write_index(index, os.path.join(SAVE_DIR, "gallery.index"))
np.save(os.path.join(SAVE_DIR, "embeddings.npy"), embeddings)

with open(os.path.join(SAVE_DIR, "mapping.jsonl"), "w", encoding="utf-8") as f:
    for m in meta_info:
        f.write(json.dumps(m, ensure_ascii=False) + "\n")

print(f"✅ 完成！共处理 {len(embeddings)} 张图片")
print(f"📁 FAISS 索引已保存到: {SAVE_DIR}")
