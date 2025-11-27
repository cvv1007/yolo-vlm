# extract_unknown_qwen_words.py
# -*- coding: utf-8 -*-
import json
from collections import Counter
from pathlib import Path

PRED_DIR = Path("submmision_tunned_140_11m")
OUT_PATH = Path("evaluate/data/unknown_qwen_words.json")


def main():
    pred_files = sorted(PRED_DIR.glob("*.json"))
    if not pred_files:
        print(f"[ERR] 没找到预测文件，检查目录: {PRED_DIR}")
        return

    counter = Counter()

    for f in pred_files:
        print(f"[READ] {f}")
        with open(f, "r", encoding="utf-8") as fp:
            preds = json.load(fp)
        if not isinstance(preds, list):
            print(f"[WARN] {f} 不是 list，跳过")
            continue

        for p in preds:
            if not p.get("is_unknown", False):
                continue
            qw = p.get("qwen_word", "")
            if not qw:
                continue
            qw_norm = qw.strip().lower()
            if not qw_norm:
                continue
            counter[qw_norm] += 1

    # 生成一个方便人工编辑的结构：列表，每条是 {qwen_word, count, match_cid}
    # match_cid 先设为 None，后面你手动填
    items = []
    for word, cnt in counter.most_common():
        items.append({
            "qwen_word": word,
            "count": cnt,
            "match_cid": None,   # 你之后手动改成具体的category id（比如 4）
            # 也可以改成 match_name 看你习惯
        })

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    print(f"[DONE] 共提取到 {len(items)} 个去重后的 unknown qwen_word，已写入 {OUT_PATH}")


if __name__ == "__main__":
    main()
