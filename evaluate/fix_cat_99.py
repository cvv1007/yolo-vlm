# apply_qwen_match_to_preds.py
# -*- coding: utf-8 -*-
import json
from pathlib import Path

PRED_DIR = Path("submmision_tunned_140_11m")          # 原始预测
OUT_DIR = Path("submmision_tunned_140_11mfixed")     # 修正后的预测输出目录
MATCH_PATH = Path("/Users/yxr/Desktop/AI7102/YOLOpractice/evaluate/data/unknown_qwen_words.json")       # 你人工编辑好的match文件
UNKNOWN_CID = 99                                   # 你当前unknown的cid设定


def load_match_table(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        items = json.load(f)

    table = {}
    for item in items:
        word = str(item["qwen_word"]).strip().lower()
        cid = item.get("match_cid", None)
        # 只记录那些你真的填了cid的词
        if word and cid is not None:
            table[word] = int(cid)

    print(f"[MATCH] 共加载到 {len(table)} 条 qwen_word -> cid 映射")
    return table


def main():
    if not MATCH_PATH.exists():
        print(f"[ERR] 匹配表 {MATCH_PATH} 不存在，先运行提取脚本并人工编辑match_cid")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    match_table = load_match_table(MATCH_PATH)

    pred_files = sorted(PRED_DIR.glob("*.json"))
    if not pred_files:
        print(f"[ERR] 没找到预测文件，检查目录: {PRED_DIR}")
        return

    total_changed = 0

    for f in pred_files:
        with open(f, "r", encoding="utf-8") as fp:
            preds = json.load(fp)

        if not isinstance(preds, list):
            print(f"[WARN] {f} 不是 list，跳过")
            continue

        changed = 0
        for p in preds:
            # 只处理你标记为unknown且cid=99的预测
            if not p.get("is_unknown", False):
                continue
            if int(p.get("category_id", UNKNOWN_CID)) != UNKNOWN_CID:
                continue

            qw = p.get("qwen_word", "")
            if not qw:
                continue
            qw_norm = qw.strip().lower()
            cid_new = match_table.get(qw_norm, None)
            if cid_new is None:
                # 没有匹配到（或者你故意留空），继续当unknown
                continue

            # 命中：修改category_id，顺便把is_unknown改成False
            p["category_id"] = int(cid_new)
            p["is_unknown"] = False
            changed += 1

        total_changed += changed

        out_path = OUT_DIR / f.name
        with open(out_path, "w", encoding="utf-8") as fp:
            json.dump(preds, fp)
        print(f"[FIX] {f.name}: 更新了 {changed} 条预测 -> {out_path}")

    print(f"[DONE] 全部文件共修正 {total_changed} 条 unknown → 已知类 的预测")


if __name__ == "__main__":
    main()
