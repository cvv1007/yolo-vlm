# make annotation json file more readable
import json
import os
def read_txt_lines(file_path): 
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    num_lines = len(lines)
    print("文本文件的行数:", num_lines)

def pretty_json(input_path, output_path):

    input_path = input_path
    
    output_path = output_path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # 读取原始文件
    with open(input_path, "r") as f:
        data = json.load(f)

    # 写入格式化后的 JSON 文件
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"✅ 已将格式化后的文件保存到: {output_path}")

input_path = "/Users/yxr/Desktop/AI7102/YOLOpractice/evaluate/data/gt_common.json"
output_path = "/Users/yxr/Desktop/AI7102/YOLOpractice/evaluate/data/gt_common_pretty.json"
pretty_json(input_path, output_path)