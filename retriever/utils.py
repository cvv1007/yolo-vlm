# ask qwen
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import torch


model_id = "Qwen/Qwen3-VL-2B-Instruct"
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForVision2Seq.from_pretrained(model_id, trust_remote_code=True).eval()

def detect_object(image_path: str) -> str:
    """输入图片路径，返回识别出的物体名（一个词）"""
    img = Image.open(image_path).convert("RGB")
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": "What is the object in this image? Reply with ONE English word only."}
        ]
    }]
    # prompt
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    # inputs
    inputs = processor(text=prompt, images=img, return_tensors="pt")
    for k, v in inputs.items():
        if torch.is_tensor(v):
            inputs[k] = v.to(model.device)
    inputs["input_ids"] = inputs["input_ids"].to(torch.long)
    inputs["attention_mask"] = inputs["attention_mask"].to(torch.long)
    # generate
    outputs = model.generate(**inputs, max_new_tokens=3)
    new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    text = processor.decode(new_tokens, skip_special_tokens=True).strip()
    return text

# 调用示例
path = "/Users/yxr/Desktop/AI7102/YOLOpractice/CODAdatasets/CODA/out_sample/000001_1616005007200/crops/000001_1616005007200_000.jpg"
print(detect_object(path))
