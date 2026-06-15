import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
import json
from tqdm import tqdm
from PIL import Image
import torch
import unsloth
from unsloth import FastVisionModel 
from transformers import AutoProcessor

# ... batch_infer 函数保持不变 ...
def batch_infer(model, processor, tokenizer, qa_data, output_path, batch_size=8):
    results = []
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    for i in tqdm(range(0, len(qa_data), batch_size)):
        batch_chunk = qa_data[i : i + batch_size]
        batch_images = []
        batch_prompts = []
        valid_items = []

        for item in batch_chunk:
            image_path = item['image']
            question = item['conversations'][0]['value'].replace('\n<image>', '').strip()

            if not os.path.exists(image_path):
                continue

            try:
                image = Image.open(image_path).convert("RGB")
                messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                
                batch_images.append(image)
                batch_prompts.append(prompt)
                valid_items.append(item)
            except Exception:
                continue

        if not batch_images: continue

        inputs = processor(text=batch_prompts, images=batch_images, return_tensors="pt", padding=True).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=1024, use_cache=True)

        input_len = inputs.input_ids.shape[1]
        generated_ids = output_ids[:, input_len:]
        answers = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        for item, ans in zip(valid_items, answers):
            results.append({"image": item['image'], "question": item['conversations'][0]['value'], "answer": ans.strip()})

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"✅ 保存完成：{output_path}, 共计 {len(results)} 条结果")


def main():
    # --- 1. 路径配置 (仅保留基础模型路径) ---
    base_model_path = "/home/zhangxw/Qwen2.5-VL-32B/"
    output_dir = "/home/zhangxw/share_data/base/PMC_VQA/"
    json_path = "/home/zhangxw/share_data/PMC-VQA/val.json"

    # --- 2. 加载数据 ---
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            qa_data = json.load(f)
    except FileNotFoundError:
        print(f"❌ 找不到 JSON 文件: {json_path}")
        return

    # --- 3. 加载基础模型 ---
    print(f"🚀 正在加载基础模型进行推理: {base_model_path}")
    model, processor = FastVisionModel.from_pretrained(
        model_name = base_model_path,
        device_map = 'auto',
        torch_dtype = torch.bfloat16,
        load_in_4bit = True, # 32B 模型建议保持 4bit 节省显存
    )
    
    tokenizer = processor.tokenizer 
    
    # 兼容性修复
    if "<image>" not in tokenizer.vocab:
        tokenizer.add_special_tokens({"additional_special_tokens": ["<image>"]})
        model.resize_token_embeddings(len(tokenizer))
    
    image_token_id = tokenizer.convert_tokens_to_ids("<image>")
    processor.image_token = "<image>"
    processor.image_token_id = image_token_id
    model.config.image_token_id = image_token_id

    # 设置聊天模板
    tokenizer.chat_template = """{% for message in messages %}
<|im_start|>{{ message['role'] }}
{% for item in message['content'] %}
{% if item['type'] == 'image' %}<image>
{% elif item['type'] == 'text' %}{{ item['text'] }}
{% endif %}{% endfor %}<|im_end|>
{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant
{% endif %}"""
    # --- 4. 执行推理 ---
    output_path = os.path.join(output_dir, "base_model_result.json")
    batch_infer(model, processor, tokenizer, qa_data, output_path, batch_size=8)
    
    print("✨ 推理任务结束。")


if __name__ == "__main__":
    main()
