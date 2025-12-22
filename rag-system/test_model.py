
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def test():

    # 1. 自动定位路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 假设你的项目结构是 rag-system/test_model.py，model 目录在同级或上级
    lora_path = os.path.join(os.path.dirname(current_dir), "rag-system", "model", "lora_adapter")

    print(f"尝试加载适配器路径: {lora_path}")

    # 2. 加载
    base_model_path = "/Users/wuzexin/.cache/modelscope/hub/models/Qwen/Qwen2___5-0___5B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        dtype=torch.float16,
        device_map="auto"
    )
    # 对齐 Padding Token（最重要）
    # Qwen 模型如果没有正确设置 pad_token，生成时会陷入死循环。请确保代码中有这两行：
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

    try:
        model = PeftModel.from_pretrained(model, lora_path)
        print("LoRA 适配器挂载成功！")
    except Exception as e:
        print(f"挂载失败: {e}")


    # 3. 准备一个测试问题
    # 注意：Prompt 格式必须和微调时的 formatting_func 一致！
    instruction = "回答关于飞跃云产品的问题"
    user_input = "免费空间"

    prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{user_input}\n\n### Response:\n"

    # 4. 生成回答
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=50,  # 先少生成点，看看对不对
            do_sample=False,  # 必须关闭
            repetition_penalty=1.2,  # 强制模型不要重复字符
            no_repeat_ngram_size=2  # 禁止连续重复的词
        )

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    print("-" * 30)
    print("模型回答：")
    print(response.split("### Response:\n")[-1]) # 只打印回答部分