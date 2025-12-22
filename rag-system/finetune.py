import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig  # 需要安装 trl 库: pip install trl
from datasets import load_dataset
from modelscope import snapshot_download

# 配置路径
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"  # 示例模型，显存占用小
DATA_PATH = "./data/output/train_data.json"
OUTPUT_DIR = "./model/lora_adapter"


def train_lora():

    # 1. 加载 Tokenizer 和 模型
    # tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    # model = AutoModelForCausalLM.from_pretrained(
    #     MODEL_ID,
    #     torch_dtype=torch.float16,
    #     device_map="auto"
    # )

    # 先将模型下载到本地（会自动处理镜像）
    model_dir = snapshot_download(MODEL_ID)

    # 然后从本地路径加载
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        device_map="auto",
        torch_dtype="auto"  # 顺便修复你 log 里的那个 deprecated 警告
    )

    # 2. 配置 LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=4,  # LoRA 秩，越大参数越多
        lora_alpha=32,
        lora_dropout=0.1
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 3. 加载数据集
    dataset = load_dataset("json", data_files=DATA_PATH, split="train")

    # 1. 定义格式化函数（保持不变）
    # 修改后的格式化函数
    def formatting_prompts_func(example):
        # 此时 example 是一个 dict，代表单条数据，例如 {"instruction": "...", "input": "...", "output": "..."}
        text = f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"
        # 注意：单条处理模式下，有些版本的 TRL 要求返回字符串，有些要求返回列表。
        # 如果下面运行还报错，可以尝试返回 [text]
        return text

    # 2. 配置 SFTConfig (替代旧的 TrainingArguments)
    # SFTConfig 继承自 TrainingArguments，所以所有训练参数都写在这里
    sft_config = SFTConfig(
        output_dir="./model/lora_adapter",
        max_length=512,  # 移到了这里
        per_device_train_batch_size=1,  # Mac 建议设为 1
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        learning_rate=1e-4,
        logging_steps=1,
        save_strategy="no",
        packing=False,  # 增加此项以确保数据按条处理
        dataset_kwargs={
            "add_special_tokens": False,  # 根据需要配置
        }
    )

    # 3. 初始化 Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=sft_config,  # 传入配置对象
        peft_config=peft_config,
        formatting_func=formatting_prompts_func,
    )

    trainer.train()

    # 6. 保存最终适配器
    model.save_pretrained(OUTPUT_DIR)
    print("LoRA 微调完成，模型已保存。")


if __name__ == "__main__":
    train_lora()