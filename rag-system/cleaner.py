import re
import pandas as pd
import json
import os


def clean_text(text):
    """基础清洗逻辑"""
    if not isinstance(text, str):
        return ""
    # 1. 去除多余的空白字符
    text = re.sub(r'\s+', ' ', text).strip()
    # 2. 去除一些特定噪声（如HTML标签，若有）
    text = re.sub(r'<.*?>', '', text)
    # 3. 可以在这里加入更多正则，比如去除乱码、敏感信息掩码等
    return text


def process_data(input_file, output_file, is_for_finetuning=False):
    """
    处理数据流程
    is_for_finetuning: 如果是微调数据，通常需要格式化为 {instruction, input, output}
    """
    # 假设输入是 CSV
    # --- 新增：确保输出目录存在 ---
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已创建目录: {output_dir}")
    # ---------------------------

    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"错误：找不到输入文件 {input_file}")
        return

    # 应用清洗
    df['output'] = df['output'].apply(clean_text)

    # 简单去重
    # df.drop_duplicates(subset=['output'], inplace=True)

    if is_for_finetuning:
        # 构造微调格式 (Alpaca 风格)
        data = []
        for _, row in df.iterrows():
            data.append({
                "instruction": "回答以下关于特定领域的问题：",
                "input": row.get('input', ''),  # 假设csv有question列
                "output": row.get('output', '')
            })
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    else:
        # 仅保存清洗后的文本用于RAG
        df.to_csv(output_file, index=False)

    print(f"数据处理完成，已保存至 {output_file}")


if __name__ == "__main__":
    # 模拟运行
    process_data("./data/input/lora_train_data.csv", "./data/output/train_data.json", is_for_finetuning=True)
    pass
