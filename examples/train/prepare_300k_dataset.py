import os
import ssl
import urllib3

# 1. 暴力破解：让整个 Python 的 SSL 模块无条件信任自签名证书
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 2. 开启国内镜像加速
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"

from datasets import load_dataset, concatenate_datasets

def download_and_prepare_300k():
    print("正在下载并构建 30 万条高质量多模态(对话 + 代码 + 数学)训练集...")
    datasets_to_combine = []

    # 1. 载入 OpenOrca (通用指令与推理，采样 10 万条)
    try:
        print("正在下载 OpenOrca (100k)...")
        openorca = load_dataset("Open-Orca/OpenOrca", split="train")
        openorca = openorca.shuffle(seed=42).select(range(min(100000, len(openorca))))
        def format_openorca(example):
            return {
                "messages": [
                    {"role": "system", "content": example.get("system_prompt", "")},
                    {"role": "user", "content": example["question"]},
                    {"role": "assistant", "content": example["response"]}
                ]
            }
        openorca = openorca.map(format_openorca, remove_columns=openorca.column_names)
        datasets_to_combine.append(openorca)
        print("-> OpenOrca 加载完成")
    except Exception as e:
        print(f"载入 OpenOrca 失败或跳过: {e}")

    # 2. 载入 ShareGPT (真实多轮对话，采样 8 万条)
    try:
        print("正在下载 ShareGPT (80k)...")
        sharegpt = load_dataset("Aeala/ShareGPT_Vicuna_unfiltered", split="train")
        def format_sharegpt(example):
            convs = example.get("conversations", [])
            messages = []
            for turn in convs:
                role = "user" if turn.get("from") in ["human", "user"] else "assistant"
                messages.append({"role": role, "content": turn.get("value", "")})
            return {"messages": messages}
        sharegpt = sharegpt.map(format_sharegpt, remove_columns=sharegpt.column_names)
        if len(sharegpt) > 80000:
            sharegpt = sharegpt.shuffle(seed=42).select(range(80000))
        datasets_to_combine.append(sharegpt)
        print("-> ShareGPT 加载完成")
    except Exception as e:
        print(f"载入 ShareGPT 失败或跳过: {e}")

    # 3. 载入 MetaMathQA (数学逻辑与步骤推理，采样 6 万条)
    try:
        print("正在下载 MetaMathQA (60k)...")
        metamath = load_dataset("meta-math/MetaMathQA", split="train")
        def format_metamath(example):
            return {
                "messages": [
                    {"role": "user", "content": example["query"]},
                    {"role": "assistant", "content": example["response"]}
                ]
            }
        metamath = metamath.map(format_metamath, remove_columns=metamath.column_names)
        metamath = metamath.shuffle(seed=42).select(range(min(60000, len(metamath))))
        datasets_to_combine.append(metamath)
        print("-> MetaMathQA 加载完成")
    except Exception as e:
        print(f"载入 MetaMathQA 失败或跳过: {e}")

    # 4. 载入 Magicoder (代码生成与代码修复能力，采样 6 万条)
    try:
        print("正在下载 Magicoder 代码指令集 (60k)...")
        code_dataset = load_dataset("ise-uiuc/Magicoder-Evol-Instruct-110K", split="train")
        def format_code(example):
            return {
                "messages": [
                    {"role": "user", "content": example["instruction"]},
                    {"role": "assistant", "content": example["response"]}
                ]
            }
        code_dataset = code_dataset.map(format_code, remove_columns=code_dataset.column_names)
        if len(code_dataset) > 60000:
            code_dataset = code_dataset.shuffle(seed=42).select(range(60000))
        datasets_to_combine.append(code_dataset)
        print("-> Magicoder 代码集加载完成")
    except Exception as e:
        print(f"载入 Magicoder 失败或跳过: {e}")

    if not datasets_to_combine:
        raise ValueError("没有成功加载任何数据集，请检查网络或 HF 镜像设置。")

    print("正在合并所有多模态数据集...")
    combined_dataset = concatenate_datasets(datasets_to_combine)
    print(f"合并完成！总原始样本数: {len(combined_dataset)}")

    # 基础清洗：去除空内容或字数太短的垃圾样本
    print("正在进行基础清洗与过滤...")
    def filter_func(example):
        msgs = example.get("messages", [])
        if len(msgs) < 2:
            return False
        total_len = sum(len(m.get("content", "")) for m in msgs)
        return total_len > 50

    filtered_dataset = combined_dataset.filter(filter_func)
    print(f"清洗后有效总样本数: {len(filtered_dataset)}")

    # 保存为标准的 jsonl 格式
    output_path = "speculator_train_300k.jsonl"
    print(f"正在保存至本地文件: {output_path}...")
    filtered_dataset.to_json(output_path, force_ascii=False)
    print("🎉 约 30w 条平衡版训练集准备完毕！")

if __name__ == "__main__":
    download_and_prepare_300k()
