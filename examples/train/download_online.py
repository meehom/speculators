import os
import ssl
import urllib3

# === 1. 终极暴力破解：全局关闭 SSL 证书验证 ===
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# === 2. 针对 huggingface 的环境变量和镜像设置 ===
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"
os.environ["CURL_CA_BUNDLE"] = ""

# 暴力替换 httpx 的客户端，防止因为单次证书失败直接关闭 client
import httpx
_orig_init = httpx.Client.__init__
def _new_init(self, *args, **kwargs):
    kwargs["verify"] = False
    return _orig_init(self, *args, **kwargs)
httpx.Client.__init__ = _new_init

_orig_async_init = httpx.AsyncClient.__init__
def _new_async_init(self, *args, **kwargs):
    kwargs["verify"] = False
    return _orig_async_init(self, *args, **kwargs)
httpx.AsyncClient.__init__ = _new_async_init


from datasets import load_dataset, concatenate_datasets

def download_and_prepare_300k():
    print("正在尝试通过镜像站直接下载并构建 30 万条数据集...")
    datasets_to_combine = []

    # 1. 载入 OpenOrca (通用指令与推理)
    try:
        print("正在下载 OpenOrca (100k)...")
        openorca = load_dataset("Open-Orca/OpenOrca", split="train", trust_remote_code=True)
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
        print("-> OpenOrca 加载成功！")
    except Exception as e:
        print(f"⚠️ 载入 OpenOrca 失败: {e}")

    # 2. 载入 ShareGPT (真实多轮对话)
    try:
        print("正在下载 ShareGPT (80k)...")
        sharegpt = load_dataset("Aeala/ShareGPT_Vicuna_unfiltered", split="train", trust_remote_code=True)
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
        print("-> ShareGPT 加载成功！")
    except Exception as e:
        print(f"⚠️ 载入 ShareGPT 失败: {e}")

    # 3. 载入 MetaMathQA (数学逻辑)
    try:
        print("正在下载 MetaMathQA (60k)...")
        metamath = load_dataset("meta-math/MetaMathQA", split="train", trust_remote_code=True)
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
        print("-> MetaMathQA 加载成功！")
    except Exception as e:
        print(f"⚠️ 载入 MetaMathQA 失败: {e}")

    # 4. 载入 Magicoder (代码生成)
    try:
        print("正在下载 Magicoder (60k)...")
        code_dataset = load_dataset("ise-uiuc/Magicoder-Evol-Instruct-110K", split="train", trust_remote_code=True)
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
        print("-> Magicoder 加载成功！")
    except Exception as e:
        print(f"⚠️ 载入 Magicoder 失败: {e}")

    if not datasets_to_combine:
        raise ValueError("❌ 所有数据集在线下载均失败。这说明内网网关完全屏蔽了外部 IP 或镜像站（hf-mirror.com）。如果依然报错，建议使用之前的离线传文件方案。")

    print("正在合并所有成功下载的数据集...")
    combined_dataset = concatenate_datasets(datasets_to_combine)
    print(f"合并完成！总样本数: {len(combined_dataset)}")

    print("正在进行基础清洗...")
    def filter_func(example):
        msgs = example.get("messages", [])
        if len(msgs) < 2:
            return False
        total_len = sum(len(m.get("content", "")) for m in msgs)
        return total_len > 50

    filtered_dataset = combined_dataset.filter(filter_func)
    print(f"清洗后有效总样本数: {len(filtered_dataset)}")

    output_path = "speculator_train_300k.jsonl"
    print(f"正在保存至本地文件: {output_path}...")
    filtered_dataset.to_json(output_path, force_ascii=False)
    print("🎉 30w 条训练集在线下载并处理完毕！")

if __name__ == "__main__":
    download_and_prepare_300k()
