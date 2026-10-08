import pandas as pd

print("正在处理本地的 OpenOrca parquet 文件...")
df = pd.read_parquet("1M-GPT4-Augmented.parquet")

# 随机打乱并截取前 10 万条
df = df.sample(frac=1, random_state=42).iloc[:100000]

print("正在格式化为标准 messages 结构...")
records = []
for _, row in df.iterrows():
    records.append({
        "messages": [
            {"role": "system", "content": row.get("system_prompt", "") or ""},
            {"role": "user", "content": row["question"]},
            {"role": "assistant", "content": row["response"]}
        ]
    })

output_file = "openorca_100k.jsonl"
print(f"正在保存至: {output_file} ...")
pd.DataFrame(records).to_json(output_file, orient="records", lines=True, force_ascii=False)
print("🎉 OpenOrca 10w 条训练集已就绪，可以立刻开始单数据源的微调测试！")
