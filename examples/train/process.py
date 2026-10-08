import json
input_file = "./magicoder_60k.jsonl"
output_file = "./magicoder_60k_processed.jsonl"
with open(input_file, "r", encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
    for line in fin:
        data = json.loads(line)
        if "messages" in data and "conversations" not in data:
            data["conversations"] = data.pop("messages")
        fout.write(json.dumps(data, ensure_ascii=False) + "\n")
