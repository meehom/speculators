
#!/bin/bash
set -euo pipefail

# ============================================================
# Qwen3.8-27B Vanilla vs dspark2
# vLLM + EvalScope Performance Benchmark
#
# ============================================================
# 测试目标
# ============================================================
#
#   Model:
#       Qwen3.8-27B
#
#   Vanilla:
#       普通 vLLM
#
#   dspark2:
#       vLLM Speculative Decoding + dspark2
#
#   Dataset:
#       /home/w00934428/lmh/sharegpt_80k_processed.jsonl
#
#   原始格式:
#
#       {
#         "conversations": [
#           {"role": "assistant", "content": "..."},
#           {"role": "user", "content": "..."},
#           {"role": "assistant", "content": "..."},
#           ...
#         ]
#       }
#
#   EvalScope 格式:
#
#       {
#         "conversations": [
#           {"from": "human", "value": "..."},
#           {"from": "gpt", "value": "..."},
#           ...
#         ]
#       }
#
#   Benchmark:
#       1000 conversations
#       32 concurrent conversations
#       multi-turn
#       stream
#       max_tokens=256
#       temperature=0
#
# ============================================================


# ============================================================
# Configuration
# ============================================================

MODEL="/home/w00934428/lmh/Qwen3.8-27B"

DRAFT="/home/w00934428/lmh/dspark/output/dspark_checkpoints/checkpoint_best"

# 原始 80K 数据
RAW_DATASET="/home/w00934428/lmh/sharegpt_80k_processed.jsonl"

# 转换后的 EvalScope ShareGPT 数据
DATASET="/home/w00934428/lmh/evalscope_dflash/sharegpt_80k_evalscope.jsonl"

# GPU
GPU="0"

# vLLM port
PORT=8199

# ============================================================
# Benchmark parameters
# ============================================================

# 总 conversation 数
NUM_CONVERSATIONS=1000

# 并发 conversation 数
PARALLEL=32

# 每一轮最大生成 token
MAX_NEW_TOKENS=256

# 确定性采样
TEMPERATURE=0

# ============================================================
# Result directory
# ============================================================

RESULT_DIR="/home/w00934428/lmh/evalscope_dspark"

mkdir -p "$RESULT_DIR"


# ============================================================
# Environment check
# ============================================================

echo
echo "============================================================"
echo "Environment"
echo "============================================================"

python3 - <<'PY'
import torch
import vllm

print("torch :", torch.__version__)
print("cuda  :", torch.version.cuda)
print("vllm  :", vllm.__version__)
print("vllm  :", vllm.__file__)
PY


echo
echo "============================================================"
echo "Configuration"
echo "============================================================"

echo "MODEL             = $MODEL"
echo "DRAFT             = $DRAFT"
echo "RAW_DATASET       = $RAW_DATASET"
echo "DATASET           = $DATASET"
echo "GPU               = $GPU"
echo "PORT              = $PORT"
echo "NUM_CONVERSATIONS = $NUM_CONVERSATIONS"
echo "PARALLEL          = $PARALLEL"
echo "MAX_NEW_TOKENS    = $MAX_NEW_TOKENS"
echo "TEMPERATURE       = $TEMPERATURE"
echo "RESULT_DIR        = $RESULT_DIR"


# ============================================================
# Basic checks
# ============================================================

echo
echo "============================================================"
echo "Checking files"
echo "============================================================"

test -d "$MODEL" || {
    echo "ERROR: model does not exist:"
    echo "$MODEL"
    exit 1
}

test -e "$DRAFT" || {
    echo "ERROR: draft checkpoint does not exist:"
    echo "$DRAFT"
    exit 1
}

test -e "$RAW_DATASET" || {
    echo "ERROR: dataset does not exist:"
    echo "$RAW_DATASET"
    exit 1
}

echo "Model       : OK"
echo "Draft       : OK"
echo "Raw dataset : OK"


# ============================================================
# Prepare EvalScope ShareGPT dataset
# ============================================================
#
# 原始格式:
#
# {
#   "conversations": [
#       {"role":"assistant","content":"..."},
#       {"role":"user","content":"..."},
#       {"role":"assistant","content":"..."},
#       ...
#   ]
# }
#
# 转换成:
#
# {
#   "conversations": [
#       {"from":"human","value":"..."},
#       {"from":"gpt","value":"..."},
#       ...
#   ]
# }
#
# 特别处理:
#
# 1. 原始数据可能以 assistant 开头
#    -> 丢弃第一个 user 之前的 assistant
#
# 2. 连续 user
#    -> 合并
#
# 3. 连续 assistant
#    -> 合并
#
# 4. Vanilla / dspark2 共用同一个转换结果
#    -> 保证 benchmark 输入完全一致
#
# ============================================================
# echo
# echo "============================================================"
# echo "Preparing EvalScope multi-turn dataset"
# echo "============================================================"

# python3 - "$RAW_DATASET" "$DATASET" "$NUM_CONVERSATIONS" <<'PY'

# import json
# import sys
# import os

# src = sys.argv[1]
# dst = sys.argv[2]
# limit = int(sys.argv[3])

# written = 0
# total = 0
# skipped = 0

# os.makedirs(os.path.dirname(dst), exist_ok=True)

# with open(src, "r", encoding="utf-8") as fin, \
#      open(dst, "w", encoding="utf-8") as fout:

#     for line_no, line in enumerate(fin, 1):

#         if written >= limit:
#             break

#         line = line.strip()

#         if not line:
#             continue

#         total += 1

#         # ----------------------------------------------------
#         # Parse JSON
#         # ----------------------------------------------------

#         try:
#             item = json.loads(line)
#         except Exception as e:
#             print(
#                 f"[WARN] line {line_no}: invalid JSON: {e}",
#                 file=sys.stderr
#             )
#             skipped += 1
#             continue

#         conversations = item.get("conversations")

#         if not isinstance(conversations, list):
#             print(
#                 f"[WARN] line {line_no}: invalid conversations",
#                 file=sys.stderr
#             )
#             skipped += 1
#             continue

#         output = []
#         started = False

#         # ----------------------------------------------------
#         # Convert to EvalScope format
#         # ----------------------------------------------------

#         for msg in conversations:

#             if not isinstance(msg, dict):
#                 continue

#             role = msg.get("role")
#             content = msg.get("content")

#             if role not in ("user", "assistant"):
#                 continue

#             if not isinstance(content, str):
#                 continue

#             if not content.strip():
#                 continue

#             # ------------------------------------------------
#             # 丢掉第一个 user 之前的 assistant
#             # ------------------------------------------------

#             if not started:

#                 if role != "user":
#                     continue

#                 started = True

#             # ------------------------------------------------
#             # EvalScope plugin format:
#             #
#             # user      -> {"human": "..."}
#             # assistant -> {"assistant": "..."}
#             # ------------------------------------------------

#             if role == "user":
#                 current = {"human": content}
#             else:
#                 current = {"assistant": content}

#             # ------------------------------------------------
#             # 连续相同角色合并
#             # ------------------------------------------------

#             if output:

#                 last_role = (
#                     "human"
#                     if "human" in output[-1]
#                     else "assistant"
#                 )

#                 current_role = (
#                     "human"
#                     if "human" in current
#                     else "assistant"
#                 )

#                 if last_role == current_role:

#                     output[-1][current_role] += (
#                         "\n\n" + content
#                     )

#                     continue

#             output.append(current)

#         # ----------------------------------------------------
#         # Validation
#         # ----------------------------------------------------

#         if not output:
#             skipped += 1
#             continue

#         # 必须从 human 开始
#         if "human" not in output[0]:
#             skipped += 1
#             continue

#         # 至少一个 human
#         if not any("human" in x for x in output):
#             skipped += 1
#             continue

#         # ----------------------------------------------------
#         # Write
#         # ----------------------------------------------------

#         result = {
#             "conversation": output
#         }

#         fout.write(
#             json.dumps(
#                 result,
#                 ensure_ascii=False
#             ) + "\n"
#         )

#         written += 1


# print()
# print("============================================================")
# print("EvalScope dataset conversion finished")
# print("============================================================")
# print(f"Input conversations  : {total}")
# print(f"Output conversations : {written}")
# print(f"Skipped              : {skipped}")
# print(f"Requested            : {limit}")
# print(f"Output file          : {dst}")
# print("============================================================")

# if written == 0:
#     raise RuntimeError(
#         "No valid conversations were generated."
#     )

# if written < limit:
#     print(
#         f"[WARN] Requested {limit} conversations, "
#         f"but only {written} valid conversations were generated."
#     )

# PY

# # ============================================================
# # Verify converted dataset
# # ============================================================

# echo
# echo "============================================================"
# echo "Verifying converted dataset"
# echo "============================================================"

# echo
# echo "First converted sample:"
# head -n 1 "$DATASET"

# echo
# echo "Number of converted conversations:"
# wc -l "$DATASET"


# # ============================================================
# # Detailed dataset validation
# # ============================================================
# python3 - "$DATASET" <<'PY'

# import json
# import sys

# path = sys.argv[1]

# count = 0
# total_turns = 0

# with open(path, "r", encoding="utf-8") as f:

#     for line_no, line in enumerate(f, 1):

#         item = json.loads(line)

#         conversation = item.get("conversation")

#         if not conversation:
#             raise RuntimeError(
#                 f"Line {line_no}: empty conversation"
#             )

#         # 必须 human 开头
#         if "human" not in conversation[0]:
#             raise RuntimeError(
#                 f"Line {line_no}: first message is not human"
#             )

#         for msg in conversation:

#             if not isinstance(msg, dict):
#                 raise RuntimeError(
#                     f"Line {line_no}: invalid message"
#                 )

#             if len(msg) != 1:
#                 raise RuntimeError(
#                     f"Line {line_no}: message must contain exactly one role"
#                 )

#             role = next(iter(msg))

#             if role not in ("human", "assistant"):
#                 raise RuntimeError(
#                     f"Line {line_no}: invalid role {role}"
#                 )

#             if not isinstance(msg[role], str):
#                 raise RuntimeError(
#                     f"Line {line_no}: message content is not string"
#                 )

#         count += 1

#         total_turns += sum(
#             1 for x in conversation
#             if "human" in x
#         )

# print()
# print("Dataset validation OK.")
# print("Conversations :", count)
# print("User turns    :", total_turns)

# if count == 0:
#     raise RuntimeError("Dataset is empty.")

# PY


# ============================================================
# vLLM server helpers
# ============================================================

SERVER_PID=""

CURRENT_SERVER_LOG=""


stop_server() {

    echo
    echo "Stopping vLLM..."

    if [ -n "${SERVER_PID:-}" ]; then

        kill "$SERVER_PID" 2>/dev/null || true

        wait "$SERVER_PID" 2>/dev/null || true

        SERVER_PID=""

    fi

    # 等 GPU / port 完全释放
    sleep 5
}


wait_server() {

    echo
    echo "Waiting for vLLM..."

    for i in $(seq 1 180); do

        if curl -sf \
            "http://localhost:${PORT}/v1/models" \
            >/dev/null 2>&1
        then

            echo
            echo "vLLM is ready."

            echo
            echo "Models:"

            curl -s \
                "http://localhost:${PORT}/v1/models"

            echo

            return 0
        fi

        sleep 2

    done


    echo
    echo "ERROR: vLLM failed to start."

    echo
    echo "============================================================"
    echo "Last 100 lines of server log"
    echo "============================================================"

    tail -n 100 "$CURRENT_SERVER_LOG" || true

    return 1
}


# ============================================================
# EvalScope performance runner
# ============================================================

run_perf() {

    local NAME="$1"

    echo
    echo "============================================================"
    echo "EvalScope Performance: $NAME"
    echo "============================================================"

    mkdir -p "$RESULT_DIR/$NAME"


    echo
    echo "Benchmark configuration:"
    echo "  dataset      = $DATASET"
    echo "  conversations= $NUM_CONVERSATIONS"
    echo "  parallel     = $PARALLEL"
    echo "  max_tokens   = $MAX_NEW_TOKENS"
    echo "  temperature  = $TEMPERATURE"
    echo "  stream       = true"
    echo "  multi-turn   = true"


    evalscope perf \
        --model "qwen3.8-27b" \
        --url "http://localhost:${PORT}/v1/chat/completions" \
        --api openai \
        --dataset share_gpt_en_multi_turn \
        --dataset-path "$DATASET" \
        --number "$NUM_CONVERSATIONS" \
        --parallel "$PARALLEL" \
        --max-tokens "$MAX_NEW_TOKENS" \
        --temperature "$TEMPERATURE" \
        --multi-turn \
        --stream \
        --outputs-dir "$RESULT_DIR/$NAME" \
        --no-timestamp


    echo
    echo "============================================================"
    echo "EvalScope $NAME finished"
    echo "============================================================"

}


# ============================================================
# Cleanup on exit
# ============================================================

cleanup() {

    echo
    echo "Cleanup..."

    if [ -n "${SERVER_PID:-}" ]; then

        kill "$SERVER_PID" 2>/dev/null || true

        wait "$SERVER_PID" 2>/dev/null || true

        SERVER_PID=""

    fi
}

trap cleanup EXIT


# ============================================================
# 1. VANILLA
# ============================================================

# echo
# echo "############################################################"
# echo "# 1. VANILLA"
# echo "############################################################"


# stop_server


# CURRENT_SERVER_LOG="$RESULT_DIR/vanilla_server.log"


# echo
# echo "Starting Vanilla vLLM..."

# CUDA_VISIBLE_DEVICES="$GPU" \
# vllm serve "$MODEL" \
#     --served-model-name qwen3.8-27b \
#     --tensor-parallel-size 1 \
#     --port "$PORT" \
#     --max-model-len 8192 \
#     --gpu-memory-utilization 0.90 \
#     --enforce-eager \
#     > "$CURRENT_SERVER_LOG" 2>&1 &

# SERVER_PID=$!


# wait_server


# run_perf "vanilla"


# stop_server


# ============================================================
# 2. dspark2
# ============================================================

echo
echo "############################################################"
echo "# 2. dspark2"
echo "############################################################"


CURRENT_SERVER_LOG="$RESULT_DIR/dspark2_server.log"


echo
echo "Starting dspark2 vLLM..."

CUDA_VISIBLE_DEVICES="$GPU" \
vllm serve "$MODEL" \
    --served-model-name qwen3.8-27b \
    --tensor-parallel-size 1 \
    --port "$PORT" \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --enforce-eager \
    --speculative-config \
    "{\"model\":\"$DRAFT\",\"num_speculative_tokens\":3}" \
    > "$CURRENT_SERVER_LOG" 2>&1 &

SERVER_PID=$!


wait_server


run_perf "dspark2"


stop_server


# ============================================================
# Final
# ============================================================

echo
echo "############################################################"
echo "# DONE"
echo "############################################################"

echo
echo "Results:"
echo
echo "  Vanilla:"
echo "    $RESULT_DIR/vanilla"
echo
echo "  dspark2:"
echo "    $RESULT_DIR/dspark2"

echo
echo "Server logs:"
echo
echo "  Vanilla:"
echo "    $RESULT_DIR/vanilla_server.log"
echo
echo "  dspark2:"
echo "    $RESULT_DIR/dspark2_server.log"

echo
echo "Converted dataset:"
echo
echo "  $DATASET"

echo
echo "============================================================"
echo "Benchmark finished."
echo "============================================================"
