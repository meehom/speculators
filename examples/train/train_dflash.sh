#!/bin/bash
# Online Eagle3 Training Script (Explicitly specifying 3 layers)
#
# Runs the full online training pipeline: data preparation, vLLM server launch,
# and training (with hidden states generated on-the-fly from the live server).
#
# Usage: Copy this script, modify the configuration variables below, then run:
#    bash examples/train/eagle3_custom_layers.sh

set -euo pipefail

# ============ Configuration ============
MODEL="/home/w00934428/lmh/Qwen3.8-27B"
DATASET="./openorca_100k_processed.jsonl"  # 可替换为你自己的预处理数据或路径
OUTPUT_DIR="./output"
VLLM_PORT=8199
DRAFT_VOCAB_SIZE=248320
MAX_SAMPLES=10000
SEQ_LENGTH=8192
EPOCHS=7
LR=1e-4

# 显式指定的 3 个目标层 ID（根据你的目标模型层数调整，例如 Qwen3-8B 常见可选 4 16 28）
TARGET_LAYERS="2 32 61"
# A100 显存极大：27B 模型单卡跑 vLLM 绰绰有余
VLLM_GPUS="0"
NUM_VLLM_GPUS=1

# 训练使用剩余的 A100 显卡（例如用 2 卡进行分布式训练）
TRAIN_GPUS="2,3,4,5"
NUM_TRAIN_GPUS=4
# =======================================

# Step 2: 使用原汁原味的 vLLM 命令启动服务
# echo "=== Step 2: Launching vLLM server on A100 ==="
# CUDA_VISIBLE_DEVICES="$VLLM_GPUS" vllm serve "$MODEL" \
#     --tensor-parallel-size "$NUM_VLLM_GPUS" \
#     --port "$VLLM_PORT" \
#     --max-model-len 8192 \
#     --gpu-memory-utilization 0.85 &
# VLLM_PID=$!
# Step 2: 使用刚刚下载的专用脚本启动 vLLM
echo "=== Step 2: Launching vLLM server with speculators support ==="
# 强制使用旧版稳定引擎（V0），并适当调大显存利用率，减小模型最大长度
CUDA_VISIBLE_DEVICES="$VLLM_GPUS" python3 /home/w00934428/lmh/speculators-main/scripts/launch_vllm.py "$MODEL" \
    -- --tensor-parallel-size "$NUM_VLLM_GPUS" \
    --port "$VLLM_PORT" \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --renderer-num-workers 1 \
    --no-disable-hybrid-kv-cache-manager \
    --enforce-eager &

VLLM_PID=$!
# Ensure vLLM is cleaned up on exit
cleanup() {
    echo "Stopping vLLM server..."
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "Waiting for vLLM server to be ready..."
until curl -sf "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; do
    sleep 3
done
echo "vLLM server ready."

# Step 1: Prepare data
echo "=== Step 1: Preparing data ==="
speculators prepare-data \
    --model "$MODEL" \
    --data "$DATASET" \
    --output "$OUTPUT_DIR" \
    --max-samples "$MAX_SAMPLES" \
    --seq-length "$SEQ_LENGTH" \
    --render-endpoint "http://localhost:${VLLM_PORT}"

# Step 2: Launch vLLM server in the background (A100 单卡 80G 跑 27B 推理)
# echo "=== Step 2: Launching vLLM server on A100 ==="
# CUDA_VISIBLE_DEVICES="$VLLM_GPUS" python scripts/launch_vllm.py "$MODEL" \
#     -- --tensor-parallel-size "$NUM_VLLM_GPUS" --port "$VLLM_PORT" --gpu-memory-utilization 0.85 &
# VLLM_PID=$!

# Step 3: Train against the live vLLM server
echo "=== Step 3: Training with explicit layers ($TARGET_LAYERS) ==="
CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" torchrun \
    --standalone --nproc_per_node "$NUM_TRAIN_GPUS" \
    -m speculators.train \
    --speculator-type dflash2 \
    --verifier-name-or-path "$MODEL" \
    --data-path "$OUTPUT_DIR" \
    --vllm-endpoint "http://localhost:${VLLM_PORT}/v1" \
    --save-path "$OUTPUT_DIR/dflash2_checkpoints" \
    --draft-vocab-size "$DRAFT_VOCAB_SIZE" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --total-seq-len "$SEQ_LENGTH" \
    --on-missing generate \
    --on-generate delete \
    --draft-arch qwen3 \
    --num-layers 5 \
    --draft-hidden-act silu

echo "Done. Checkpoints saved to $OUTPUT_DIR/checkpoints/"
