#!/bin/bash
# R3Bench Step 1: Run Reflection Task
#
# Usage: bash run_reflection.sh [NUM_GPUS] [MODEL_TYPE] [PROMPT_STYLE]
#
# MODEL_TYPE defaults to qwen2.5vl.

set -e

# Load common configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

# Parameters
NUM_GPUS=${1:-8}
MODEL_TYPE=${2:-qwen2.5vl}
# Prompt style: 'default' (benchmark standard prompt)
# or 'refiner' (the released R3-Refiner model's training format).
PROMPT_STYLE=${3:-default}

# Use data version configuration
R3BENCH_DIR="$(dirname "$SCRIPT_DIR")"
INPUT_FILE="${R3BENCH_DIR}/r3bench/data/${DATA_FILE}"
BENCHMARK_DIR="${R3BENCH_DATA_DIR:-${R3BENCH_DIR}/benchmark_data}"

echo "=========================================="
echo "R3Bench Step 1: Reflection Task"
echo "=========================================="
echo "Model: ${MODEL_TYPE}"
echo "Prompt style: ${PROMPT_STYLE}"
echo "GPUs: ${NUM_GPUS}"

# Get model path
MODEL_PATH=$(get_verifier_path "$MODEL_TYPE") || exit 1
echo "Model path: ${MODEL_PATH}"

# Set output directory
OUTPUT_DIR="${R3BENCH_DIR}/output/reflection/${MODEL_TYPE}"
mkdir -p "$OUTPUT_DIR"
FINAL_OUTPUT_FILE="${OUTPUT_DIR}/reflection_res.jsonl"
LOG_FILE="${OUTPUT_DIR}/$(date +%Y%m%d_%H%M%S)_reflection.log"

echo "Input: ${INPUT_FILE}"
echo "Output: ${FINAL_OUTPUT_FILE}"
echo "Log: ${LOG_FILE}"
echo "=========================================="

# Run reflection task.
# Launch via ``-m`` from the repo root so ``import r3bench`` resolves via cwd
# even when the package is not pip-installed in the active environment.
cd "$R3BENCH_DIR"
torchrun \
    --nnodes=1 --node_rank=0 --nproc_per_node=$NUM_GPUS \
    --master_addr=127.0.0.1 --master_port=12346 \
    -m r3bench.step1_reflection \
    --model-type "$MODEL_TYPE" \
    --input-jsonl "$INPUT_FILE" \
    --output-jsonl "$FINAL_OUTPUT_FILE" \
    --benchmark-dir "$BENCHMARK_DIR" \
    --model-path "$MODEL_PATH" \
    --prompt "$PROMPT_STYLE" 2>&1 | tee ${LOG_FILE}

echo "=========================================="
echo "Reflection task completed."
echo "Output: ${FINAL_OUTPUT_FILE}"
echo "=========================================="
