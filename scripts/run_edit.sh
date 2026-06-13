#!/bin/bash
# R3Bench Step 2: Run Image Editing Task
#
# Usage: bash run_edit.sh [NUM_GPUS] [VERIFIER_NAME] [EDITOR_NAME]
#
# EDITOR_NAME defaults to qwen_image_2511.

set -e

# Load common configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

# Parameters
NUM_GPUS=${1:-8}
VERIFIER_NAME=${2:-qwen2.5vl}
EDITOR_NAME=${3:-qwen_image_2511}

# torchrun launch parameters
MASTER_PORT=${MASTER_PORT:-12339}
MASTER_ADDR=${MASTER_ADDR:-"localhost"}

R3BENCH_DIR="$(dirname "$SCRIPT_DIR")"
BENCHMARK_DIR="${R3BENCH_DATA_DIR:-${R3BENCH_DIR}/benchmark_data}"

echo "=========================================="
echo "R3Bench Step 2: Image Editing Task"
echo "=========================================="
echo "Verifier: ${VERIFIER_NAME}"
echo "Editor: ${EDITOR_NAME}"
echo "GPUs: ${NUM_GPUS}"

# Get model path
MODEL_PATH=$(get_editor_path "$EDITOR_NAME") || exit 1
echo "Model path: ${MODEL_PATH}"

# Set input/output paths
INPUT_JSONL="${R3BENCH_DIR}/output/reflection/${VERIFIER_NAME}/reflection_res.jsonl"
OUTPUT_DIR="${R3BENCH_DIR}/output/edit/${EDITOR_NAME}/${VERIFIER_NAME}"

# Check input file
check_file_exists "$INPUT_JSONL" "Reflection result file not found"

# Create output directory
mkdir -p "$OUTPUT_DIR"
LOG_FILE="${OUTPUT_DIR}/$(date +%Y%m%d_%H%M%S)_edit.log"

echo "Input: ${INPUT_JSONL}"
echo "Output: ${OUTPUT_DIR}"
echo "Log: ${LOG_FILE}"
echo "=========================================="

# Run editing task.
# Launch via ``-m`` from the repo root so ``import r3bench`` resolves via cwd
# even when the package is not pip-installed in the active environment.
cd "$R3BENCH_DIR"
torchrun --nproc_per_node=${NUM_GPUS} \
    --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT} \
    -m r3bench.step2_edit \
    --input-jsonl ${INPUT_JSONL} \
    --output-dir ${OUTPUT_DIR} \
    --benchmark-dir ${BENCHMARK_DIR} \
    --model-path ${MODEL_PATH} \
    --editor-name ${EDITOR_NAME} 2>&1 | tee ${LOG_FILE}

echo "=========================================="
echo "Editing task completed."
echo "Output: ${OUTPUT_DIR}"
echo "=========================================="
