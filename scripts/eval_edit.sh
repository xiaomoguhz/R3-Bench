#!/bin/bash
# R3Bench Step 4: Evaluate Image Editing Results
#
# Usage: bash eval_edit.sh [VERIFIER_NAME] [EDITOR_NAME]
#
# This script evaluates edited images using QA-based evaluation,
# computing the Rectification Score (S_rect).
#
# Prerequisites:
#   - Start a vLLM service for the vision-language judge (we use Qwen3-VL-235B-A22B-Instruct)
#   - Default port: 8000

set -e

# Load common configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

# Parameters
VERIFIER_NAME=${1:-qwen2.5vl}
EDITOR_NAME=${2:-qwen_image_2511}

# vLLM service configuration
BASE_PORT=${BASE_PORT:-8000}
NUM_INSTANCES=${NUM_INSTANCES:-1}
HOST="127.0.0.1"
NUM_WORKERS=32

R3BENCH_DIR="$(dirname "$SCRIPT_DIR")"
BENCHMARK_DIR="${R3BENCH_DATA_DIR:-${R3BENCH_DIR}/benchmark_data}"

echo "=========================================="
echo "R3Bench Step 4: Evaluate Image Editing"
echo "=========================================="
echo "Verifier: ${VERIFIER_NAME}"
echo "Editor: ${EDITOR_NAME}"

REFLECTION_DIR="${R3BENCH_DIR}/output/reflection/${VERIFIER_NAME}"
EDIT_DIR="${R3BENCH_DIR}/output/edit/${EDITOR_NAME}/${VERIFIER_NAME}"

INPUT_FILE="${REFLECTION_DIR}/reflection_eval_res.jsonl"
OUTPUT_FILE="${EDIT_DIR}/image_eval_res.jsonl"
EDITED_IMAGES_DIR="${EDIT_DIR}"

check_file_exists "$INPUT_FILE" "Input file not found. Run step3 first."

# Generate API URL list
read -ra API_URLS <<< "$(generate_api_urls $BASE_PORT $NUM_INSTANCES $HOST)"

echo "Input: ${INPUT_FILE}"
echo "Output: ${OUTPUT_FILE}"
echo "Edited images: ${EDITED_IMAGES_DIR}"
echo "API endpoints: ${#API_URLS[@]}"
echo "Workers: ${NUM_WORKERS}"
echo "=========================================="

# Run evaluation script.
# Use ``-m`` from the repo root so ``import r3bench`` resolves via cwd —
# invoking the file by absolute path would set sys.path[0] to ``r3bench/``
# itself and miss the package.
cd "$R3BENCH_DIR"
python3 -m r3bench.step4_eval_edit \
    --backend qwen \
    --input-jsonl "$INPUT_FILE" \
    --output-jsonl "$OUTPUT_FILE" \
    --benchmark-dir "${BENCHMARK_DIR}" \
    --edited-images-dir "$EDITED_IMAGES_DIR" \
    --api-urls "${API_URLS[@]}" \
    --num-workers "$NUM_WORKERS"

echo "=========================================="
echo "Image editing evaluation completed."
echo "Output: ${OUTPUT_FILE}"
echo "=========================================="
