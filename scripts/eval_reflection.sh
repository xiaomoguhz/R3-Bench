#!/bin/bash
# R3Bench Step 3: Evaluate Reflection Results
#
# Usage: bash eval_reflection.sh [MODEL_TYPE]
#
# This script evaluates the semantic equivalence between model-generated
# reflections and ground truth reflections using a text-only LLM.
#
# Prerequisites:
#   - Start a vLLM service for the text judge (we use Qwen3-Next-80B-A3B-Instruct)
#   - Default port: 8000

set -e

# Load common configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

# Parameters
MODEL_TYPE=${1:-qwen2.5vl}

# vLLM configuration
BASE_PORT=${BASE_PORT:-8000}
NUM_INSTANCES=${NUM_INSTANCES:-1}

R3BENCH_DIR="$(dirname "$SCRIPT_DIR")"
BENCHMARK_DIR="${R3BENCH_DATA_DIR:-${R3BENCH_DIR}/benchmark_data}"

echo "=========================================="
echo "R3Bench Step 3: Evaluate Reflection"
echo "=========================================="
echo "Model: ${MODEL_TYPE}"

INPUT_JSONL="${R3BENCH_DIR}/output/reflection/${MODEL_TYPE}/reflection_res.jsonl"
OUTPUT_JSONL="${R3BENCH_DIR}/output/reflection/${MODEL_TYPE}/reflection_eval_res.jsonl"

check_file_exists "$INPUT_JSONL" "Input file not found"

echo "Input: ${INPUT_JSONL}"
echo "Output: ${OUTPUT_JSONL}"

# Generate API URL list
read -ra API_URLS <<< "$(generate_api_urls $BASE_PORT $NUM_INSTANCES)"
echo "Using ${#API_URLS[@]} API endpoint(s) for evaluation..."
echo "=========================================="

# Run evaluation script.
# Use ``-m`` from the repo root so ``import r3bench`` resolves via cwd —
# invoking the file by absolute path would set sys.path[0] to ``r3bench/``
# itself and miss the package.
cd "$R3BENCH_DIR"
python3 -m r3bench.step3_eval_reflection \
    --backend qwen \
    --input-jsonl "$INPUT_JSONL" \
    --output-jsonl "$OUTPUT_JSONL" \
    --benchmark-dir "${BENCHMARK_DIR}" \
    --api-urls "${API_URLS[@]}" \
    --num-workers 32 \
    --temperature 0.1 \
    --top-p 0.9 \
    --max-tokens 1024

echo "=========================================="
echo "Reflection evaluation completed."
echo "Output: ${OUTPUT_JSONL}"
echo "=========================================="
