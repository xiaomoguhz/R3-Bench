#!/bin/bash
# R3Bench Common Configuration
# Shared functions and configurations for all scripts

# ============================================================================
# Data Configuration
# ============================================================================
# R3Bench: 670 high-quality samples for closed-loop evaluation
# - Covers 8 error dimensions: color, object, numeracy, spatial, shape, texture, complex, non

DATA_FILE="r3bench.jsonl"

# ============================================================================
# Model Path Configuration
# ============================================================================

# Verifier model paths (for reflection task - Step 1)
declare -A VERIFIER_PATHS=(
    ["qwen2.5vl"]="${R3BENCH_QWEN2_5VL_PATH:-Qwen/Qwen2.5-VL-7B-Instruct}"
    ["qwen3vl"]="${R3BENCH_QWEN3VL_PATH:-Qwen/Qwen3-VL-8B-Instruct}"
    ["bagel"]="ByteDance-Seed/BAGEL-7B-MoT"
    ["gpt"]="N/A"
)

# Editor model paths (for editing task - Step 2)
declare -A EDITOR_PATHS=(
    ["qwen_image_2511"]="${R3BENCH_QWEN_IMAGE_2511_PATH:-Qwen/Qwen-Image-Edit-2511}"
    ["bagel"]="ByteDance-Seed/BAGEL-7B-MoT"
    ["gpt_image"]="gpt-image-1"
)

# ============================================================================
# Helper Functions
# ============================================================================

# Get verifier model path
get_verifier_path() {
    local model_name="$1"
    local model_path="${VERIFIER_PATHS[$model_name]}"

    if [ -z "$model_path" ]; then
        echo "Error: Invalid verifier '$model_name'" >&2
        echo "Available options: ${!VERIFIER_PATHS[@]}" >&2
        return 1
    fi

    echo "$model_path"
}

# Get editor model path
get_editor_path() {
    local model_name="$1"
    local model_path="${EDITOR_PATHS[$model_name]}"

    if [ -z "$model_path" ]; then
        echo "Error: Invalid editor '$model_name'" >&2
        echo "Available options: ${!EDITOR_PATHS[@]}" >&2
        return 1
    fi

    echo "$model_path"
}

# Generate vLLM API URL list
generate_api_urls() {
    local base_port=${1:-8000}
    local num_instances=${2:-8}
    local host=${3:-127.0.0.1}
    local urls=()

    for (( i=0; i<$num_instances; i++ )); do
        local port=$((base_port + i))
        urls+=("http://$host:$port")
    done

    echo "${urls[@]}"
}

# Check if file exists, exit if not
check_file_exists() {
    local file="$1"
    local error_msg="${2:-File not found}"

    if [ ! -f "$file" ]; then
        echo "Error: $error_msg: $file" >&2
        exit 1
    fi
}
