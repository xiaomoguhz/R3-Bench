#!/bin/bash
# Distributed service launcher

# Usage:
#   On the edit-service node:    ./deploy_services.sh edit_server
#   On the reward-service node:  ./deploy_services.sh reward_server
#   On the training node:        ./deploy_services.sh get_config

set -euo pipefail

# Resolve paths relative to this script.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${DISTRIBUTED_SERVICES_ROOT:-}" ]]; then
    DISTRIBUTED_SERVICES_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi

REPO_ROOT="$(cd "${DISTRIBUTED_SERVICES_ROOT}/.." && pwd)"
export PROJECT_ROOT="${PROJECT_ROOT:-${REPO_ROOT}}"
export LOG_DIR="${LOG_DIR:-${DISTRIBUTED_SERVICES_ROOT}/logs}"
export CONFIG_DIR="${CONFIG_DIR:-${DISTRIBUTED_SERVICES_ROOT}/config}"

SERVICE_TYPE=${1:-"help"}

# Load config variables from YAML and export them.
# The "help" sub-command does not depend on PyYAML, so users can view usage
# before installing dependencies.
if [[ "$SERVICE_TYPE" != "help" && "$SERVICE_TYPE" != "--help" && "$SERVICE_TYPE" != "-h" ]]; then
    CONFIG_PYTHON="${CONFIG_PYTHON:-}"
    if [[ -z "$CONFIG_PYTHON" ]]; then
        for candidate in python3 python; do
            if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c "import yaml" >/dev/null 2>&1; then
                CONFIG_PYTHON="$candidate"
                break
            fi
        done
    fi
    CONFIG_PYTHON="${CONFIG_PYTHON:-python3}"

    if ! config_exports="$("$CONFIG_PYTHON" "${DISTRIBUTED_SERVICES_ROOT}/config/load_config.py")"; then
        echo "[ERROR] Failed to load config with CONFIG_PYTHON=${CONFIG_PYTHON}." >&2
        echo "[ERROR] Install PyYAML in that environment or run with CONFIG_PYTHON=/path/to/python." >&2
        exit 1
    fi
    eval "${config_exports}"
fi

mkdir -p "${LOG_DIR:-${DISTRIBUTED_SERVICES_ROOT}/logs}"

# ============================================================================
# Runtime variables
# ============================================================================

# Service node labels used in status output.
EDIT_SERVER_NODE="${EDIT_SERVER_NODE:-$(hostname)}"
REWARD_SERVER_NODE="${REWARD_SERVER_NODE:-$(hostname)}"

# ============================================================================
# Constants
# ============================================================================
readonly MAX_EDIT_INSTANCES_PER_GPU=2
readonly MAX_REWARD_INSTANCES_PER_GPU=32
readonly SERVICE_KEEPALIVE_SECONDS=259200  # 72 hours

# ============================================================================
# Utility functions
# ============================================================================

# Return a node identifier suitable for log file names.
get_node_id() {
    local node_id=$(hostname | sed 's/[^a-zA-Z0-9]/_/g' | head -c 20)
    if [[ -z "$node_id" ]]; then
        node_id=$(hostname -I 2>/dev/null | awk '{print $1}' | tr -d '[:space:]' | sed 's/[^a-zA-Z0-9]/_/g' | head -c 20)
    fi
    echo "$node_id"
}

# Return the server's primary IP address.
get_server_address() {
    local server_addr=$(hostname -I 2>/dev/null | awk '{print $1}' | tr -d '[:space:]')
    if [[ -z "$server_addr" ]]; then
        server_addr=$(hostname)
        echo "[WARNING] Could not determine IP address; using hostname: $server_addr" >&2
    fi
    echo "$server_addr"
}

# Return the Python executable for the given Conda environment.
get_python_cmd() {
    local conda_env="${1:-}"
    local python_cmd="python3"

    if [[ -n "$conda_env" ]] && command -v conda &> /dev/null; then
        if conda env list | grep -q "^${conda_env} "; then
            local conda_base=$(conda info --base)
            python_cmd="${conda_base}/envs/${conda_env}/bin/python3"
            if [[ ! -f "$python_cmd" ]]; then
                python_cmd="python3"  # Use system python if the Conda path is missing.
            fi
        fi
    fi

    echo "$python_cmd"
}

# Parse instance info: gpu_id:instance_id:port:log_file[:reward_type].
parse_instance_info() {
    local info="$1"
    local -n parts_ref=$2
    IFS=':' read -ra parts_ref <<< "$info"
}

normalize_reward_type() {
    local reward_type
    reward_type="$(echo "${1:-self_reward}" | tr '[:upper:]' '[:lower:]' | tr '-' '_')"
    case "$reward_type" in
        self_reward|clip|sam3|mixed)
            echo "$reward_type"
            ;;
        *)
            echo "$reward_type"
            ;;
    esac
}

is_self_reward_type() {
    local reward_type
    reward_type="$(normalize_reward_type "$1")"
    [[ "$reward_type" == "self_reward" ]]
}

normalize_self_reward_model_type() {
    local model_type
    model_type="$(echo "${1:-qwen2_5vl}" | tr '[:upper:]' '[:lower:]' | tr '-' '_')"
    case "$model_type" in
        qwen3|qwen3vl|qwen3_vl)
            echo "qwen3vl"
            ;;
        qwen2_5|qwen2_5vl|qwen2.5|qwen2.5vl|qwen25|qwen25vl)
            echo "qwen2_5vl"
            ;;
        *)
            echo "$model_type"
            ;;
    esac
}

normalize_edit_model_type() {
    local model_type
    model_type="$(echo "${1:-bagel}" | tr '[:upper:]' '[:lower:]' | tr '-' '_')"
    case "$model_type" in
        qwen|qwen_image_edit)
            echo "qwen_image_edit"
            ;;
        bagel|bagel_edit)
            echo "bagel"
            ;;
        *)
            echo "$model_type"
            ;;
    esac
}

# Compute the total number of instances across all GPUs.
calculate_total_instances() {
    local num_gpus=$1
    local -n gpu_instances_ref=$2
    local total=0
    for gpu_id in $(seq 0 $((num_gpus - 1))); do
        total=$((total + ${gpu_instances_ref[$gpu_id]}))
    done
    echo "$total"
}

# Check service status and print a summary.
check_and_print_service_status() {
    local service_name="$1"
    local loading_msg="$2"
    local server_node="$3"
    local -n pids_ref=$4
    local -n instance_info_ref=$5

    echo ""
    echo "Service processes started."
    echo "   $loading_msg"
    echo ""
    echo "Follow loading progress in real time:"
    for info in "${instance_info_ref[@]}"; do
        local parts
        parse_instance_info "$info" "parts"
        local gpu_id="${parts[0]}"
        local instance_id="${parts[1]}"
        local log_file="${parts[3]}"
        local reward_type="${parts[4]:-}"
        if [[ -n "$reward_type" ]]; then
            echo "   tail -f ${log_file}  # GPU $gpu_id, instance $instance_id ($reward_type)"
        else
            echo "   tail -f ${log_file}  # GPU $gpu_id, instance $instance_id"
        fi
    done
    echo ""

    # Check whether processes are still alive
    echo "Checking service startup status..."
    local all_running=true
    for i in "${!pids_ref[@]}"; do
        local pid=${pids_ref[$i]}
        local info="${instance_info_ref[$i]}"
        local parts
        parse_instance_info "$info" "parts"
        local gpu_id="${parts[0]}"
        local instance_id="${parts[1]}"
        local log_file="${parts[3]}"

        sleep 3
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "  [FAILED] GPU $gpu_id instance $instance_id: process exited (PID: $pid)"
            echo "     Check log: $log_file"
            echo "     Last log lines:"
            tail -10 "$log_file" 2>/dev/null | sed 's/^/     /' || echo "     (log file is empty or missing)"
            all_running=false
        else
            echo "  [OK] GPU $gpu_id instance $instance_id: process running (PID: $pid)"
        fi
    done

    if [[ "$all_running" == "false" ]]; then
        echo ""
        echo "[WARNING] Some service instances failed to start; running instances remain available."
    fi

    echo ""
    echo "Running service processes are loading models."
    echo ""
    echo "Service endpoint list (available once the model finishes loading):"
    for info in "${instance_info_ref[@]}"; do
        local parts
        parse_instance_info "$info" "parts"
        local gpu_id="${parts[0]}"
        local instance_id="${parts[1]}"
        local port="${parts[2]}"
        local reward_type="${parts[4]:-}"
        if [[ -n "$reward_type" ]]; then
            echo "  - http://${server_node}:$port  (GPU $gpu_id, instance $instance_id, $reward_type)"
        else
            echo "  - http://${server_node}:$port  (GPU $gpu_id, instance $instance_id)"
        fi
    done
    echo ""
    echo "Check readiness with:"
    if [[ ${#instance_info_ref[@]} -gt 0 ]]; then
        local first_parts
        parse_instance_info "${instance_info_ref[0]}" "first_parts"
        local first_port="${first_parts[2]}"
        echo "   curl http://${server_node}:$first_port/health"
    fi
}

# ============================================================================
# Config-parsing functions
# ============================================================================

# Parse per-GPU instance-count configuration.
# Supported formats:
#   - Single integer: all GPUs use the same count, e.g. "2"
#   - Per-GPU spec:   "0:2,1:2,2:1,3:1" — GPU0 and GPU1 get 2 instances each,
#                     GPU2 and GPU3 get 1 instance each
parse_instances_per_gpu() {
    local instances_config="${INSTANCES_PER_GPU:-1}"
    local num_gpus=$1
    declare -gA gpu_instances  # global associative array

    # Single integer: every GPU gets the same instance count.
    if [[ "$instances_config" =~ ^[0-9]+$ ]]; then
        local instances_per_gpu=$instances_config
        for gpu_id in $(seq 0 $((num_gpus - 1))); do
            gpu_instances[$gpu_id]=$instances_per_gpu
        done
        echo "Each GPU will start $instances_per_gpu instance(s)"
    else
        # Per-GPU specification.
        IFS=',' read -ra entries <<< "$instances_config"
        for entry in "${entries[@]}"; do
            IFS=':' read -ra parts <<< "$entry"
            if [[ ${#parts[@]} -eq 2 ]]; then
                local gpu_idx="${parts[0]}"
                local instances="${parts[1]}"
                if [[ "$gpu_idx" -ge 0 && "$gpu_idx" -lt "$num_gpus" ]]; then
                    gpu_instances["$gpu_idx"]="$instances"
                fi
            fi
        done

        # Unspecified GPUs use one instance.
        for gpu_id in $(seq 0 $((num_gpus - 1))); do
            if [[ -z "${gpu_instances[$gpu_id]:-}" ]]; then
                gpu_instances[$gpu_id]=1
            fi
        done

        echo "Per-GPU instance counts:"
        for gpu_id in $(seq 0 $((num_gpus - 1))); do
            echo "  GPU $gpu_id: ${gpu_instances[$gpu_id]} instance(s)"
        done
    fi
}

# Start the image-editing service.
start_edit_server() {
    echo "========================================="
    echo "Starting image-editing service"
    echo "========================================="

    local model_path="${1:-${EDIT_MODEL_PATH:-/path/to/BAGEL-7B-MoT}}"
    local edit_model_type
    edit_model_type="$(normalize_edit_model_type "${2:-${EDIT_MODEL_TYPE:-bagel}}")"

    # GPU count is loaded from YAML or auto-detected by load_config.py.
    local num_gpus="${GPUS_PER_NODE:-0}"

    echo "Using $num_gpus GPU(s) (from YAML config or auto-detection)"

    # BAGEL runs one service instance per GPU.
    INSTANCES_PER_GPU=1

    # Parse per-GPU instance counts.
    parse_instances_per_gpu "$num_gpus"

    echo "Model path: $model_path"
    echo "Image-editing model type: $edit_model_type"

    # Calculate total instance count.
    local total_instances=$(calculate_total_instances "$num_gpus" "gpu_instances")
    echo "Total instances to start: $total_instances"

    # Resolve Python executable.
    local python_cmd=$(get_python_cmd "${EDIT_CONDA_ENV:-bagelfast}")
    local node_id=$(get_node_id)

    # Launch service instances.
    local pids=()
    local instance_info=()

    # Choose server script based on edit model type.
    local server_script=""
    case "$edit_model_type" in
        qwen_image_edit)
            server_script="${DISTRIBUTED_SERVICES_ROOT}/servers/qwen_image_edit_server.py"
            ;;
        bagel)
            server_script="${DISTRIBUTED_SERVICES_ROOT}/servers/bagel_edit_server.py"
            ;;
        *)
            echo "[ERROR] Unknown image-editing model type: $edit_model_type"
            echo "Supported EDIT_MODEL_TYPE values: bagel, qwen_image_edit"
            exit 1
            ;;
    esac

    for gpu_id in $(seq 0 $((num_gpus - 1))); do
        local instances_per_gpu=${gpu_instances[$gpu_id]}
        for instance_id in $(seq 0 $((instances_per_gpu - 1))); do
            local port=$((EDIT_SERVER_BASE_PORT + gpu_id * MAX_EDIT_INSTANCES_PER_GPU + instance_id))
            local log_file="${LOG_DIR}/edit_server_${node_id}_gpu${gpu_id}_inst${instance_id}.log"

            echo "Starting instance $instance_id/$((instances_per_gpu - 1)) on GPU $gpu_id, port: $port"

            CUDA_VISIBLE_DEVICES=$gpu_id nohup "$python_cmd" -u "$server_script" \
                --model_path "$model_path" \
                --port "$port" \
                --host "0.0.0.0" \
                > "$log_file" 2>&1 &

            pids+=($!)
            instance_info+=("$gpu_id:$instance_id:$port:$log_file")
            echo "  PID: ${pids[-1]}, log: $log_file"
            sleep 3
        done
        sleep 3
    done

    # Check and report service status.
    check_and_print_service_status "image-edit" "Model loading may take several minutes." \
        "$EDIT_SERVER_NODE" "pids" "instance_info"

    # Write endpoint file for the training node.
    generate_edit_server_config "$total_instances"
}

# Start reward services. Supports self_reward, CLIP, SAM3, and mixed per-GPU assignment.
start_reward_server() {
    echo "========================================="
    echo "Starting reward service"
    echo "========================================="

    # GPU count is loaded from YAML or auto-detected by load_config.py.
    local num_gpus="${GPUS_PER_NODE:-0}"

    if [[ "$num_gpus" -le 0 ]]; then
        echo "[ERROR] No GPUs detected." >&2
        echo "[ERROR] Please check:" >&2
        echo "  1. Set other.gpus_per_node in config.yaml" >&2
        echo "  2. Or ensure nvidia-smi is available and detects GPUs" >&2
        exit 1
    fi

    echo "Using $num_gpus GPU(s) (from YAML config or auto-detection)"

    local reward_type
    reward_type="$(normalize_reward_type "${REWARD_TYPE:-self_reward}")"
    local reward_type_per_gpu="${REWARD_TYPE_PER_GPU:-}"

    # Parse per-GPU reward-type assignments.
    declare -A gpu_reward_types
    if [[ -n "$reward_type_per_gpu" ]]; then
        echo "Per-GPU reward type detected: $reward_type_per_gpu"
        IFS=',' read -ra entries <<< "$reward_type_per_gpu"
        for entry in "${entries[@]}"; do
            IFS=':' read -ra parts <<< "$entry"
            if [[ ${#parts[@]} -eq 2 ]]; then
                local gpu_idx="${parts[0]}"
                local gpu_reward_type
                gpu_reward_type="$(normalize_reward_type "${parts[1]}")"
                gpu_reward_types["$gpu_idx"]="$gpu_reward_type"
                echo "  GPU $gpu_idx: $gpu_reward_type"
            fi
        done
    fi

    # If no per-GPU config was found, apply the default reward type to all GPUs.
    local has_any_gpu_config=false
    for gpu_id in $(seq 0 $((num_gpus - 1))); do
        if [[ -n "${gpu_reward_types[$gpu_id]:-}" ]]; then
            has_any_gpu_config=true
            break
        fi
    done

    if [[ "$has_any_gpu_config" == "false" ]]; then
        echo "No per-GPU reward type found; using default type for all GPUs: $reward_type"
        for gpu_id in $(seq 0 $((num_gpus - 1))); do
            gpu_reward_types["$gpu_id"]="$reward_type"
            echo "  GPU $gpu_id: $reward_type"
        done
    fi

    # Prepare service arguments.
    local sam3_bpe_path="${SAM3_BPE_PATH:-}"
    local sam3_ckpt_path="${SAM3_CKPT_PATH:-}"
    local sam3_metadata_jsonl="${SAM3_METADATA_JSONL:-}"
    local self_reward_model_path="${SELF_REWARD_MODEL_PATH:-${MODEL_PATH:-}}"
    local self_reward_model_type
    self_reward_model_type="$(normalize_self_reward_model_type "${SELF_REWARD_MODEL_TYPE:-qwen2_5vl}")"
    local clip_model_name="${1:-ViT-B/32}"  # First argument is the CLIP model name.

    # Validate required arguments.
    if [[ -n "$reward_type_per_gpu" ]]; then
        # Mixed mode: check requirements for each type in use.
        local has_sam3=false
        local has_clip=false
        local has_self_reward=false

        for gpu_id in $(seq 0 $((num_gpus - 1))); do
            local gpu_reward_type="${gpu_reward_types[$gpu_id]:-$reward_type}"
            if [[ "$gpu_reward_type" == "sam3" ]]; then
                has_sam3=true
            elif [[ "$gpu_reward_type" == "clip" ]]; then
                has_clip=true
            elif is_self_reward_type "$gpu_reward_type"; then
                has_self_reward=true
            fi
        done

        if [[ "$has_sam3" == "true" ]]; then
            if [[ -z "$sam3_bpe_path" || -z "$sam3_ckpt_path" ]]; then
                echo "[ERROR] SAM3 service requires SAM3_BPE_PATH and SAM3_CKPT_PATH to be set."
                exit 1
            fi
        fi

        if [[ "$has_self_reward" == "true" ]]; then
            if [[ -z "$self_reward_model_path" || ! -d "$self_reward_model_path" ]]; then
                echo "[ERROR] R3-Refiner self-reward model path does not exist or is not set: $self_reward_model_path"
                echo "Set SELF_REWARD_MODEL_PATH or configure service.self_reward_model_path in config.yaml"
                exit 1
            fi
            case "$self_reward_model_type" in
                qwen2_5vl|qwen3vl) ;;
                *)
                    echo "[ERROR] Unknown SELF_REWARD_MODEL_TYPE: $self_reward_model_type"
                    echo "Supported SELF_REWARD_MODEL_TYPE values: qwen2_5vl, qwen3vl"
                    exit 1
                    ;;
            esac
        fi
    else
        # Single-type mode: validate based on reward_type.
        if is_self_reward_type "$reward_type"; then
            if [[ -z "$self_reward_model_path" || ! -d "$self_reward_model_path" ]]; then
                echo "[ERROR] R3-Refiner self-reward model path does not exist or is not set: $self_reward_model_path"
                echo "Set SELF_REWARD_MODEL_PATH or configure service.self_reward_model_path in config.yaml"
                exit 1
            fi
            case "$self_reward_model_type" in
                qwen2_5vl|qwen3vl) ;;
                *)
                    echo "[ERROR] Unknown SELF_REWARD_MODEL_TYPE: $self_reward_model_type"
                    echo "Supported SELF_REWARD_MODEL_TYPE values: qwen2_5vl, qwen3vl"
                    exit 1
                    ;;
            esac
        elif [[ "$reward_type" == "sam3" ]]; then
            if [[ -z "$sam3_bpe_path" || -z "$sam3_ckpt_path" ]]; then
                echo "[ERROR] SAM3 service requires SAM3_BPE_PATH and SAM3_CKPT_PATH to be set."
                exit 1
            fi
        fi
    fi

    # Parse per-GPU instance counts.
    parse_instances_per_gpu "$num_gpus"

    # Calculate total instance count.
    local total_instances=$(calculate_total_instances "$num_gpus" "gpu_instances")
    echo "Total instances to start: $total_instances"

    mkdir -p "${LOG_DIR}"

    # Launch service instances.
    local pids=()
    local reward_instance_info=()
    local node_id=$(get_node_id)
    local python_cmd=$(get_python_cmd "${REWARD_CONDA_ENV:-}")

    for gpu_id in $(seq 0 $((num_gpus - 1))); do
        local gpu_reward_type="${gpu_reward_types[$gpu_id]:-$reward_type}"
        local instances_per_gpu=${gpu_instances[$gpu_id]}

        # Choose the server script and base port based on reward type.
        local server_script=""
        local base_port=""
        case "$gpu_reward_type" in
            sam3)
                server_script="${DISTRIBUTED_SERVICES_ROOT}/servers/sam3_reward_server.py"
                base_port="${SAM3_REWARD_SERVER_BASE_PORT:-7001}"
                ;;
            self_reward)
                case "$self_reward_model_type" in
                    qwen2_5vl)
                        server_script="${DISTRIBUTED_SERVICES_ROOT}/servers/self_reward_qwen2_5vl_server.py"
                        ;;
                    qwen3vl)
                        server_script="${DISTRIBUTED_SERVICES_ROOT}/servers/self_reward_qwen3vl_server.py"
                        ;;
                    *)
                        echo "[ERROR] Unknown SELF_REWARD_MODEL_TYPE: $self_reward_model_type"
                        exit 1
                        ;;
                esac
                base_port="${REWARD_SERVER_BASE_PORT:-6001}"
                ;;
            *)
                server_script="${DISTRIBUTED_SERVICES_ROOT}/servers/clip_reward_server.py"
                base_port="${REWARD_SERVER_BASE_PORT:-6001}"
                ;;
        esac

        # Start the configured instances on this GPU.
        for instance_id in $(seq 0 $((instances_per_gpu - 1))); do
            local port=$((base_port + gpu_id * MAX_REWARD_INSTANCES_PER_GPU + instance_id))
            local log_suffix="$gpu_reward_type"
            if [[ "$gpu_reward_type" == "self_reward" ]]; then
                log_suffix="${gpu_reward_type}_${self_reward_model_type}"
            fi
            local log_file="${LOG_DIR}/reward_server_${node_id}_gpu${gpu_id}_inst${instance_id}_${log_suffix}.log"

            echo "Starting instance $instance_id/$((instances_per_gpu - 1)) on GPU $gpu_id (type: $log_suffix), port: $port"

            case "$gpu_reward_type" in
                self_reward)
                    CUDA_VISIBLE_DEVICES=$gpu_id nohup "$python_cmd" -u "$server_script" \
                        --model_path "$self_reward_model_path" \
                        --port "$port" \
                        --host "0.0.0.0" \
                        --device "$gpu_id" \
                        > "$log_file" 2>&1 &
                    ;;
                sam3)
                    CUDA_VISIBLE_DEVICES=$gpu_id nohup "$python_cmd" -u "$server_script" \
                        --bpe_path "$sam3_bpe_path" \
                        --ckpt_path "$sam3_ckpt_path" \
                        --metadata_jsonl "$sam3_metadata_jsonl" \
                        --port "$port" \
                        --host "0.0.0.0" \
                        --device "$gpu_id" \
                        > "$log_file" 2>&1 &
                    ;;
                *)
                    local cmd_args=(
                        --model_name "$clip_model_name"
                        --port "$port"
                        --host "0.0.0.0"
                        --device "$gpu_id"
                    )
                    [[ -n "${CLIP_MODULE_PATH}" && -f "${CLIP_MODULE_PATH}" ]] && \
                        cmd_args+=(--clip_module_path "${CLIP_MODULE_PATH}")
                    [[ -n "${CLIP_MODEL_PATH}" && -f "${CLIP_MODEL_PATH}" ]] && \
                        cmd_args+=(--model_path "${CLIP_MODEL_PATH}")

                    CUDA_VISIBLE_DEVICES=$gpu_id nohup "$python_cmd" -u "$server_script" \
                        "${cmd_args[@]}" \
                        > "$log_file" 2>&1 &
                    ;;
            esac

            pids+=($!)
            reward_instance_info+=("$gpu_id:$instance_id:$port:$log_file:$gpu_reward_type")
            echo "  PID: ${pids[-1]}, log: $log_file"
            sleep 1
        done
        sleep 1
    done

    # Check and report service status.
    check_and_print_service_status "reward-service" "Model loading may take several minutes." \
        "$REWARD_SERVER_NODE" "pids" "reward_instance_info"

    # Write endpoint files for the training node.
    generate_reward_server_config "$total_instances" "${reward_instance_info[@]}"
}

# Read existing endpoints into an associative array.
read_existing_endpoints() {
    local config_file="$1"
    local -n endpoints_ref=$2
    if [[ -f "$config_file" ]]; then
        while IFS= read -r line || [[ -n "$line" ]]; do
            line=$(echo "$line" | tr -d '[:space:]')
            [[ -n "$line" ]] && endpoints_ref["$line"]=1
        done < "$config_file"
    fi
}

# Append an endpoint to a config file if it is not already present.
append_endpoint_to_file() {
    local config_file="$1"
    local endpoint="$2"
    local existing_array_name="$3"
    local count_var_name="$4"

    # The associative-array name is passed by string for shell compatibility.
    local exists=0
    eval "exists=\${${existing_array_name}[\"$endpoint\"]:-0}"

    if [[ $exists -eq 0 ]]; then
        echo "$endpoint" >> "$config_file"
        eval "${existing_array_name}[\"$endpoint\"]=1"
        eval "${count_var_name}=\$((\${${count_var_name}} + 1))"
        return 0
    else
        echo "[INFO] Endpoint already exists, skipping: $endpoint" >&2
        return 1
    fi
}

# Write the image-editing endpoint file.
generate_edit_server_config() {
    local num_instances=$1
    local config_file="${CONFIG_DIR}/edit_server_endpoints.txt"

    mkdir -p "${CONFIG_DIR}"
    local server_addr=$(get_server_address)
    echo "[INFO] Using IP address: $server_addr"

    # Load existing endpoints for deduplication.
    declare -A existing_endpoints
    read_existing_endpoints "$config_file" "existing_endpoints"

    local added_count=0

    # Prefer the instance list produced by the current launch.
    if [[ -n "${instance_info[@]:-}" ]]; then
        for info in "${instance_info[@]}"; do
            local parts
            parse_instance_info "$info" "parts"
            local port="${parts[2]}"
            local endpoint="http://${server_addr}:$port"
            append_endpoint_to_file "$config_file" "$endpoint" "existing_endpoints" "added_count" >/dev/null
        done
    else
        # Reconstruct endpoint ports when instance_info is unavailable.
        for gpu_id in $(seq 0 $((num_instances - 1))); do
            local endpoint="http://${server_addr}:$((EDIT_SERVER_BASE_PORT + gpu_id))"
            append_endpoint_to_file "$config_file" "$endpoint" "existing_endpoints" "added_count" >/dev/null
        done
    fi

    if [[ $added_count -gt 0 ]]; then
        echo "Config file updated: $config_file ($added_count new endpoint(s) appended)"
    else
        echo "Config file unchanged: $config_file (all endpoints already present)"
    fi
}

# Write reward endpoint files.
generate_reward_server_config() {
    local num_instances=$1
    shift
    local reward_instance_info=("$@")

    local config_file="${CONFIG_DIR}/reward_server_endpoints.txt"
    local clip_config_file="${CONFIG_DIR}/clip_reward_server_endpoints.txt"
    local self_reward_config_file="${CONFIG_DIR}/self_reward_server_endpoints.txt"
    local sam3_config_file="${CONFIG_DIR}/sam3_reward_server_endpoints.txt"

    mkdir -p "${CONFIG_DIR}"
    local server_addr=$(get_server_address)
    echo "[INFO] Using IP address: $server_addr"

    # Load existing endpoints for deduplication.
    declare -A existing_endpoints
    declare -A existing_clip_endpoints
    declare -A existing_self_reward_endpoints
    declare -A existing_sam3_endpoints

    read_existing_endpoints "$config_file" "existing_endpoints"
    read_existing_endpoints "$clip_config_file" "existing_clip_endpoints"
    read_existing_endpoints "$self_reward_config_file" "existing_self_reward_endpoints"
    read_existing_endpoints "$sam3_config_file" "existing_sam3_endpoints"

    # Append endpoints.
    local added_count=0
    local added_clip_count=0
    local added_self_reward_count=0
    local added_sam3_count=0

    if [[ ${#reward_instance_info[@]} -gt 0 ]]; then
        for info in "${reward_instance_info[@]}"; do
            local parts
            parse_instance_info "$info" "parts"
            local port="${parts[2]}"
            local reward_type="${parts[4]:-clip}"
            local endpoint="http://${server_addr}:$port"

            # Select the per-type endpoint file.
            local target_file=""
            local existing_array_name="existing_clip_endpoints"
            local count_var_name="added_clip_count"

            case "$reward_type" in
                sam3)
                    target_file="$sam3_config_file"
                    existing_array_name="existing_sam3_endpoints"
                    count_var_name="added_sam3_count"
                    ;;
                self_reward)
                    target_file="$self_reward_config_file"
                    existing_array_name="existing_self_reward_endpoints"
                    count_var_name="added_self_reward_count"
                    ;;
                *)
                    target_file="$clip_config_file"
                    existing_array_name="existing_clip_endpoints"
                    count_var_name="added_clip_count"
                    ;;
            esac

            if append_endpoint_to_file "$target_file" "$endpoint" "$existing_array_name" "$count_var_name"; then
                added_count=$((added_count + 1))
                append_endpoint_to_file "$config_file" "$endpoint" "existing_endpoints" "added_count" >/dev/null
            fi
        done
    else
        # Reconstruct endpoint ports when instance_info is unavailable.
        local default_reward_type
        default_reward_type="$(normalize_reward_type "${REWARD_TYPE:-self_reward}")"
        local target_file=""
        local existing_array_name="existing_clip_endpoints"
        local count_var_name="added_clip_count"
        case "$default_reward_type" in
            sam3)
                target_file="$sam3_config_file"
                existing_array_name="existing_sam3_endpoints"
                count_var_name="added_sam3_count"
                ;;
            self_reward)
                target_file="$self_reward_config_file"
                existing_array_name="existing_self_reward_endpoints"
                count_var_name="added_self_reward_count"
                ;;
            *)
                target_file="$clip_config_file"
                ;;
        esac

        for gpu_id in $(seq 0 $((num_instances - 1))); do
            local endpoint="http://${server_addr}:$((REWARD_SERVER_BASE_PORT + gpu_id))"
            append_endpoint_to_file "$config_file" "$endpoint" "existing_endpoints" "added_count" >/dev/null
            append_endpoint_to_file "$target_file" "$endpoint" "$existing_array_name" "$count_var_name" >/dev/null
        done
    fi

    # Count endpoints started on this node.
    local total_clip_count=0
    local total_self_reward_count=0
    local total_sam3_count=0
    for info in "${reward_instance_info[@]}"; do
        local parts
        parse_instance_info "$info" "parts"
        case "${parts[4]:-clip}" in
            sam3) total_sam3_count=$((total_sam3_count + 1)) ;;
            self_reward) total_self_reward_count=$((total_self_reward_count + 1)) ;;
            *) total_clip_count=$((total_clip_count + 1)) ;;
        esac
    done

    echo ""
    echo "========================================="
    echo "Config file generation results"
    echo "========================================="
    echo "Current node IP: $server_addr"
    echo "Config file paths:"
    echo "  - $config_file (unified)"
    echo "  - $clip_config_file (CLIP endpoints: $total_clip_count instance(s) on this node)"
    echo "  - $self_reward_config_file (R3-Refiner self-reward endpoints: $total_self_reward_count instance(s) on this node)"
    echo "  - $sam3_config_file (SAM3 endpoints: $total_sam3_count instance(s) on this node)"

    if [[ $added_count -gt 0 ]]; then
        echo ""
        echo "Config files updated (new endpoints appended):"
        echo "  - $config_file ($added_count new endpoint(s))"
        if [[ $added_clip_count -gt 0 ]]; then
            echo "  - $clip_config_file ($added_clip_count new CLIP endpoint(s))"
        fi
        if [[ $added_self_reward_count -gt 0 ]]; then
            echo "  - $self_reward_config_file ($added_self_reward_count new R3-Refiner self-reward endpoint(s))"
        fi
        if [[ $added_sam3_count -gt 0 ]]; then
            echo "  - $sam3_config_file ($added_sam3_count new SAM3 endpoint(s))"
        fi
    else
        echo ""
        echo "[INFO] Config files checked; all endpoints already present, nothing to update."
        if [[ $total_clip_count -gt 0 ]]; then
            echo "  - CLIP endpoints: $total_clip_count instance(s) configured"
        fi
        if [[ $total_self_reward_count -gt 0 ]]; then
            echo "  - R3-Refiner self-reward endpoints: $total_self_reward_count instance(s) configured"
        fi
        if [[ $total_sam3_count -gt 0 ]]; then
            echo "  - SAM3 endpoints: $total_sam3_count instance(s) configured"
        fi
    fi
    echo "========================================="
    echo ""
}

# Stop the image-editing service
stop_edit_server() {
    echo "Stopping image-editing service..."

    pkill -f "distributed_services/servers/bagel_edit_server.py" || pkill -f "bagel_edit_server.py" || true
    pkill -f "distributed_services/servers/qwen_image_edit_server.py" || pkill -f "qwen_image_edit_server.py" || true

    echo "Image-editing service stopped."
    echo "Reward service remains running."
}

# Stop the reward service only
stop_reward_server() {
    echo "Stopping reward service..."

    # Stop reward service processes.
    pkill -f "distributed_services/servers/clip_reward_server.py" || pkill -f "clip_reward_server.py" || true
    pkill -f "distributed_services/servers/self_reward_qwen2_5vl_server.py" || pkill -f "self_reward_qwen2_5vl_server.py" || true
    pkill -f "distributed_services/servers/self_reward_qwen3vl_server.py" || pkill -f "self_reward_qwen3vl_server.py" || true
    pkill -f "distributed_services/servers/sam3_reward_server.py" || pkill -f "sam3_reward_server.py" || true

    echo "Reward service stopped."
    echo "Image-editing service remains running."
}

# Stop all services
stop_services() {
    echo "Stopping all services..."

    # Stop image-editing services.
    pkill -f "distributed_services/servers/bagel_edit_server.py" || pkill -f "bagel_edit_server.py" || true
    pkill -f "distributed_services/servers/qwen_image_edit_server.py" || pkill -f "qwen_image_edit_server.py" || true

    # Stop reward services.
    pkill -f "distributed_services/servers/clip_reward_server.py" || pkill -f "clip_reward_server.py" || true
    pkill -f "distributed_services/servers/sam3_reward_server.py" || pkill -f "sam3_reward_server.py" || true
    pkill -f "distributed_services/servers/self_reward_qwen2_5vl_server.py" || pkill -f "self_reward_qwen2_5vl_server.py" || true
    pkill -f "distributed_services/servers/self_reward_qwen3vl_server.py" || pkill -f "self_reward_qwen3vl_server.py" || true

    echo "All services stopped."
}
# ============================================================================
# Config-generation functions
# ============================================================================

# Read endpoints from a file with deduplication.
read_endpoints_from_file() {
    local config_file="$1"
    local -n seen_array="$2"  # nameref — references the caller's associative array directly
    local endpoints=""

    if [[ ! -f "$config_file" ]]; then
        echo ""
        return
    fi

    while IFS= read -r line || [[ -n "$line" ]]; do
        line=$(echo "$line" | tr -d '[:space:]')
        if [[ -n "$line" && "$line" =~ ^http:// ]]; then
            if [[ -z "${seen_array[$line]:-}" ]]; then
                seen_array[$line]=1
                if [[ -n "$endpoints" ]]; then
                    endpoints="${endpoints},"
                fi
                endpoints="${endpoints}${line}"
            fi
        fi
    done < "$config_file"

    echo "$endpoints"
}

# Read image-editing service endpoints.
read_edit_server_endpoints() {
    local edit_config="${CONFIG_DIR}/edit_server_endpoints.txt"
    declare -A seen_edit_endpoints

    if [[ -f "$edit_config" ]]; then
        echo "[INFO] Reading edit-service endpoints from config file: $edit_config" >&2
        local endpoints=$(read_endpoints_from_file "$edit_config" "seen_edit_endpoints")
        local count=$(echo "$endpoints" | tr ',' '\n' | grep -c '^http://' 2>/dev/null || echo "0")
        echo "[INFO] Found $count edit-service endpoint(s)" >&2
        echo "$endpoints"
    else
        echo ""
    fi
}

# Read reward service endpoints by type.
read_reward_server_endpoints() {
    local clip_config_file="${CONFIG_DIR}/clip_reward_server_endpoints.txt"
    local self_reward_config_file="${CONFIG_DIR}/self_reward_server_endpoints.txt"
    local sam3_config_file="${CONFIG_DIR}/sam3_reward_server_endpoints.txt"
    local reward_config="${CONFIG_DIR}/reward_server_endpoints.txt"
    local sam3_base_port="${SAM3_REWARD_SERVER_BASE_PORT:-7001}"

    declare -A seen_clip_endpoints
    declare -A seen_self_reward_endpoints
    declare -A seen_sam3_endpoints

    local reward_endpoints_clip=""
    local reward_endpoints_self_reward=""
    local reward_endpoints_sam3=""
    local has_classified_configs=false

    # Prefer non-empty per-type endpoint files.
    if [[ -f "$clip_config_file" ]]; then
        reward_endpoints_clip=$(read_endpoints_from_file "$clip_config_file" "seen_clip_endpoints")
        if [[ -n "$reward_endpoints_clip" ]]; then
            has_classified_configs=true
            echo "[INFO] Reading CLIP service endpoints from: $clip_config_file"
        fi
    fi

    if [[ -f "$self_reward_config_file" ]]; then
        reward_endpoints_self_reward=$(read_endpoints_from_file "$self_reward_config_file" "seen_self_reward_endpoints")
        if [[ -n "$reward_endpoints_self_reward" ]]; then
            has_classified_configs=true
            echo "[INFO] Reading R3-Refiner self-reward service endpoints from: $self_reward_config_file"
        fi
    fi

    if [[ -f "$sam3_config_file" ]]; then
        reward_endpoints_sam3=$(read_endpoints_from_file "$sam3_config_file" "seen_sam3_endpoints")
        if [[ -n "$reward_endpoints_sam3" ]]; then
            has_classified_configs=true
            echo "[INFO] Reading SAM3 service endpoints from: $sam3_config_file"
        fi
    fi

    # Read the unified endpoint file if no non-empty per-type files exist.
    if [[ "$has_classified_configs" == "false" && -f "$reward_config" ]]; then
        echo "[INFO] No per-type config files found; reading reward-service endpoints from unified file: $reward_config"
        while IFS= read -r line || [[ -n "$line" ]]; do
            line=$(echo "$line" | tr -d '[:space:]')
            if [[ -n "$line" && "$line" =~ ^http:// ]]; then
                    if [[ "$line" =~ :([0-9]+)$ ]]; then
                    local port="${BASH_REMATCH[1]}"
                    if [[ $port -ge $sam3_base_port ]]; then
                        if [[ -z "${seen_sam3_endpoints[$line]:-}" ]]; then
                            seen_sam3_endpoints[$line]=1
                            if [[ -n "$reward_endpoints_sam3" ]]; then
                                reward_endpoints_sam3="${reward_endpoints_sam3},"
                            fi
                            reward_endpoints_sam3="${reward_endpoints_sam3}${line}"
                        fi
                    else
                        # Treat non-SAM3 endpoints as the current REWARD_TYPE.
                        local default_reward_type
                        default_reward_type="$(normalize_reward_type "${REWARD_TYPE:-self_reward}")"
                        if [[ "$default_reward_type" == "self_reward" ]]; then
                            if [[ -z "${seen_self_reward_endpoints[$line]:-}" ]]; then
                                seen_self_reward_endpoints[$line]=1
                                if [[ -n "$reward_endpoints_self_reward" ]]; then
                                    reward_endpoints_self_reward="${reward_endpoints_self_reward},"
                                fi
                                reward_endpoints_self_reward="${reward_endpoints_self_reward}${line}"
                            fi
                            continue
                        fi
                        if [[ -z "${seen_clip_endpoints[$line]:-}" ]]; then
                            seen_clip_endpoints[$line]=1
                            if [[ -n "$reward_endpoints_clip" ]]; then
                                reward_endpoints_clip="${reward_endpoints_clip},"
                            fi
                            reward_endpoints_clip="${reward_endpoints_clip}${line}"
                        fi
                    fi
                else
                    # If the port cannot be parsed, keep the endpoint under CLIP.
                    if [[ -z "${seen_clip_endpoints[$line]:-}" ]]; then
                        seen_clip_endpoints[$line]=1
                        if [[ -n "$reward_endpoints_clip" ]]; then
                            reward_endpoints_clip="${reward_endpoints_clip},"
                        fi
                        reward_endpoints_clip="${reward_endpoints_clip}${line}"
                    fi
                fi
            fi
        done < "$reward_config"
    fi

    # Report endpoint counts by type.
    local clip_count=0
    local self_reward_count=0
    local sam3_count=0

    if [[ -n "$reward_endpoints_clip" ]]; then
        clip_count=$(echo "$reward_endpoints_clip" | tr ',' '\n' | grep -c '^http://' 2>/dev/null || echo "0")
    fi
    if [[ -n "$reward_endpoints_self_reward" ]]; then
        self_reward_count=$(echo "$reward_endpoints_self_reward" | tr ',' '\n' | grep -c '^http://' 2>/dev/null || echo "0")
    fi
    if [[ -n "$reward_endpoints_sam3" ]]; then
        sam3_count=$(echo "$reward_endpoints_sam3" | tr ',' '\n' | grep -c '^http://' 2>/dev/null || echo "0")
    fi

    echo "[INFO] Found $clip_count CLIP service endpoint(s)"
    echo "[INFO] Found $self_reward_count R3-Refiner self-reward service endpoint(s)"
    echo "[INFO] Found $sam3_count SAM3 service endpoint(s)"

    REWARD_ENDPOINTS_CLIP="$reward_endpoints_clip"
    REWARD_ENDPOINTS_SELF_REWARD="$reward_endpoints_self_reward"
    REWARD_ENDPOINTS_SAM3="$reward_endpoints_sam3"
}

# Check whether endpoint config files exist.
warn_missing_endpoint_configs() {
    local edit_endpoints="$1"
    local reward_endpoints_clip="$2"
    local reward_endpoints_self_reward="$3"
    local reward_endpoints_sam3="$4"

    if [[ -z "$edit_endpoints" ]]; then
        echo "[WARNING] No edit-service endpoints found. Start edit_server first, copy edit_server_endpoints.txt, or set EDIT_SERVER_ENDPOINTS manually before training."
    fi

    if [[ -z "$reward_endpoints_clip" && -z "$reward_endpoints_self_reward" && -z "$reward_endpoints_sam3" ]]; then
        echo "[WARNING] No reward-service endpoints found. Start reward_server first, copy the corresponding reward endpoint file, or set REWARD_SERVER_ENDPOINTS manually before training."
    fi
}

# Select reward endpoints based on REWARD_TYPE.
select_reward_endpoints() {
    local reward_type
    reward_type="$(normalize_reward_type "$1")"
    local reward_type_per_gpu="$2"
    local reward_endpoints_clip="$3"
    local reward_endpoints_self_reward="$4"
    local reward_endpoints_sam3="$5"

    local selected_reward_endpoints="${reward_endpoints_clip}"

    if [[ "$reward_type" == "self_reward" && -n "$reward_endpoints_self_reward" ]]; then
        selected_reward_endpoints="${reward_endpoints_self_reward}"
        echo "[INFO] REWARD_TYPE=self_reward; using R3-Refiner self-reward service endpoints" >&2
    elif [[ "$reward_type" == "clip" && -n "$reward_endpoints_clip" ]]; then
        selected_reward_endpoints="${reward_endpoints_clip}"
        echo "[INFO] REWARD_TYPE=clip; using CLIP service endpoints" >&2
    elif [[ "$reward_type" == "sam3" && -n "$reward_endpoints_sam3" ]]; then
        selected_reward_endpoints=""
        echo "[INFO] REWARD_TYPE=sam3; using SAM3 service endpoints" >&2
    elif [[ "$reward_type" == "mixed" ]]; then
        # Mixed mode exposes CLIP/self-reward endpoints via REWARD_SERVER_ENDPOINTS.
        selected_reward_endpoints=""
        for endpoints in "$reward_endpoints_clip" "$reward_endpoints_self_reward"; do
            if [[ -z "$endpoints" ]]; then
                continue
            fi
            if [[ -n "$selected_reward_endpoints" ]]; then
                selected_reward_endpoints="${selected_reward_endpoints},${endpoints}"
            else
                selected_reward_endpoints="${endpoints}"
            fi
        done
        echo "[INFO] REWARD_TYPE=mixed; merging CLIP and self-reward endpoints (differentiated via REWARD_TYPE_PER_GPU)" >&2
    else
        echo "[WARNING] Unknown REWARD_TYPE=$reward_type; using CLIP endpoints" >&2
    fi

    if [[ -n "$reward_endpoints_sam3" && -n "$selected_reward_endpoints" && ( "$reward_type" == "clip" || "$reward_type" == "self_reward" ) && -z "$reward_type_per_gpu" ]]; then
        echo "[INFO] SAM3 endpoints detected; keeping REWARD_TYPE=$reward_type and exporting SAM3 endpoints separately" >&2
    fi

    UPDATED_REWARD_TYPE="$reward_type"

    echo "$selected_reward_endpoints"
}

# Generate service_endpoints.env.
generate_service_endpoints_env() {
    local export_file="$1"
    local edit_endpoints="$2"
    local selected_reward_endpoints="$3"
    local reward_endpoints_clip="$4"
    local reward_endpoints_self_reward="$5"
    local reward_endpoints_sam3="$6"
    local reward_type="$7"
    local reward_type_per_gpu="$8"

    edit_endpoints="${edit_endpoints:-}"
    selected_reward_endpoints="${selected_reward_endpoints:-}"
    reward_endpoints_clip="${reward_endpoints_clip:-}"
    reward_endpoints_self_reward="${reward_endpoints_self_reward:-}"
    reward_endpoints_sam3="${reward_endpoints_sam3:-}"
    reward_type="${reward_type:-}"
    reward_type_per_gpu="${reward_type_per_gpu:-}"

    # Values are double-quoted; endpoint URLs are not expected to contain
    # special characters that would require single-quote escaping.
    cat > "$export_file" << EOF
# Image-editing service endpoints (comma-separated)
export EDIT_SERVER_ENDPOINTS="${edit_endpoints}"

# Reward service endpoints (clip/self_reward; selected by REWARD_TYPE)
export REWARD_SERVER_ENDPOINTS="${selected_reward_endpoints}"

# CLIP reward service endpoints (standalone; used in mixed mode)
export CLIP_REWARD_SERVER_ENDPOINTS="${reward_endpoints_clip}"

# R3-Refiner self-reward service endpoints (standalone; used in self_reward or mixed mode)
export SELF_REWARD_SERVER_ENDPOINTS="${reward_endpoints_self_reward}"

# SAM3 reward service endpoints
export SAM3_REWARD_SERVER_ENDPOINTS="${reward_endpoints_sam3}"

# Reward service type (self_reward, clip, sam3, or mixed)
export REWARD_TYPE="${reward_type}"

# Per-GPU reward type assignment
export REWARD_TYPE_PER_GPU="${reward_type_per_gpu}"

EOF
}

# Print service config for VERL training nodes.
get_config() {
    echo "========================================="
    echo "Service configuration"
    echo "========================================="

    local export_file="${CONFIG_DIR}/service_endpoints.env"
    mkdir -p "${CONFIG_DIR}"

    # Print the current node IP for quick sanity checks.
    local current_ip=$(get_server_address)
    echo "[INFO] Current node IP address: $current_ip"

    local edit_endpoints=$(read_edit_server_endpoints)

    read_reward_server_endpoints
    local reward_endpoints_clip="$REWARD_ENDPOINTS_CLIP"
    local reward_endpoints_self_reward="$REWARD_ENDPOINTS_SELF_REWARD"
    local reward_endpoints_sam3="$REWARD_ENDPOINTS_SAM3"

    warn_missing_endpoint_configs "$edit_endpoints" "$reward_endpoints_clip" "$reward_endpoints_self_reward" "$reward_endpoints_sam3"

    local reward_type
    reward_type="$(normalize_reward_type "${REWARD_TYPE:-self_reward}")"
    local reward_type_per_gpu="${REWARD_TYPE_PER_GPU:-}"

    local selected_reward_endpoints=$(select_reward_endpoints \
        "$reward_type" \
        "$reward_type_per_gpu" \
        "$reward_endpoints_clip" \
        "$reward_endpoints_self_reward" \
        "$reward_endpoints_sam3")

    if [[ -n "${UPDATED_REWARD_TYPE:-}" ]]; then
        reward_type="$UPDATED_REWARD_TYPE"
    fi

    generate_service_endpoints_env \
        "$export_file" \
        "$edit_endpoints" \
        "$selected_reward_endpoints" \
        "$reward_endpoints_clip" \
        "$reward_endpoints_self_reward" \
        "$reward_endpoints_sam3" \
        "$reward_type" \
        "$reward_type_per_gpu"

    echo "Config file written: $export_file"
    echo ""
    echo "On the VERL training node, source the config with:"
    echo "  source $export_file"
    echo ""
    cat "$export_file"
}

# Print usage information
show_help() {
    cat << EOF
Multi-node distributed service deployment script

Usage:
  $0 <command> [options]

Commands:
  edit_server           - Start the image-editing service (BAGEL or Qwen-Image-Edit)
                          Optional args: <model_path> <edit_model_type>
                          Example: $0 edit_server /path/to/model bagel
                          Example: $0 edit_server /path/to/model qwen_image_edit

  reward_server         - Start the reward service (self_reward/CLIP/SAM3/mixed)
                          The specific self-reward model is set by SELF_REWARD_MODEL_TYPE.
                          Supported: qwen2_5vl, qwen3vl
                          Example: SELF_REWARD_MODEL_TYPE=qwen3vl $0 reward_server
                          Example: REWARD_TYPE=mixed REWARD_TYPE_PER_GPU="0:self_reward,1:self_reward,2:sam3,3:sam3" $0 reward_server

  stop                  - Stop all services
  stop_edit             - Stop the image-editing service only
  stop_reward           - Stop the reward service only

  get_config            - Generate service config file (run on any node)

  help                  - Show this help message

Architecture:
  Node 0:   VERL training
  Node 1-N: Image-editing service (BAGEL or Qwen-Image-Edit)
  Node M:   Reward service (self_reward/CLIP/SAM3/mixed, controlled by REWARD_TYPE and REWARD_TYPE_PER_GPU)

Environment variables:
  EDIT_MODEL_PATH       - Edit model path (loaded from service.edit_model_path in config.yaml)
  EDIT_MODEL_TYPE       - Image-editing model type: bagel or qwen_image_edit
  SELF_REWARD_MODEL_PATH - self_reward model path
  SELF_REWARD_MODEL_TYPE - self_reward model type: qwen2_5vl or qwen3vl
  CLIP_MODULE_PATH      - Custom CLIP module path (optional; uses the standard clip package if unset)
  CLIP_MODEL_PATH       - CLIP model checkpoint path (optional; .pt file for custom-trained models)
  GPUS_PER_NODE         - Maximum number of GPUs to use per node (0 = all detected GPUs)
  EDIT_SERVER_NODE      - Image-editing service node address (default: current hostname)
  REWARD_SERVER_NODE    - Reward service node address (default: current hostname)

Example deployment workflow:
  # Start the image-editing service on a service node
  EDIT_MODEL_TYPE=bagel bash distributed_services/scripts/deploy_services.sh edit_server

  # Start the reward service on a service node
  REWARD_TYPE=self_reward SELF_REWARD_MODEL_TYPE=qwen3vl bash distributed_services/scripts/deploy_services.sh reward_server

  # On a multi-node setup, generate the config on the training node
  bash distributed_services/scripts/deploy_services.sh get_config

  # Source the config on the training node
  source distributed_services/config/service_endpoints.env

  # Start VERL training on the training node
  bash distributed_services/scripts/train_quick.sh
EOF
}

# Main dispatch.
case "$SERVICE_TYPE" in
    edit_server)
        model_path="${2:-${EDIT_MODEL_PATH:-/path/to/BAGEL-7B-MoT}}"
        edit_model_type="${3:-${EDIT_MODEL_TYPE:-bagel}}"
        start_edit_server "$model_path" "$edit_model_type"

        echo "========================================="
        echo "Output is being redirected to log files."
        echo "========================================="

        sleep $SERVICE_KEEPALIVE_SECONDS
        echo "Keepalive period elapsed; exiting."
        ;;
    reward_server)
        model_name="${2:-ViT-B/32}"  # First argument is the CLIP model name.
        start_reward_server "$model_name"

        echo "========================================="
        echo "Output is being redirected to log files."
        echo "========================================="

        sleep $SERVICE_KEEPALIVE_SECONDS
        echo "Keepalive period elapsed; exiting."
        ;;
    stop)
        stop_services
        ;;
    stop_edit)
        stop_edit_server
        ;;
    stop_reward)
        stop_reward_server
        ;;
    get_config)
        get_config
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo "[ERROR] Unknown command: $SERVICE_TYPE"
        echo ""
        show_help
        exit 1
        ;;
esac
