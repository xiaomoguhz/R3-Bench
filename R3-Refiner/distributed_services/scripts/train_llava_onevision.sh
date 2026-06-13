#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${PROJECT_ROOT}"

export VLLM_DISABLE_SYMMETRIC_MEMORY="${VLLM_DISABLE_SYMMETRIC_MEMORY:-1}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export HF_HOME="${HF_HOME:-${PROJECT_ROOT}/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"

project_name="${PROJECT_NAME:-R3-Refiner}"
experiment_name="${EXPERIMENT_NAME:-llava-onevision-r3-refiner}"
model_path="${MODEL_PATH:?Set MODEL_PATH to the LLaVA-OneVision checkpoint.}"

train_data_path="${TRAIN_DATA_PATH:-examples/data/demo_train.json}"
val_data_path="${VAL_DATA_PATH:-examples/data/demo_train.json}"
image_dir="${IMAGE_DIR:-examples/data/images}"

save_checkpoint_path="${SAVE_CHECKPOINT_PATH:-checkpoints/${project_name}/${experiment_name}}"
config_file="${CONFIG_FILE:-distributed_services/config/config.yaml}"
verl_base_config_rel="${VERL_BASE_CONFIG:-distributed_services/config/verl_config.yaml}"
format_prompt="${FORMAT_PROMPT:-examples/format_prompt/llava_ov_edit.jinja}"
reward_function="${REWARD_FUNCTION:-examples/reward_function/self_reward_staged_reward_api_llava.py:compute_score}"

normalize_bool() {
    local value
    value="$(printf '%s' "${1:-}" | tr '[:upper:]' '[:lower:]')"
    case "${value}" in
        1|true|yes|on) printf 'true' ;;
        0|false|no|off) printf 'false' ;;
        *)
            echo "Invalid boolean value: ${1}. Use true/false or 1/0." >&2
            exit 1
            ;;
    esac
}

is_demo_train_data() {
    local path_spec="${1%%,*}"
    path_spec="${path_spec%%@*}"
    path_spec="${path_spec#\"}"
    path_spec="${path_spec%\"}"
    path_spec="${path_spec#\'}"
    path_spec="${path_spec%\'}"
    path_spec="${path_spec#./}"

    if [[ "${path_spec}" == "examples/data/demo_train.json" ]]; then
        return 0
    fi

    if command -v realpath >/dev/null 2>&1; then
        local demo_path
        local resolved_path
        demo_path="$(realpath -m "examples/data/demo_train.json" 2>/dev/null || true)"
        resolved_path="$(realpath -m "${path_spec}" 2>/dev/null || true)"
        [[ -n "${resolved_path}" && "${resolved_path}" == "${demo_path}" ]]
        return
    fi

    return 1
}

if [[ -n "${REWARD_KWARGS:-}" ]]; then
    reward_kwargs="${REWARD_KWARGS}"
else
    enable_stage2="$(normalize_bool "${ENABLE_STAGE2:-true}")"
    reward_kwargs="{\"think_format_weight\":0.1,\"json_format_weight\":0.05,\"stage1_weight\":0.2,\"stage2_weight\":0.8,\"image_dir\":\"${image_dir}\",\"default_reward_type\":\"self_reward\",\"enable_stage2\":${enable_stage2}}"
fi

if [[ -n "${ROLLOUT_BATCH_SIZE:-}" ]]; then
    rollout_batch_size="${ROLLOUT_BATCH_SIZE}"
elif is_demo_train_data "${train_data_path}"; then
    rollout_batch_size=64
else
    rollout_batch_size=128
fi
n_gpus_per_node="${N_GPUS_PER_NODE:-8}"
trainer_nnodes="${NNODES:-1}"
actor_micro_batch_update="${ACTOR_MICRO_BATCH_UPDATE:-2}"
actor_micro_batch_experience="${ACTOR_MICRO_BATCH_EXPERIENCE:-4}"
max_prompt_length="${MAX_PROMPT_LENGTH:-8192}"
max_response_length="${MAX_RESPONSE_LENGTH:-2048}"
min_pixels="${MIN_PIXELS:-0}"
max_pixels="${MAX_PIXELS:-262144}"
rollout_max_model_len="${ROLLOUT_MAX_MODEL_LEN:-12288}"
rollout_max_num_batched_tokens="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-12288}"

train_log_file="${LOG_FILE:-distributed_services/logs/${project_name}_${experiment_name}_$(date +"%Y%m%d_%H%M%S").log}"
mkdir -p "$(dirname "${train_log_file}")"

if [[ -f "distributed_services/config/service_endpoints.env" ]]; then
    source distributed_services/config/service_endpoints.env
fi

python_bin="${PYTHON_BIN:-python3}"
config_exports="$("${python_bin}" distributed_services/config/load_config.py --config "${config_file}")"
eval "${config_exports}"
if [[ "${HF_ENDPOINT+x}" == "x" && -z "${HF_ENDPOINT}" ]]; then
    unset HF_ENDPOINT
fi

"${python_bin}" -m verl.trainer.main \
    config="${verl_base_config_rel}" \
    data.image_dir="${image_dir}" \
    "data.train_files='${train_data_path}'" \
    "data.val_files='${val_data_path}'" \
    data.max_prompt_length="${max_prompt_length}" \
    data.max_response_length="${max_response_length}" \
    data.min_pixels="${min_pixels}" \
    data.max_pixels="${max_pixels}" \
    data.rollout_batch_size="${rollout_batch_size}" \
    data.val_filter_overlong_prompts=true \
    data.train_filter_overlong_prompts=true \
    data.format_prompt="${format_prompt}" \
    worker.actor.model.model_path="${model_path}" \
    worker.actor.model.trust_remote_code=true \
    worker.actor.use_torch_compile=false \
    worker.actor.micro_batch_size_per_device_for_update="${actor_micro_batch_update}" \
    worker.actor.micro_batch_size_per_device_for_experience="${actor_micro_batch_experience}" \
    worker.rollout.max_model_len="${rollout_max_model_len}" \
    worker.rollout.max_num_batched_tokens="${rollout_max_num_batched_tokens}" \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${experiment_name}" \
    trainer.nnodes="${trainer_nnodes}" \
    trainer.n_gpus_per_node="${n_gpus_per_node}" \
    trainer.save_checkpoint_path="${save_checkpoint_path}" \
    worker.reward.reward_function="${reward_function}" \
    worker.reward.reward_function_kwargs="${reward_kwargs}" \
    "$@" 2>&1 | tee "${train_log_file}"
