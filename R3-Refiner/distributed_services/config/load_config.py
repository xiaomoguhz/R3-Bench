#!/usr/bin/env python3
"""Load config.yaml and emit shell export statements."""

import sys
import os
import yaml
import argparse
import shlex
import subprocess

def detect_gpu_count():
    """Detect the number of available GPUs."""
    # Prefer nvidia-smi because it matches the runtime visible to launch scripts.
    try:
        result = subprocess.run(
            ["nvidia-smi", "--list-gpus"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            gpu_count = len([line for line in result.stdout.strip().split('\n') if line.strip()])
            if gpu_count > 0:
                return gpu_count
    except (OSError, subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        pass

    # Use PyTorch when nvidia-smi is unavailable.
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            if gpu_count > 0:
                return gpu_count
    except ImportError:
        pass

    # Last resort: parse CUDA_VISIBLE_DEVICES.
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_visible:
        # Parse CUDA_VISIBLE_DEVICES, e.g. "0,1,2,3" or "0-3"
        if "," in cuda_visible:
            return len([x.strip() for x in cuda_visible.split(",") if x.strip()])
        elif "-" in cuda_visible:
            parts = cuda_visible.split("-")
            if len(parts) == 2:
                try:
                    start = int(parts[0].strip())
                    end = int(parts[1].strip())
                    return end - start + 1
                except ValueError:
                    pass

    return 0


def get_env_vars(config):
    """Extract environment variables from a YAML config dict."""
    env_vars = {}

    service = config.get("service", {})

    # Service ports.
    env_vars["EDIT_SERVER_BASE_PORT"] = str(service.get("edit_server_base_port", 5001))
    env_vars["REWARD_SERVER_BASE_PORT"] = str(service.get("reward_server_base_port", 6001))
    env_vars["SAM3_REWARD_SERVER_BASE_PORT"] = str(service.get("sam3_reward_server_base_port", 7001))

    # Model paths.
    env_vars["EDIT_MODEL_PATH"] = service.get("edit_model_path", "")
    env_vars["SELF_REWARD_MODEL_PATH"] = service.get("self_reward_model_path", "")
    env_vars["SELF_REWARD_MODEL_TYPE"] = service.get("self_reward_model_type", "qwen2_5vl")

    # SAM3 model resource paths
    env_vars["SAM3_BPE_PATH"] = service.get("sam3_bpe_path", "")
    env_vars["SAM3_CKPT_PATH"] = service.get("sam3_ckpt_path", "")
    env_vars["SAM3_METADATA_JSONL"] = service.get("sam3_metadata_jsonl", "")

    # Reward routing.
    env_vars["REWARD_TYPE"] = service.get("reward_type", "self_reward")
    env_vars["REWARD_TYPE_PER_GPU"] = service.get("reward_type_per_gpu", "")

    # GPU config.
    gpus_per_node = service.get("gpus_per_node", 0)
    if gpus_per_node == 0:
        detected_gpus = detect_gpu_count()
        env_vars["GPUS_PER_NODE"] = str(detected_gpus) if detected_gpus > 0 else "0"
    else:
        env_vars["GPUS_PER_NODE"] = str(gpus_per_node)
    instances_per_gpu = service.get("instances_per_gpu", "1")
    env_vars["INSTANCES_PER_GPU"] = str(instances_per_gpu) if isinstance(instances_per_gpu, (int, float)) else instances_per_gpu

    # Image-edit model config.
    env_vars["EDIT_MODEL_TYPE"] = service.get("edit_model_type", "bagel")

    # CLIP config.
    env_vars["CLIP_MODULE_PATH"] = service.get("clip_module_path", "")
    env_vars["CLIP_MODEL_PATH"] = service.get("clip_model_path", "")

    # API mode config.
    env_vars["USE_API_MODE"] = str(service.get("use_api_mode", True)).lower()
    env_vars["EDIT_SERVER_THREADED"] = str(service.get("edit_server_threaded", True)).lower()

    # API client config.
    api_client = config.get("api_client", {})
    env_vars["API_ENABLE_HEALTH_CHECK"] = str(api_client.get("enable_health_check", True)).lower()
    env_vars["API_HEALTH_CHECK_FAILURE_THRESHOLD"] = str(api_client.get("health_check_failure_threshold", 5))
    env_vars["API_REQUEST_TIMEOUT"] = str(api_client.get("request_timeout", 180))

    # Other config.
    other = config.get("other", {})
    hf_endpoint = other.get("hf_endpoint", "")
    if hf_endpoint:
        env_vars["HF_ENDPOINT"] = hf_endpoint
    wandb = other.get("wandb", {})
    env_vars["WANDB_API_KEY"] = wandb.get("api_key", "")
    env_vars["WANDB_MODE"] = wandb.get("mode", "offline")
    env_vars["PYTHONUNBUFFERED"] = str(other.get("pythonunbuffered", 1))
    env_vars["EDIT_CONDA_ENV"] = other.get("edit_conda_env", "bagelfast")
    env_vars["REWARD_CONDA_ENV"] = other.get("reward_conda_env", "")

    # Path config.
    env_vars["PROJECT_ROOT"] = other.get("project_root", "")

    # Derived paths.
    project_root = env_vars.get("PROJECT_ROOT", "")
    distributed_services = os.path.join(project_root, "distributed_services")
    logs = os.path.join(distributed_services, "logs")
    config = os.path.join(distributed_services, "config")
    env_vars["LOG_DIR"] = logs
    env_vars["CONFIG_DIR"] = config

    return env_vars


def main():
    parser = argparse.ArgumentParser(description="Load config from a YAML file and export as environment variables")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to the YAML config file (default: config.yaml in the same directory as this script)"
    )
    args = parser.parse_args()

    if args.config:
        config_file = args.config
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        config_file = os.path.join(base_dir, "config.yaml")

    if not os.path.exists(config_file):
        print(f"# [ERROR] YAML config file not found: {config_file}", file=sys.stderr)
        sys.exit(1)

    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"# [ERROR] Failed to load YAML config: {e}", file=sys.stderr)
        sys.exit(1)

    env_vars = get_env_vars(config)

    # Shell variables already set by the caller take precedence.
    for key, value in env_vars.items():
        if key not in os.environ:
            print(f"export {key}={shlex.quote(str(value))}")

    # Include locally generated service endpoints when available.
    base_dir = os.path.dirname(os.path.abspath(__file__))
    env_file = os.path.join(base_dir, "service_endpoints.env")

    if os.path.exists(env_file):
        print(f"# [INFO] Loading service endpoint config file: {env_file}", file=sys.stderr)
        try:
            with open(env_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()

                    if not line or line.startswith('#'):
                        continue

                    if not line.startswith('export '):
                        continue

                    export_part = line[7:].strip()  # strip 'export '
                    if '=' not in export_part:
                        continue

                    eq_pos = export_part.find('=')
                    key = export_part[:eq_pos].strip()
                    value_part = export_part[eq_pos+1:].strip()

                    if value_part.startswith('"') and value_part.endswith('"'):
                        value = value_part[1:-1]
                    elif value_part.startswith("'") and value_part.endswith("'"):
                        value = value_part[1:-1]
                    else:
                        value = value_part
                        if '#' in value:
                            comment_pos = value.find('#')
                            if comment_pos >= 0:
                                value = value[:comment_pos].strip()

                    if key and key not in os.environ:
                        print(f"export {key}={shlex.quote(value)}")

        except Exception as e:
            print(f"# [WARNING] Failed to load service endpoint config: {e}", file=sys.stderr)
    else:
        print(f"# [WARNING] Service endpoint config file not found: {env_file}", file=sys.stderr)


if __name__ == "__main__":
    main()
