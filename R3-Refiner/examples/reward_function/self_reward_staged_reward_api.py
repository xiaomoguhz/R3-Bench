"""Staged reward function used by R3-Refiner training."""

import re
import json
import os
import sys
from typing import Any, Optional, Dict, Tuple
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

# Resolve project-local distributed service clients.
CURRENT_DIR = os.path.dirname(__file__)
PARENT_DIR = os.path.dirname(CURRENT_DIR)  # examples/
GRANDPARENT_DIR = os.path.dirname(PARENT_DIR)  # project root/
DISTRIBUTED_SERVICES_DIR = os.path.join(GRANDPARENT_DIR, "distributed_services")
if DISTRIBUTED_SERVICES_DIR not in sys.path:
    sys.path.insert(0, DISTRIBUTED_SERVICES_DIR)

try:
    from clients.api_client import (
        ImageEditClient,
        RewardClient,
        create_clients_from_env,
    )
    API_CLIENT_AVAILABLE = True
except ImportError:
    API_CLIENT_AVAILABLE = False
    print("Warning: distributed_services/clients/api_client.py is not importable.")

# Global client instances (lazily initialized)
_edit_client: Optional[ImageEditClient] = None
_reward_client: Optional[RewardClient] = None
_sam3_reward_client: Optional[RewardClient] = None

# Log-once flags (suppress duplicate warnings)
_client_warning_printed: Dict[str, bool] = {
    "edit_client": False,
    "reward_client": False,
    "sam3_reward_client": False,
    "max_workers": False,
    "self_reward_client_error": False,
    "clip_client_error": False,
    "self_reward_endpoints_empty": False,
    "reward_type_default_endpoints": False,
    "default_reward_type": False,
}


def _get_env_int(key: str, default: int) -> int:
    """Read an integer environment variable."""
    value = os.environ.get(key, str(default))
    # Strip quotes that may be left by config-file parsers
    value = value.strip().strip('"').strip("'")
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def _get_env_bool(key: str, default: bool = True) -> bool:
    """Read a boolean environment variable."""
    value = os.environ.get(key, str(default).lower())
    # Strip surrounding quotes
    value = value.strip().strip('"').strip("'").lower()
    return value == "true"


# Mapping from category to reward_type
_CATEGORY_TO_REWARD_TYPE = {
    "color": "clip",
    "shape": "clip",
    "texture": "clip",
    "spatial": "clip",
    "numeracy": "clip",
    "object": "clip",
    "complex": "clip",
    "non": "clip",
}

# Supported reward types. The specific self-reward model is determined by the server-side SELF_REWARD_MODEL_TYPE.
_SUPPORTED_REWARD_TYPES = {"clip", "self_reward", "sam3", "mixed"}


def _normalize_reward_type(reward_type: Optional[str]) -> str:
    reward_type = (reward_type or "").strip().lower()
    return reward_type.replace("-", "_")


def _is_self_reward_type(reward_type: Optional[str]) -> bool:
    return _normalize_reward_type(reward_type) == "self_reward"


def _get_self_reward_endpoints() -> str:
    return os.environ.get("SELF_REWARD_SERVER_ENDPOINTS", "")


def _get_reward_type_for_current_gpu() -> Optional[str]:
    """
    Look up the reward type for the current GPU using the REWARD_TYPE_PER_GPU env var.

    Returns:
        Reward type for the current GPU, or None if it cannot be determined.
    """
    reward_type_per_gpu = os.environ.get("REWARD_TYPE_PER_GPU", "")
    if not reward_type_per_gpu:
        return None

    # LOCAL_RANK maps the worker to its local GPU. Single-process runs use GPU 0.
    gpu_id = 0
    local_rank = os.environ.get("LOCAL_RANK", None)
    if local_rank is not None:
        try:
            gpu_id = int(local_rank)
        except (ValueError, TypeError):
            gpu_id = 0

    # Parse REWARD_TYPE_PER_GPU format: "0:self_reward,1:self_reward,2:self_reward,3:sam3"
    try:
        mappings = {}
        for mapping in reward_type_per_gpu.split(","):
            mapping = mapping.strip()
            if ":" in mapping:
                gpu_str, reward_type = mapping.split(":", 1)
                gpu_str = gpu_str.strip()
                reward_type = _normalize_reward_type(reward_type.strip())
                try:
                    gpu_num = int(gpu_str)
                    mappings[gpu_num] = reward_type
                except (ValueError, TypeError):
                    continue

        return mappings.get(gpu_id)
    except Exception:
        return None


def initialize_clients():
    """Initialize API clients from environment variables."""
    global _edit_client, _reward_client, _sam3_reward_client, _client_warning_printed

    if not API_CLIENT_AVAILABLE:
        raise RuntimeError("API client not available. Cannot initialize clients.")

    # Existing clients pick up endpoint changes from service_endpoints.env.
    if _edit_client is not None or _reward_client is not None or _sam3_reward_client is not None:
        try:
            from clients.api_client import _check_and_reload_config
            if _check_and_reload_config(_edit_client, _reward_client, _sam3_reward_client):
                print("[INFO] Config file change detected; client endpoints reloaded.")
        except Exception:
            pass

    if _edit_client is None or _reward_client is None or _sam3_reward_client is None:
        _edit_client, _reward_client, _sam3_reward_client = create_clients_from_env()

    # Print missing-client warnings once per process.
    if _edit_client is None and not _client_warning_printed.get("edit_client", False):
        print("Warning: Image edit client not initialized. Check EDIT_SERVER_ENDPOINTS environment variable.")
        _client_warning_printed["edit_client"] = True

    if _reward_client is None and not _client_warning_printed.get("reward_client", False):
        print("Warning: Reward client not initialized. Check REWARD_SERVER_ENDPOINTS environment variable.")
        _client_warning_printed["reward_client"] = True

    return _edit_client, _reward_client, _sam3_reward_client


def get_reward_client_for_type(reward_type: str):
    """
    Return the reward client for a reward type.

    Args:
        reward_type: reward type (clip, self_reward, or sam3)

    Returns:
        Reward client, or None if unavailable.
    """
    global _reward_client, _sam3_reward_client, _client_warning_printed

    reward_type = _normalize_reward_type(reward_type)

    if reward_type == "sam3":
        # SAM3 uses the same HTTP client class with a dedicated endpoint list.
        return _sam3_reward_client
    elif _is_self_reward_type(reward_type):
        # Prefer the dedicated SELF_REWARD_SERVER_ENDPOINTS
        self_reward_endpoints = _get_self_reward_endpoints()
        if self_reward_endpoints:
            try:
                from clients.api_client import RewardClient
                endpoints = [e.strip() for e in self_reward_endpoints.split(",") if e.strip()]
                if endpoints:
                    timeout = _get_env_int("API_REQUEST_TIMEOUT", 180)
                    health_check_timeout = _get_env_int("API_HEALTH_CHECK_TIMEOUT", 5)
                    enable_health_check = _get_env_bool("API_ENABLE_HEALTH_CHECK", True)
                    return RewardClient(
                        endpoints,
                        timeout=timeout,
                        health_check_timeout=health_check_timeout,
                        enable_health_check=enable_health_check
                    )
            except Exception as e:
                if not _client_warning_printed.get("self_reward_client_error", False):
                    print(f"[WARNING] Failed to create self-reward client: {e}")
                    _client_warning_printed["self_reward_client_error"] = True

        # Without a dedicated self-reward endpoint, check whether generic
        # reward endpoints can serve self_reward requests.
        env_reward_type = _normalize_reward_type(os.environ.get("REWARD_TYPE", ""))
        if env_reward_type in ["mixed", "clip"]:
            if not _client_warning_printed.get("self_reward_endpoints_empty", False):
                print(f"[WARNING] self_reward reward type requested but SELF_REWARD_SERVER_ENDPOINTS is empty")
                print(f"[WARNING] Current REWARD_TYPE={env_reward_type}; available endpoints may only support clip")
                print(f"[WARNING] Deploy R3-Refiner self-reward service and set SELF_REWARD_SERVER_ENDPOINTS")
                print(f"[WARNING] Or set default_reward_type to 'clip'")
                _client_warning_printed["self_reward_endpoints_empty"] = True
            return None

        # In self_reward mode, REWARD_SERVER_ENDPOINTS is allowed as the selected endpoint list.
        if _is_self_reward_type(env_reward_type):
            if not _client_warning_printed.get("reward_type_default_endpoints", False):
                print(f"[WARNING] REWARD_TYPE={env_reward_type} but SELF_REWARD_SERVER_ENDPOINTS is empty; using REWARD_SERVER_ENDPOINTS")
                print(f"[WARNING] Ensure the endpoints in REWARD_SERVER_ENDPOINTS support the self_reward type")
                _client_warning_printed["reward_type_default_endpoints"] = True

        return _reward_client
    else:
        # CLIP uses CLIP_REWARD_SERVER_ENDPOINTS when provided, otherwise the selected reward client.
        clip_endpoints = os.environ.get("CLIP_REWARD_SERVER_ENDPOINTS", "")
        if clip_endpoints:
            try:
                from clients.api_client import RewardClient
                endpoints = [e.strip() for e in clip_endpoints.split(",") if e.strip()]
                if endpoints:
                    timeout = _get_env_int("API_REQUEST_TIMEOUT", 180)
                    health_check_timeout = _get_env_int("API_HEALTH_CHECK_TIMEOUT", 5)
                    enable_health_check = _get_env_bool("API_ENABLE_HEALTH_CHECK", True)
                    return RewardClient(
                        endpoints,
                        timeout=timeout,
                        health_check_timeout=health_check_timeout,
                        enable_health_check=enable_health_check
                    )
            except Exception as e:
                if not _client_warning_printed.get("clip_client_error", False):
                    print(f"[WARNING] Failed to create CLIP client: {e}")
                    _client_warning_printed["clip_client_error"] = True
        return _reward_client


def filter_thinking_part(response, eos_token=None):
    """
    Extract the answer portion of the response (strips the think tag).
    """
    response_start = 0
    success = False
    tag_pairs = (("<thinking>", "</thinking>"), ("<think>", "</think>"))

    for think_tag_start, think_tag_end in tag_pairs:
        think_end = response.rfind(think_tag_end)
        if think_end != -1:
            response_start = think_end + len(think_tag_end)
            success = True
            break

        think_start = response.find(think_tag_start)
        if think_start != -1:
            response_start = think_start + len(think_tag_start)
            success = True
            break

    if eos_token is not None:
        response_end = response.find(eos_token, response_start)
    else:
        response_end = len(response)
    response = response[response_start:response_end]
    return response, success


def think_format_reward(response: str) -> float:
    """
    Compute the think-tag format reward.
    """
    r = (response or "").strip()

    for think_tag_start, think_tag_end in (("<thinking>", "</thinking>"), ("<think>", "</think>")):
        think_end = r.rfind(think_tag_end)
        if think_end == -1:
            continue

        think_start = r.find(think_tag_start, 0, think_end)
        if think_start != -1:
            think = r[think_start + len(think_tag_start):think_end]
        else:
            think = r[:think_end]

        ans = r[think_end + len(think_tag_end):].strip()
        if think.strip() and ans.strip() and think_tag_start not in ans:
            return 1.0

    return 0.0


def is_valid_edit_prompt(edit_prompt: str) -> bool:
    """Validate edit_prompt before sending it to the edit service."""
    if not edit_prompt:
        return False

    edit_prompt_clean = edit_prompt.strip()

    # Empty or whitespace-only strings are not actionable edit instructions.
    if not edit_prompt_clean:
        return False

    # Very short strings are usually fragments rather than instructions.
    if len(edit_prompt_clean) <= 10:
        return False

    # Explicit no-op values should not trigger image editing.
    edit_prompt_lower = edit_prompt_clean.lower()
    if edit_prompt_lower in ["remain unchanged", "no edit", ""]:
        return False

    # Reject copied template placeholders.
    template_patterns = [
        "a concrete, location-specific editing instruction",
        "concrete, location-specific editing instruction to fix the error",
        "provide a concrete, location-specific editing instruction",
        "location-specific editing instruction",
    ]
    for pattern in template_patterns:
        if pattern in edit_prompt_lower:
            return False

    # Short prompts need at least one clear editing verb.
    action_words = ['add', 'remove', 'replace', 'change', 'move', 'delete', 'place', 'position', 'shift', 'make', 'modify', 'update']
    has_action_word = any(word in edit_prompt_lower for word in action_words)

    if not has_action_word:
        if len(edit_prompt_clean) < 20:
            return False

    return True


def check_format_collapse(text: str, min_words: int = 5, max_consecutive_repeat: int = 5) -> bool:
    """Return True when text contains repeated-character or repeated-word collapse."""
    if not text:
        return False

    # Character-level collapse, such as repeated spaces or quotes.
    if len(text) > 10:
        consecutive_char_count = 1
        max_char_repeat = 1
        for i in range(1, len(text)):
            if text[i] == text[i-1] and text[i] in [' ', '"', "'", '\n', '\t']:
                consecutive_char_count += 1
                max_char_repeat = max(max_char_repeat, consecutive_char_count)
            else:
                consecutive_char_count = 1

        if max_char_repeat > 20:
            return True

    # Word-level collapse.
    words = text.split()
    if len(words) <= min_words:
        return False

    consecutive_repeat_count = 1
    max_repeat = 1
    for i in range(1, len(words)):
        if words[i].lower() == words[i-1].lower():
            consecutive_repeat_count += 1
            max_repeat = max(max_repeat, consecutive_repeat_count)
        else:
            consecutive_repeat_count = 1

    return max_repeat > max_consecutive_repeat


def json_format_reward(response: str, require_edit_prompt_for_true: bool = False) -> float:
    """
    Check JSON format and required fields; return a format reward score.

    Args:
        response: model response string.
        require_edit_prompt_for_true: require edit_prompt even when answer=true.

    Returns:
        JSON format reward score in [0, 1].
    """
    r = (response or "").strip()

    ans = r
    for think_tag_start, think_tag_end in (("<thinking>", "</thinking>"), ("<think>", "</think>")):
        think_end = r.rfind(think_tag_end)
        if think_end != -1:
            ans = r[think_end + len(think_tag_end):].strip()
            break

        think_start = r.find(think_tag_start)
        if think_start != -1:
            ans = r[think_start + len(think_tag_start):].strip()
            break

    try:
        response_json = json.loads(ans.strip())

        # Double-encoded JSON parses as a string and is not accepted.
        if not isinstance(response_json, dict):
            return 0.0

        # The answer field is required before any other JSON checks.
        if "answer" not in response_json or not isinstance(response_json.get("answer"), bool):
            return 0.0

        answer = response_json.get("answer")

        if not ("explanation" in response_json and isinstance(response_json.get("explanation"), str)):
            return 0.0

        explanation = response_json.get("explanation", "").strip()

        if explanation and check_format_collapse(explanation):
            return 0.0

        if answer is False:
            if not explanation:
                return 0.0

        # Reject copied explanation placeholders.
        if explanation:
            explanation_lower = explanation.lower().strip()

            explicit_template_patterns = [
                "a brief, specific description of the main error (if answer is false)",
                "a brief, specific description of the main error",
                "brief, specific description of the main error (if answer is false)",
                "brief, specific description of the main error",
            ]
            for pattern in explicit_template_patterns:
                if explanation_lower == pattern or explanation_lower.startswith(pattern + ".") or explanation_lower.startswith(pattern + ","):
                    return 0.0

            has_brief_specific = "brief, specific description" in explanation_lower or "brief specific description" in explanation_lower
            has_main_error = "of the main error" in explanation_lower
            has_if_answer_false = "(if answer is false)" in explanation_lower or "if answer is false" in explanation_lower

            if len(explanation) < 80 and (has_brief_specific and has_main_error):
                return 0.0

            if len(explanation) < 120 and has_brief_specific and has_main_error and has_if_answer_false:
                return 0.0

        needs_edit_prompt = (answer is False) or require_edit_prompt_for_true

        if needs_edit_prompt:
            if not ("edit_prompt" in response_json and isinstance(response_json.get("edit_prompt"), str)):
                return 0.0

            edit_prompt = response_json.get("edit_prompt", "").strip()

            if answer is False:
                if not edit_prompt or len(edit_prompt) <= 10:
                    return 0.0

            if edit_prompt and check_format_collapse(edit_prompt):
                return 0.0

            # Use the same edit_prompt validator as extract_edit_info.
            if answer is False:
                if not is_valid_edit_prompt(edit_prompt):
                    return 0.0
            elif edit_prompt and not is_valid_edit_prompt(edit_prompt):
                return 0.0

        return 1.0
    except (json.JSONDecodeError, ValueError, TypeError):
        return 0.0


def accuracy_reward(response: str, ground_truth: str) -> float:
    """Compute the judgment accuracy reward."""
    try:
        answer = json.loads(ground_truth)
        match = re.search(r'"answer"\s*:\s*(true|false)', response, re.IGNORECASE)
        if match and response is not None:
            extracted_value = match.group(1).lower() == "true"
            is_same = extracted_value == answer.get('answer', False)
            if is_same:
                return 1.0
            else:
                return 0.0
    except:
        pass
    return 0.0


def extract_edit_info(response: str) -> Optional[Dict[str, str]]:
    """
    Extract edit_prompt and explanation from a response.

    The edit_prompt validator is shared with json_format_reward.
    """
    try:
        # First parse the answer region after the thinking tag.
        response_clean, has_think_tag = filter_thinking_part(response)

        try:
            response_json = json.loads(response_clean.strip())
            if isinstance(response_json, dict):
                edit_prompt = response_json.get("edit_prompt", "")
                explanation = response_json.get("explanation", "")

                if is_valid_edit_prompt(edit_prompt):
                    return {
                        "edit_prompt": edit_prompt.strip(),
                        "explanation": explanation,
                    }
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

        # Then search the full response for a balanced JSON object.
        json_start = response.find('{')
        if json_start != -1:
            brace_count = 0
            json_end = -1
            for i in range(json_start, len(response)):
                if response[i] == '{':
                    brace_count += 1
                elif response[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_end = i + 1
                        break

            if json_end != -1:
                try:
                    json_str = response[json_start:json_end]
                    response_json = json.loads(json_str)
                    if isinstance(response_json, dict):
                        edit_prompt = response_json.get("edit_prompt", "")
                        explanation = response_json.get("explanation", "")

                        if is_valid_edit_prompt(edit_prompt):
                            return {
                                "edit_prompt": edit_prompt.strip(),
                                "explanation": explanation,
                            }
                except (json.JSONDecodeError, ValueError, TypeError):
                    pass

        # Finally search inside the thinking region.
        think_end = -1
        for think_tag_end in ("</thinking>", "</think>"):
            think_end = response.rfind(think_tag_end)
            if think_end != -1:
                break
        if think_end != -1:
            think_content = response[:think_end]
            json_start = think_content.find('{')
            if json_start != -1:
                brace_count = 0
                json_end = -1
                for i in range(json_start, len(think_content)):
                    if think_content[i] == '{':
                        brace_count += 1
                    elif think_content[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            json_end = i + 1
                            break

                if json_end != -1:
                    try:
                        json_str = think_content[json_start:json_end]
                        response_json = json.loads(json_str)
                        if isinstance(response_json, dict):
                            edit_prompt = response_json.get("edit_prompt", "")
                            explanation = response_json.get("explanation", "")

                            if is_valid_edit_prompt(edit_prompt):
                                return {
                                    "edit_prompt": edit_prompt.strip(),
                                    "explanation": explanation,
                                }
                    except (json.JSONDecodeError, ValueError, TypeError):
                        pass

        return None

    except (json.JSONDecodeError, ValueError, TypeError, AttributeError, KeyError, Exception):
        return None


def compute_stage2_reward_api(
    edit_prompt: str,
    original_image_path: str,
    original_prompt: str,
    reward_type: str = "clip",
    edit_config: Optional[dict] = None,
    explanation: Optional[str] = None,
    ground_truth: Optional[str] = None,
) -> dict:
    """
    Compute the stage-2 reward via API.

    Args:
        edit_prompt: editing instruction
        original_image_path: path to the original image
        original_prompt: original text prompt
        reward_type: reward type
        edit_config: editing config (num_timesteps, cfg_scale, etc.)
        explanation: accepted for interface compatibility; currently unused
        ground_truth: optional JSON string; may carry a "category" field (SAM3 category,
                      defaults to "object") and a "generated_qa" payload passed to the
                      reward services

    Returns:
        {
            "score": reward score,
            "success": whether computation succeeded,
            "error": error message (if any)
        }
    """
    try:
        # Initialize clients.
        edit_client, reward_client, sam3_client = initialize_clients()

        if edit_client is None:
            error_msg = "Image edit client not available"
            print(f"[ERROR] compute_stage2_reward_api: {error_msg}", flush=True)
            return {"score": 0.0, "success": False, "error": error_msg}

        # Guard against a stale client object after endpoint reload.
        if not hasattr(edit_client, 'health_check_timeout'):
            error_msg = f"ImageEditClient missing required attributes. Please restart the training process."
            print(f"[ERROR] compute_stage2_reward_api: {error_msg}", flush=True)
            return {"score": 0.0, "success": False, "error": error_msg}

        # reward_client is only used in clip/self_reward mode.

        # Empty or no-op edit prompts cannot be evaluated by the edit service.
        edit_prompt_clean = edit_prompt.strip() if edit_prompt else ""
        if not edit_prompt_clean or edit_prompt_clean.lower() in ["remain unchanged", "no edit"]:
            return {
                "score": 0.0,
                "success": False,
                "error": "edit_prompt is empty or invalid"
            }

        # Load the original image.
        if not os.path.exists(original_image_path):
            return {
                "score": 0.0,
                "success": False,
                "error": f"Image file not found: {original_image_path}"
            }
        if not os.path.isfile(original_image_path):
            return {
                "score": 0.0,
                "success": False,
                "error": f"Path is not a file: {original_image_path}"
            }
        original_image = Image.open(original_image_path).convert("RGB")

        edit_instruction = edit_prompt_clean

        # Call the image editing service.
        edit_config = edit_config or {}
        num_timesteps = edit_config.get("num_timesteps", 40)
        cfg_scale = edit_config.get("cfg_scale", 4.0)
        true_cfg_scale = edit_config.get("true_cfg_scale", 4.0)
        timestep_shift = edit_config.get("timestep_shift", 3.0)
        resolution_scale = edit_config.get("resolution_scale", 0.75)

        edited_image = edit_client.edit_image(
            image=original_image,
            edit_prompt=edit_instruction,
            num_timesteps=num_timesteps,
            cfg_scale=cfg_scale,
            true_cfg_scale=true_cfg_scale,
            timestep_shift=timestep_shift,
            resolution_scale=resolution_scale,
            model_type="qwen_image_edit",
        )

        # Reward services expect RGB images.
        edited_image = edited_image.convert("RGB")

        generated_qa = None
        if ground_truth:
            try:
                gt_data_for_reward = json.loads(ground_truth)
                if isinstance(gt_data_for_reward, dict):
                    generated_qa = gt_data_for_reward.get("generated_qa")
            except (json.JSONDecodeError, TypeError, AttributeError):
                generated_qa = None

        # Call the reward service selected by reward_type.
        score = 0.0
        if reward_type == "sam3":
            if sam3_client is None:
                return {"score": 0.0, "success": False, "error": "SAM3 reward client not available"}

            category = None
            if ground_truth:
                try:
                    gt_data = json.loads(ground_truth)
                    if isinstance(gt_data, dict):
                        category = gt_data.get("category", "")
                except (json.JSONDecodeError, TypeError, AttributeError):
                    pass
            else:
                print("No ground truth provided, using default category: object", flush=True)

            if not category:
                print("No category found in ground truth, using default category: object", flush=True)
                category = "object"

            try:
                sam3_result = sam3_client.compute_reward(
                    image=edited_image,
                    prompt=original_prompt,
                    reward_type="sam3",
                    category=category,
                    ground_truth=ground_truth,
                )
                score = sam3_result.get("score", 0.0)
            except Exception as e:
                return {"score": 0.0, "success": False, "error": str(e)}
        elif reward_type == "mixed":
            # Combine SAM3 and the base reward with equal weight.
            if sam3_client is None:
                error_msg = "SAM3 reward client not available"
                print(f"[ERROR] compute_stage2_reward_api: {error_msg}", flush=True)
                return {"score": 0.0, "success": False, "error": error_msg}

            category = None
            if ground_truth:
                try:
                    gt_data = json.loads(ground_truth)
                    if isinstance(gt_data, dict):
                        category = gt_data.get("category", "")
                except (json.JSONDecodeError, TypeError, AttributeError):
                    pass

            if not category:
                category = "object"

            # REWARD_TYPE_PER_GPU can route workers to different base reward types.
            base_reward_type = _get_reward_type_for_current_gpu()

            if base_reward_type is None:
                base_reward_type = os.environ.get("REWARD_TYPE", "self_reward")
            base_reward_type = _normalize_reward_type(base_reward_type)
            if base_reward_type == "mixed":
                    base_reward_type = "self_reward"

            # The base branch handles CLIP/self_reward; SAM3 is evaluated separately.
            if base_reward_type in ["mixed", "sam3"]:
                base_reward_type = "self_reward"

            base_reward_client = get_reward_client_for_type(base_reward_type)
            if base_reward_client is None:
                error_msg = f"{base_reward_type} reward client not available"
                print(f"[ERROR] compute_stage2_reward_api: {error_msg}", flush=True)
                return {"score": 0.0, "success": False, "error": error_msg}

            def _call_sam3():
                try:
                    return sam3_client.compute_reward(
                        image=edited_image,
                        prompt=original_prompt,
                        reward_type="sam3",
                        category=category,
                        ground_truth=ground_truth,
                    )
                except RuntimeError as e:
                    error_msg = str(e) if e and str(e) else "Unknown RuntimeError in _call_sam3"
                    print(f"[ERROR] _call_sam3 RuntimeError: {error_msg}", flush=True)
                    return {"success": False, "error": error_msg, "score": 0.0}
                except Exception as e:
                    import traceback
                    error_msg = str(e) if e and str(e) else "Unknown exception in _call_sam3"
                    error_detail = f"{error_msg}\n{traceback.format_exc()}"
                    print(f"[ERROR] _call_sam3 exception: {error_detail}", flush=True)
                    return {"success": False, "error": error_msg, "score": 0.0}

            def _call_base():
                try:
                    result = base_reward_client.compute_reward(
                        image=edited_image,
                        prompt=original_prompt,
                        reward_type=base_reward_type,
                        generated_qa=generated_qa,
                    )
                    return result
                except RuntimeError as e:
                    error_msg = str(e) if e and str(e) else "Unknown RuntimeError in _call_base"
                    print(f"[ERROR] _call_base RuntimeError: {error_msg}", flush=True)
                    return {"success": False, "error": error_msg, "raw_score": 0.0}
                except Exception as e:
                    import traceback
                    error_msg = str(e) if e and str(e) else "Unknown exception in _call_base"
                    error_detail = f"{error_msg}\n{traceback.format_exc()}"
                    print(f"[ERROR] _call_base exception: {error_detail}", flush=True)
                    return {"success": False, "error": error_msg, "raw_score": 0.0}

            with ThreadPoolExecutor(max_workers=2) as executor:
                f_sam3 = executor.submit(_call_sam3)
                f_base = executor.submit(_call_base)
                try:
                    sam3_result = f_sam3.result()
                    base_result = f_base.result()
                except Exception as e:
                    return {"score": 0.0, "success": False, "error": f"Thread execution failed: {str(e)}"}

            # Validate SAM3 result.
            if not isinstance(sam3_result, dict):
                return {"score": 0.0, "success": False, "error": f"SAM3 result is not a dict: {type(sam3_result)}"}

            if sam3_result.get("success") is False:
                error_msg = sam3_result.get("error", "")
                if not error_msg:
                    error_msg = "SAM3 reward computation failed (no error message provided)"
                print(f"[ERROR] compute_stage2_reward_api: sam3_result failed: {error_msg}", flush=True)
                return {"score": 0.0, "success": False, "error": str(error_msg)}

            # Validate base reward result.
            if not base_result:
                return {"score": 0.0, "success": False, "error": "Base reward result is empty"}

            if not isinstance(base_result, dict):
                return {"score": 0.0, "success": False, "error": f"Base result is not a dict: {type(base_result)}"}

            if base_result.get("success") is False:
                error_msg = base_result.get("error", "")
                if not error_msg:
                    error_msg = "Base reward computation failed (no error message provided)"
                print(f"[ERROR] compute_stage2_reward_api: base_reward call failed: {error_msg}", flush=True)
                return {"score": 0.0, "success": False, "error": error_msg}

            if "score" not in sam3_result:
                return {"score": 0.0, "success": False, "error": "SAM3 reward result missing score field"}

            sam3_score = sam3_result.get("score", 0.0)

            if "raw_score" not in base_result and "score" not in base_result:
                return {"score": 0.0, "success": False, "error": "Base reward result missing score/raw_score field"}

            base_score = base_result.get("raw_score", base_result.get("score", 0.0))
            score = 0.5 * sam3_score + 0.5 * base_score

            return {
                "score": score,
                "success": True,
                "sam3_score": sam3_score,
                "base_score": base_score,
                "edited_image": edited_image
            }
        else:
            # CLIP/self_reward branch.
            target_reward_client = get_reward_client_for_type(reward_type)
            if target_reward_client is None:
                return {"score": 0.0, "success": False, "error": f"{reward_type} reward client not available"}

            reward_result = target_reward_client.compute_reward(
                image=edited_image,
                prompt=original_prompt,
                reward_type=reward_type,
                generated_qa=generated_qa,
            )
            score = reward_result["raw_score"]

        return {
            "score": score,
            "success": True,
            "edited_image": edited_image
        }

    except Exception as e:
        error_msg = str(e) if e else "Unknown exception in compute_stage2_reward_api"
        if not error_msg:
            error_msg = "Unknown exception in compute_stage2_reward_api"
        import traceback
        error_detail = f"{error_msg}\n{traceback.format_exc()}"
        print(f"[ERROR] compute_stage2_reward_api exception: {error_detail[:500]}", flush=True)
        return {
            "score": 0.0,
            "success": False,
            "error": error_msg
        }


def compute_score(
    reward_inputs: list[dict[str, Any]],
    think_format_weight: float = 0.1,
    json_format_weight: float = 0.1,
    stage1_weight: float = 0.4,
    stage2_weight: float = 0.6,
    enable_stage2: bool = True,
    image_dir: Optional[str] = None,
    edit_config: Optional[dict] = None,
    max_workers: Optional[int] = None,
    default_reward_type: Optional[str] = None,
    virtual_correct_reward: float = 0.0,
    require_edit_prompt_for_true: bool = True,
    **kwargs
) -> list[dict[str, float]]:
    """
    Compute staged rewards using remote edit and reward services.

    Args:
        max_workers: parallel stage-2 worker count. If None, it matches the edit endpoint count.
        default_reward_type: override reward type for all samples.
        virtual_correct_reward: optional reward for true-positive samples.
        image_dir: base directory for resolving relative image paths.
        require_edit_prompt_for_true: require edit_prompt even when answer=true.
    """
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for self_reward.")

    # Argument wins over the environment variable.
    if default_reward_type is None:
        default_reward_type = os.environ.get("REWARD_TYPE", None)
    if default_reward_type is not None:
        default_reward_type = _normalize_reward_type(default_reward_type)

    if default_reward_type is not None:
        if default_reward_type not in _SUPPORTED_REWARD_TYPES:
            if not _client_warning_printed.get("default_reward_type", False):
                print(f"Warning: unsupported reward type: {default_reward_type}, supported types: {_SUPPORTED_REWARD_TYPES}")
                print("  Ignoring the override and using per-sample reward routing.")
                _client_warning_printed["default_reward_type"] = True
            default_reward_type = None
        else:
            if not _client_warning_printed.get("default_reward_type", False):
                print(f"[INFO] Using externally specified reward type: {default_reward_type}")
                _client_warning_printed["default_reward_type"] = True

    # Initialize clients.
    try:
        edit_client, reward_client, _ = initialize_clients()
    except Exception as e:
        if enable_stage2:
            raise RuntimeError(
                f"Stage-2 reward requires the image edit/reward services, but API client "
                f"initialization failed: {e}. Start the services (see "
                f"R3-Refiner/docs/SERVICES.md), or pass enable_stage2=False to run a "
                f"stage-1-only reward intentionally."
            ) from e
        edit_client = None
        reward_client = None

    # Stage-2 needs the edit service; do not silently change the training objective.
    if enable_stage2 and edit_client is None:
        raise RuntimeError(
            "Stage-2 reward is enabled but no image edit service endpoint is reachable. "
            "Start the edit/reward services (see R3-Refiner/docs/SERVICES.md), or pass "
            "enable_stage2=False to run a stage-1-only reward intentionally."
        )

    # Match worker count to endpoint count unless the caller overrides it.
    if max_workers is None:
        if edit_client is not None and len(edit_client.endpoints) > 0:
            max_workers = len(edit_client.endpoints)
            if not _client_warning_printed.get("max_workers", False):
                print(f"[INFO] Auto-setting max_workers={max_workers} (endpoint count={len(edit_client.endpoints)})")
                _client_warning_printed["max_workers"] = True
        else:
            max_workers = 1
            if not _client_warning_printed.get("max_workers", False):
                print(f"[INFO] No edit_client detected; using default max_workers=1")
                _client_warning_printed["max_workers"] = True

    batch_image_paths = kwargs.get("image_paths", None)
    batch_prompts = kwargs.get("prompts", None)

    # Stage 1: format and judgment accuracy.
    stage1_results = []
    stage2_tasks = []

    for idx, reward_input in enumerate(reward_inputs):
        if not isinstance(reward_input, dict):
            print(f"Warning: reward_input[{idx}] is not a dict, skipping")
            continue
        if "response" not in reward_input:
            print(f"Warning: reward_input[{idx}] missing 'response' key, skipping")
            continue
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])

        # Think format is required before the answer JSON can be parsed.
        think_score = think_format_reward(response)

        response_clean, _ = filter_thinking_part(response)
        accuracy_score = accuracy_reward(response_clean, reward_input["ground_truth"])

        # Stage-1 reward: judgment accuracy plus a small format reward.
        accuracy_weight = 1.0 - think_format_weight
        if accuracy_weight < 0:
            print(f"[WARNING] Invalid weight assignment: think_format_weight={think_format_weight}, accuracy_weight={accuracy_weight}")
            print(f"  Ensure think_format_weight <= 1.0")
            accuracy_weight = max(0.0, accuracy_weight)

        stage1_reward = (
            accuracy_weight * accuracy_score +
            think_format_weight * think_score
        )

        # JSON score is used by stage 2 because edit_prompt must be parseable.
        json_score = json_format_reward(response, require_edit_prompt_for_true=require_edit_prompt_for_true)

        model_answer = None
        try:
            model_answer_match = re.search(r'"answer"\s*:\s*(true|false)', response_clean, re.IGNORECASE)
            if model_answer_match:
                model_answer = model_answer_match.group(1).lower() == "true"
        except (AttributeError, IndexError):
            model_answer = False

        stage1_results.append({
            "idx": idx,
            "think_score": think_score,
            "json_score": json_score,
            "model_answer": model_answer,
            "accuracy_score": accuracy_score,
            "stage1_reward": stage1_reward,
            "response_clean": response_clean,
        })

        # Prepare stage-2 reward task.
        stage2_task = None

        if enable_stage2:
            try:
                if "ground_truth" not in reward_input:
                    print(f"Warning: reward_input[{idx}] missing 'ground_truth' key, skipping stage2")
                    continue
                gt_data = json.loads(reward_input["ground_truth"])
                if not isinstance(gt_data, dict):
                    print(f"Warning: reward_input[{idx}] ground_truth is not a dict, skipping stage2")
                    continue
                gt_answer = gt_data.get("answer", False)

                category = gt_data.get("category", "")
                # Reward type priority: function argument, data field, category mapping.
                if default_reward_type is not None:
                    reward_type = default_reward_type
                else:
                    reward_type = gt_data.get("reward_type")
                    if not reward_type:
                        reward_type = _CATEGORY_TO_REWARD_TYPE.get(category, "self_reward")
                reward_type = _normalize_reward_type(reward_type)

                model_answer_match = re.search(r'"answer"\s*:\s*(true|false)', response_clean, re.IGNORECASE)
                model_answer = False
                if model_answer_match:
                    try:
                        model_answer = model_answer_match.group(1).lower() == "true"
                    except (AttributeError, IndexError):
                        model_answer = False

                # Image editing is evaluated only for true-negative judgments.
                if not model_answer and not gt_answer:
                    edit_info = extract_edit_info(response)

                    if edit_info and edit_info.get("edit_prompt"):
                        edit_prompt = edit_info.get("edit_prompt")
                        explanation = edit_info.get("explanation", "")

                        image_path = ""
                        # Prefer the images field used by the training dataloader.
                        if "images" in reward_input and isinstance(reward_input["images"], list) and len(reward_input["images"]) > 0:
                            image_path = reward_input["images"][0]
                        if not image_path and batch_image_paths is not None and idx < len(batch_image_paths):
                            image_path = batch_image_paths[idx]
                        if not image_path:
                            image_path = reward_input.get("image_path", "")
                        if not image_path:
                            image_path = gt_data.get("image_path", "")

                        prompt = reward_input.get("prompt", "")
                        if not prompt:
                            prompt = gt_data.get("prompt", "")
                        if not prompt and batch_prompts is not None and idx < len(batch_prompts):
                            prompt = batch_prompts[idx]

                        # Resolve relative image paths against image_dir.
                        if image_path and image_dir and not os.path.isabs(image_path):
                            image_path = os.path.join(image_dir, image_path)

                        if image_path and os.path.exists(image_path) and os.path.isfile(image_path) and prompt:
                            stage2_task = {
                                "idx": idx,
                                "edit_prompt": edit_prompt,
                                "explanation": explanation,
                                "image_path": image_path,
                                "prompt": prompt,
                                "reward_type": reward_type,
                                "edit_config": edit_config,
                                "json_score": json_score,
                                "ground_truth": reward_input["ground_truth"],
                                "is_false_false": True,
                            }
                        else:
                            stage2_task = {
                                "idx": idx,
                                "error": "Missing image_path or prompt",
                                "stage2_reward": 0.0,
                                "json_score": json_score,
                                "is_false_false": True,
                            }
                    else:
                        stage2_task = {
                            "idx": idx,
                            "error": "No edit_prompt extracted",
                            "stage2_reward": 0.0,
                            "json_score": json_score,
                            "is_false_false": True,
                        }
                elif model_answer and gt_answer:
                    # True-positive samples can receive an optional virtual reward.
                    stage2_reward = (1 - json_format_weight) * virtual_correct_reward + json_format_weight * json_score

                    stage2_task = {
                        "idx": idx,
                        "json_score": json_score,
                        "stage2_reward": stage2_reward,
                        "stage2_details": {
                            "type": "TP",
                            "json_score": json_score,
                        },
                    }
                elif model_answer and not gt_answer:
                    stage2_task = {
                        "idx": idx,
                        "stage2_reward": 0.0,
                        "stage2_details": {
                            "type": "FP",
                            "json_score": 0.0,
                        },
                    }
                elif not model_answer and gt_answer:
                    stage2_task = {
                        "idx": idx,
                        "stage2_reward": 0.0,
                        "stage2_details": {
                            "type": "FN",
                            "json_score": 0.0,
                        },
                    }

            except Exception as e:
                # Ground-truth parsing failed; skip stage-2 scoring for this sample.
                stage2_task = {
                    "idx": idx,
                    "error": f"Failed to parse ground_truth (data issue): {e}",
                    "stage2_reward": 0.0,
                    "stage2_details": {"type": "error", "reason": f"Parse error: {e}"},
                    "json_score": 0.0,
                }

        if stage2_task is not None:
            stage2_tasks.append(stage2_task)

    # Stage 2: edit and score selected samples.
    stage2_results = {}

    # compute_stage2_reward_api resolves the reward client from reward_type.
    if stage2_tasks and enable_stage2 and edit_client is not None:

        def process_stage2_task(task: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
            """Process a single stage-2 reward task."""
            idx = task["idx"]

            # Some tasks already have a deterministic stage-2 result.
            if "error" in task or "stage2_reward" in task:
                stage2_details = task.get("stage2_details", {})
                if not stage2_details or "type" not in stage2_details:
                    stage2_details = {
                        "type": "error",
                        "reason": task.get("error", "Unknown error")
                    }
                if task.get("is_false_false", False):
                    stage2_details["is_false_false"] = True

                stage2_details["json_score"] = task.get("json_score", 0.0)
                stage2_reward = task.get("stage2_reward", 0.0)

                return idx, {
                    "stage2_reward": stage2_reward,
                    "stage2_details": stage2_details,
                }

            # Call the edit and reward services.
            try:
                stage2_result = compute_stage2_reward_api(
                    edit_prompt=task["edit_prompt"],
                    original_image_path=task["image_path"],
                    original_prompt=task["prompt"],
                    reward_type=task["reward_type"],
                    edit_config=task["edit_config"],
                    explanation=task.get("explanation"),
                    ground_truth=task.get("ground_truth"),
                )

                if not stage2_result.get("success", False):
                    error_msg = stage2_result.get("error", "")
                    if not error_msg:
                        error_msg = "Stage2 reward computation failed (no error message provided)"
                    json_score = task.get("json_score", 0.0)
                    stage2_reward = json_format_weight * json_score
                    return idx, {
                        "stage2_reward": stage2_reward,
                        "stage2_details": {
                            "type": "error",
                            "reason": error_msg,
                            "json_score": json_score,
                            "is_false_false": task.get("is_false_false", False),
                        },
                    }

                # GRPO handles score normalization within each group.
                edit_reward = stage2_result["score"]

                # Stage-2 reward combines JSON validity and edit quality.
                edit_reward_weight = 1.0 - json_format_weight
                if edit_reward_weight < 0:
                    edit_reward_weight = max(0.0, edit_reward_weight)

                json_score = task.get("json_score", 0.0)

                stage2_reward = (
                    json_format_weight * json_score +
                    edit_reward_weight * edit_reward
                )

                # Display all edited false+false cases as TN.
                display_type = "TN" if task.get("is_false_false", False) else task["reward_type"]

                stage2_details = {
                    "type": display_type,
                    "reward_type": task["reward_type"],
                    "success": True,
                    "edit_reward": edit_reward,
                    "json_score": json_score,
                    "is_false_false": task.get("is_false_false", False),
                }
                if stage2_result.get("error"):
                    stage2_details["error"] = stage2_result.get("error")

                return idx, {
                    "stage2_reward": stage2_reward,
                    "stage2_details": stage2_details,
                }
            except Exception as e:
                json_score = task.get("json_score", 0.0)
                stage2_reward = json_format_weight * json_score
                return idx, {
                    "stage2_reward": stage2_reward,
                    "stage2_details": {
                        "type": "error",
                        "reason": str(e),
                        "json_score": json_score,
                    },
                }

        # Run stage-2 tasks in parallel.
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {executor.submit(process_stage2_task, task): task for task in stage2_tasks}

            for future in as_completed(future_to_task):
                try:
                    idx, result = future.result()
                    stage2_results[idx] = result
                except Exception as e:
                    task = future_to_task[future]
                    idx = task["idx"]
                    stage2_results[idx] = {
                        "stage2_reward": 0.0,
                        "stage2_details": {"type": "error", "reason": f"Task execution failed: {e}"},
                    }

    # Merge staged rewards.
    scores = []
    for idx, stage1_result in enumerate(stage1_results):
        model_answer = stage1_result.get("model_answer")
        stage2_result = stage2_results.get(idx, {
            "stage2_reward": 0.0,
            "stage2_details": {"type": "none"}
        })

        overall_reward = (
            stage1_weight * stage1_result["stage1_reward"] +
            stage2_weight * stage2_result["stage2_reward"]
        )

        stage2_edit_reward = stage2_result["stage2_details"].get("edit_reward", 0.0)
        is_false_false = stage2_result["stage2_details"].get("is_false_false", False)

        scores.append({
            "overall": overall_reward,
            "stage1_reward": stage1_result["stage1_reward"],
            "stage2_reward": stage2_result["stage2_reward"],
            "think_format": stage1_result["think_score"],
            "json_format": stage1_result["json_score"],
            "stage1_accuracy": stage1_result["accuracy_score"],
            "stage2_accuracy": stage2_edit_reward,
            "TN_counts": 1.0 if is_false_false else 0.0,
            "stage2_details": stage2_result["stage2_details"]
        })

    # Training logs summarize the stage-2 routing and failure modes.
    stage2_type_counts = {}
    stage2_stats = {
        "virtual_correct_count": 0,
        "virtual_correct_reward_sum": 0.0,
        "computed_count": 0,
        "computed_reward_sum": 0.0,
        "error_count": 0,
        "error_reasons": {},
        "virtual_neutral_fp_count": 0,
        "virtual_neutral_fn_count": 0,
        "false_false_count": 0,
        "false_false_computed_count": 0,
        "false_false_computed_rewards": [],
        "false_false_error_count": 0,
        "false_false_error_reasons": {},
    }

    for score in scores:
        stage2_type = score.get("stage2_details", {}).get("type", "none")
        stage2_type_counts[stage2_type] = stage2_type_counts.get(stage2_type, 0) + 1

        stage2_reward = score.get("stage2_reward", 0.0)
        stage2_details = score.get("stage2_details", {})
        stage2_reason = stage2_details.get("reason", "")
        stage2_error = stage2_details.get("error")

        is_false_false = stage2_details.get("is_false_false", False)

        if stage2_type == "TP":
            stage2_stats["virtual_correct_count"] += 1
            stage2_stats["virtual_correct_reward_sum"] += stage2_reward
        elif stage2_type == "TN" or stage2_type in ["clip", "self_reward", "sam3", "mixed"]:
            stage2_stats["computed_count"] += 1
            stage2_stats["computed_reward_sum"] += stage2_reward
            if is_false_false:
                stage2_stats["false_false_computed_count"] += 1
                stage2_stats["false_false_computed_rewards"].append(stage2_reward)
        elif stage2_type == "error":
            stage2_stats["error_count"] += 1
            error_reason = stage2_reason or "Unknown error"
            if stage2_error:
                error_reason = f"{error_reason} ({stage2_error[:50]})"
            stage2_stats["error_reasons"][error_reason] = stage2_stats["error_reasons"].get(error_reason, 0) + 1
            if is_false_false:
                stage2_stats["false_false_error_count"] += 1
                stage2_stats["false_false_error_reasons"][error_reason] = stage2_stats["false_false_error_reasons"].get(error_reason, 0) + 1
        elif stage2_type == "FP":
            stage2_stats["virtual_neutral_fp_count"] += 1
        elif stage2_type == "FN":
            stage2_stats["virtual_neutral_fn_count"] += 1


        if is_false_false:
            stage2_stats["false_false_count"] += 1

    if stage2_type_counts:
        batch_size = len(scores)
        total = sum(stage2_type_counts.values())

        # Merge all image-edit computed types under TN for display.
        unified_type_counts = {}
        tn_count = 0
        for k, v in stage2_type_counts.items():
            if k == "TN":
                tn_count += v
            elif k in ["self_reward", "clip", "sam3", "mixed"]:
                tn_count += v
            else:
                unified_type_counts[k] = v
        if tn_count > 0:
            unified_type_counts["TN"] = tn_count

        print(f"[INFO] Stage2 reward type distribution (batch_size={batch_size}): {dict(sorted(unified_type_counts.items(), key=lambda x: x[1], reverse=True))}")
        print(f"[INFO] Stage2 reward type ratios: {', '.join([f'{k}: {v/total*100:.1f}%' for k, v in sorted(unified_type_counts.items(), key=lambda x: x[1], reverse=True)])}")
        print()

        if stage2_stats["virtual_correct_count"] > 0:
            avg_virtual = stage2_stats["virtual_correct_reward_sum"] / stage2_stats["virtual_correct_count"]
            if virtual_correct_reward > 0:
                print(f"[INFO] Stage2 detail - TP (model=true, gt=true, stage2_reward=(1-json_format_weight)x{virtual_correct_reward} + json_format_weight*json_score): {stage2_stats['virtual_correct_count']} ({stage2_stats['virtual_correct_count']/total*100:.1f}%), avg stage2_reward: {avg_virtual:.3f}")
            else:
                print(f"[INFO] Stage2 detail - TP (model=true, gt=true, stage2_reward=json_format_weight*json_score): {stage2_stats['virtual_correct_count']} ({stage2_stats['virtual_correct_count']/total*100:.1f}%), avg stage2_reward: {avg_virtual:.3f}")

        # TN breakdown.
        if stage2_stats["false_false_count"] > 0:
            print(f"[INFO] Stage2 detail - TN (model=false, gt=false, edit required): {stage2_stats['false_false_count']} ({stage2_stats['false_false_count']/total*100:.1f}%)")

            if stage2_stats["false_false_computed_count"] > 0:
                false_false_rewards = stage2_stats["false_false_computed_rewards"]
                avg_false_false = sum(false_false_rewards) / len(false_false_rewards)

                print(f"      OK  Edit reward computed successfully: {stage2_stats['false_false_computed_count']}")
                print(f"         - avg reward: {avg_false_false:.3f}")


            if stage2_stats["false_false_error_count"] > 0:
                print(f"      Error  Edit reward could not be computed: {stage2_stats['false_false_error_count']}")
                for reason, count in sorted(stage2_stats["false_false_error_reasons"].items(), key=lambda x: x[1], reverse=True):
                    print(f"         - {reason}: {count}")

        non_tn_error_count = stage2_stats["error_count"] - stage2_stats["false_false_error_count"]
        if non_tn_error_count > 0:
            print(f"[INFO] Stage2 detail - Error (non-TN errors, stage-2 reward could not be computed): {non_tn_error_count} ({non_tn_error_count/total*100:.1f}%)")
            non_tn_error_reasons = {}
            for reason, count in stage2_stats["error_reasons"].items():
                tn_error_count = stage2_stats["false_false_error_reasons"].get(reason, 0)
                if count > tn_error_count:
                    non_tn_error_reasons[reason] = count - tn_error_count
            for reason, count in sorted(non_tn_error_reasons.items(), key=lambda x: x[1], reverse=True):
                print(f"      - {reason}: {count}")

        virtual_neutral_total = stage2_stats["virtual_neutral_fp_count"] + stage2_stats["virtual_neutral_fn_count"]
        if virtual_neutral_total > 0:
            print(f"[INFO] Stage2 detail - Mismatch (model and gt disagree, stage2_reward=0.0): {virtual_neutral_total} ({virtual_neutral_total/total*100:.1f}%)")
            if stage2_stats["virtual_neutral_fp_count"] > 0:
                print(f"      - FP (False Positive, model=true, gt=false): {stage2_stats['virtual_neutral_fp_count']}")
            if stage2_stats["virtual_neutral_fn_count"] > 0:
                print(f"      - FN (False Negative, model=false, gt=true): {stage2_stats['virtual_neutral_fn_count']}")

    return scores
