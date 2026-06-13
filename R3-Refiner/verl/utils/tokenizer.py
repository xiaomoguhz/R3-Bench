
"""Utils for tokenization."""

import json
import os
from typing import Optional

from transformers import (
    AutoProcessor,
    AutoTokenizer,
    PreTrainedTokenizer,
    ProcessorMixin,
)

def _is_llava_onevision_model(model_path: str) -> bool:
    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        return False

    try:
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
    except Exception:
        return False

    architectures = config.get("architectures") or []
    auto_map = config.get("auto_map") or {}
    model_type = config.get("model_type")
    auto_config = str(auto_map.get("AutoConfig", ""))

    return (
        model_type == "llavaonevision1_5"
        or any("LLaVAOneVision1_5" in arch for arch in architectures)
        or "llavaonevision1_5" in auto_config.lower()
    )


def _load_llava_onevision_processor(model_path: str, **kwargs) -> ProcessorMixin:
    try:
        from transformers import Qwen2_5_VLProcessor
    except ImportError as exc:
        raise RuntimeError(
            "LLaVA-OneVision-1.5 fallback requires transformers with Qwen2.5-VL processor support."
        ) from exc

    return Qwen2_5_VLProcessor.from_pretrained(model_path, **kwargs)


def get_tokenizer(model_path: str, override_chat_template: Optional[str] = None, **kwargs) -> PreTrainedTokenizer:
    """Create a huggingface pretrained tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_path, **kwargs)
    if override_chat_template is not None:
        tokenizer.chat_template = override_chat_template

    if tokenizer.bos_token == "<bos>" and tokenizer.eos_token == "<eos>":
        # the EOS token in gemma2 & gemma3 is ambiguious, which may worsen RL performance.
        # https://huggingface.co/google/gemma-2-2b-it/commit/17a01657f5c87135bcdd0ec7abb4b2dece04408a
        print("Found gemma model. Set eos_token and eos_token_id to <end_of_turn> and 107.")
        tokenizer.eos_token = "<end_of_turn>"

    if tokenizer.pad_token_id is None:
        print("Pad token is None. Set it to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def get_processor(model_path: str, override_chat_template: Optional[str] = None, **kwargs) -> Optional[ProcessorMixin]:
    """Create a huggingface pretrained processor."""
    try:
        processor = AutoProcessor.from_pretrained(model_path, **kwargs)
    except Exception:
        if _is_llava_onevision_model(model_path):
            processor = _load_llava_onevision_processor(model_path, **kwargs)
        else:
            raise
    if override_chat_template is not None:
        processor.chat_template = override_chat_template

    # Avoid load tokenizer, see:
    # https://github.com/huggingface/transformers/blob/v4.52.4/src/transformers/models/auto/processing_auto.py#L386
    if processor is not None and "Processor" not in processor.__class__.__name__:
        if _is_llava_onevision_model(model_path):
            processor = _load_llava_onevision_processor(model_path, **kwargs)
            if override_chat_template is not None:
                processor.chat_template = override_chat_template
        else:
            processor = None

    return processor
