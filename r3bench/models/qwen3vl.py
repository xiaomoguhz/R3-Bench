# Copyright 2026 R3-Bench Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Qwen3-VL reflection backend (uses Qwen3-VL-8B-Instruct by default)."""

from pathlib import Path
from typing import Any, Dict

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from r3bench.prompts import format_reflection_prompt
from r3bench.utils import parse_reflection_response_qwen3, get_logger

logger = get_logger("models.qwen3vl")


def load_model(args, device):
    """Load Qwen3-VL model and processor."""
    model_path = getattr(args, 'model_path', 'Qwen/Qwen3-VL-8B-Instruct')
    logger.info(f"Loading Qwen3-VL: {model_path}")

    model_kwargs = {"torch_dtype": torch.bfloat16}
    attn_implementation = getattr(args, "attn_implementation", None)
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path, **model_kwargs
    ).to(device)
    processor_kwargs = {}
    qwen_use_fast = getattr(args, "qwen_use_fast", "auto")
    if qwen_use_fast != "auto":
        processor_kwargs["use_fast"] = qwen_use_fast == "true"
    processor = AutoProcessor.from_pretrained(model_path, **processor_kwargs)

    logger.info("Qwen3-VL loaded")
    return model, processor


def _generate_with_params(model, inputs, max_new_tokens, temperature=None, use_cache=True):
    gen_kwargs = {"max_new_tokens": max_new_tokens, "use_cache": use_cache}
    if temperature is not None and temperature > 0:
        gen_kwargs.update({
            "do_sample": True,
            "temperature": temperature,
            "top_p": 0.95,
        })
    else:
        gen_kwargs["do_sample"] = False
    return model.generate(**inputs, **gen_kwargs)


def _is_parse_failed(result: Dict[str, Any]) -> bool:
    if result.get("explanation") == "Failed to parse model output":
        return True
    if (not result.get("answer")) and not result.get("edit_prompt", "").strip():
        return True
    return False


def reflect(
    record: Dict[str, Any],
    benchmark_dir: Path,
    model_components,
    device: torch.device,
    **kwargs,
) -> Dict[str, Any]:
    """Run reflection on a single record using Qwen3-VL, retrying with temperature
    sampling if greedy decoding produces an unparseable response."""
    model, processor = model_components
    origin_prompt = record["original_prompt"]
    bad_image_rel = record["bad_image"]
    bad_image_path = benchmark_dir / bad_image_rel

    # Qwen3-VL occasionally emits malformed JSON under greedy decoding; falling
    # back to ascending-temperature samples is a cheap way to escape the bad
    # mode without re-prompting and biases the retries toward the parsable
    # neighbourhood of the greedy output.
    max_retries = kwargs.get("max_retries", 9)
    retry_temperatures = kwargs.get("retry_temperatures", [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    question = format_reflection_prompt(
        origin_prompt,
        with_system=True,
        style=kwargs.get("prompt_style", "default"),
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(bad_image_path)},
                {"type": "text", "text": question},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    logger.debug("[attempt 1] greedy decoding (temperature=0)")
    generated_ids = _generate_with_params(
        model,
        inputs,
        max_new_tokens=2048,
        temperature=None,
        use_cache=kwargs.get("use_cache", True),
    )
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    result = parse_reflection_response_qwen3(output_text)
    if not _is_parse_failed(result):
        logger.debug("greedy decoding parsed successfully")
        return result

    logger.warning(f"Greedy decoding parse failed; retrying (up to {max_retries} times)")
    for idx, temperature in enumerate(retry_temperatures[:max_retries], start=1):
        logger.info(f"[attempt {idx + 1}] temperature sampling (temperature={temperature})")
        with torch.no_grad():
            generated_ids = _generate_with_params(
                model, inputs,
                max_new_tokens=2048,
                temperature=temperature,
                use_cache=kwargs.get("use_cache", True),
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
        result = parse_reflection_response_qwen3(output_text)
        if not _is_parse_failed(result):
            logger.info(f"retry succeeded at temperature={temperature}")
            return result
        else:
            logger.warning(f"retry {idx} failed at temperature={temperature}")

    logger.error("All retries failed; returning last (likely parse-failure) result")
    return result
