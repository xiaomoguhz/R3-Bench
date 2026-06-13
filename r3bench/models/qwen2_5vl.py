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

"""
Qwen2.5-VL reflection backend.

Uses Qwen/Qwen2.5-VL-7B-Instruct (or any compatible Qwen2.5-VL checkpoint passed
via --model-path) to inspect a generated image against its prompt.
"""

from pathlib import Path
from typing import Any, Dict

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from r3bench.prompts import format_reflection_prompt
from r3bench.utils import parse_reflection_response, get_logger

logger = get_logger("models.qwen2_5vl")


def load_model(args, device):
    """Load Qwen2.5-VL model and processor."""
    model_kwargs = {"torch_dtype": torch.bfloat16}
    attn_implementation = getattr(args, "attn_implementation", None)
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, **model_kwargs
    ).to(device)
    processor_kwargs = {}
    qwen_use_fast = getattr(args, "qwen_use_fast", "auto")
    if qwen_use_fast != "auto":
        processor_kwargs["use_fast"] = qwen_use_fast == "true"
    processor = AutoProcessor.from_pretrained(args.model_path, **processor_kwargs)

    return model, processor


def reflect(
    record: Dict[str, Any],
    benchmark_dir: Path,
    model_components,
    device: torch.device,
    **kwargs,
) -> Dict[str, Any]:
    """Run reflection on a single record using Qwen2.5-VL."""
    model, processor = model_components
    origin_prompt = record["original_prompt"]
    bad_image_rel = record["bad_image"]
    bad_image_path = benchmark_dir / bad_image_rel
    
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
    
    generation_kwargs = {"max_new_tokens": 2048}
    if kwargs.get("force_greedy", False):
        generation_kwargs["do_sample"] = False
    generation_kwargs["use_cache"] = kwargs.get("use_cache", True)
    generated_ids = model.generate(**inputs, **generation_kwargs)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] 
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False
    )
    return parse_reflection_response(output_text)
