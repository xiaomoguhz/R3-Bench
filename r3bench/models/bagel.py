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

"""BAGEL reflection backend for R3Bench step 1.

BAGEL is a *unified multimodal model* (UMM): one set of weights does both
image understanding (used here for reflection) and image generation (used by
:mod:`r3bench.editors.bagel_editor` for rectification). This module is the
reflection half — it loads BAGEL and runs its chat / understanding mode to
judge whether ``bad_image`` matches ``original_prompt`` and, if not, produce a
concrete edit instruction.

The companion editor (:class:`r3bench.editors.bagel_editor.BagelImageEditor`)
loads the *same* checkpoint and runs BAGEL's generation mode. Pointing both at
the same ``model_path`` is exactly the UMM pattern documented in the README.

Vendored BAGEL modeling code lives under ``r3bench.third_party.bagel``.
"""

import os
from pathlib import Path
from typing import Any, Dict

import torch
from PIL import Image
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

from r3bench.third_party.bagel import (
    BagelConfig,
    Bagel,
    Qwen2Config,
    Qwen2ForCausalLM,
    SiglipVisionConfig,
    SiglipVisionModel,
    Qwen2Tokenizer,
    load_ae,
    add_special_tokens,
    ImageTransform,
)
from r3bench.prompts import format_reflection_prompt
from r3bench.utils import get_logger, parse_reflection_response

logger = get_logger("models.bagel")

# From upstream BAGEL data/configs/example.yaml :: vlm_sft.image_transform_args.
_IMAGE_TRANSFORM_ARGS = dict(
    max_image_size=980,
    min_image_size=378,
    image_stride=14,
    max_pixels=2_007_040,
)


def load_model(args, device):
    """Load BAGEL and its tokenizer for reflection.

    ``args.model_path`` must contain ``llm_config.json``, ``vit_config.json``,
    ``ae.safetensors``, ``ema.safetensors`` and the tokenizer files (the layout
    of the official ``ByteDance-Seed/BAGEL-7B-MoT`` release).

    Returns ``(model, tokenizer, new_token_ids, image_transform)``.
    """
    max_latent_size = getattr(args, "max_latent_size", 64)

    llm_config = Qwen2Config.from_json_file(
        os.path.join(args.model_path, "llm_config.json")
    )
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    vit_config = SiglipVisionConfig.from_json_file(
        os.path.join(args.model_path, "vit_config.json")
    )
    vit_config.rope = False
    vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

    _, vae_config = load_ae(local_path=os.path.join(args.model_path, "ae.safetensors"))

    config = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config,
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act="gelu_pytorch_tanh",
        latent_patch_size=2,
        max_latent_size=max_latent_size,
    )

    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model = SiglipVisionModel(vit_config)
        model = Bagel(language_model, vit_model, config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config)

    tokenizer = Qwen2Tokenizer.from_pretrained(args.model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    model_state_dict_path = os.path.join(args.model_path, "ema.safetensors")
    # safetensors cannot consume a torch.device object; stringify it.
    device_str = str(device) if isinstance(device, torch.device) else device

    model = load_checkpoint_and_dispatch(
        model,
        model_state_dict_path,
        device_map={"": device_str},
        dtype=torch.bfloat16,
    )
    model = model.to(device).eval()

    image_transform = ImageTransform(**_IMAGE_TRANSFORM_ARGS)
    return model, tokenizer, new_token_ids, image_transform


def reflect(
    record: Dict[str, Any],
    benchmark_dir: Path,
    model_components,
    device,
    **kwargs,
) -> Dict[str, Any]:
    """Run BAGEL's understanding mode on one sample.

    Returns ``{"answer": bool, "explanation": str, "edit_prompt": str}``.
    ``device`` is part of the public step-1 contract; BAGEL is already on its
    device after :func:`load_model`, so it is accepted but unused here.
    """
    model, tokenizer, new_token_ids, image_transform = model_components
    origin_prompt = record["original_prompt"]
    bad_image_path = benchmark_dir / record["bad_image"]

    if not bad_image_path.exists():
        logger.warning(f"Image not found at {bad_image_path}, skipping record.")
        return {"answer": True, "explanation": "Image not found", "edit_prompt": ""}

    image = Image.open(bad_image_path).convert("RGB")
    prompt_style = kwargs.get("prompt_style", "default")
    # BAGEL reflection uses the user prompt only (no system-prompt prefix).
    question = format_reflection_prompt(
        origin_prompt, with_system=False, style=prompt_style
    )

    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            output_text = model.chat(
                tokenizer=tokenizer,
                new_token_ids=new_token_ids,
                image_transform=image_transform,
                images=[image],
                prompt=question,
                max_length=2048,
            )

    # Shared parser (same as the qwen2.5vl backend): tolerates raw JSON,
    # ```json fences, and the default prompt's <think>...</think> wrapper.
    return parse_reflection_response(output_text)
