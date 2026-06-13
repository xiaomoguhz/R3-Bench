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

"""BAGEL rectification backend for R3Bench step 2.

This is the *generation* half of the BAGEL unified multimodal model (UMM).
The companion reflection backend (:mod:`r3bench.models.bagel`) loads the same
checkpoint and runs BAGEL's understanding mode. Pointing both at the same
``model_path`` is the UMM usage pattern.

``edit`` runs BAGEL's KV-cache image-generation path: it conditions on the
original prompt, the flawed image (via both VAE and ViT encoders), and the
reflection's edit instruction, then samples a rectified image.

Vendored BAGEL modeling code lives under ``r3bench.third_party.bagel``.
"""

import copy
import os
from typing import Dict, Tuple

import torch
from PIL import Image

from .base_editor import BaseImageEditor
from r3bench.config import DEFAULT_BAGEL_GEN_CONFIG
from r3bench.utils import get_logger
from r3bench.third_party.bagel import (
    Bagel,
    BagelConfig,
    Qwen2Config,
    Qwen2ForCausalLM,
    SiglipVisionConfig,
    SiglipVisionModel,
    NaiveCache,
    Qwen2Tokenizer,
    load_ae,
    add_special_tokens,
    ImageTransform,
)

logger = get_logger("editors.bagel")



def move_to_device(batch: Dict, device) -> Dict:
    """Move every tensor in ``batch`` to ``device`` (floats -> bfloat16)."""
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            dtype = torch.bfloat16 if value.is_floating_point() else value.dtype
            batch[key] = value.to(device=device, dtype=dtype)
    return batch


def normalize_model_path(model_path: str) -> str:
    """Return a local path; download from the Hub if given a repo id.

    Requires ``huggingface_hub`` (declared in ``setup.py`` extras ``bagel``).
    """
    if os.path.exists(model_path):
        return model_path

    if "/" in model_path and not os.path.isabs(model_path):
        from huggingface_hub import snapshot_download

        logger.info(f"Resolving HuggingFace repo id: {model_path}")
        local_dir = snapshot_download(
            repo_id=model_path,
            allow_patterns=["*.json", "*.safetensors", "*.bin", "*.py", "*.md", "*.txt", "*.model"],
        )
        logger.info(f"Model downloaded to: {local_dir}")
        return local_dir

    return model_path


def load_bagel_model(
    model_path: str,
    device,
    max_latent_size: int = 64,
) -> Tuple[Bagel, Qwen2Tokenizer, Dict, torch.nn.Module]:
    """Load BAGEL + tokenizer + VAE for generation.

    Returns ``(model, tokenizer, new_token_ids, vae_model)``. The checkpoint
    is dispatched straight onto ``device`` in bfloat16 (same as the reflection
    backend :func:`r3bench.models.bagel.load_model`). Going via a CPU fp32
    copy would transiently need ~2x the bf16 size in host RAM and can OOM in
    memory-limited containers.
    """
    model_path = normalize_model_path(model_path)

    llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers -= 1

    vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))

    bagel_config = BagelConfig(
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

    from accelerate import init_empty_weights, load_checkpoint_and_dispatch

    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model = SiglipVisionModel(vit_config)
        model = Bagel(language_model, vit_model, bagel_config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config)

    tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    checkpoint_path = os.path.join(model_path, "ema.safetensors")
    # safetensors cannot consume a torch.device object; stringify it.
    device_str = str(device) if isinstance(device, torch.device) else device
    model = load_checkpoint_and_dispatch(
        model, checkpoint_path, device_map={"": device_str}, dtype=torch.bfloat16,
    )

    model = model.to(device=device, dtype=torch.bfloat16).eval()
    vae_model = vae_model.to(device=device, dtype=torch.bfloat16).eval()
    return model, tokenizer, new_token_ids, vae_model


def create_image_transforms() -> Tuple[ImageTransform, ImageTransform]:
    """Return ``(vae_transform, vit_transform)`` (BAGEL ViT patch size 224)."""
    vae_transform = ImageTransform(max_image_size=1024, min_image_size=512, image_stride=16)
    vit_transform = ImageTransform(max_image_size=980, min_image_size=224, image_stride=14)
    return vae_transform, vit_transform


def decode_latent_to_image(latent: torch.Tensor, vae_model, height: int, width: int, device):
    """Decode a packed latent into a ``PIL.Image``."""
    latent = latent.reshape(1, height // 16, width // 16, 2, 2, 16)
    latent = torch.einsum("nhwpqc->nchpwq", latent).reshape(1, 16, height // 8, width // 8)
    latent = latent.to(device=device, dtype=torch.bfloat16)

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        decoded = vae_model.decode(latent)

    np_img = (
        (decoded * 0.5 + 0.5).clamp(0, 1)[0]
        .permute(1, 2, 0)
        .mul(255)
        .to(torch.uint8)
        .cpu()
        .numpy()
    )
    return Image.fromarray(np_img)


class BagelImageEditor(BaseImageEditor):
    """Rectify a flawed image with the BAGEL UMM (generation mode)."""

    def __init__(self, model_path: str, device: str, **kwargs):
        """Load BAGEL for generation.

        ``model_path`` is a local checkpoint dir or a HuggingFace repo id
        (auto-downloaded). Unrecognised kwargs (e.g. ``seed``) are silently
        ignored, per the Editor Protocol.
        """
        self.device = device
        self.config = DEFAULT_BAGEL_GEN_CONFIG

        self.model, self.tokenizer, self.new_token_ids, self.vae_model = load_bagel_model(
            model_path=model_path,
            device=device,
            max_latent_size=self.config.max_latent_size,
        )
        self.vae_transform, self.vit_transform = create_image_transforms()
        logger.info("BAGEL model loaded.")

    @torch.no_grad()
    def edit(
        self,
        bad_image: Image.Image,
        edit_prompt: str,
        origin_prompt: str = None,
        explanation: str = None,
        **kwargs,
    ) -> Image.Image:
        """Generate a rectified image conditioned on prompt + flawed image."""
        h = w = self.config.resolution

        past_key_values = NaiveCache(self.model.config.llm_config.num_hidden_layers)
        newlens, new_rope = [0], [0]

        # 1. original prompt
        generation_input, newlens, new_rope = self.model.prepare_prompts(
            curr_kvlens=newlens,
            curr_rope=new_rope,
            prompts=[origin_prompt],
            tokenizer=self.tokenizer,
            new_token_ids=self.new_token_ids,
        )
        generation_input = move_to_device(generation_input, self.device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            past_key_values = self.model.forward_cache_update_text(
                past_key_values, **generation_input
            )

        # 2. flawed image (VAE branch)
        generation_input, newlens, new_rope = self.model.prepare_vae_images(
            curr_kvlens=newlens,
            curr_rope=new_rope,
            images=[bad_image.copy()],
            transforms=self.vae_transform,
            new_token_ids=self.new_token_ids,
        )
        generation_input = move_to_device(generation_input, self.device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            past_key_values = self.model.forward_cache_update_vae(
                self.vae_model, past_key_values, **generation_input
            )

        # 3. flawed image (ViT branch)
        generation_input, newlens, new_rope = self.model.prepare_vit_images(
            curr_kvlens=newlens,
            curr_rope=new_rope,
            images=[bad_image.copy()],
            transforms=self.vit_transform,
            new_token_ids=self.new_token_ids,
        )
        generation_input = move_to_device(generation_input, self.device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            past_key_values = self.model.forward_cache_update_vit(
                past_key_values, **generation_input
            )

        # 4. CFG text branch
        cfg_text_past_key_values = copy.deepcopy(past_key_values)
        cfg_generation_input = self.model.prepare_vae_latent_cfg(
            curr_kvlens=newlens,
            curr_rope=new_rope,
            image_sizes=[(h, w)],
        )
        cfg_generation_input = move_to_device(cfg_generation_input, self.device)

        # 5. CFG image branch (edit instruction only)
        cfg_img_past_key_values = NaiveCache(self.model.config.llm_config.num_hidden_layers)
        cfg_img_newlens, cfg_img_new_rope = [0], [0]
        tmp_input, cfg_img_newlens, cfg_img_new_rope = self.model.prepare_prompts(
            curr_kvlens=cfg_img_newlens,
            curr_rope=cfg_img_new_rope,
            prompts=[edit_prompt],
            tokenizer=self.tokenizer,
            new_token_ids=self.new_token_ids,
        )
        tmp_input = move_to_device(tmp_input, self.device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            cfg_img_past_key_values = self.model.forward_cache_update_text(
                cfg_img_past_key_values, **tmp_input
            )

        # 6. edit instruction on the main branch
        generation_input, newlens, new_rope = self.model.prepare_prompts(
            curr_kvlens=newlens,
            curr_rope=new_rope,
            prompts=[edit_prompt],
            tokenizer=self.tokenizer,
            new_token_ids=self.new_token_ids,
        )
        generation_input = move_to_device(generation_input, self.device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            past_key_values = self.model.forward_cache_update_text(
                past_key_values, **generation_input
            )

        # 7. CFG image-branch generation input
        cfg_img_generation_input = self.model.prepare_vae_latent_cfg(
            curr_kvlens=cfg_img_newlens,
            curr_rope=cfg_img_new_rope,
            image_sizes=[(h, w)],
        )
        cfg_img_generation_input = move_to_device(cfg_img_generation_input, self.device)

        # 8. sample + decode
        generation_input = self.model.prepare_vae_latent(
            curr_kvlens=newlens,
            curr_rope=new_rope,
            image_sizes=[(h, w)],
            new_token_ids=self.new_token_ids,
        )
        generation_input = move_to_device(generation_input, self.device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            unpacked_latent = self.model.generate_image(
                past_key_values=past_key_values,
                num_timesteps=self.config.num_timesteps,
                cfg_text_scale=self.config.cfg_text_scale,
                cfg_img_scale=self.config.cfg_img_scale,
                cfg_interval=self.config.cfg_interval,
                cfg_renorm_min=self.config.cfg_renorm_min,
                cfg_type=self.config.cfg_type,
                cfg_renorm_type=self.config.cfg_renorm_type,
                timestep_shift=self.config.timestep_shift,
                cfg_text_past_key_values=cfg_text_past_key_values,
                cfg_img_past_key_values=cfg_img_past_key_values,
                cfg_text_packed_position_ids=cfg_generation_input["cfg_packed_position_ids"],
                cfg_text_key_values_lens=cfg_generation_input["cfg_key_values_lens"],
                cfg_text_packed_query_indexes=cfg_generation_input["cfg_packed_query_indexes"],
                cfg_text_packed_key_value_indexes=cfg_generation_input["cfg_packed_key_value_indexes"],
                cfg_img_packed_position_ids=cfg_img_generation_input["cfg_packed_position_ids"],
                cfg_img_key_values_lens=cfg_img_generation_input["cfg_key_values_lens"],
                cfg_img_packed_query_indexes=cfg_img_generation_input["cfg_packed_query_indexes"],
                cfg_img_packed_key_value_indexes=cfg_img_generation_input["cfg_packed_key_value_indexes"],
                **generation_input,
            )

        return decode_latent_to_image(unpacked_latent[0], self.vae_model, h, w, self.device)
