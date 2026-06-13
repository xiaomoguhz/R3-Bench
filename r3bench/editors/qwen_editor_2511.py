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

"""Image editor backed by the Qwen-Image-Edit-2511 diffusers pipeline.

Reference: https://huggingface.co/Qwen/Qwen-Image-Edit-2511
"""

import torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline
from .base_editor import BaseImageEditor
from r3bench.utils.logging import get_logger

logger = get_logger("editors.qwen_image_2511")


class QwenImageEditor2511(BaseImageEditor):
    """Image editor backed by the Qwen-Image-Edit-2511 pipeline."""

    def __init__(self, model_path: str, device: str, **kwargs):
        self.device = device
        self.pipeline = QwenImageEditPlusPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16
        )
        self.pipeline.to(self.device)
        self.pipeline.set_progress_bar_config(disable=None)

        self.seed = kwargs.get("seed", 42)

        logger.info("Qwen-Image-Edit-2511 loaded.")

    def edit(self,
             bad_image: Image.Image,
             edit_prompt: str,
             origin_prompt: str = None,
             explanation: str = None,
             bad_image_path: str = None,
             **kwargs) -> Image.Image:
        full_prompt = edit_prompt

        inputs = {
            "image": bad_image,
            "prompt": full_prompt,
            "generator": torch.manual_seed(self.seed),
            "true_cfg_scale": 4.0,
            "negative_prompt": " ",
            "num_inference_steps": 40,
            "guidance_scale": 1.0,
            "num_images_per_prompt": 1,
        }

        with torch.inference_mode():
            output = self.pipeline(**inputs)
            edited_image = output.images[0]

        return edited_image
