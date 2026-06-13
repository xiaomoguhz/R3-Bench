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

"""Image editor backed by OpenAI's ``gpt-image-1`` endpoint."""

import base64
import io
import os
import time
from openai import OpenAI
from PIL import Image
from .base_editor import BaseImageEditor
from r3bench.utils.logging import get_logger

logger = get_logger("editors.gpt_image")


class GptImageEditor(BaseImageEditor):
    """Image editor backed by the OpenAI-compatible gpt-image-1 endpoint."""

    def __init__(self, model_path: str, device: str, **kwargs):
        """Initialize the GPT-Image editor from OPENAI_API_KEY / OPENAI_BASE_URL."""
        self.model = model_path
        self.api_key = os.environ.get("OPENAI_API_KEY")
        self.base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        if not self.api_key:
            raise RuntimeError(
                "GPT-Image editor requires OPENAI_API_KEY in the environment."
            )
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )
        logger.info(f"GPT-Image Editor initialized (model={self.model})")

    def edit(self,
             bad_image: Image.Image,
             edit_prompt: str,
             origin_prompt: str = None,
             explanation: str = None,
             bad_image_path: str = None,
             **kwargs) -> Image.Image:
        full_prompt = edit_prompt

        buffered = io.BytesIO()
        bad_image.save(buffered, format="PNG")
        buffered.name = "image.png"

        max_retries = 10
        for attempt in range(max_retries):
            try:
                buffered.seek(0)
                result = self.client.images.edit(
                    model=self.model,
                    image=buffered,
                    prompt=full_prompt,
                    extra_headers={"api-key": self.api_key},
                )

                image_base64 = result.data[0].b64_json
                img_data = base64.b64decode(image_base64)
                edited_image = Image.open(io.BytesIO(img_data))
                return edited_image

            except Exception as e:
                error_str = str(e).lower()
                if any(code in error_str for code in ["429", "500", "502", "503", "timeout", "timed out"]):
                    wait = 10 * (2**attempt)
                    logger.warning(
                        f"[API Error: {str(e)[:100]}] Waiting {wait}s before retry... (attempt {attempt+1}/{max_retries})"
                    )
                    time.sleep(wait)
                else:
                    # Non-retriable error (e.g. content moderation rejection).
                    logger.error(f"GPT-Image API error: {e}")
                    if bad_image_path:
                        logger.error(f"Error while processing: {bad_image_path}")
                    return None

        logger.error("Max retries exceeded for GPT-Image API.")
        return None