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
R3Bench GPT evaluator (Azure-compatible OpenAI client).

api_key / api_base may be supplied explicitly (e.g. via step3/step4 CLI flags) or
fall back to OPENAI_API_KEY / OPENAI_BASE_URL environment variables.
"""

import os
from typing import Optional
import openai

from .base import ReflectionEvaluator, ImageEditEvaluator
from r3bench.utils import encode_image, get_logger

logger = get_logger("evaluators.gpt")


class GPTEvaluatorMixin:
    """Shared logic for GPT-based evaluators."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        api_version: str = "2024-03-01-preview",
        model_name: str = "gpt-4o",
        max_tokens: int = 512,
        max_retries: int = 5,
        request_timeout: float = 300.0,
        **kwargs,
    ):
        super().__init__(
            max_tokens=max_tokens,
            max_retries=max_retries,
            request_timeout=request_timeout,
        )

        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        resolved_base = api_base or os.environ.get("OPENAI_BASE_URL")
        if not resolved_key or not resolved_base:
            raise RuntimeError(
                "GPT evaluator requires OPENAI_API_KEY and OPENAI_BASE_URL (or "
                "explicit api_key / api_base arguments)."
            )

        self.model_name = model_name
        self.client = openai.AzureOpenAI(
            api_key=resolved_key,
            azure_endpoint=resolved_base,
            api_version=api_version,
        )
    
    def call_api(
        self,
        prompt: str,
        image_path: Optional[str] = None,
        **kwargs,
    ) -> str:
        content = [{"type": "input_text", "text": prompt}]

        if image_path:
            data_url = encode_image(image_path)
            content.append({"type": "input_image", "image_url": data_url})

        payload = [{"role": "user", "content": content}]

        response = self.client.responses.create(
            model=self.model_name,
            input=payload,
            max_output_tokens=self.max_tokens,
        )

        if response.output:
            for item in response.output:
                if (item.type == 'message' and
                    item.role == 'assistant' and
                    item.content):
                    for content_part in item.content:
                        if (content_part.type == 'output_text' and
                            hasattr(content_part, 'text')):
                            text = content_part.text.strip()
                            if text:
                                return text

        raise RuntimeError("No valid response from GPT API")


class GPTReflectionEvaluator(GPTEvaluatorMixin, ReflectionEvaluator):
    pass


class GPTImageEditEvaluator(GPTEvaluatorMixin, ImageEditEvaluator):
    pass

