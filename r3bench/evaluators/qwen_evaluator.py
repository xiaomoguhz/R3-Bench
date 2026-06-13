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

"""Qwen-based evaluators backed by a vLLM-compatible HTTP endpoint.

Reflection evaluation uses a text-only LLM; image-edit evaluation uses a VL model.
"""

from typing import List, Optional
import requests

from .base import ReflectionEvaluator, ImageEditEvaluator
from r3bench.utils import encode_image, get_logger

logger = get_logger("evaluators.qwen")


class QwenEvaluatorMixin:
    """Shared logic for Qwen-based evaluators."""

    def __init__(
        self,
        api_urls: List[str] = None,
        model_name: Optional[str] = None,
        temperature: float = 0.3,
        top_p: float = 0.9,
        seed: Optional[int] = None,
        max_tokens: int = 256,
        max_retries: int = 3,
        request_timeout: float = 300.0,
        **kwargs,
    ):
        super().__init__(
            max_tokens=max_tokens,
            max_retries=max_retries,
            request_timeout=request_timeout,
        )

        self.api_urls = api_urls or ["http://127.0.0.1:8000"]
        self.temperature = temperature
        self.top_p = top_p
        self.seed = seed
        self.model_name = model_name or self._get_model_name()
        self._api_index = 0

    def _get_model_name(self) -> Optional[str]:
        try:
            resp = requests.get(
                f"{self.api_urls[0]}/v1/models",
                timeout=5.0,
            )
            resp.raise_for_status()
            data = resp.json()
            models = data.get("data", [])
            if models:
                return models[0].get("id")
        except Exception as e:
            logger.warning(f"Failed to get model name: {e}")
        return None

    def _get_next_api_url(self) -> str:
        url = self.api_urls[self._api_index % len(self.api_urls)]
        self._api_index += 1
        return url

    def call_api(
        self,
        prompt: str,
        image_path: Optional[str] = None,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> str:
        # Text-only LLMs reject the multimodal content list shape, so switch
        # message format based on whether an image is present.
        if image_path:
            data_url = encode_image(image_path)
            content = [
                {
                    "type": "image_url",
                    "image_url": {"url": data_url},
                },
                {"type": "text", "text": prompt},
            ]
        else:
            content = prompt

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})

        payload = {
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }

        if self.model_name:
            payload["model"] = self.model_name
        if self.seed is not None:
            payload["seed"] = self.seed

        api_url = self._get_next_api_url()

        response = requests.post(
            f"{api_url}/v1/chat/completions",
            json=payload,
            timeout=self.request_timeout,
        )
        response.raise_for_status()

        data = response.json()
        result_content = data["choices"][0]["message"]["content"].strip()

        if not result_content:
            raise RuntimeError("Empty response from vLLM API")

        return result_content


class QwenReflectionEvaluator(QwenEvaluatorMixin, ReflectionEvaluator):
    def __init__(self, **kwargs):
        kwargs.setdefault("temperature", 0.3)
        kwargs.setdefault("top_p", 0.9)
        kwargs.setdefault("max_tokens", 256)
        super().__init__(**kwargs)


class QwenImageEditEvaluator(QwenEvaluatorMixin, ImageEditEvaluator):
    def __init__(self, **kwargs):
        kwargs.setdefault("temperature", 0.0)
        kwargs.setdefault("top_p", 1.0)
        kwargs.setdefault("max_tokens", 512)
        super().__init__(**kwargs)
