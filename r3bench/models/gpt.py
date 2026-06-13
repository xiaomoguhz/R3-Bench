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

"""GPT reflection backend (Azure-compatible OpenAI client).

The target model is configurable via ``R3BENCH_GPT_MODEL`` (default:
``gpt-5.2``).
"""

import base64
import json
import os
import time
from pathlib import Path
from typing import Any, Dict

from openai import AzureOpenAI

from r3bench.prompts import REFLECTION_USER_PROMPT
from r3bench.utils.logging import get_logger

logger = get_logger("models.gpt")

# Configure via environment variables.
API_VERSION = os.environ.get("OPENAI_API_VERSION", "2024-02-01")
MODEL = os.environ.get("R3BENCH_GPT_MODEL", "gpt-5.2")


def load_model(args, device):
    """Initializes the AzureOpenAI client."""
    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL")
    if not api_key or not base_url:
        raise RuntimeError(
            "GPT backend requires the OPENAI_API_KEY and OPENAI_BASE_URL "
            "environment variables."
        )
    client = AzureOpenAI(
        azure_endpoint=base_url,
        api_version=API_VERSION,
        api_key=api_key,
    )
    return client, MODEL


def image_to_base64(image_path: Path) -> str | None:
    """Converts an image file to a base64 encoded string."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        logger.warning(f"File {image_path} not found.")
        return None


def reflect(record: Dict[str, Any], benchmark_dir: Path, model_components, device, **kwargs) -> Dict[str, Any]:
    """Processes a single record to get model reflection using the GPT judge model."""
    client, model = model_components

    origin_prompt = record["original_prompt"]
    bad_image_rel = record["bad_image"]
    bad_image_path = benchmark_dir / bad_image_rel

    bad_image_base64 = image_to_base64(bad_image_path)
    if not bad_image_base64:
        return {"answer": True, "explanation": f"Image not found at {bad_image_path}", "edit_prompt": ""}

    question = REFLECTION_USER_PROMPT.format(origin_prompt=origin_prompt)

    max_retries = 10
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                stream=False,
                max_tokens=500,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{bad_image_base64}"}},
                    ]
                }]
            )
            output_text = response.choices[0].message.content
            return parse_json_response(output_text)
        except Exception as e:
            if "429" in str(e):
                wait = 10 * (2 ** attempt)
                logger.warning(f"[Rate limit] Waiting {wait}s before retry... (attempt {attempt+1})")
                time.sleep(wait)
            else:
                logger.error(f"Error in calling GPT API: {e}")
                return {"answer": True, "explanation": f"API call failed: {e}", "edit_prompt": ""}

    return {"answer": True, "explanation": "Max retries exceeded", "edit_prompt": ""}


def parse_json_response(output_text: str) -> Dict[str, Any]:
    """Parses the JSON response from the model."""
    try:
        if "```json" in output_text:
            json_part = output_text.split("```json")[1].split("```")[0].strip()
        else:
            json_part = output_text

        output_json = json.loads(json_part)
        answer = output_json.get("answer", False)
        explanation = output_json.get("explanation", '')
        edit_prompt = output_json.get("edit_prompt", '')
        return {"answer": answer, "explanation": explanation, "edit_prompt": edit_prompt}
    except Exception as e:
        logger.error(f"Failed to parse model output: {e}")
        logger.error(f"Original output: {output_text}")
        return {"answer": True, "explanation": "Failed to parse model output", "edit_prompt": ""}
