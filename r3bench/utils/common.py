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

"""Shared helpers: image encoding, JSON parsing, JSONL I/O, and checkpointing."""

import base64
import json
import mimetypes
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .exceptions import OutputFormatError, ParseError
from .logging import get_logger

logger = get_logger("utils.common")


def encode_image(image_path: str) -> str:
    """Encode an image file as a base64 data URL."""
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        mime_type = "image/jpeg"

    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")

    return f"data:{mime_type};base64,{encoded}"


def parse_json_response(
    response_text: str,
    key_field: str = "is_correct",
    strict: bool = False,
) -> Dict[str, Any]:
    """Parse a JSON object from a model response.

    First tries to locate an object containing ``key_field``; falls back to
    parsing the entire response. Returns an empty dict on failure unless
    ``strict`` is True.
    """
    try:
        json_match = re.search(
            rf'\{{[^{{}}]*"{key_field}"[^{{}}]*\}}',
            response_text,
            re.DOTALL
        )
        if json_match:
            return json.loads(json_match.group(0))

        return json.loads(response_text)

    except (json.JSONDecodeError, AttributeError) as e:
        if strict:
            raise ParseError(f"Failed to parse JSON response: {e}\nOriginal: {response_text[:200]}...")
        logger.warning(f"JSON parse failed, returning empty dict: {e}")
        return {}


def parse_reflection_response(output_text: str) -> Dict[str, Any]:
    """Parse a reflection response (Qwen2.5-VL flavour).

    Handles raw JSON, markdown code fences, and ``<think>...</think>`` or
    ``<thinking>...</thinking>`` wrappers (thinking-style checkpoints may emit
    the long form).
    """
    default_result = {
        "answer": True,
        "explanation": "Failed to parse model output",
        "edit_prompt": ""
    }

    try:
        if isinstance(output_text, list):
            if not output_text:
                return default_result
            output_text = output_text[0]

        # Long tag first so ``</thinking>`` doesn't accidentally match a
        # ``</think>`` prefix inside it.
        for close_tag in ("</thinking>", "</think>"):
            if close_tag in output_text:
                output_text = output_text.split(close_tag)[-1]
                break

        if "```json" in output_text:
            output_text = output_text.split("```json")[1].split("```")[0]
        elif "```" in output_text:
            parts = output_text.split("```")
            if len(parts) >= 2:
                output_text = parts[1]

        output_text = output_text.strip()
        output_json = json.loads(output_text)

        answer_val = output_json.get("answer", False)
        if isinstance(answer_val, str):
            answer = answer_val.lower() == "true"
        else:
            answer = bool(answer_val)

        return {
            "answer": answer,
            "explanation": output_json.get("explanation", ""),
            "edit_prompt": output_json.get("edit_prompt", ""),
        }

    except (json.JSONDecodeError, IndexError, KeyError, AttributeError, TypeError) as e:
        logger.warning(f"Failed to parse reflection response: {e}")
        logger.warning(f"Original output: {output_text}")
        return default_result


def is_valid_edit_prompt(edit_prompt: str) -> bool:
    """Heuristically reject empty/template/non-actionable edit instructions."""
    if not edit_prompt:
        return False

    edit_prompt_clean = edit_prompt.strip()

    if not edit_prompt_clean:
        return False

    if len(edit_prompt_clean) <= 10:
        return False

    edit_prompt_lower = edit_prompt_clean.lower()
    if edit_prompt_lower in ["remain unchanged", "no edit", ""]:
        return False

    template_patterns = [
        "a concrete, location-specific editing instruction",
        "concrete, location-specific editing instruction to fix the error",
        "provide a concrete, location-specific editing instruction",
        "location-specific editing instruction",
    ]
    for pattern in template_patterns:
        if pattern in edit_prompt_lower:
            return False

    action_words = ['add', 'remove', 'replace', 'change', 'move', 'delete',
                   'place', 'position', 'shift', 'make', 'modify', 'update']
    has_action_word = any(word in edit_prompt_lower for word in action_words)

    if not has_action_word:
        if len(edit_prompt_clean) < 20:
            return False

    return True


def filter_thinking_part(response: str, eos_token: Optional[str] = None) -> tuple:
    """Strip the chain-of-thought prefix from a model response.

    Handles both ``<think>...</think>answer`` (the benchmark prompts) and
    ``<thinking>...</thinking>answer`` (Qwen3-VL "thinking" checkpoints),
    plus the variant where only the closing tag is emitted.
    """
    response_start = 0
    success = False
    # Long tag first so ``</thinking>`` doesn't get accidentally matched by
    # the ``</think>`` prefix substring inside it.
    tag_pairs = (("<thinking>", "</thinking>"), ("<think>", "</think>"))

    for open_tag, close_tag in tag_pairs:
        end_idx = response.rfind(close_tag)
        if end_idx != -1:
            response_start = end_idx + len(close_tag)
            success = True
            break
        start_idx = response.find(open_tag)
        if start_idx != -1:
            response_start = start_idx + len(open_tag)
            success = True
            break

    if eos_token is not None:
        response_end = response.find(eos_token, response_start)
        if response_end == -1:
            response_end = len(response)
    else:
        response_end = len(response)

    response_filtered = response[response_start:response_end]
    return response_filtered, success


def parse_reflection_response_qwen3(output_text: str) -> Dict[str, Any]:
    """Parse a Qwen3-VL reflection response with three fallback strategies:

    1. Strip ``<think>`` prefix and any markdown fences, then JSON-parse.
    2. Scan the whole response for the first balanced ``{...}`` block.
    3. Search inside the ``<think>`` block in case the model put JSON there.
    """
    default_result = {
        "answer": True,
        "explanation": "Failed to parse model output",
        "edit_prompt": ""
    }

    try:
        if isinstance(output_text, list):
            if not output_text:
                return default_result
            output_text = output_text[0]

        response_clean, has_think_tag = filter_thinking_part(output_text)

        if "```json" in response_clean:
            response_clean = response_clean.split("```json")[1].split("```")[0]
        elif "```" in response_clean:
            parts = response_clean.split("```")
            if len(parts) >= 2:
                response_clean = parts[1]

        try:
            response_json = json.loads(response_clean.strip())
            if isinstance(response_json, dict):
                edit_prompt = response_json.get("edit_prompt", "")
                explanation = response_json.get("explanation", "")
                answer_val = response_json.get("answer", False)

                if isinstance(answer_val, str):
                    answer = answer_val.lower() == "true"
                else:
                    answer = bool(answer_val)

                if is_valid_edit_prompt(edit_prompt) or answer:
                    return {
                        "answer": answer,
                        "explanation": explanation,
                        "edit_prompt": edit_prompt.strip() if edit_prompt else "",
                    }
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

        json_start = output_text.find('{')
        if json_start != -1:
            brace_count = 0
            json_end = -1
            for i in range(json_start, len(output_text)):
                if output_text[i] == '{':
                    brace_count += 1
                elif output_text[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_end = i + 1
                        break

            if json_end != -1:
                try:
                    json_str = output_text[json_start:json_end]
                    response_json = json.loads(json_str)
                    if isinstance(response_json, dict):
                        edit_prompt = response_json.get("edit_prompt", "")
                        explanation = response_json.get("explanation", "")
                        answer_val = response_json.get("answer", False)

                        if isinstance(answer_val, str):
                            answer = answer_val.lower() == "true"
                        else:
                            answer = bool(answer_val)

                        if is_valid_edit_prompt(edit_prompt) or answer:
                            return {
                                "answer": answer,
                                "explanation": explanation,
                                "edit_prompt": edit_prompt.strip() if edit_prompt else "",
                            }
                except (json.JSONDecodeError, ValueError, TypeError):
                    pass

        think_tag_end = '</think>'
        think_end = output_text.rfind(think_tag_end)
        if think_end != -1:
            think_content = output_text[:think_end]
            json_start = think_content.find('{')
            if json_start != -1:
                brace_count = 0
                json_end = -1
                for i in range(json_start, len(think_content)):
                    if think_content[i] == '{':
                        brace_count += 1
                    elif think_content[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            json_end = i + 1
                            break

                if json_end != -1:
                    try:
                        json_str = think_content[json_start:json_end]
                        response_json = json.loads(json_str)
                        if isinstance(response_json, dict):
                            edit_prompt = response_json.get("edit_prompt", "")
                            explanation = response_json.get("explanation", "")
                            answer_val = response_json.get("answer", False)

                            if isinstance(answer_val, str):
                                answer = answer_val.lower() == "true"
                            else:
                                answer = bool(answer_val)

                            if is_valid_edit_prompt(edit_prompt) or answer:
                                return {
                                    "answer": answer,
                                    "explanation": explanation,
                                    "edit_prompt": edit_prompt.strip() if edit_prompt else "",
                                }
                    except (json.JSONDecodeError, ValueError, TypeError):
                        pass

        logger.warning(f"Failed to parse Qwen3 reflection response: {output_text[:200]}")
        return default_result

    except Exception as e:
        logger.warning(f"Unexpected error in parse_reflection_response_qwen3: {e}")
        return default_result


def extract_yes_no(model_output: str, num_questions: int) -> List[str]:
    """Extract a yes/no answer per question from a model output.

    Raises OutputFormatError if the number of extracted answers does not match
    ``num_questions``.
    """
    lines = [line.strip() for line in model_output.strip().split('\n') if line.strip()]
    preds = []

    for line in lines:
        match = re.match(r'^(yes|no)\b', line.strip(), flags=re.IGNORECASE)
        if match:
            preds.append(match.group(1).lower())

    if len(preds) != num_questions:
        raise OutputFormatError(
            f"Answer count mismatch: got {len(preds)}, expected {num_questions}. "
            f"Output: {model_output[:200]}..."
        )

    return preds


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load a JSONL file, returning [] if the file does not exist."""
    records = []
    if not os.path.exists(file_path):
        return records

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse line: {e}")
                    continue

    return records


def save_jsonl(
    records: List[Dict[str, Any]],
    file_path: str,
    sort_key: Optional[str] = None,
) -> None:
    """Save a list of records to a JSONL file, optionally sorted by ``sort_key``."""
    if sort_key:
        records = sorted(records, key=lambda x: x.get(sort_key, 0))

    with open(file_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_records_with_checkpoint(
    input_path: str,
    output_path: str,
    index_field: str = "index",
    result_field: str = "reflection_eval",
    overwrite: bool = False,
) -> Tuple[Dict[int, Dict[str, Any]], Set[int]]:
    """Load input records and merge in any completed results from a prior run."""
    records = load_jsonl(input_path)

    if not records:
        return {}, set()

    for i, record in enumerate(records):
        if index_field not in record:
            record[index_field] = i

    records_by_index = {r[index_field]: r for r in records}
    completed_indices: Set[int] = set()

    if os.path.exists(output_path) and not overwrite:
        logger.info(f"Loading checkpoint file: {output_path}")

        existing_records = load_jsonl(output_path)
        for rec in existing_records:
            idx = rec.get(index_field)
            if idx is not None and rec.get(result_field):
                completed_indices.add(idx)
                if idx in records_by_index:
                    records_by_index[idx][result_field] = rec[result_field]

        logger.info(f"Restored {len(completed_indices)} completed results")

    return records_by_index, completed_indices


def save_checkpoint(
    records_by_index: Dict[int, Dict[str, Any]],
    output_path: str,
    index_field: str = "index",
    filter_field: Optional[str] = None,
) -> None:
    """Save the current records dict as a JSONL checkpoint."""
    records = list(records_by_index.values())

    if filter_field:
        records = [r for r in records if r.get(filter_field)]

    save_jsonl(records, output_path, sort_key=index_field)
