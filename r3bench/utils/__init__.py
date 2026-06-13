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

"""Shared utility functions, exception classes, and logging setup."""

from .common import (
    encode_image,
    parse_json_response,
    parse_reflection_response,
    parse_reflection_response_qwen3,
    is_valid_edit_prompt,
    filter_thinking_part,
    extract_yes_no,
    load_jsonl,
    save_jsonl,
    load_records_with_checkpoint,
    save_checkpoint,
)
from .exceptions import (
    R3BenchError,
    ParseError,
    OutputFormatError,
)
from .logging import get_logger, setup_logging

__all__ = [
    "encode_image",
    "parse_json_response",
    "parse_reflection_response",
    "parse_reflection_response_qwen3",
    "is_valid_edit_prompt",
    "filter_thinking_part",
    "extract_yes_no",
    "load_jsonl",
    "save_jsonl",
    "load_records_with_checkpoint",
    "save_checkpoint",
    "R3BenchError",
    "ParseError",
    "OutputFormatError",
    "get_logger",
    "setup_logging",
]
