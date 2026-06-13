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

"""Evaluator backends for R3Bench steps 3 and 4.

An evaluator is the *judge* that turns model outputs into scores.
Two backend families ship with R3Bench:

* ``gpt`` — Azure-compatible OpenAI GPT-4o judge (paid).
* ``qwen`` — self-hosted Qwen judge served by vLLM (free).

Class hierarchy (see :mod:`.base` for the abstract definitions)::

    BaseEvaluator
    ├── ReflectionEvaluator   # step 3: scores explanation semantics
    └── ImageEditEvaluator    # step 4: scores edited-image QA accuracy

Each backend provides one subclass per task — e.g. ``GPTReflectionEvaluator``
and ``GPTImageEditEvaluator``. Use the factory helpers below from CLI code:

    evaluator = get_reflection_evaluator("gpt", api_key=..., api_base=...)

Evaluator Protocol
------------------

Subclasses inherit ``evaluate`` / ``retry_with_backoff`` from the base classes
and must implement only ``call_api(prompt, image_path=None, **kwargs) -> str``.
The base class handles JSON parsing, retry/backoff, and metric computation.
See :class:`r3bench.evaluators.base.ReflectionEvaluator` for the
Reflective Verdict Score ``S_ref`` definition and
:class:`r3bench.evaluators.base.ImageEditEvaluator` for the
Rectification Score ``S_rect`` formula.
"""

from .base import BaseEvaluator, ReflectionEvaluator, ImageEditEvaluator
from .gpt_evaluator import GPTReflectionEvaluator, GPTImageEditEvaluator
from .qwen_evaluator import QwenReflectionEvaluator, QwenImageEditEvaluator

__all__ = [
    "BaseEvaluator",
    "ReflectionEvaluator",
    "ImageEditEvaluator",
    "GPTReflectionEvaluator",
    "GPTImageEditEvaluator",
    "QwenReflectionEvaluator",
    "QwenImageEditEvaluator",
]


def get_reflection_evaluator(backend: str, **kwargs):
    """Get reflection evaluator instance for backend in {gpt, qwen}."""
    evaluators = {
        "gpt": GPTReflectionEvaluator,
        "qwen": QwenReflectionEvaluator,
    }

    if backend not in evaluators:
        raise ValueError(f"Unknown backend: {backend}. Available: {list(evaluators.keys())}")

    return evaluators[backend](**kwargs)


def get_image_edit_evaluator(backend: str, **kwargs):
    """Get image-edit evaluator instance for backend in {gpt, qwen}."""
    evaluators = {
        "gpt": GPTImageEditEvaluator,
        "qwen": QwenImageEditEvaluator,
    }

    if backend not in evaluators:
        raise ValueError(f"Unknown backend: {backend}. Available: {list(evaluators.keys())}")

    return evaluators[backend](**kwargs)
