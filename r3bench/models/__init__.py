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

"""Reflection backends for R3Bench step 1.

A reflection backend is a Python module that implements two functions:

    def load_model(args, device) -> Any
    def reflect(record, benchmark_dir, model_components, device, **kwargs) -> Dict

The handler is selected by name via :func:`get_model_handler` and invoked by
:mod:`r3bench.step1_reflection`.

Backend Protocol
----------------

``load_model(args, device)`` is called once per process and should return any
state (a model, a client, a (model, processor) tuple, ...) that ``reflect`` will
need. ``args`` is the namespace produced by ``step1_reflection``'s argparse —
backends typically read ``args.model_path`` and may also read ``args.model_type``.

``reflect`` is called once per sample. Its contract:

* ``record`` is one JSONL line from the benchmark and must carry at least
  ``original_prompt`` (str) and ``bad_image`` (path relative to
  ``benchmark_dir``).
* ``benchmark_dir`` (``pathlib.Path``) roots the relative image paths.
* ``model_components`` is whatever ``load_model`` returned.
* ``device`` is the target ``torch.device`` (irrelevant for API backends).
* ``**kwargs`` may include ``prompt_style`` (``"default"`` / ``"refiner"``)
  and backend-specific tuning knobs (``max_retries``, ``retry_temperatures``).

The return value must be a dict with exactly these keys:

    {
        "answer": bool,        # True if the image matches the prompt
        "explanation": str,    # Why the image is wrong (empty if answer=True)
        "edit_prompt": str,    # Concrete edit instruction (empty if answer=True)
    }
"""

import importlib

# Map model type -> submodule name. Handlers are imported lazily by
# get_model_handler so that selecting one backend never drags in another
# backend's version-specific deps.
_MODEL_HANDLERS = {
    "qwen2.5vl": "qwen2_5vl",
    "qwen3vl": "qwen3vl",
    "bagel": "bagel",
    "gpt": "gpt",
}

def get_model_handler(model_type: str):
    """Return the model handler module for a registered model type.

    The submodule is imported on demand (not at package import time) so an
    environment that only needs one backend is never broken by another
    backend's imports.
    """
    module_name = _MODEL_HANDLERS.get(model_type)
    if module_name:
        return importlib.import_module(f".{module_name}", __name__)

    raise ValueError(
        f"Unsupported model type: {model_type}. "
        f"Supported types are: {sorted(_MODEL_HANDLERS.keys())}"
    )


def list_models() -> list:
    """List all supported model types."""
    return sorted(_MODEL_HANDLERS.keys())


__all__ = [
    "get_model_handler",
    "list_models",
]
