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

"""Image editor backends for R3Bench step 2.

An editor is a subclass of :class:`BaseImageEditor` that applies a model's
edit instruction to a flawed image. Editors are selected by name through
:func:`get_editor` and invoked by :mod:`r3bench.step2_edit`.

Editor Protocol
---------------

Subclasses implement two methods on top of :class:`BaseImageEditor`:

    def __init__(self, model_path: str, device: str, **kwargs) -> None
    def edit(
        self,
        bad_image: PIL.Image.Image,
        edit_prompt: str,
        origin_prompt: str | None = None,
        explanation: str | None = None,
        bad_image_path: str | None = None,
        **kwargs,
    ) -> PIL.Image.Image

* ``model_path`` is the local checkpoint dir or HuggingFace repo id (ignored
  by API-only editors).
* ``device`` is a string such as ``"cuda:0"``.
* Keyword arguments forwarded by ``step2_edit`` include ``seed`` (for
  reproducibility); unknown kwargs must be tolerated.
* ``edit`` must return the edited image as a ``PIL.Image.Image``. On
  unrecoverable failure raise an exception; transient failures should be
  retried inside ``edit`` and surface as a ``None`` return so the caller can
  skip the sample.
"""

import importlib

from .base_editor import BaseImageEditor

# Editor name -> (submodule, class name). The editor class is imported
# lazily by get_editor so that selecting one editor never drags in another
# editor's version-specific deps.
_EDITOR_MODULES = {
    "qwen_image_2511": ("qwen_editor_2511", "QwenImageEditor2511"),
    "bagel": ("bagel_editor", "BagelImageEditor"),
    "gpt_image": ("gpt_image_editor", "GptImageEditor"),
}

# Public registry: keys are the valid ``--editor-name`` values. Exposed as a
# dict so ``name in EDITORS`` / ``list(EDITORS.keys())`` work without
# importing any editor backend.
EDITORS = dict.fromkeys(_EDITOR_MODULES)


def get_editor(name: str, **kwargs) -> BaseImageEditor:
    """Get editor instance by name (editor backend imported on demand)."""
    if name not in _EDITOR_MODULES:
        available = ", ".join(sorted(_EDITOR_MODULES.keys()))
        raise ValueError(f"Unknown editor: {name}. Available options: {available}")

    module_name, class_name = _EDITOR_MODULES[name]
    module = importlib.import_module(f".{module_name}", __name__)
    editor_class = getattr(module, class_name)
    return editor_class(**kwargs)


def list_editors() -> list:
    """List all available editors."""
    return sorted(_EDITOR_MODULES.keys())


__all__ = [
    "BaseImageEditor",
    "EDITORS",
    "get_editor",
    "list_editors",
]
