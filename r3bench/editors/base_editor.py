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

"""Abstract base for R3Bench image editors."""

from abc import ABC, abstractmethod
from PIL import Image


class BaseImageEditor(ABC):
    """Base class for every editor backend.

    See :mod:`r3bench.editors` for the full Editor Protocol. Concrete
    subclasses must implement :meth:`__init__` (load the model) and
    :meth:`edit` (apply the edit instruction to one image).
    """

    @abstractmethod
    def __init__(self, model_path: str, device: str, **kwargs):
        """Load the editor model.

        Args:
            model_path: Local checkpoint directory or HuggingFace repo id.
                API-only editors may ignore this argument.
            device: Target device string, e.g. ``"cuda:0"``.
            **kwargs: Optional knobs forwarded from ``step2_edit`` (such as
                ``seed``). Unrecognised kwargs must be tolerated silently.
        """

    @abstractmethod
    def edit(self,
             bad_image: Image.Image,
             edit_prompt: str,
             origin_prompt: str = None,
             explanation: str = None) -> Image.Image:
        """Apply ``edit_prompt`` to ``bad_image`` and return the result.

        Args:
            bad_image: Flawed image generated from ``origin_prompt``.
            edit_prompt: Concrete edit instruction produced by the reflection
                step (e.g. ``"replace the red apple with a green one"``).
            origin_prompt: Original generation prompt. Optional; some editors
                use it as additional context.
            explanation: Free-form description of the error. Optional.

        Returns:
            Edited image, or ``None`` when retries are exhausted on a
            transient failure — ``step2_edit`` then skips saving so the next
            run can resume from this sample. Implementations may raise on
            unrecoverable failures so the caller can skip the sample.
        """
