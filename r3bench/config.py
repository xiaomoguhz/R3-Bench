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

"""Configuration dataclasses for R3Bench."""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class BagelGenConfig:
    """Generation settings for the BAGEL UMM rectification editor.

    BAGEL is a unified multimodal model: the same checkpoint serves both the
    ``bagel`` reflection backend (:mod:`r3bench.models.bagel`) and this editor.
    Defaults mirror the upstream BAGEL inference config.

    Attributes:
        resolution: square output side length (pixels) for the rectified image.
        num_timesteps: number of denoising steps.
        cfg_text_scale: classifier-free guidance scale for the text branch.
        cfg_img_scale: classifier-free guidance scale for the image branch.
        cfg_interval: (start, end) fraction of the schedule over which CFG is
            applied.
        cfg_renorm_min: lower bound for CFG renormalization.
        cfg_type: CFG composition strategy (upstream BAGEL default).
        cfg_renorm_type: CFG renormalization mode (upstream BAGEL default).
        timestep_shift: schedule shift applied to the sampling timesteps.
        max_latent_size: maximum latent grid side; also passed to BAGEL's
            config at model-load time.
    """

    resolution: int = 1024
    num_timesteps: int = 50
    cfg_text_scale: float = 4.0
    cfg_img_scale: float = 2.0
    cfg_interval: Tuple[float, float] = (0.0, 1.0)
    cfg_renorm_min: float = 0.0
    cfg_type: str = "serial_text_img"
    cfg_renorm_type: str = "text_channel"
    timestep_shift: float = 3.0
    max_latent_size: int = 64


DEFAULT_BAGEL_GEN_CONFIG = BagelGenConfig()
