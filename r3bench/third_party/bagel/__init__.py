# Vendored from ByteDance BAGEL (https://github.com/ByteDance-Seed/Bagel).
# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
#
# Only the transitive closure required by R3-Bench's BAGEL UMM backend
# (reflection + rectification) is vendored here. Import paths have been
# rebased from ``modeling.* / data.*`` to ``r3bench.third_party.bagel.*``;
# model logic is unmodified.

from .modeling_bagel import (
    BagelConfig,
    Bagel,
    Qwen2Config,
    Qwen2Model,
    Qwen2ForCausalLM,
    SiglipVisionConfig,
    SiglipVisionModel,
)
from .modeling_bagel.qwen2_navit import NaiveCache
from .qwen2 import Qwen2Tokenizer
from .autoencoder import load_ae
from .data_utils import add_special_tokens, pil_img2rgb
from .transforms import ImageTransform

__all__ = [
    "BagelConfig",
    "Bagel",
    "Qwen2Config",
    "Qwen2Model",
    "Qwen2ForCausalLM",
    "SiglipVisionConfig",
    "SiglipVisionModel",
    "NaiveCache",
    "Qwen2Tokenizer",
    "load_ae",
    "add_special_tokens",
    "pil_img2rgb",
    "ImageTransform",
]
