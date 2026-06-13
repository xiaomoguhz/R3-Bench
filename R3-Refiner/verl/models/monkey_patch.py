


from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from ..utils.py_functional import is_transformers_version_greater_than
from .transformers.flash_attention_utils import flash_attention_forward
from .transformers.qwen2_vl import (
    qwen2_vl_attn_forward,
    qwen2_vl_base_forward_new,
    qwen2_vl_forward_new,
    qwen2_vl_forward_old,
)
from .transformers.qwen3_vl import (
    qwen3_vl_base_forward,
    forward_with_normal_backend,
)


def apply_ulysses_patch(model_type: str) -> None:
    # Map qwen2_5_vl_text to qwen2_5_vl (text-only variant shares the Qwen2.5-VL architecture)
    if model_type == "qwen2_5_vl_text":
        model_type = "qwen2_5_vl"
    elif model_type == "llavaonevision1_5":
        # LLaVA-OneVision-1.5 has its own HF modeling classes and should use the
        # generic Llava/Llama flash-attention path instead of Qwen2.5-VL patches.
        model_type = "llava_onevision"
    
    if model_type in (
            "llama",
            "llava",
            "llava_next",
            "llava_onevision",
            "gemma",
            "gemma2",
            "mistral",
            "qwen2",
            "qwen3",
            "qwen3_moe",):
        ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = flash_attention_forward
    elif model_type in ("qwen3_vl", "qwen3_vl_moe"):
        # Qwen3-VL uses standard flash attention, no special attention patch needed
        ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = flash_attention_forward
        
        # Patch model forward to support image-text mixed data
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            Qwen3VLForConditionalGeneration,
            Qwen3VLModel,
        )
        
        Qwen3VLModel.forward = qwen3_vl_base_forward
        Qwen3VLForConditionalGeneration.forward = forward_with_normal_backend
        
        # Support for Qwen3-VL-MoE if available
        try:
            from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
                Qwen3VLMoeForConditionalGeneration,
                Qwen3VLMoeModel,
            )
            Qwen3VLMoeModel.forward = qwen3_vl_base_forward
            Qwen3VLMoeForConditionalGeneration.forward = forward_with_normal_backend
        except ImportError:
            pass  # Qwen3-VL-MoE not available
    elif model_type in ("qwen2_vl", "qwen2_5_vl"):
        if is_transformers_version_greater_than("4.54.0"):
            # transformers 4.54.0 does not need special patch: https://github.com/huggingface/transformers/pull/39447
            ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = flash_attention_forward
        elif is_transformers_version_greater_than("4.53.0"):
            raise NotImplementedError("Transformers 4.53.* is not compatible with Qwen2-VL. Use 4.54.0 or later.")
        else:
            from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLFlashAttention2
            from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLFlashAttention2

            Qwen2VLFlashAttention2.forward = qwen2_vl_attn_forward
            Qwen2_5_VLFlashAttention2.forward = qwen2_vl_attn_forward

        if is_transformers_version_greater_than("4.52.0"):
            from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
                Qwen2_5_VLForConditionalGeneration,
                Qwen2_5_VLModel,
            )
            from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLForConditionalGeneration, Qwen2VLModel

            Qwen2VLModel.forward = qwen2_vl_base_forward_new
            Qwen2_5_VLModel.forward = qwen2_vl_base_forward_new
            Qwen2VLForConditionalGeneration.forward = qwen2_vl_forward_new
            Qwen2_5_VLForConditionalGeneration.forward = qwen2_vl_forward_new
        else:
            from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
            from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLForConditionalGeneration

            Qwen2VLForConditionalGeneration.forward = qwen2_vl_forward_old
            Qwen2_5_VLForConditionalGeneration.forward = qwen2_vl_forward_old
    else:
        raise NotImplementedError(f"Model architecture {model_type} is not supported yet.")
