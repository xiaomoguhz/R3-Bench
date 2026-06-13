

import gc
from collections import defaultdict
from functools import partial
from typing import Callable, Union

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp._runtime_utils import _lazy_init
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.optim import Optimizer
from transformers import PreTrainedModel
from transformers.trainer_pt_utils import get_module_class_from_name

# text_config.model_type -> the actual DecoderLayer class name present in the model tree
#                           (matches the transformers definition).
# Note: models such as Llava-OneVision may still declare LlamaDecoderLayer in _no_split_modules;
# this table is used to correct that.
_TEXT_MODEL_TYPE_TO_DECODER_LAYER: dict[str, str] = {
    "qwen2": "Qwen2DecoderLayer",
    "qwen3": "Qwen3DecoderLayer",
    "qwen2_vl": "Qwen2VLDecoderLayer",
    "qwen2_vl_text": "Qwen2VLDecoderLayer",
    "qwen2_5_vl": "Qwen2_5_VLDecoderLayer",
    "qwen2_5_vl_text": "Qwen2_5_VLDecoderLayer",
    "qwen3_vl": "Qwen3VLTextDecoderLayer",
    "qwen3_vl_text": "Qwen3VLTextDecoderLayer",
    "llama": "LlamaDecoderLayer",
    "mistral": "MistralDecoderLayer",
    "mixtral": "MixtralDecoderLayer",
    "gemma": "GemmaDecoderLayer",
    "gemma2": "Gemma2DecoderLayer",
    "phi3": "Phi3DecoderLayer",
}

# Tier-3 fallback: probe common DecoderLayer names in order when the table lookup also fails
# (only covers well-known backbones to avoid pulling in unrelated modules).
_FSDP_DECODER_LAYER_FALLBACK: tuple[str, ...] = (
    "Qwen2DecoderLayer",
    "Qwen3DecoderLayer",
    "Qwen2VLDecoderLayer",
    "Qwen2_5_VLDecoderLayer",
    "Qwen3VLTextDecoderLayer",
    "LlamaDecoderLayer",
    "MistralDecoderLayer",
    "GemmaDecoderLayer",
    "Gemma2DecoderLayer",
    "Phi3DecoderLayer",
    "MixtralDecoderLayer",
)


def _decoder_layer_class_for_text_config(model: PreTrainedModel) -> str | None:
    """Infer the DecoderLayer class name that FSDP should wrap, from config.text_config.model_type."""
    cfg = getattr(model, "config", None)
    if cfg is None:
        return None
    text_cfg = getattr(cfg, "text_config", None)
    if text_cfg is None:
        return None
    mt = getattr(text_cfg, "model_type", None)
    if not mt:
        return None
    return _TEXT_MODEL_TYPE_TO_DECODER_LAYER.get(mt)


def get_init_fn(model: nn.Module, device: Union[str, torch.device]) -> Callable[[nn.Module], None]:
    param_occurrence = defaultdict(int)
    for _, param in model.named_parameters(remove_duplicate=False):
        param_occurrence[param] += 1

    duplicated_params = {param for param in param_occurrence.keys() if param_occurrence[param] > 1}
    materialized_params = {}

    def init_fn(module: nn.Module):
        for name, param in module.named_parameters(recurse=False):
            if param in duplicated_params:
                module._parameters[name] = materialized_params.setdefault(
                    param, nn.Parameter(torch.empty_like(param.data, device=device), requires_grad=param.requires_grad)
                )
            else:
                module._parameters[name] = nn.Parameter(
                    torch.empty_like(param.data, device=device), requires_grad=param.requires_grad
                )

    return init_fn


def get_fsdp_wrap_policy(model: PreTrainedModel):
    """Get FSDP wrap policy for the model.

    Three-tier logic (do not blindly trust _no_split_modules, especially for multimodal + Qwen backbones):
    1) Look up the class name declared in model._no_split_modules inside the model tree.
    2) If not found, use config.text_config.model_type to look up the real DecoderLayer class name.
    3) If still not found, fall back to probing a small set of common DecoderLayer names.
    """
    transformer_cls_to_wrap = set()
    for module in model._no_split_modules:
        # 1) Try the declared name as-is
        transformer_cls = get_module_class_from_name(model, module)
        # 2) Fall back to the text_config.model_type table (e.g. declared as LlamaDecoderLayer but actually Qwen2DecoderLayer)
        if transformer_cls is None:
            alt = _decoder_layer_class_for_text_config(model)
            if alt:
                transformer_cls = get_module_class_from_name(model, alt)
        # 3) Narrow fallback probe
        if transformer_cls is None:
            for candidate in _FSDP_DECODER_LAYER_FALLBACK:
                if candidate == module:
                    continue
                transformer_cls = get_module_class_from_name(model, candidate)
                if transformer_cls is not None:
                    break
        if transformer_cls is None:
            raise Exception(
                f"Cannot find {module} (or a known text backbone DecoderLayer) in pretrained model."
            )
        transformer_cls_to_wrap.add(transformer_cls)

    return partial(transformer_auto_wrap_policy, transformer_layer_cls=transformer_cls_to_wrap)


@torch.no_grad()
def offload_fsdp_model(model: FSDP, empty_cache: bool = True):
    # lazy init FSDP model
    _lazy_init(model, model)
    assert model._is_root, "Only support root model offloading to CPU"
    for handle in model._all_handles:
        if handle._offload_params:
            continue

        flat_param = handle.flat_param
        assert (
            flat_param.data.data_ptr() == flat_param._local_shard.data_ptr()
            and id(flat_param.data) != id(flat_param._local_shard)
            and flat_param.data.size() == flat_param._local_shard.size()
        )
        handle.flat_param_to("cpu", non_blocking=True)
        # the following still keeps id(._local_shard) != id(.data)
        flat_param._local_shard = flat_param.data
        assert id(flat_param._local_shard) != id(flat_param.data)

    if empty_cache:
        torch.cuda.empty_cache()


@torch.no_grad()
def load_fsdp_model(model: FSDP, empty_cache: bool = True):
    # lazy init FSDP model
    _lazy_init(model, model)
    assert model._is_root, "Only support root model loading to GPU"
    for handle in model._all_handles:
        if handle._offload_params:
            continue

        flat_param = handle.flat_param
        handle.flat_param_to("cuda", non_blocking=True)
        # the following still keeps id(._local_shard) != id(.data)
        flat_param._local_shard = flat_param.data

    if empty_cache:
        gc.collect()


@torch.no_grad()
def offload_fsdp_optimizer(optimizer: Optimizer, empty_cache: bool = True):
    if not optimizer.state:
        return

    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            state = optimizer.state[param]
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to("cpu", non_blocking=True)

    if empty_cache:
        torch.cuda.empty_cache()


@torch.no_grad()
def load_fsdp_optimizer(optimizer: Optimizer, empty_cache: bool = True):
    if not optimizer.state:
        return

    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            state = optimizer.state[param]
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to("cuda", non_blocking=True)

    if empty_cache:
        gc.collect()
