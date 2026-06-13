
"""Merge FSDP-sharded VERL actor checkpoints into a Hugging Face model.

Usage:
    python3 tools/model_merger.py --local_dir /path/to/checkpoint/global_step_x/actor
"""

import argparse
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
from torch.distributed._tensor import DTensor, Placement, Shard
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForTokenClassification,
    AutoModelForVision2Seq,
    PretrainedConfig,
    PreTrainedModel,
)


def merge_by_placement(tensors: list[torch.Tensor], placement: Placement):
    if placement.is_replicate():
        return tensors[0]
    elif placement.is_partial():
        raise NotImplementedError("Partial placement is not supported yet")
    elif placement.is_shard():
        return torch.cat(tensors, dim=placement.dim).contiguous()
    else:
        raise ValueError(f"Unsupported placement: {placement}")


def upload_model_to_huggingface(local_path: str, remote_path: str):
    """Upload a merged checkpoint directory to a Hugging Face model repo."""
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=remote_path, private=False, exist_ok=True)
    api.upload_folder(repo_id=remote_path, folder_path=local_path, repo_type="model")


def _collect_auto_map_py_files(cfg) -> set[str]:
    """Collect custom modeling files referenced by top-level and nested auto_map entries."""
    needed: set[str] = set()
    if isinstance(cfg, dict):
        am = cfg.get("auto_map")
        if isinstance(am, dict):
            for class_ref in am.values():
                if isinstance(class_ref, str) and "." in class_ref:
                    needed.add(f"{class_ref.split('.')[0]}.py")
        for v in cfg.values():
            needed |= _collect_auto_map_py_files(v)
    elif isinstance(cfg, list):
        for item in cfg:
            needed |= _collect_auto_map_py_files(item)
    return needed


def auto_copy_custom_modeling_files(hf_path: str):
    """Copy missing custom modeling files referenced by config auto_map entries."""
    import json
    import shutil

    config_file = os.path.join(hf_path, "config.json")
    if not os.path.exists(config_file):
        return

    with open(config_file, "r") as f:
        cfg = json.load(f)

    needed_py_files = _collect_auto_map_py_files(cfg)

    missing = [f for f in sorted(needed_py_files) if not os.path.exists(os.path.join(hf_path, f))]
    if not missing:
        return

    print(f"[auto_copy] Missing custom modeling files: {missing}")

    candidate_dirs = []

    name_or_path = cfg.get("_name_or_path", "")
    if name_or_path and os.path.isdir(name_or_path):
        candidate_dirs.append(name_or_path)

    for sub in (cfg.get("llm_config"), cfg.get("vision_config")):
        if isinstance(sub, dict):
            p = sub.get("_name_or_path", "")
            if p and os.path.isdir(p):
                candidate_dirs.append(p)

    custom_model_src = os.environ.get("CUSTOM_MODEL_SOURCE", "")
    if os.path.isdir(custom_model_src):
        candidate_dirs.append(custom_model_src)

    run_root = os.path.dirname(os.path.dirname(hf_path))  # .../global_step_N
    experiment_dir = os.path.dirname(run_root)  # .../<exp_name>
    if os.path.isdir(experiment_dir):
        for entry in sorted(os.listdir(experiment_dir), reverse=True):
            sibling_hf = os.path.join(experiment_dir, entry, "actor", "huggingface")
            if os.path.isdir(sibling_hf) and os.path.abspath(sibling_hf) != os.path.abspath(hf_path):
                candidate_dirs.append(sibling_hf)

    for missing_file in list(missing):
        for src_dir in candidate_dirs:
            src_file = os.path.join(src_dir, missing_file)
            if os.path.exists(src_file):
                shutil.copy2(src_file, os.path.join(hf_path, missing_file))
                print(f"[auto_copy] Copied {missing_file} from {src_dir}")
                missing.remove(missing_file)
                break

    if missing:
        print(f"[WARNING] These files are still missing and model loading may fail: {missing}")
        print(f"[WARNING] Copy them from the original model directory to: {hf_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge FSDP-sharded actor checkpoints.")
    parser.add_argument("--local_dir", required=True, type=str, help="Actor checkpoint directory.")
    parser.add_argument("--hf_upload_path", default=False, type=str, help="Optional Hugging Face model repo id to upload to.")
    args = parser.parse_args()
    local_dir: str = args.local_dir

    assert not local_dir.endswith("huggingface"), "The local_dir should not end with huggingface."

    # Load rank 0 first to infer the device mesh and sharding layout.
    rank = 0
    world_size = 0
    for filename in os.listdir(local_dir):
        match = re.match(r"model_world_size_(\d+)_rank_0\.pt", filename)
        if match:
            world_size = match.group(1)
            break

    assert world_size, "No model file with the proper format."

    rank0_weight_path = os.path.join(local_dir, f"model_world_size_{world_size}_rank_{rank}.pt")
    state_dict = torch.load(rank0_weight_path, map_location="cpu", weights_only=False)
    pivot_key = sorted(state_dict.keys())[0]
    weight = state_dict[pivot_key]
    if isinstance(weight, DTensor):
        device_mesh = weight.device_mesh
        mesh = device_mesh.mesh
        mesh_dim_names = device_mesh.mesh_dim_names
    else:
        mesh = np.array([int(world_size)], dtype=np.int64)
        mesh_dim_names = ("fsdp",)

    print(f"Got device mesh {mesh}, mesh_dim_names {mesh_dim_names}")

    assert mesh_dim_names in (("fsdp",), ("ddp", "fsdp")), f"Unsupported mesh_dim_names {mesh_dim_names}."

    if "tp" in mesh_dim_names:
        total_shards = mesh.shape[-1] * mesh.shape[-2]
        mesh_shape = (mesh.shape[-2], mesh.shape[-1])
    else:
        total_shards = mesh.shape[-1]
        mesh_shape = (mesh.shape[-1],)

    print(f"Processing {total_shards} model shards in total.")
    model_state_dict_lst = []
    model_state_dict_lst.append(state_dict)
    model_state_dict_lst.extend([""] * (total_shards - 1))

    def process_one_shard(rank, model_state_dict_lst):
        model_path = os.path.join(local_dir, f"model_world_size_{world_size}_rank_{rank}.pt")
        state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
        model_state_dict_lst[rank] = state_dict
        return state_dict

    max_workers = min(32, os.cpu_count() or 1)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_one_shard, rank, model_state_dict_lst) for rank in range(1, total_shards)]
        for future in as_completed(futures):
            future.result()

    state_dict: dict[str, list[torch.Tensor]] = {}
    param_placements: dict[str, list[Placement]] = {}
    keys = set(model_state_dict_lst[0].keys())
    for key in keys:
        state_dict[key] = []
        for shard_rank, model_state_dict in enumerate(model_state_dict_lst):
            try:
                tensor = model_state_dict.pop(key)
            except KeyError as exc:
                raise KeyError(f"Cannot find key {key} in rank {shard_rank}.") from exc

            if isinstance(tensor, DTensor):
                state_dict[key].append(tensor._local_tensor.bfloat16())
                placements = tuple(tensor.placements)
                # DDP replication does not affect the tensor merge.
                if mesh_dim_names[0] == "ddp":
                    placements = placements[1:]

                if key not in param_placements:
                    param_placements[key] = placements
                else:
                    assert param_placements[key] == placements
            else:
                state_dict[key].append(tensor.bfloat16())

    del model_state_dict_lst

    for key in sorted(state_dict):
        if not isinstance(state_dict[key], list):
            print(f"No need to merge key {key}")
            continue

        if key in param_placements:
            placements: tuple[Shard] = param_placements[key]
            if len(mesh_shape) == 1:
                assert len(placements) == 1
                shards = state_dict[key]
                state_dict[key] = merge_by_placement(shards, placements[0])
            else:
                raise NotImplementedError("FSDP + TP is not supported yet.")
        else:
            state_dict[key] = torch.cat(state_dict[key], dim=0)

    print("Merge completed.")
    hf_path = os.path.join(local_dir, "huggingface")
    auto_copy_custom_modeling_files(hf_path)
    config: PretrainedConfig = AutoConfig.from_pretrained(hf_path, trust_remote_code=True)
    architectures: list[str] = getattr(config, "architectures", ["Unknown"])

    if "ForTokenClassification" in architectures[0]:
        AutoClass = AutoModelForTokenClassification
    elif "ForCausalLM" in architectures[0]:
        AutoClass = AutoModelForCausalLM
    elif "ForConditionalGeneration" in architectures[0]:
        AutoClass = AutoModelForVision2Seq
    else:
        raise NotImplementedError(f"Unknown architecture {architectures}.")

    with torch.device("meta"):
        model: PreTrainedModel = AutoClass.from_config(config, torch_dtype=torch.bfloat16)

    assert isinstance(model, PreTrainedModel)
    model.to_empty(device="cpu")

    print(f"Saving model to {hf_path}...")
    model.save_pretrained(hf_path, state_dict=state_dict)
    del state_dict, model

    if args.hf_upload_path:
        upload_model_to_huggingface(hf_path, args.hf_upload_path)
