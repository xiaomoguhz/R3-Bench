#!/usr/bin/env python3
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

"""Step 1: run reflection on each benchmark sample.

Loads a reflection backend (selected by ``--model-type``; see
:func:`r3bench.models.list_models`), shards the input JSONL across distributed
ranks, and writes one ``model_reflection`` field per record to the output
JSONL. Launched via ``torchrun``; usage examples live in the project README.
"""

import argparse
import datetime
import json
import os
import random
import shutil
import torch.distributed as dist
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

from r3bench.models import get_model_handler

def setup_distributed():
    """Initializes the distributed process group."""
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    backend = os.environ.get("R3BENCH_DIST_BACKEND", "nccl")
    dist.init_process_group(backend=backend, timeout=datetime.timedelta(hours=2))

def main() -> None:
    parser = argparse.ArgumentParser(description="Run model reflection on the benchmark.")
    parser.add_argument("--input-jsonl", required=True, help="Path to the input data in JSONL format.")
    parser.add_argument("--output-jsonl", required=True, help="Path to save the output results in JSONL format.")
    parser.add_argument("--benchmark-dir", required=True, help="Directory where benchmark assets (e.g., images) are stored.")
    parser.add_argument("--model-type", required=True, help="Reflection backend (default: qwen2.5vl).")
    parser.add_argument("--model-path", required=True, help="Path to the model directory or HuggingFace repo id (ignored by API backends).")
    parser.add_argument("--overwrite", action="store_true", help="If set, overwrite existing model reflections.")
    parser.add_argument("--prompt", dest="prompt_style", type=str, default="default", choices=["default", "refiner"], help="Reflection prompt style: 'default' (the benchmark's standard prompt) or 'refiner' (the released R3-Refiner model's training format).")
    parser.add_argument("--disable-cudnn-sdp", action="store_true", help="Disable cuDNN SDPA before model loading.")
    parser.add_argument("--sdpa-backend", choices=["default", "math"], default="default", help="Force the SDPA backend ('math' uses the reference kernel).")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic algorithms where available.")
    parser.add_argument("--qwen-use-fast", choices=["auto", "true", "false"], default="auto", help="Override the Qwen processor use_fast ('auto' keeps defaults).")
    parser.add_argument("--attn-implementation", default=None, help="Attention implementation for Qwen loaders (eager/sdpa/flash_attention_2).")
    parser.add_argument("--force-greedy", action="store_true", help="Force greedy decoding (do_sample=False).")
    parser.add_argument("--disable-cache", action="store_true", help="Disable the generation KV cache (use_cache=False).")
    parser.add_argument("--reflection-max-retries", type=int, default=None, help="Override the reflection retry count.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for Python, NumPy, and PyTorch, applied before model loading.")
    args = parser.parse_args()

    if args.disable_cudnn_sdp:
        torch.backends.cuda.enable_cudnn_sdp(False)
    if args.sdpa_backend == "math":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    if args.deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)

    setup_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    try:
        model_handler = get_model_handler(args.model_type)
    except ValueError as e:
        if rank == 0:
            print(f"Error: {e}")
        dist.barrier()
        exit(1)

    if rank == 0:
        print(f"Loading model '{args.model_type}'...")
    model_components = model_handler.load_model(args, device)
    if rank == 0:
        print("Model loaded successfully.")

    benchmark_dir = Path(args.benchmark_dir)
    output_path = Path(args.output_jsonl)
    part_dir = output_path.parent / (output_path.name + ".parts")

    if rank == 0:
        if part_dir.exists():
            shutil.rmtree(part_dir)
        part_dir.mkdir(parents=True, exist_ok=True)

    dist.barrier()

    part_output_path = part_dir / f"part_{rank}.jsonl"

    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        all_records = [json.loads(line) for line in f if line.strip()]

    records_per_shard = (len(all_records) + world_size - 1) // world_size
    start_index = rank * records_per_shard
    end_index = min((rank + 1) * records_per_shard, len(all_records))
    records = all_records[start_index:end_index]

    if not records:
        part_output_path.touch()
    else:
        processed_records = []
        record_iterator = records
        if rank == 0:
            record_iterator = tqdm(records, desc=f"Processing records with {args.model_type}")

        for record in record_iterator:
            if not args.overwrite and record.get("model_reflection"):
                processed_records.append(record)
                continue

            reflect_kwargs = {
                "prompt_style": args.prompt_style,
                "force_greedy": args.force_greedy,
                "use_cache": not args.disable_cache,
            }
            if args.reflection_max_retries is not None:
                reflect_kwargs["max_retries"] = args.reflection_max_retries

            model_reflection = model_handler.reflect(
                record=record,
                benchmark_dir=benchmark_dir,
                model_components=model_components,
                device=device,
                **reflect_kwargs,
            )
            record["model_reflection"] = model_reflection
            processed_records.append(record)

        with open(part_output_path, "w", encoding="utf-8") as f:
            for r in processed_records:
                r.pop("GT_reflection_cot", None)
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    dist.barrier()

    if rank == 0:
        total_lines = 0
        with open(output_path, "w", encoding="utf-8") as f_out:
            for i in range(world_size):
                part_file = part_dir / f"part_{i}.jsonl"
                if part_file.is_file():
                    with open(part_file, "r", encoding="utf-8") as f_in:
                        lines = f_in.readlines()
                        f_out.writelines(lines)
                        total_lines += len(lines)

        shutil.rmtree(part_dir)
        print(f"Total records processed: {total_lines}")
        print(f"Processing complete. Final output: {output_path}")

if __name__ == "__main__":
    main()
