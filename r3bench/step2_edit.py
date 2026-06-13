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

"""Step 2: apply the reflection model's edit instruction to each bad image.

Reads the JSONL produced by ``step1_reflection``, skips records where the
reflection answered ``True`` (the model thinks the image is fine, so nothing
to edit), and writes one edited PNG per remaining sample to ``--output-dir``.
The editor backend is selected by ``--editor-name``; see
:func:`r3bench.editors.list_editors`. Launched via ``torchrun``.
"""
import argparse
import json
import math
import os
from pathlib import Path
import torch
import torch.distributed as dist
from PIL import Image

from r3bench.editors import get_editor, EDITORS


def setup_distributed():
    dist.init_process_group(backend=os.environ.get("R3BENCH_DIST_BACKEND", "nccl"))
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser(description="Step 2: Unified image editing script")
    parser.add_argument("--editor-name", required=True, choices=list(EDITORS.keys()),
                        help="Name of the editor to use")
    parser.add_argument("--model-path", required=True, help="Model path or HuggingFace ID")
    parser.add_argument("--input-jsonl", required=True,
                        help="Input JSONL file path (output from step 1)")
    parser.add_argument("--output-dir", required=True, help="Output directory path")
    parser.add_argument("--benchmark-dir", required=True,
                        help="Benchmark directory path (for resolving image relative paths)")

    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--overwrite", action="store_true",
                        help="Whether to overwrite existing output files")

    return parser.parse_args()


def main():
    import random
    import numpy as np

    args = parse_args()

    setup_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{os.environ['LOCAL_RANK']}")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    benchmark_dir = Path(args.benchmark_dir)

    if rank == 0:
        print(f"Loading editor: {args.editor_name}...")

    editor = get_editor(
        name=args.editor_name,
        model_path=args.model_path,
        device=device,
        seed=args.seed,
    )

    records = []
    if os.path.exists(args.input_jsonl):
        with open(args.input_jsonl, "r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]

    if not records:
        if rank == 0:
            print(f"Input file does not exist or is empty: {args.input_jsonl}")
        cleanup_distributed()
        return

    total = len(records)
    per_rank = math.ceil(total / world_size)
    start, end = rank * per_rank, min((rank + 1) * per_rank, total)

    if rank == 0:
        print(f"Found {total} samples, each process handles approximately {per_rank}.")

    for idx in range(start, end):
        record = records[idx]
        record_index = record.get("index", idx)
        sample_id = record.get("prompt_id", record_index)

        bad_image_rel = record.get("bad_image")
        if not bad_image_rel:
            print(f"Rank {rank} Warning: Sample {sample_id} missing 'bad_image' field, skipped.")
            continue

        bad_image_path_obj = Path(bad_image_rel)
        edited_filename = f"{bad_image_path_obj.stem}_edited{bad_image_path_obj.suffix}"
        output_image_path = output_root / edited_filename

        if output_image_path.exists() and not args.overwrite:
            if rank == 0:
                print(f"Skipping sample {sample_id}: File {edited_filename} already exists.")
            continue

        origin_prompt = record.get("original_prompt")
        model_reflection = record.get("model_reflection", {})

        if not all([origin_prompt, model_reflection]):
            print(f"Rank {rank} Warning: Sample {sample_id} missing key information, skipped.")
            continue

        bad_image_path = benchmark_dir / bad_image_rel
        if not bad_image_path.exists():
            print(f"Rank {rank} Warning: Image file for sample {sample_id} does not exist: "
                  f"{bad_image_path}, skipped.")
            continue

        bad_image = Image.open(bad_image_path).convert("RGB")

        if str(model_reflection.get("answer")).lower() == 'true':
            edited_image = bad_image
            if rank == 0:
                print(f"Processing sample {sample_id}: Model determined no editing needed, "
                      "using original image.")
        else:
            edit_prompt = model_reflection.get("edit_prompt")
            if not edit_prompt:
                edited_image = bad_image
                if rank == 0:
                    print(f"Rank {rank} Warning: Sample {sample_id} needs editing but "
                          "'edit_prompt' is empty, using original image.")
            else:
                if rank == 0:
                    print(f"Processing sample {sample_id}: Using editor '{args.editor_name}'...")
                    print(f"  - Edit prompt: {edit_prompt}")

                edited_image = editor.edit(
                    bad_image=bad_image,
                    edit_prompt=edit_prompt,
                    origin_prompt=origin_prompt,
                    explanation=model_reflection.get("explanation", ""),
                    bad_image_path=str(bad_image_path),
                )

                # Returning None lets the next run resume from this sample.
                if edited_image is None:
                    if rank == 0:
                        print(f"Warning: Sample {sample_id} editing failed (API retry limit exceeded), "
                              "skipping save, can retry on next run.")
                    continue

        if edited_image:
            edited_image.save(output_image_path)
            metadata_path = output_root / f"{Path(edited_filename).stem}.json"
            metadata = {
                "index": record_index,
                "prompt_id": sample_id,
                "original_prompt": origin_prompt,
                "explanation": model_reflection.get("explanation", ""),
                "edit_prompt": model_reflection.get("edit_prompt", ""),
                "edited_image": edited_filename,
                "model_answer": model_reflection.get("answer"),
            }
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

    if rank == 0:
        print("\nAll tasks completed.")


if __name__ == "__main__":
    main()
