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

"""
Step 4: Image Editing Evaluation

Supports GPT and Qwen evaluation backends for QA-based evaluation of model's
image editing capability.

Usage:
    # Using Qwen
    python3 -m r3bench.step4_eval_edit --backend qwen --input-jsonl ... --output-jsonl ... \
        --edited-images-dir ...

    # Using GPT
    python3 -m r3bench.step4_eval_edit --backend gpt --input-jsonl ... --output-jsonl ... \
        --edited-images-dir ... --api-key YOUR_KEY --api-base YOUR_ENDPOINT
"""

import argparse
import shutil
from pathlib import Path
from typing import Any, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from r3bench.evaluators import get_image_edit_evaluator
from r3bench.utils import (
    get_logger,
    load_jsonl,
    save_jsonl,
)

logger = get_logger("step4")


def parse_args():
    parser = argparse.ArgumentParser(description="Step 4: Image Editing Evaluation")

    parser.add_argument("--backend", choices=["gpt", "qwen"], required=True,
                        help="Evaluation backend: gpt or qwen")
    parser.add_argument("--input-jsonl", required=True, help="Input JSONL file path")
    parser.add_argument("--output-jsonl", required=True, help="Output JSONL file path")
    parser.add_argument("--benchmark-dir", required=True, help="Benchmark directory path")
    parser.add_argument("--edited-images-dir", required=True, help="Edited images directory path")

    parser.add_argument("--num-workers", type=int, default=8, help="Number of concurrent workers")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Maximum output tokens")
    parser.add_argument("--max-retries", type=int, default=5, help="Maximum retry count")
    parser.add_argument("--overwrite", action="store_true", help="Whether to overwrite existing results")
    parser.add_argument("--qualitative-output-dir", type=str, default=None,
                        help="Qualitative analysis output directory")

    parser.add_argument("--api-key", help="Azure API Key (GPT)")
    parser.add_argument("--api-base", help="Azure Endpoint URL (GPT)")
    parser.add_argument("--api-version", default="2024-03-01-preview")
    parser.add_argument("--model-name", default="gpt-4o", help="Model name")

    parser.add_argument("--api-urls", nargs="+", default=["http://127.0.0.1:8000"],
                        help="vLLM API URL list (Qwen)")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p sampling")
    parser.add_argument("--seed", type=int, default=None,
                        help="Optional request seed for vLLM/OpenAI-compatible Qwen backends")
    parser.add_argument("--request-timeout", type=float, default=300.0, help="Request timeout")

    return parser.parse_args()


def save_qualitative_sample(
    record: Dict[str, Any],
    benchmark_dir: Path,
    edited_images_dir: Path,
    dest_dir: Path,
):
    bad_image_rel = record.get("bad_image", "")
    if not bad_image_rel:
        return

    bad_image_path_obj = Path(bad_image_rel)
    edited_filename = f"{bad_image_path_obj.stem}_edited{bad_image_path_obj.suffix}"

    bad_image_path = benchmark_dir / bad_image_rel
    edited_image_path = edited_images_dir / edited_filename

    base_name = bad_image_path_obj.stem
    category = record.get("category", "unknown")
    sample_dir = dest_dir / f"{base_name}_{category}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    if bad_image_path.exists():
        shutil.copy(bad_image_path, sample_dir / "bad.png")
    if edited_image_path.exists():
        shutil.copy(edited_image_path, sample_dir / "image_edited.png")

    import json
    with open(sample_dir / "data.json", "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=4)


def print_statistics(
    records_by_index: Dict[int, Dict[str, Any]],
    benchmark_dir: Path,
    edited_images_dir: Path,
    qualitative_dir: Path = None,
    output_file: str = None,
):
    """Print evaluation statistics and optionally save qualitative analysis samples."""
    category_stats = {}

    processed_records = [
        r for r in records_by_index.values()
        if r.get("answer") is False
        and r.get("image_eval") is not None
        and r["image_eval"].get("s_rect") is not None
    ]

    improved_dir = worsened_dir = unchanged_dir = None
    if qualitative_dir:
        improved_dir = qualitative_dir / "improved"
        worsened_dir = qualitative_dir / "worsened"
        unchanged_dir = qualitative_dir / "unchanged"
        for d in [improved_dir, worsened_dir, unchanged_dir]:
            d.mkdir(parents=True, exist_ok=True)
        logger.info(f"Qualitative analysis samples will be saved to: {qualitative_dir}")

    for record in processed_records:
        category = record.get("category", "Uncategorized")
        if category not in category_stats:
            category_stats[category] = {
                "total_s_rect": 0.0,
                "count": 0,
                "improved": 0,
                "worsened": 0,
                "unchanged": 0,
            }

        stats = category_stats[category]
        stats["total_s_rect"] += record["image_eval"]["s_rect"]
        stats["count"] += 1

        acc_bad = record["image_eval"]["bad_image_eval"]["accuracy"]
        acc_edited = record["image_eval"]["edited_image_eval"]["accuracy"]

        if acc_edited > acc_bad:
            stats["improved"] += 1
            dest_dir = improved_dir
        elif acc_edited < acc_bad:
            stats["worsened"] += 1
            dest_dir = worsened_dir
        else:
            stats["unchanged"] += 1
            dest_dir = unchanged_dir

        if dest_dir:
            save_qualitative_sample(record, benchmark_dir, edited_images_dir, dest_dir)

    header = (f"{'Category':<16}{'Samples':>8}{'S_rect':>9}"
              f"{'Improved':>10}{'Worsened':>10}{'Unchanged':>11}")
    lines = []
    lines.append("\n" + " R3-Bench Step 4: Rectification (S_rect) ".center(len(header), "="))
    lines.append(header)
    lines.append("-" * len(header))

    for category in sorted(category_stats.keys()):
        stats = category_stats[category]
        avg_s_rect = stats["total_s_rect"] / stats["count"] if stats["count"] > 0 else 0.0
        lines.append(f"{category:<16}{stats['count']:>8d}{avg_s_rect:>9.2f}"
                     f"{stats['improved']:>10d}{stats['worsened']:>10d}{stats['unchanged']:>11d}")

    total_s_rect = sum(r["image_eval"]["s_rect"] for r in processed_records)
    total_samples = len(processed_records)
    overall_avg = total_s_rect / total_samples if total_samples > 0 else 0.0

    total_improved = sum(s["improved"] for s in category_stats.values())
    total_worsened = sum(s["worsened"] for s in category_stats.values())
    total_unchanged = sum(s["unchanged"] for s in category_stats.values())

    lines.append("-" * len(header))
    lines.append(f"{'overall':<16}{total_samples:>8d}{overall_avg:>9.2f}"
                 f"{total_improved:>10d}{total_worsened:>10d}{total_unchanged:>11d}")
    lines.append("=" * len(header))

    for line in lines:
        print(line)

    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        logger.info(f"Evaluation summary saved to: {output_file}")


def main():
    args = parse_args()

    logger.info("Loading data...")
    benchmark_dir = Path(args.benchmark_dir)
    edited_images_dir = Path(args.edited_images_dir)

    records = load_jsonl(args.input_jsonl)

    if not records:
        logger.error(f"Input file does not exist or is empty: {args.input_jsonl}")
        return

    if not all("idx" in r for r in records):
        logger.error("Records missing 'idx' field")
        return

    records_by_index = {r["idx"]: r for r in records}

    completed_indices = set()
    if not args.overwrite and Path(args.output_jsonl).exists():
        logger.info(f"Loading checkpoint file: {args.output_jsonl}")
        existing_records = load_jsonl(args.output_jsonl)
        for rec in existing_records:
            idx = rec.get("idx")
            if (idx is not None and
                rec.get("image_eval") and
                rec["image_eval"].get("s_rect") is not None):
                completed_indices.add(idx)
                if idx in records_by_index:
                    records_by_index[idx]["image_eval"] = rec["image_eval"]

        if completed_indices:
            logger.info(f"Restored {len(completed_indices)} completed results")

    need_process = [r for r in records_by_index.values() if r["idx"] not in completed_indices]
    logger.info(f"To process: {len(need_process)} / {len(records)} "
                f"(completed: {len(completed_indices)})")

    if not need_process:
        logger.info("All completed")
    else:
        evaluator_kwargs = {
            "max_tokens": args.max_tokens,
            "max_retries": args.max_retries,
            "request_timeout": args.request_timeout,
        }

        if args.backend == "gpt":
            if not args.api_key or not args.api_base:
                logger.error("GPT backend requires --api-key and --api-base parameters")
                return
            evaluator_kwargs.update({
                "api_key": args.api_key,
                "api_base": args.api_base,
                "api_version": args.api_version,
                "model_name": args.model_name,
            })
        else:
            evaluator_kwargs.update({
                "api_urls": args.api_urls,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "seed": args.seed,
            })

        evaluator = get_image_edit_evaluator(args.backend, **evaluator_kwargs)

        errors = []
        completed_count = 0
        save_interval = 20

        logger.info(f"Starting evaluation with backend: {args.backend}")

        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {
                executor.submit(
                    evaluator.process_record,
                    record,
                    benchmark_dir,
                    edited_images_dir,
                    args.overwrite,
                    completed_indices,
                ): record["idx"]
                for record in need_process
            }

            for future in tqdm(as_completed(futures), total=len(futures),
                               desc=f"{args.backend.upper()} Evaluating"):
                idx, result, err = future.result()

                if result:
                    records_by_index[idx]["image_eval"] = result
                if err:
                    errors.append(err)

                completed_count += 1
                if completed_count % save_interval == 0:
                    valid_records = [
                        r for r in records_by_index.values()
                        if r.get("image_eval") and r["image_eval"].get("s_rect") is not None
                    ]
                    save_jsonl(valid_records, args.output_jsonl, sort_key="idx")

        if errors:
            error_log_path = args.output_jsonl.replace(".jsonl", ".errors.log")
            with open(error_log_path, "w") as f:
                f.write("\n".join(errors))
            logger.warning(f"{len(errors)} errors, details in: {error_log_path}")

    valid_records = [
        r for r in records_by_index.values()
        if r.get("image_eval") and r["image_eval"].get("s_rect") is not None
    ]
    save_jsonl(valid_records, args.output_jsonl, sort_key="idx")
    logger.info(f"Output saved to: {args.output_jsonl}")

    qualitative_dir = Path(args.qualitative_output_dir) if args.qualitative_output_dir else None
    eval_summary_file = edited_images_dir / "eval_summary.txt"
    print_statistics(records_by_index, benchmark_dir, edited_images_dir,
                     qualitative_dir, str(eval_summary_file))


if __name__ == "__main__":
    main()
