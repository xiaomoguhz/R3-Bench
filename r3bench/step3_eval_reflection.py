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
Step 3: Reflection Evaluation

Supports GPT and Qwen evaluation backends for evaluating semantic equivalence
between model-generated reflections and ground truth reflections.

Usage:
    # Using Qwen
    python3 -m r3bench.step3_eval_reflection --backend qwen --input-jsonl ... --output-jsonl ...

    # Using GPT
    python3 -m r3bench.step3_eval_reflection --backend gpt --input-jsonl ... --output-jsonl ... \
        --api-key YOUR_KEY --api-base YOUR_ENDPOINT --model-name gpt-4o
"""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict

from tqdm import tqdm

from r3bench.evaluators import get_reflection_evaluator
from r3bench.utils import (
    get_logger,
    load_records_with_checkpoint,
    save_checkpoint,
    save_jsonl,
)

logger = get_logger("step3")


def parse_args():
    parser = argparse.ArgumentParser(description="Step 3: Reflection Evaluation")

    parser.add_argument("--backend", choices=["gpt", "qwen"], required=True,
                        help="Evaluation backend: gpt or qwen")
    parser.add_argument("--input-jsonl", required=True, help="Input JSONL file path")
    parser.add_argument("--output-jsonl", required=True, help="Output JSONL file path")
    parser.add_argument("--benchmark-dir", default=None,
                        help="Benchmark directory path (unused by the text-only reflection judge; "
                             "accepted for interface symmetry with step 4)")

    parser.add_argument("--num-workers", type=int, default=8, help="Number of concurrent workers")
    parser.add_argument("--max-tokens", type=int, default=512, help="Maximum output tokens")
    parser.add_argument("--max-retries", type=int, default=5, help="Maximum retry count")
    parser.add_argument("--overwrite", action="store_true", help="Whether to overwrite existing results")

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


def print_statistics(records_by_index: Dict[int, Dict[str, Any]], result_field: str,
                     output_file: str = None):
    """Print evaluation statistics and optionally save to file."""
    category_stats = defaultdict(lambda: {
        "total": 0,
        "answer_correct": 0,
        "correct_true_cases": 0,
        "both_false_cases": 0,
        "explanation_correct": 0,
    })

    for record in records_by_index.values():
        if not (record.get("model_reflection") and record.get("answer") is not None):
            continue

        category = record.get("category", "uncategorized")
        stats = category_stats[category]
        stats["total"] += 1

        model_answer = record.get("model_reflection", {}).get("answer")
        gt_answer = record.get("answer")

        if model_answer is not None and model_answer == gt_answer:
            stats["answer_correct"] += 1

        if model_answer is True and gt_answer is True:
            stats["correct_true_cases"] += 1
        elif model_answer is False and gt_answer is False:
            stats["both_false_cases"] += 1
            if record.get(result_field, {}).get("is_correct", False):
                stats["explanation_correct"] += 1

    header = f"{'Category':<16}{'Samples':>8}{'Verdict Acc':>14}{'S_ref':>10}"
    lines = []
    lines.append("\n" + " R3-Bench Step 3: Reflection (S_ref) ".center(len(header), "="))
    lines.append(header)
    lines.append("-" * len(header))

    for category in sorted(category_stats.keys()):
        stats = category_stats[category]
        total = stats["total"]

        answer_correct = stats["answer_correct"]
        answer_accuracy = answer_correct / total if total > 0 else 0.0

        holistic_correct = stats["correct_true_cases"] + stats["explanation_correct"]
        holistic_accuracy = holistic_correct / total if total > 0 else 0.0

        lines.append(f"{category:<16}{total:>8d}{answer_accuracy * 100:>14.2f}{holistic_accuracy * 100:>10.2f}")

    total_samples = sum(s["total"] for s in category_stats.values())
    total_answer_correct = sum(s["answer_correct"] for s in category_stats.values())
    total_correct_true = sum(s["correct_true_cases"] for s in category_stats.values())
    total_explanation_correct = sum(s["explanation_correct"] for s in category_stats.values())

    overall_answer_acc = total_answer_correct / total_samples if total_samples > 0 else 0.0
    total_holistic_correct = total_correct_true + total_explanation_correct
    overall_holistic_acc = total_holistic_correct / total_samples if total_samples > 0 else 0.0

    lines.append("-" * len(header))
    lines.append(f"{'overall':<16}{total_samples:>8d}{overall_answer_acc * 100:>14.2f}{overall_holistic_acc * 100:>10.2f}")
    lines.append("=" * len(header))

    for line in lines:
        print(line)

    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        logger.info(f"Evaluation summary saved to: {output_file}")


def main():
    args = parse_args()

    if args.backend == "gpt":
        result_field = "reflection_eval_gpt"
    else:
        result_field = "reflection_eval"

    logger.info("Loading data...")
    benchmark_dir = Path(args.benchmark_dir) if args.benchmark_dir else None

    records_by_index, completed_indices = load_records_with_checkpoint(
        input_path=args.input_jsonl,
        output_path=args.output_jsonl,
        index_field="index",
        result_field=result_field,
        overwrite=args.overwrite,
    )

    if not records_by_index:
        logger.error(f"Input file does not exist or is empty: {args.input_jsonl}")
        return

    need_process = [
        r for r in records_by_index.values()
        if r["index"] not in completed_indices
        and r.get("model_reflection", {}).get("answer") is False
        and r.get("answer") is False
    ]

    logger.info(f"To process: {len(need_process)} / {len(records_by_index)} "
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

        evaluator = get_reflection_evaluator(args.backend, **evaluator_kwargs)

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
                    args.overwrite,
                    completed_indices,
                    result_field,
                ): record["index"]
                for record in need_process
            }

            for future in tqdm(as_completed(futures), total=len(futures),
                               desc=f"{args.backend.upper()} Evaluating"):
                idx, result, err = future.result()

                if result:
                    records_by_index[idx][result_field] = result
                if err:
                    errors.append(err)

                completed_count += 1
                if completed_count % save_interval == 0:
                    save_checkpoint(records_by_index, args.output_jsonl, "index")

        if errors:
            logger.warning(f"{len(errors)} errors:")
            for e in errors[:5]:
                print(f"  - {e}")

    save_jsonl(list(records_by_index.values()), args.output_jsonl, sort_key="index")
    logger.info(f"Output saved to: {args.output_jsonl}")

    output_dir = Path(args.output_jsonl).parent
    eval_summary_file = output_dir / "eval_summary.txt"
    print_statistics(records_by_index, result_field, str(eval_summary_file))


if __name__ == "__main__":
    main()
