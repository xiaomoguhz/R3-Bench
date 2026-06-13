#!/usr/bin/env python3
"""Merge optional SAM3 metadata into training-data ground_truth fields."""
import json
import argparse
from pathlib import Path
from typing import Optional
from metadata_loader import load_metadata_loader


def merge_metadata_to_ground_truth_batch(
    train_data_path: str,
    metadata_jsonl_path: str,
    output_path: Optional[str] = None
):
    """Merge metadata into a JSON training file."""
    metadata_loader = load_metadata_loader(metadata_jsonl_path)
    if metadata_loader is None:
        print("[ERROR] Failed to load metadata")
        return

    print(f"[INFO] Loading training data: {train_data_path}")
    with open(train_data_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    print(f"[INFO] {len(train_data)} training records loaded")

    merged_count = 0
    for i, item in enumerate(train_data):
        ground_truth_str = item.get("ground_truth", "")
        if ground_truth_str:
            try:
                merged_gt = metadata_loader.merge_metadata_to_ground_truth(ground_truth_str)
                if merged_gt != ground_truth_str:
                    item["ground_truth"] = merged_gt
                    merged_count += 1
            except Exception as e:
                print(f"[WARNING] Failed to merge record {i}: {e}")
                continue
        
        if (i + 1) % 1000 == 0:
            print(f"[INFO] Processed {i + 1}/{len(train_data)} records, merged {merged_count}")
    
    output_file = output_path or train_data_path
    print(f"[INFO] Saving results to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    
    print(f"[INFO] Done. Merged metadata for {merged_count} records")


def main():
    parser = argparse.ArgumentParser(description="Merge metadata into the ground_truth of training data")
    parser.add_argument(
        "--train_data",
        type=str,
        required=True,
        help="path to the training data JSON file"
    )
    parser.add_argument(
        "--metadata_jsonl",
        type=str,
        required=True,
        help="path to the metadata JSONL file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="output file path; if omitted, the input file is overwritten"
    )
    
    args = parser.parse_args()
    
    merge_metadata_to_ground_truth_batch(
        train_data_path=args.train_data,
        metadata_jsonl_path=args.metadata_jsonl,
        output_path=args.output
    )


if __name__ == "__main__":
    main()
