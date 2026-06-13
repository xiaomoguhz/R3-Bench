"""Load optional SAM3 metadata from JSONL."""
import json
import os
from typing import Dict, Optional, List
from pathlib import Path


class MetadataLoader:
    """Metadata lookup keyed by prompt text."""
    
    def __init__(self, jsonl_path: str):
        """Load metadata from a JSONL file."""
        self.jsonl_path = jsonl_path
        self.metadata_dict: Dict[str, Dict] = {}
        self._load_metadata()
    
    def _load_metadata(self):
        """Load all metadata into memory."""
        if not os.path.exists(self.jsonl_path):
            print(f"[WARNING] Metadata file not found: {self.jsonl_path}")
            return
        
        count = 0
        with open(self.jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    metadata = json.loads(line)
                    prompt = metadata.get("prompt", "").strip()
                    if prompt:
                        # Keyed by raw prompt; SAM3 prompts need no normalization.
                        self.metadata_dict[prompt] = metadata
                        count += 1
                except Exception as e:
                    print(f"[WARNING] Failed to parse metadata line: {e}")
                    continue
        
        print(f"[INFO] Loaded {count} metadata records")
    
    def get_metadata(self, prompt: str) -> Optional[Dict]:
        """
        Retrieve metadata for the given prompt.

        Args:
            prompt: image-generation prompt

        Returns:
            metadata dict, or None if not found
        """
        prompt = prompt.strip()
        
        if prompt in self.metadata_dict:
            return self.metadata_dict[prompt]
        
        # Allow prompt wrappers around the original prompt.
        for key, metadata in self.metadata_dict.items():
            if key in prompt or prompt in key:
                return metadata
        
        return None
    
    def merge_metadata_to_ground_truth(self, ground_truth_str: str) -> str:
        """
        Merge metadata into a ground_truth JSON string.

        Args:
            ground_truth_str: original ground_truth JSON string

        Returns:
            merged ground_truth JSON string
        """
        try:
            gt_data = json.loads(ground_truth_str)
            prompt = gt_data.get("prompt", "")
            
            if not prompt:
                return ground_truth_str
            
            metadata = self.get_metadata(prompt)
            if metadata:
                for key, value in metadata.items():
                    if key == "task_type":
                        if "category" not in gt_data:
                            gt_data["category"] = value
                        continue

                    if key not in gt_data:
                        gt_data[key] = value
            
            return json.dumps(gt_data, ensure_ascii=False)
        except Exception as e:
            print(f"[WARNING] Failed to merge metadata: {e}")
            return ground_truth_str


def load_metadata_loader(jsonl_path: Optional[str] = None) -> Optional[MetadataLoader]:
    """
    Create a MetadataLoader from a path or the SAM3_METADATA_JSONL env var.

    Args:
        jsonl_path: path to the metadata file; if None, read from the SAM3_METADATA_JSONL env var.

    Returns:
        MetadataLoader instance, or None if the file does not exist.
    """
    if jsonl_path is None:
        jsonl_path = os.environ.get("SAM3_METADATA_JSONL", "")
        if not jsonl_path:
            print("[WARNING] SAM3_METADATA_JSONL not set; skipping metadata loading")
            return None
    
    if not os.path.exists(jsonl_path):
        print(f"[WARNING] Metadata file not found: {jsonl_path}")
        return None
    
    return MetadataLoader(jsonl_path)
