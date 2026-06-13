"""SAM3 reward entry points used by the HTTP service."""
import json
import os
from typing import Dict, Optional, Any
from PIL import Image

from .metadata_loader import MetadataLoader, load_metadata_loader
from .sam3_detector import SAM3Detector
from .score_calculator import SAM3ScoreCalculator

_metadata_loader: Optional[MetadataLoader] = None
_score_calculator: Optional[SAM3ScoreCalculator] = None


def initialize_sam3_reward(
    metadata_jsonl_path: Optional[str] = None,
    device: str = "cuda",
    bpe_path: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
):
    """Load the optional metadata index and the SAM3 detector."""
    global _metadata_loader, _score_calculator
    
    if _metadata_loader is None and metadata_jsonl_path:
        _metadata_loader = load_metadata_loader(metadata_jsonl_path)
    
    if _score_calculator is None:
        detector = SAM3Detector(
            device=device,
            bpe_path=bpe_path,
            checkpoint_path=checkpoint_path,
        )
        _score_calculator = SAM3ScoreCalculator(detector=detector, device=device)
    
    return _metadata_loader, _score_calculator


def compute_sam3_reward(
    image_path: Optional[str],
    prompt: str,
    category: str,
    ground_truth: Optional[str] = None,
    metadata_jsonl_path: Optional[str] = None,
    device: str = "cuda",
    bpe_path: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    image: Optional[Image.Image] = None,
) -> Dict[str, Any]:
    """
    Compute the SAM3 reward score.

    Args:
        image_path: path to the image
        prompt: image-generation prompt
        category: task category (spatial, numeracy, color, shape, texture, object, non, complex)
        ground_truth: ground_truth JSON string (optional; metadata is extracted from it if provided)
        metadata_jsonl_path: path to the metadata JSONL file (optional)
        device: device to use
        bpe_path: path to the BPE tokenizer file for the SAM3 detector (optional)
        checkpoint_path: path to the SAM3 model checkpoint (optional)
        image: PIL Image passed directly (optional; takes priority over image_path)

    Returns:
        {
            "score": float,  # score (0.0-1.0)
            "success": bool,  # whether the computation succeeded
            "error": str,  # error message if any
        }
    """
    try:
        metadata_loader, score_calculator = initialize_sam3_reward(
            metadata_jsonl_path,
            device,
            bpe_path=bpe_path,
            checkpoint_path=checkpoint_path,
        )
        if not score_calculator.detector.is_available():
            return {
                "score": 0.0,
                "success": False,
                "error": "SAM3 detector not available"
            }
        
        if image is None:
            if not image_path or not os.path.exists(image_path):
                return {
                    "score": 0.0,
                    "success": False,
                    "error": f"Image not found: {image_path}"
                }
            image = Image.open(image_path).convert("RGB")
        
        metadata = None
        
        if ground_truth:
            try:
                gt_data = json.loads(ground_truth)
                if "nouns" in gt_data or "spatial_info" in gt_data or "numeracy_info" in gt_data:
                    metadata = gt_data
            except json.JSONDecodeError:
                pass
        
        if metadata is None:
            if metadata_loader is not None:
                metadata = metadata_loader.get_metadata(prompt)
            if metadata is None:
                return {
                    "score": 0.0,
                    "success": False,
                    "error": f"Metadata not found for prompt: {prompt}"
                }
        
        score = score_calculator.calculate_score(image, metadata, category)
        
        return {
            "score": score,
            "success": True,
            "error": None
        }
        
    except Exception as e:
        return {
            "score": 0.0,
            "success": False,
            "error": str(e)
        }
