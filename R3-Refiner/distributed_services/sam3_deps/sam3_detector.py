"""SAM3 object detector wrapper used by the reward service."""
import os
import sys
import torch
from PIL import Image
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

try:
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    SAM3_AVAILABLE = True
except ImportError:
    SAM3_AVAILABLE = False
    print("[WARNING] SAM3 module import failed; SAM3 detection will be unavailable")

DEFAULT_CONFIDENCE_THRESHOLD = 0.5  # matches SAM3Processor default


class SAM3Detector:
    """Thin wrapper around SAM3 image-text detection."""
    
    def __init__(
        self,
        device: str = "cuda",
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        bpe_path: str | None = None,
        checkpoint_path: str | None = None,
    ):
        """Load SAM3 and its processor."""
        self.device = device
        self.model = None
        self.processor = None
        
        if not SAM3_AVAILABLE:
            print("[WARNING] SAM3 not available; detector will not function")
            return
        
        try:
            if bpe_path or checkpoint_path:
                self.model = build_sam3_image_model(
                    bpe_path=bpe_path,
                    checkpoint_path=checkpoint_path,
                )
            else:
                self.model = build_sam3_image_model()
            self.processor = Sam3Processor(self.model, confidence_threshold=confidence_threshold)
            if device != "cpu" and torch.cuda.is_available():
                self.model = self.model.to(device)
            print(f"[INFO] SAM3 model loaded successfully (device: {device})")
        except Exception as e:
            print(f"[ERROR] SAM3 model load failed: {e}")
            self.model = None
            self.processor = None
    
    def is_available(self) -> bool:
        """Return whether the detector finished loading."""
        return self.model is not None and self.processor is not None
    
    def detect_objects(
        self,
        image: Image.Image,
        object_names: List[str],
        return_bbox: bool = True,
        per_object_prompt: bool = True,
    ) -> Tuple[List[str], List[float], Optional[List[List[float]]]]:
        """
        Detect specified objects in an image using SAM3.

        Args:
            image: PIL Image object
            object_names: list of object names to detect (e.g. ["chair", "table"])
            return_bbox: whether to return bounding box coordinates
            per_object_prompt: whether to issue a separate text prompt per object (more robust)

        Returns:
            detected_objects: list of detected object names
            confidences: corresponding confidence scores
            bboxes: list of bounding boxes (when return_bbox=True), format [[x1, y1, x2, y2], ...]
        """
        if not self.is_available():
            return [], [], None if return_bbox else []
        
        try:
            if not isinstance(image, Image.Image):
                image = Image.fromarray(image)
            image = image.convert("RGB")
            
            inference_state = self.processor.set_image(image)
            
            detected_objects = []
            confidences = []
            bboxes_list = []
            
            img_width, img_height = image.size
            
            prompts = object_names if per_object_prompt else [". ".join(object_names) + "."]
            
            # Prompt one object at a time for counting and attribute/position alignment.
            for idx, prompt in enumerate(prompts):
                with torch.no_grad():
                    output = self.processor.set_text_prompt(
                        state=inference_state,
                        prompt=prompt
                    )
                
                boxes = output.get("boxes", [])
                scores = output.get("scores", [])

                # Convert tensors before boolean checks.
                if torch.is_tensor(boxes):
                    boxes = boxes.detach().cpu().tolist()
                if torch.is_tensor(scores):
                    scores = scores.detach().cpu().tolist()

                if len(boxes) == 0 or len(scores) == 0:
                    continue
                
                # Use the prompt object name when per-object prompting is enabled.
                matched_obj = prompt if per_object_prompt else (object_names[idx] if idx < len(object_names) else (object_names[0] if object_names else None))
                
                for box, score in zip(boxes, scores):
                    name_to_use = matched_obj
                    if name_to_use is None:
                        continue
                    
                    detected_objects.append(name_to_use)
                    confidences.append(float(score))
                    
                    if return_bbox and box is not None:
                        if isinstance(box, (list, tuple)) and len(box) >= 4:
                            bbox = [float(box[0]), float(box[1]), float(box[2]), float(box[3])]
                            bboxes_list.append(bbox)
                        elif hasattr(box, 'tolist'):
                            bbox = box.tolist()
                            if len(bbox) >= 4:
                                bboxes_list.append(bbox[:4])
                            else:
                                bboxes_list.append([0, 0, img_width, img_height])
                        else:
                            bboxes_list.append([0, 0, img_width, img_height])
            
            return detected_objects, confidences, bboxes_list if return_bbox else None
            
        except Exception as e:
            print(f"[ERROR] SAM3 detection failed: {e}")
            return [], [], None if return_bbox else []
