"""Category-specific SAM3 reward scoring."""
from typing import Dict, List, Optional, Tuple, Any
from PIL import Image
import numpy as np

from .sam3_detector import SAM3Detector

def determine_position(
    locality: str,
    box1: Dict[str, float],
    box2: Dict[str, float],
    image_size: Optional[Tuple[int, int]] = None,
    iou_threshold: float = 0.1,
    distance_ratio: float = 0.15,
) -> float:
    """
    Compute a spatial-relation score (adapted from T2I-CompBench-style evaluation logic).

    Args:
        locality: spatial relation phrase (e.g. "on the right of" or "below")
        box1: bounding box of the first object {"x_min": float, "y_min": float, "x_max": float, "y_max": float}
        box2: bounding box of the second object
        image_size: (width, height), used for adaptive distance threshold
        iou_threshold: IoU threshold
        distance_ratio: distance threshold as a fraction of the longest edge (default 15%)

    Returns:
        score (0.0-1.0)
    """
    # Dataset phrasing -> canonical relation names used below.
    locality_mapping = {
        "below": "on the bottom of",
        "above": "on the top of",
        "right of": "on the right of",
        "left of": "on the left of",
        "top of": "on the top of",
        "bottom of": "on the bottom of",
        "on right of": "on the right of",
        "on left of": "on the left of",
        "on top of": "on the top of",
        "on bottom of": "on the bottom of",
        "on the bottom of": "on the bottom of",
        "on the top of": "on the top of",
        "on the right of": "on the right of",
        "on the left of": "on the left of",
        "next to": "next to",
        "near": "near",
        "on side of": "on side of",
    }
    
    locality = locality.lower().strip()
    locality = locality_mapping.get(locality, locality)
    
    box1_center = ((box1['x_min'] + box1['x_max']) / 2, (box1['y_min'] + box1['y_max']) / 2)
    box2_center = ((box2['x_min'] + box2['x_max']) / 2, (box2['y_min'] + box2['y_max']) / 2)
    
    x_distance = box2_center[0] - box1_center[0]
    y_distance = box2_center[1] - box1_center[1]
    max_dim = max(image_size) if image_size else 1024
    distance_threshold = max_dim * distance_ratio
    
    x_overlap = max(0, min(box1['x_max'], box2['x_max']) - max(box1['x_min'], box2['x_min']))
    y_overlap = max(0, min(box1['y_max'], box2['y_max']) - max(box1['y_min'], box2['y_min']))
    intersection = x_overlap * y_overlap
    box1_area = (box1['x_max'] - box1['x_min']) * (box1['y_max'] - box1['y_min'])
    box2_area = (box2['x_max'] - box2['x_min']) * (box2['y_max'] - box2['y_min'])
    union = box1_area + box2_area - intersection
    iou = intersection / union if union > 0 else 0
    
    score = 0.0
    
    if locality in ['next to', 'on side of', 'near']:
        dist = min(abs(x_distance), abs(y_distance))
        score = max(0.0, min(1.0, distance_threshold / max(dist, 1e-5)))
    elif locality == 'on the right of':
        if x_distance < 0:
            if abs(x_distance) > abs(y_distance) and iou < iou_threshold:
                score = 1.0
            elif abs(x_distance) > abs(y_distance) and iou >= iou_threshold:
                score = iou_threshold / iou if iou > 0 else 0.0
        else:
            score = 0.0
    elif locality == 'on the left of':
        if x_distance > 0:
            if abs(x_distance) > abs(y_distance) and iou < iou_threshold:
                score = 1.0
            elif abs(x_distance) > abs(y_distance) and iou >= iou_threshold:
                score = iou_threshold / iou if iou > 0 else 0.0
        else:
            score = 0.0
    elif locality == 'on the bottom of':
        if y_distance < 0:
            if abs(y_distance) > abs(x_distance) and iou < iou_threshold:
                score = 1.0
            elif abs(y_distance) > abs(x_distance) and iou >= iou_threshold:
                score = iou_threshold / iou if iou > 0 else 0.0
        else:
            score = 0.0
    elif locality == 'on the top of':
        if y_distance > 0:
            if abs(y_distance) > abs(x_distance) and iou < iou_threshold:
                score = 1.0
            elif abs(y_distance) > abs(x_distance) and iou >= iou_threshold:
                score = iou_threshold / iou if iou > 0 else 0.0
        else:
            score = 0.0
    
    return score


class SAM3ScoreCalculator:
    """Compute SAM3 reward scores from detection outputs."""
    
    def __init__(self, detector: Optional[SAM3Detector] = None, device: str = "cuda"):
        """Use an existing detector or create one."""
        if detector is None:
            self.detector = SAM3Detector(device=device)
        else:
            self.detector = detector
    
    def calculate_spatial_score(
        self,
        image: Image.Image,
        obj1: str,
        obj2: str,
        locality: str
    ) -> float:
        """
        Compute the spatial-relation score.

        Args:
            image: PIL Image object
            obj1: name of the first object
            obj2: name of the second object
            locality: spatial relation phrase

        Returns:
            score (0.0-1.0)
        """
        if not self.detector.is_available():
            return 0.0
        
        object_names = [obj1, obj2]
        detected_objects, confidences, bboxes = self.detector.detect_objects(
            image,
            object_names,
            return_bbox=True,
            per_object_prompt=True,
        )
        
        obj1_pos = None
        obj2_pos = None
        obj1_lower = obj1.lower()
        obj2_lower = obj2.lower()
        
        obj1_best = (-1.0, None)  # (score, idx)
        obj2_best = (-1.0, None)
        
        for i, detected_obj in enumerate(detected_objects):
            detected_obj_lower = detected_obj.lower() if isinstance(detected_obj, str) else str(detected_obj).lower()
            score = confidences[i]
            
            if obj1_lower == detected_obj_lower or obj1_lower in detected_obj_lower or detected_obj_lower in obj1_lower:
                if score > obj1_best[0]:
                    obj1_best = (score, i)
            if obj2_lower == detected_obj_lower or obj2_lower in detected_obj_lower or detected_obj_lower in obj2_lower:
                if score > obj2_best[0]:
                    obj2_best = (score, i)
        
        obj1_pos = obj1_best[1]
        obj2_pos = obj2_best[1]
        
        # Award partial credit if only one object is found.
        if obj1_pos is None or obj2_pos is None:
            partial_score = 0.0
            if obj1_pos is not None:
                partial_score += 0.25 * confidences[obj1_pos]
            if obj2_pos is not None:
                partial_score += 0.25 * confidences[obj2_pos]
            return partial_score
        
        box1 = {
            "x_min": bboxes[obj1_pos][0],
            "y_min": bboxes[obj1_pos][1],
            "x_max": bboxes[obj1_pos][2],
            "y_max": bboxes[obj1_pos][3],
        }
        box2 = {
            "x_min": bboxes[obj2_pos][0],
            "y_min": bboxes[obj2_pos][1],
            "x_max": bboxes[obj2_pos][2],
            "y_max": bboxes[obj2_pos][3],
        }
        
        position_score = determine_position(
            locality,
            box1,
            box2,
            image_size=(image.width, image.height),
        )
        
        obj_score = 0.25 * confidences[obj1_pos] + 0.25 * confidences[obj2_pos]
        spatial_score = position_score / 2
        score = obj_score + spatial_score

        return score
    
    def calculate_numeracy_score(
        self,
        image: Image.Image,
        expected_objects: List[str],
        expected_counts: List[int]
    ) -> float:
        """
        Compute the numeracy score (adapted from T2I-CompBench-style evaluation logic).

        Args:
            image: PIL Image object
            expected_objects: list of expected object names
            expected_counts: list of expected counts

        Returns:
            score (0.0-1.0)
        """
        if not self.detector.is_available():
            return 0.0
        
        if len(expected_objects) != len(expected_counts):
            return 0.0
        
        detected_objects, confidences, _ = self.detector.detect_objects(
            image,
            expected_objects,
            return_bbox=False,
            per_object_prompt=True,
        )

        score = 0.0
        weight = 1.0 / len(expected_objects)

        for i, expected_obj in enumerate(expected_objects):
            detected_count = 0
            matched_confidences: List[float] = []
            expected_obj_lower = expected_obj.lower() if isinstance(expected_obj, str) else str(expected_obj).lower()
            
            for idx, detected_obj in enumerate(detected_objects):
                detected_obj_lower = detected_obj.lower() if isinstance(detected_obj, str) else str(detected_obj).lower()
                if expected_obj_lower == detected_obj_lower or expected_obj_lower in detected_obj_lower or detected_obj_lower in expected_obj_lower:
                    detected_count += 1
                    matched_confidences.append(confidences[idx])
            
            # Presence and exact-count accuracy contribute equally.
            expected_num = max(1, expected_counts[i])
            presence_score = np.mean(matched_confidences) if detected_count > 0 else 0.0
            count_score = 1 if (matched_confidences and detected_count == expected_num) else 0.0
          
            score += weight * (0.5 * presence_score + 0.5 * count_score)
        
        return score
    
    def calculate_object_score(
        self,
        image: Image.Image,
        expected_objects: List[str],
    ) -> float:
        """
        Compute the object-presence score (adapted from T2I-CompBench-style evaluation logic).

        Args:
            image: PIL Image object
            expected_objects: list of expected object names (plain names or attribute-qualified
                descriptions such as "purple elephant")

        Returns:
            score (0.0-1.0) = detected objects / expected objects
        """
        if not self.detector.is_available():
            return 0.0
        
        detected_objects, confidences, _ = self.detector.detect_objects(
            image,
            expected_objects,
            return_bbox=False,
            per_object_prompt=True,
        )

        detected_count = 0
        conf_accum = 0.0
        for expected_obj in expected_objects:
            expected_obj_lower = expected_obj.lower() if isinstance(expected_obj, str) else str(expected_obj).lower()
            for idx, detected_obj in enumerate(detected_objects):
                detected_obj_lower = detected_obj.lower() if isinstance(detected_obj, str) else str(detected_obj).lower()
                if expected_obj_lower == detected_obj_lower or expected_obj_lower in detected_obj_lower or detected_obj_lower in expected_obj_lower:
                    detected_count += 1
                    conf_accum += confidences[idx]
                    break  # each expected object is matched at most once
        
        if len(expected_objects) > 0:
            base_score = detected_count / len(expected_objects)
            avg_conf = conf_accum / detected_count if detected_count > 0 else 0.0
            score = base_score * avg_conf
        else:
            score = 0.0
        
        return score
    
    def calculate_complex_score(
        self,
        image: Image.Image,
        expected_objects: List[str],
        spatial_info: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Compute the complex-task score.

        Args:
            image: PIL Image object
            expected_objects: list of expected object names
            spatial_info: optional spatial context {"obj1": str, "obj2": str, "locality": str}

        Returns:
            score (0.0-1.0)
        """
        if not self.detector.is_available():
            return 0.0
        
        object_score = self.calculate_object_score(image, expected_objects)
        
        if spatial_info:
            obj1 = spatial_info.get("obj1", "")
            obj2 = spatial_info.get("obj2", "")
            locality = spatial_info.get("locality", "")
            
            if obj1 and obj2 and locality:
                spatial_score = self.calculate_spatial_score(image, obj1, obj2, locality)
                return spatial_score
        
        return object_score
    
    def calculate_score(
        self,
        image: Image.Image,
        metadata: Dict[str, Any],
        category: str
    ) -> float:
        """
        Compute a score by category (unified entry point).

        Args:
            image: PIL Image object
            metadata: metadata dict (contains nouns, attr_nouns, spatial_info, numeracy_info, etc.)
            category: task category

        Returns:
            score (0.0-1.0)
        """
        if category == "spatial":
            spatial_info = metadata.get("spatial_info", {})
            obj1 = spatial_info.get("obj1", "")
            obj2 = spatial_info.get("obj2", "")
            locality = spatial_info.get("locality", "")
            if obj1 and obj2 and locality:
                return self.calculate_spatial_score(image, obj1, obj2, locality)
            return 0.0
        
        elif category == "numeracy":
            numeracy_info = metadata.get("numeracy_info", [])
            if not numeracy_info:
                return 0.0
            expected_objects = [item.get("obj_name", "") for item in numeracy_info]
            expected_counts = [item.get("num", 1) for item in numeracy_info]
            return self.calculate_numeracy_score(image, expected_objects, expected_counts)
        
        elif category in ["color", "shape", "texture"]:
            # Use attribute-qualified targets such as "purple elephant".
            attr_nouns = metadata.get("attr_nouns", [])
            if attr_nouns:
                return self.calculate_object_score(image, attr_nouns)
            
            nouns = metadata.get("nouns", [])
            if nouns:
                return self.calculate_object_score(image, nouns)
            return 0.0
        
        elif category == "object":
            nouns = metadata.get("nouns", [])
            if nouns:
                return self.calculate_object_score(image, nouns)
            return 0.0

        elif category == "non":
            nouns = metadata.get("nouns", [])
            if nouns:
                return self.calculate_object_score(image, nouns)
            return 0.0
        
        elif category == "complex":
            nouns = metadata.get("nouns", [])
            spatial_info = metadata.get("spatial_info")
            return self.calculate_complex_score(image, nouns, spatial_info)
        
        else:
            return 0.0
