"""YOLO inference utilities."""
import os
import torch
from ultralytics import YOLO
from typing import List, Tuple, Optional
import numpy as np
from pathlib import Path


class YOLODetector:
    """YOLO-based person detector."""

    def __init__(self, weights_path: Optional[str] = None, device: str = "auto"):
        """
        Initialize YOLO detector.

        Args:
            weights_path: Path to custom weights. If None, uses pretrained YOLOv8n.
            device: Device to run on ('cuda', 'cpu', or 'auto')
        """
        self.device = self._get_device(device)

        if weights_path and os.path.exists(weights_path):
            self.model = YOLO(weights_path)
        else:
            # Use pretrained YOLOv8n
            self.model = YOLO('yolov8n.pt')

        # Move to device
        if self.device == 'cuda':
            self.model.to('cuda')

    @staticmethod
    def _get_device(device: str) -> str:
        """Determine the device to use."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def detect_people(
        self,
        image_path: str,
        confidence: float = 0.25,
        iou_threshold: float = 0.45
    ) -> Tuple[List[List[float]], List[float], np.ndarray]:
        """
        Detect people in an image.

        Args:
            image_path: Path to the image
            confidence: Confidence threshold
            iou_threshold: IOU threshold for NMS

        Returns:
            Tuple of (bboxes, confidences, image_array)
            - bboxes: List of [x1, y1, x2, y2] coordinates
            - confidences: List of confidence scores
            - image_array: Original image as numpy array
        """
        results = self.model.predict(
            source=image_path,
            conf=confidence,
            iou=iou_threshold,
            classes=[0],  # Person class only
            device=self.device,
            verbose=False
        )

        bboxes = []
        confidences = []

        if len(results) > 0:
            result = results[0]
            boxes = result.boxes

            if boxes is not None:
                for box in boxes:
                    # Get bbox coordinates
                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())

                    bboxes.append(xyxy.tolist())
                    confidences.append(conf)

            # Get original image
            image_array = result.orig_img

        return bboxes, confidences, image_array

    def batch_detect(
        self,
        image_paths: List[str],
        confidence: float = 0.25
    ) -> List[Tuple[str, List[List[float]], List[float]]]:
        """
        Detect people in multiple images.

        Args:
            image_paths: List of image paths
            confidence: Confidence threshold

        Returns:
            List of tuples (image_path, bboxes, confidences)
        """
        results_list = []

        for image_path in image_paths:
            bboxes, confidences, _ = self.detect_people(image_path, confidence)
            results_list.append((image_path, bboxes, confidences))

        return results_list


def get_detector(camera_id: int, models_dir: str = "models") -> YOLODetector:
    """
    Get YOLO detector for a specific camera.

    Args:
        camera_id: Camera ID
        models_dir: Directory containing model weights

    Returns:
        YOLODetector instance
    """
    weights_path = os.path.join(models_dir, f"camera{camera_id}_weights.pt")

    if os.path.exists(weights_path):
        return YOLODetector(weights_path=weights_path)
    else:
        # Use default pretrained model
        return YOLODetector()
