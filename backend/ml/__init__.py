"""Machine learning utilities for people detection and tracking."""
from .yolo_inference import YOLODetector, get_detector
from .tracking import PersonTracker, SimpleIOUTracker
from .zone_logic import ZoneDetector, ZoneManager, calculate_crossing_rate
from .training import YOLOTrainer, convert_labelme_to_yolo

__all__ = [
    'YOLODetector',
    'get_detector',
    'PersonTracker',
    'SimpleIOUTracker',
    'ZoneDetector',
    'ZoneManager',
    'calculate_crossing_rate',
    'YOLOTrainer',
    'convert_labelme_to_yolo'
]
