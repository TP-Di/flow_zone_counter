"""Person tracking and Re-ID utilities."""
from typing import List, Tuple, Set
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort


class PersonTracker:
    """Person tracker using DeepSORT."""

    def __init__(
        self,
        max_age: int = 30,
        n_init: int = 3,
        max_iou_distance: float = 0.7,
        max_cosine_distance: float = 0.3,
        nn_budget: int = 100
    ):
        """
        Initialize person tracker.

        Args:
            max_age: Maximum frames to keep alive a track without detections
            n_init: Number of consecutive detections before track is confirmed
            max_iou_distance: Maximum IOU distance for matching
            max_cosine_distance: Maximum cosine distance for appearance matching
            nn_budget: Maximum size of appearance descriptors gallery
        """
        self.tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            max_iou_distance=max_iou_distance,
            max_cosine_distance=max_cosine_distance,
            nn_budget=nn_budget,
            embedder="mobilenet",  # Lightweight embedder
            embedder_gpu=False  # CPU for embedder to save GPU memory
        )

        self.track_history = {}  # track_id -> list of centroids

    def update(
        self,
        bboxes: List[List[float]],
        confidences: List[float],
        frame: np.ndarray
    ) -> Tuple[List[int], List[List[float]]]:
        """
        Update tracker with new detections.

        Args:
            bboxes: List of bounding boxes [[x1, y1, x2, y2], ...]
            confidences: List of confidence scores
            frame: Current frame as numpy array

        Returns:
            Tuple of (track_ids, tracked_bboxes)
        """
        # Convert to DeepSORT format: ([left, top, width, height], confidence, class)
        detections = []
        for bbox, conf in zip(bboxes, confidences):
            x1, y1, x2, y2 = bbox
            w = x2 - x1
            h = y2 - y1
            detections.append(([x1, y1, w, h], conf, 'person'))

        # Update tracker
        tracks = self.tracker.update_tracks(detections, frame=frame)

        # Extract confirmed tracks
        track_ids = []
        tracked_bboxes = []

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            ltrb = track.to_ltrb()  # left, top, right, bottom

            track_ids.append(track_id)
            tracked_bboxes.append(ltrb.tolist())

            # Update history
            centroid = self._get_centroid(ltrb.tolist())
            if track_id not in self.track_history:
                self.track_history[track_id] = []
            self.track_history[track_id].append(centroid)

        return track_ids, tracked_bboxes

    def get_unique_ids(self) -> Set[int]:
        """Get all unique track IDs seen so far."""
        return set(self.track_history.keys())

    def reset(self):
        """Reset tracker state."""
        self.tracker = DeepSort(
            max_age=30,
            n_init=3,
            embedder="mobilenet"
        )
        self.track_history = {}

    @staticmethod
    def _get_centroid(bbox: List[float]) -> Tuple[float, float]:
        """Get centroid of bounding box."""
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        return (cx, cy)


class SimpleIOUTracker:
    """
    Simple IOU-based tracker for fixed camera scenarios.
    Lightweight alternative to DeepSORT.
    """

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 30):
        """
        Initialize simple IOU tracker.

        Args:
            iou_threshold: Minimum IOU for matching
            max_age: Maximum frames to keep track alive
        """
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.next_id = 0
        self.tracks = {}  # track_id -> {'bbox': [...], 'age': int, 'hits': int}

    def update(
        self,
        bboxes: List[List[float]],
        confidences: List[float]
    ) -> Tuple[List[int], List[List[float]]]:
        """
        Update tracker with new detections.

        Args:
            bboxes: List of bounding boxes
            confidences: List of confidence scores

        Returns:
            Tuple of (track_ids, tracked_bboxes)
        """
        # Match detections to existing tracks
        matched_tracks = []
        unmatched_detections = list(range(len(bboxes)))

        for track_id, track_data in list(self.tracks.items()):
            best_match_idx = None
            best_iou = self.iou_threshold

            for det_idx in unmatched_detections:
                iou = self._calculate_iou(track_data['bbox'], bboxes[det_idx])
                if iou > best_iou:
                    best_iou = iou
                    best_match_idx = det_idx

            if best_match_idx is not None:
                # Update track
                self.tracks[track_id]['bbox'] = bboxes[best_match_idx]
                self.tracks[track_id]['age'] = 0
                self.tracks[track_id]['hits'] += 1
                matched_tracks.append((track_id, bboxes[best_match_idx]))
                unmatched_detections.remove(best_match_idx)
            else:
                # Increment age
                self.tracks[track_id]['age'] += 1

        # Create new tracks for unmatched detections
        for det_idx in unmatched_detections:
            track_id = self.next_id
            self.next_id += 1
            self.tracks[track_id] = {
                'bbox': bboxes[det_idx],
                'age': 0,
                'hits': 1
            }
            matched_tracks.append((track_id, bboxes[det_idx]))

        # Remove old tracks
        self.tracks = {
            tid: data for tid, data in self.tracks.items()
            if data['age'] <= self.max_age
        }

        # Extract results
        track_ids = [tid for tid, _ in matched_tracks]
        tracked_bboxes = [bbox for _, bbox in matched_tracks]

        return track_ids, tracked_bboxes

    def get_unique_ids(self) -> Set[int]:
        """Get all unique track IDs."""
        return set(range(self.next_id))

    def reset(self):
        """Reset tracker."""
        self.next_id = 0
        self.tracks = {}

    @staticmethod
    def _calculate_iou(bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate IOU between two bboxes."""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2

        # Intersection
        xi_min = max(x1_min, x2_min)
        yi_min = max(y1_min, y2_min)
        xi_max = min(x1_max, x2_max)
        yi_max = min(y1_max, y2_max)

        inter_width = max(0, xi_max - xi_min)
        inter_height = max(0, yi_max - yi_min)
        inter_area = inter_width * inter_height

        # Union
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area

        if union_area == 0:
            return 0.0

        return inter_area / union_area
