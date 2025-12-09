"""Zone crossing detection logic."""
from typing import List, Tuple
from shapely.geometry import box, Polygon, Point
import json
import os


class ZoneDetector:
    """Detector for zone crossing and occupancy."""

    def __init__(self, zone_points: List[List[float]], threshold: float = 0.3):
        """
        Initialize zone detector.

        Args:
            zone_points: List of [x, y] points defining the polygon zone
            threshold: Overlap threshold (0.0 to 1.0)
        """
        self.zone_points = zone_points
        self.threshold = threshold
        self.zone_polygon = Polygon(zone_points)

    def is_in_zone(self, bbox: List[float]) -> bool:
        """
        Check if a bounding box is inside the zone.

        Uses two methods:
        1. Centroid method: Check if bbox centroid is inside zone
        2. Overlap method: Check if bbox overlap with zone >= threshold

        Args:
            bbox: Bounding box [x1, y1, x2, y2]

        Returns:
            True if bbox is considered inside zone
        """
        x1, y1, x2, y2 = bbox
        person_box = box(x1, y1, x2, y2)

        # Method 1: Centroid check
        centroid = person_box.centroid
        if self.zone_polygon.contains(centroid):
            return True

        # Method 2: Overlap check
        if not self.zone_polygon.intersects(person_box):
            return False

        intersection = person_box.intersection(self.zone_polygon)
        intersection_area = intersection.area
        person_area = person_box.area

        if person_area == 0:
            return False

        overlap_ratio = intersection_area / person_area
        return overlap_ratio >= self.threshold

    def count_in_zone(self, bboxes: List[List[float]]) -> int:
        """
        Count how many bboxes are in the zone.

        Args:
            bboxes: List of bounding boxes

        Returns:
            Count of bboxes in zone
        """
        count = 0
        for bbox in bboxes:
            if self.is_in_zone(bbox):
                count += 1
        return count

    def filter_in_zone(
        self,
        bboxes: List[List[float]],
        track_ids: List[int] = None
    ) -> Tuple[List[List[float]], List[int]]:
        """
        Filter bboxes that are in the zone.

        Args:
            bboxes: List of bounding boxes
            track_ids: Optional list of track IDs

        Returns:
            Tuple of (filtered_bboxes, filtered_track_ids)
        """
        filtered_bboxes = []
        filtered_ids = []

        for i, bbox in enumerate(bboxes):
            if self.is_in_zone(bbox):
                filtered_bboxes.append(bbox)
                if track_ids is not None and i < len(track_ids):
                    filtered_ids.append(track_ids[i])

        return filtered_bboxes, filtered_ids


class ZoneManager:
    """Manager for camera zone configurations."""

    def __init__(self, config_path: str = "config/camera_zones.json"):
        """
        Initialize zone manager.

        Args:
            config_path: Path to zone configuration file
        """
        self.config_path = config_path
        self.zones = self._load_zones()

    def _load_zones(self) -> dict:
        """Load zone configurations from file."""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                return json.load(f)
        return {}

    def save_zones(self):
        """Save zone configurations to file."""
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump(self.zones, f, indent=2)

    def set_zone(
        self,
        camera_id: int,
        points: List[List[float]],
        threshold: float = 0.3
    ):
        """
        Set zone configuration for a camera.

        Args:
            camera_id: Camera ID
            points: Zone polygon points
            threshold: Overlap threshold
        """
        self.zones[str(camera_id)] = {
            "points": points,
            "threshold": threshold
        }
        self.save_zones()

    def get_zone(self, camera_id: int) -> ZoneDetector:
        """
        Get zone detector for a camera.

        Args:
            camera_id: Camera ID

        Returns:
            ZoneDetector instance or None if no zone configured
        """
        zone_config = self.zones.get(str(camera_id))
        if zone_config is None:
            return None

        return ZoneDetector(
            zone_points=zone_config["points"],
            threshold=zone_config.get("threshold", 0.3)
        )

    def has_zone(self, camera_id: int) -> bool:
        """Check if a camera has a zone configured."""
        return str(camera_id) in self.zones

    def delete_zone(self, camera_id: int):
        """Delete zone configuration for a camera."""
        if str(camera_id) in self.zones:
            del self.zones[str(camera_id)]
            self.save_zones()

    def get_all_zones(self) -> dict:
        """Get all zone configurations."""
        return self.zones


def calculate_crossing_rate(total_unique: int, people_in_zone: int) -> float:
    """
    Calculate crossing conversion rate.

    Args:
        total_unique: Total unique people detected
        people_in_zone: Number of people who crossed into zone

    Returns:
        Crossing rate as percentage (0-100)
    """
    if total_unique == 0:
        return 0.0
    return (people_in_zone / total_unique) * 100.0
