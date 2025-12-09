"""Dashboard and analytics endpoints."""
from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List, Optional
from datetime import datetime, timedelta
from pathlib import Path
import json
import csv
import io

from ..database import get_db
from ..models import Camera, Detection
from ..schemas import (
    DashboardStats,
    DetectionLog,
    DetectionLogFilter
)
from ..ml.zone_logic import ZoneManager, calculate_crossing_rate

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])

# Global zone manager
zone_manager = ZoneManager()


@router.get("/stats/{camera_id}", response_model=DashboardStats)
def get_dashboard_stats(
    camera_id: int,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    db: Session = Depends(get_db)
):
    """
    Get dashboard statistics for a camera.

    Args:
        camera_id: Camera ID
        start_date: Optional start date filter
        end_date: Optional end date filter
        db: Database session

    Returns:
        Dashboard statistics
    """
    # Verify camera exists
    camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")

    # Build query
    query = db.query(Detection).filter(Detection.camera_id == camera_id)

    if start_date:
        query = query.filter(Detection.timestamp >= start_date)
    if end_date:
        query = query.filter(Detection.timestamp <= end_date)

    detections = query.order_by(Detection.timestamp.asc()).all()

    # Calculate statistics
    total_unique_people = 0
    people_crossed_zone = 0
    current_in_zone = 0

    # Track unique IDs across all detections
    all_unique_ids = set()
    zone_unique_ids = set()

    # Time series data
    time_series = []

    for detection in detections:
        # Parse unique IDs
        if detection.unique_ids:
            try:
                ids = json.loads(detection.unique_ids)
                all_unique_ids.update(ids)
            except:
                pass

        # Count people in zone
        if detection.people_in_zone > 0:
            if detection.unique_ids:
                try:
                    ids = json.loads(detection.unique_ids)
                    zone_unique_ids.update(ids)
                except:
                    pass

        # Add to time series
        time_series.append({
            "timestamp": detection.timestamp.isoformat(),
            "total_people": detection.total_people,
            "people_in_zone": detection.people_in_zone
        })

    total_unique_people = len(all_unique_ids)
    people_crossed_zone = len(zone_unique_ids)

    # Get current count (most recent detection)
    if detections:
        current_in_zone = detections[-1].people_in_zone

    # Calculate crossing rate
    crossing_rate = calculate_crossing_rate(total_unique_people, people_crossed_zone)

    return DashboardStats(
        camera_id=camera_id,
        camera_name=camera.name,
        total_unique_people=total_unique_people,
        current_people_in_zone=current_in_zone,
        crossing_conversion_rate=crossing_rate,
        time_series=time_series
    )


@router.post("/logs", response_model=List[DetectionLog])
def get_detection_logs(
    filters: DetectionLogFilter,
    db: Session = Depends(get_db)
):
    """
    Get detection logs with filtering.

    Args:
        filters: Filter parameters
        db: Database session

    Returns:
        List of detection logs
    """
    # Build query
    query = db.query(Detection)

    if filters.camera_id:
        query = query.filter(Detection.camera_id == filters.camera_id)

    if filters.start_date:
        query = query.filter(Detection.timestamp >= filters.start_date)

    if filters.end_date:
        query = query.filter(Detection.timestamp <= filters.end_date)

    # Order and paginate
    query = query.order_by(Detection.timestamp.desc())
    query = query.offset(filters.offset).limit(filters.limit)

    detections = query.all()

    return detections


@router.get("/export/{camera_id}")
def export_logs_csv(
    camera_id: int,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    db: Session = Depends(get_db)
):
    """
    Export detection logs as CSV.

    Args:
        camera_id: Camera ID
        start_date: Optional start date
        end_date: Optional end date
        db: Database session

    Returns:
        CSV file
    """
    # Verify camera exists
    camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")

    # Get detections
    query = db.query(Detection).filter(Detection.camera_id == camera_id)

    if start_date:
        query = query.filter(Detection.timestamp >= start_date)
    if end_date:
        query = query.filter(Detection.timestamp <= end_date)

    detections = query.order_by(Detection.timestamp.asc()).all()

    # Create CSV
    output = io.StringIO()
    writer = csv.writer(output)

    # Write header
    writer.writerow([
        'Timestamp',
        'Image Path',
        'Total People',
        'People in Zone',
        'Unique IDs'
    ])

    # Write data
    for detection in detections:
        writer.writerow([
            detection.timestamp.isoformat(),
            detection.image_path,
            detection.total_people,
            detection.people_in_zone,
            detection.unique_ids or ''
        ])

    # Return CSV response
    output.seek(0)
    return Response(
        content=output.getvalue(),
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename=detections_{camera.name}_{datetime.utcnow().strftime('%Y%m%d')}.csv"
        }
    )


@router.post("/process-image/{camera_id}")
def process_and_log_detection(
    camera_id: int,
    image_name: str,
    db: Session = Depends(get_db)
):
    """
    Process a single image and log detection results.

    This endpoint runs full detection pipeline:
    1. YOLO detection
    2. Tracking
    3. Zone crossing
    4. Log to database

    Args:
        camera_id: Camera ID
        image_name: Image filename
        db: Database session

    Returns:
        Detection results
    """
    from ..ml.yolo_inference import get_detector
    from ..ml.tracking import SimpleIOUTracker

    # Verify camera exists
    camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")

    # Get image path
    camera_dir = Path("cameras") / camera.name
    image_path = None

    for subdir in ["labeled", "unlabeled"]:
        test_path = camera_dir / subdir / image_name
        if test_path.exists():
            image_path = test_path
            break

    if not image_path:
        raise HTTPException(status_code=404, detail="Image not found")

    # Run detection
    detector = get_detector(camera_id)
    bboxes, confidences, img_array = detector.detect_people(str(image_path))

    total_people = len(bboxes)

    # Run tracking (simple IOU for demo)
    tracker = SimpleIOUTracker()
    track_ids, tracked_bboxes = tracker.update(bboxes, confidences)

    # Check zone crossing
    people_in_zone = 0
    zone_ids = []

    zone_detector = zone_manager.get_zone(camera_id)
    if zone_detector:
        for bbox, track_id in zip(tracked_bboxes, track_ids):
            if zone_detector.is_in_zone(bbox):
                people_in_zone += 1
                zone_ids.append(track_id)

    # Log detection
    detection = Detection(
        camera_id=camera_id,
        image_path=str(image_path.relative_to(Path("cameras"))),
        timestamp=datetime.utcnow(),
        total_people=total_people,
        people_in_zone=people_in_zone,
        unique_ids=json.dumps(track_ids)
    )

    db.add(detection)
    db.commit()
    db.refresh(detection)

    return {
        "detection_id": detection.id,
        "camera_id": camera_id,
        "image_name": image_name,
        "total_people": total_people,
        "people_in_zone": people_in_zone,
        "unique_ids": track_ids,
        "zone_ids": zone_ids,
        "timestamp": detection.timestamp
    }
