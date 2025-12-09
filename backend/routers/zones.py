"""Zone configuration endpoints."""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import Optional
from pathlib import Path
import os
import random

from ..database import get_db
from ..models import Camera
from ..schemas import ZoneConfig, ZoneConfigResponse
from ..ml.zone_logic import ZoneManager

router = APIRouter(prefix="/api/zones", tags=["zones"])

# Global zone manager
zone_manager = ZoneManager()


@router.post("/set", response_model=ZoneConfigResponse)
def set_zone(
    zone_config: ZoneConfig,
    db: Session = Depends(get_db)
):
    """
    Set zone configuration for a camera.

    Args:
        zone_config: Zone configuration data
        db: Database session

    Returns:
        Saved zone configuration
    """
    # Verify camera exists
    camera = db.query(Camera).filter(Camera.id == zone_config.camera_id).first()
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")

    # Validate points
    if len(zone_config.points) < 3:
        raise HTTPException(
            status_code=400,
            detail="Zone must have at least 3 points to form a polygon"
        )

    # Save zone configuration
    zone_manager.set_zone(
        camera_id=zone_config.camera_id,
        points=zone_config.points,
        threshold=zone_config.threshold
    )

    return ZoneConfigResponse(
        camera_id=zone_config.camera_id,
        points=zone_config.points,
        threshold=zone_config.threshold
    )


@router.get("/{camera_id}", response_model=Optional[ZoneConfigResponse])
def get_zone(
    camera_id: int,
    db: Session = Depends(get_db)
):
    """
    Get zone configuration for a camera.

    Args:
        camera_id: Camera ID
        db: Database session

    Returns:
        Zone configuration or None
    """
    # Verify camera exists
    camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")

    # Get zone configuration
    if not zone_manager.has_zone(camera_id):
        return None

    zone_config = zone_manager.zones.get(str(camera_id))

    return ZoneConfigResponse(
        camera_id=camera_id,
        points=zone_config["points"],
        threshold=zone_config.get("threshold", 0.3)
    )


@router.delete("/{camera_id}")
def delete_zone(
    camera_id: int,
    db: Session = Depends(get_db)
):
    """
    Delete zone configuration for a camera.

    Args:
        camera_id: Camera ID
        db: Database session

    Returns:
        Deletion status
    """
    # Verify camera exists
    camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")

    zone_manager.delete_zone(camera_id)

    return {"status": "deleted", "camera_id": camera_id}


@router.get("/{camera_id}/sample-image")
def get_sample_image(
    camera_id: int,
    use_random: bool = Query(False, description="Get random image instead of first"),
    db: Session = Depends(get_db)
):
    """
    Get a sample image from camera for zone drawing.

    Args:
        camera_id: Camera ID
        use_random: If True, return a random image instead of the first one
        db: Database session

    Returns:
        Sample image path
    """
    camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")

    # Look for images in both labeled and unlabeled directories
    camera_dir = Path("cameras") / camera.name
    all_images = []

    for subdir in ["labeled", "unlabeled"]:
        search_dir = camera_dir / subdir
        if search_dir.exists():
            images = [
                f for f in search_dir.iterdir()
                if f.suffix.lower() in ['.jpg', '.jpeg', '.png']
            ]
            all_images.extend(images)

    if not all_images:
        raise HTTPException(
            status_code=404,
            detail="No images found for this camera. Please upload images first."
        )

    # Select image based on random parameter
    selected_image = random.choice(all_images) if use_random else all_images[0]

    return {
        "camera_id": camera_id,
        "image_path": str(selected_image.relative_to(Path("cameras"))),
        "image_name": selected_image.name
    }
