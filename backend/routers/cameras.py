"""Camera management endpoints."""
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from typing import List
import os
import shutil
from pathlib import Path

from ..database import get_db
from ..models import Camera, Annotation
from ..schemas import CameraCreate, CameraResponse, CameraStats

router = APIRouter(prefix="/api/cameras", tags=["cameras"])


@router.post("/create", response_model=CameraResponse)
def create_camera(
    camera: CameraCreate,
    db: Session = Depends(get_db)
):
    """
    Create a new camera.

    Args:
        camera: Camera creation data
        db: Database session

    Returns:
        Created camera
    """
    # Check if camera with same name exists
    existing = db.query(Camera).filter(Camera.name == camera.name).first()
    if existing:
        raise HTTPException(status_code=400, detail="Camera with this name already exists")

    # Create camera
    db_camera = Camera(
        name=camera.name,
        location=camera.location,
        description=camera.description
    )

    db.add(db_camera)
    db.commit()
    db.refresh(db_camera)

    # Create directories for camera
    camera_dir = Path("cameras") / camera.name
    (camera_dir / "labeled").mkdir(parents=True, exist_ok=True)
    (camera_dir / "unlabeled").mkdir(parents=True, exist_ok=True)

    return db_camera


@router.get("/list", response_model=List[CameraResponse])
def list_cameras(db: Session = Depends(get_db)):
    """
    List all cameras.

    Args:
        db: Database session

    Returns:
        List of cameras
    """
    cameras = db.query(Camera).all()
    return cameras


@router.get("/{camera_id}", response_model=CameraResponse)
def get_camera(
    camera_id: int,
    db: Session = Depends(get_db)
):
    """
    Get camera by ID.

    Args:
        camera_id: Camera ID
        db: Database session

    Returns:
        Camera details
    """
    camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")

    return camera


@router.get("/{camera_id}/stats", response_model=CameraStats)
def get_camera_stats(
    camera_id: int,
    db: Session = Depends(get_db)
):
    """
    Get statistics for a camera.

    Args:
        camera_id: Camera ID
        db: Database session

    Returns:
        Camera statistics
    """
    camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")

    # Count labeled and unlabeled images
    camera_dir = Path("cameras") / camera.name
    labeled_dir = camera_dir / "labeled"
    unlabeled_dir = camera_dir / "unlabeled"

    labeled_count = 0
    unlabeled_count = 0

    if labeled_dir.exists():
        # Count image files (not JSON)
        labeled_count = len([
            f for f in labeled_dir.iterdir()
            if f.suffix.lower() in ['.jpg', '.jpeg', '.png']
        ])

    if unlabeled_dir.exists():
        unlabeled_count = len([
            f for f in unlabeled_dir.iterdir()
            if f.suffix.lower() in ['.jpg', '.jpeg', '.png']
        ])

    return CameraStats(
        camera_id=camera.id,
        camera_name=camera.name,
        labeled_count=labeled_count,
        unlabeled_count=unlabeled_count,
        total_count=labeled_count + unlabeled_count
    )


@router.post("/{camera_id}/upload")
async def upload_images(
    camera_id: int,
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db)
):
    """
    Upload images to camera's unlabeled directory.

    Args:
        camera_id: Camera ID
        files: List of image files
        db: Database session

    Returns:
        Upload status
    """
    camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")

    # Get unlabeled directory
    unlabeled_dir = Path("cameras") / camera.name / "unlabeled"
    unlabeled_dir.mkdir(parents=True, exist_ok=True)

    uploaded_files = []
    errors = []

    for file in files:
        try:
            # Validate file type
            if not file.content_type.startswith('image/'):
                errors.append(f"{file.filename}: Not an image file")
                continue

            # Save file
            file_path = unlabeled_dir / file.filename
            with open(file_path, 'wb') as f:
                content = await file.read()
                f.write(content)

            uploaded_files.append(file.filename)

        except Exception as e:
            errors.append(f"{file.filename}: {str(e)}")

    return {
        "camera_id": camera_id,
        "uploaded": len(uploaded_files),
        "failed": len(errors),
        "uploaded_files": uploaded_files,
        "errors": errors
    }


@router.delete("/{camera_id}")
def delete_camera(
    camera_id: int,
    db: Session = Depends(get_db)
):
    """
    Delete a camera and all its data.

    Args:
        camera_id: Camera ID
        db: Database session

    Returns:
        Deletion status
    """
    camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")

    # Delete camera directory
    camera_dir = Path("cameras") / camera.name
    if camera_dir.exists():
        shutil.rmtree(camera_dir)

    # Delete from database (cascades to detections and annotations)
    db.delete(camera)
    db.commit()

    return {"status": "deleted", "camera_id": camera_id}
