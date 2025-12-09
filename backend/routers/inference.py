"""Inference and annotation endpoints."""
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List
from pathlib import Path
import json
import shutil
from datetime import datetime
import cv2

from ..database import get_db
from ..models import Camera, Annotation
from ..schemas import (
    InferenceRequest,
    InferenceResult,
    BatchInferenceResponse,
    AnnotationApproval,
    BoundingBox
)
from ..ml.yolo_inference import get_detector

router = APIRouter(prefix="/api/inference", tags=["inference"])


@router.post("/run", response_model=BatchInferenceResponse)
def run_inference(
    request: InferenceRequest,
    db: Session = Depends(get_db)
):
    """
    Run YOLO inference on unlabeled images.

    Args:
        request: Inference request parameters
        db: Database session

    Returns:
        Batch inference results
    """
    # Verify camera exists
    camera = db.query(Camera).filter(Camera.id == request.camera_id).first()
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")

    # Get unlabeled directory
    unlabeled_dir = Path("cameras") / camera.name / "unlabeled"
    if not unlabeled_dir.exists():
        raise HTTPException(status_code=404, detail="No unlabeled directory found")

    # Get images to process
    if request.image_names:
        image_paths = [unlabeled_dir / name for name in request.image_names]
        # Verify all images exist
        for img_path in image_paths:
            if not img_path.exists():
                raise HTTPException(
                    status_code=404,
                    detail=f"Image not found: {img_path.name}"
                )
    else:
        # Process all unlabeled images
        image_paths = [
            f for f in unlabeled_dir.iterdir()
            if f.suffix.lower() in ['.jpg', '.jpeg', '.png']
        ]

    if not image_paths:
        raise HTTPException(status_code=404, detail="No images to process")

    # Get detector
    detector = get_detector(request.camera_id)

    # Run inference
    results = []

    for img_path in image_paths:
        try:
            bboxes, confidences, img_array = detector.detect_people(
                str(img_path),
                confidence=request.confidence
            )

            # Get image dimensions
            img_height, img_width = img_array.shape[:2]

            # Convert to schema format
            detections = []
            for bbox, conf in zip(bboxes, confidences):
                detections.append(BoundingBox(
                    label="person",
                    points=[[bbox[0], bbox[1]], [bbox[2], bbox[3]]],
                    shape_type="rectangle",
                    confidence=conf
                ))

            results.append(InferenceResult(
                image_name=img_path.name,
                detections=detections,
                image_width=img_width,
                image_height=img_height
            ))

        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")
            continue

    return BatchInferenceResponse(
        camera_id=request.camera_id,
        results=results,
        total_images=len(results)
    )


@router.post("/approve")
def approve_annotation(
    approval: AnnotationApproval,
    db: Session = Depends(get_db)
):
    """
    Approve and save annotation for an image.

    Moves image and creates JSON from unlabeled to labeled directory.

    Args:
        approval: Annotation approval data
        db: Database session

    Returns:
        Save status
    """
    # Verify camera exists
    camera = db.query(Camera).filter(Camera.id == approval.camera_id).first()
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")

    # Get directories
    camera_dir = Path("cameras") / camera.name
    unlabeled_dir = camera_dir / "unlabeled"
    labeled_dir = camera_dir / "labeled"

    # Ensure labeled directory exists
    labeled_dir.mkdir(parents=True, exist_ok=True)

    # Get source image path
    src_image_path = unlabeled_dir / approval.image_name
    if not src_image_path.exists():
        raise HTTPException(status_code=404, detail="Image not found in unlabeled directory")

    # Create JSON annotation
    annotation_dict = approval.annotations.model_dump()

    # Save JSON file
    json_filename = src_image_path.stem + ".json"
    json_path = labeled_dir / json_filename

    with open(json_path, 'w') as f:
        json.dump(annotation_dict, f, indent=2)

    # Move image to labeled directory
    dst_image_path = labeled_dir / approval.image_name
    shutil.move(str(src_image_path), str(dst_image_path))

    # Create annotation record in database
    db_annotation = Annotation(
        camera_id=approval.camera_id,
        image_path=str(dst_image_path.relative_to(Path("cameras"))),
        json_path=str(json_path.relative_to(Path("cameras"))),
        is_labeled=True
    )

    db.add(db_annotation)
    db.commit()

    return {
        "status": "approved",
        "camera_id": approval.camera_id,
        "image_name": approval.image_name,
        "json_path": str(json_path.relative_to(camera_dir)),
        "detections_count": len(approval.annotations.shapes)
    }


@router.get("/unlabeled/{camera_id}")
def get_unlabeled_images(
    camera_id: int,
    db: Session = Depends(get_db)
):
    """
    Get list of unlabeled images for a camera.

    Args:
        camera_id: Camera ID
        db: Database session

    Returns:
        List of unlabeled image filenames
    """
    # Verify camera exists
    camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")

    # Get unlabeled directory
    unlabeled_dir = Path("cameras") / camera.name / "unlabeled"

    if not unlabeled_dir.exists():
        return {
            "camera_id": camera_id,
            "images": [],
            "count": 0
        }

    # Get image files
    images = [
        {
            "filename": f.name,
            "path": str(f.relative_to(Path("cameras"))),
            "size": f.stat().st_size
        }
        for f in unlabeled_dir.iterdir()
        if f.suffix.lower() in ['.jpg', '.jpeg', '.png']
    ]

    return {
        "camera_id": camera_id,
        "images": images,
        "count": len(images)
    }
