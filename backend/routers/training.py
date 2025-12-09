"""Model training endpoints."""
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from datetime import datetime
from pathlib import Path

from ..database import get_db
from ..models import Camera, TrainingJob
from ..schemas import TrainingRequest, TrainingStatus
from ..ml.training import YOLOTrainer

router = APIRouter(prefix="/api/training", tags=["training"])


def train_model_task(
    job_id: int,
    camera_id: int,
    epochs: int,
    batch_size: int,
    image_size: int,
    db_path: str = "people_counter.db"
):
    """
    Background task for model training.

    Args:
        job_id: Training job ID
        camera_id: Camera ID
        epochs: Number of epochs
        batch_size: Batch size
        image_size: Image size
        db_path: Path to database
    """
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from ..database import Base

    # Create new database session for background task
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionLocal()

    try:
        # Get training job
        job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
        if not job:
            return

        # Update job status
        job.status = "running"
        job.started_at = datetime.utcnow()
        db.commit()

        # Get camera name
        camera = db.query(Camera).filter(Camera.id == camera_id).first()
        if not camera:
            return

        # Initialize trainer
        trainer = YOLOTrainer(camera_id=camera_id, camera_name=camera.name)

        # Train model
        metrics = trainer.train(
            epochs=epochs,
            batch_size=batch_size,
            image_size=image_size
        )

        # Update job with results
        job.status = "completed"
        job.completed_at = datetime.utcnow()
        job.weights_path = metrics['weights_path']
        job.final_map = metrics['final_map']
        job.final_precision = metrics['final_precision']
        job.final_recall = metrics['final_recall']
        db.commit()

        # Cleanup temporary files
        trainer.cleanup()

    except Exception as e:
        # Mark job as failed
        job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
        if job:
            job.status = "failed"
            job.completed_at = datetime.utcnow()
            db.commit()

        print(f"Training failed: {e}")

    finally:
        db.close()


@router.post("/start", response_model=TrainingStatus)
def start_training(
    request: TrainingRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Start model training for a camera.

    Args:
        request: Training request parameters
        background_tasks: FastAPI background tasks
        db: Database session

    Returns:
        Training job status
    """
    # Verify camera exists
    camera = db.query(Camera).filter(Camera.id == request.camera_id).first()
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")

    # Check if there are labeled images
    labeled_dir = Path("cameras") / camera.name / "labeled"
    if not labeled_dir.exists():
        raise HTTPException(
            status_code=400,
            detail="No labeled directory found. Please annotate images first."
        )

    json_files = list(labeled_dir.glob("*.json"))
    if len(json_files) == 0:
        raise HTTPException(
            status_code=400,
            detail="No labeled images found. Please annotate images first."
        )

    # Check if there's already a running training job
    running_job = db.query(TrainingJob).filter(
        TrainingJob.camera_id == request.camera_id,
        TrainingJob.status == "running"
    ).first()

    if running_job:
        raise HTTPException(
            status_code=400,
            detail="Training already in progress for this camera"
        )

    # Create training job
    job = TrainingJob(
        camera_id=request.camera_id,
        status="pending",
        epochs=request.epochs,
        batch_size=request.batch_size
    )

    db.add(job)
    db.commit()
    db.refresh(job)

    # Start training in background
    background_tasks.add_task(
        train_model_task,
        job_id=job.id,
        camera_id=request.camera_id,
        epochs=request.epochs,
        batch_size=request.batch_size,
        image_size=request.image_size
    )

    return TrainingStatus(
        job_id=job.id,
        camera_id=job.camera_id,
        status=job.status,
        started_at=job.started_at,
        completed_at=job.completed_at,
        progress={"message": "Training job queued"}
    )


@router.get("/status/{job_id}", response_model=TrainingStatus)
def get_training_status(
    job_id: int,
    db: Session = Depends(get_db)
):
    """
    Get training job status.

    Args:
        job_id: Training job ID
        db: Database session

    Returns:
        Training job status
    """
    job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")

    progress = {
        "message": f"Training {job.status}"
    }

    if job.status == "completed":
        progress.update({
            "mAP": job.final_map,
            "precision": job.final_precision,
            "recall": job.final_recall,
            "weights_path": job.weights_path
        })

    return TrainingStatus(
        job_id=job.id,
        camera_id=job.camera_id,
        status=job.status,
        started_at=job.started_at,
        completed_at=job.completed_at,
        progress=progress
    )


@router.get("/camera/{camera_id}/jobs")
def get_camera_training_jobs(
    camera_id: int,
    db: Session = Depends(get_db)
):
    """
    Get all training jobs for a camera.

    Args:
        camera_id: Camera ID
        db: Database session

    Returns:
        List of training jobs
    """
    camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")

    jobs = db.query(TrainingJob).filter(
        TrainingJob.camera_id == camera_id
    ).order_by(TrainingJob.started_at.desc()).all()

    return {
        "camera_id": camera_id,
        "jobs": [
            {
                "job_id": job.id,
                "status": job.status,
                "started_at": job.started_at,
                "completed_at": job.completed_at,
                "epochs": job.epochs,
                "final_map": job.final_map,
                "weights_path": job.weights_path
            }
            for job in jobs
        ],
        "total": len(jobs)
    }
