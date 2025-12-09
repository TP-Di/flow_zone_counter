"""Database models."""
from sqlalchemy import Column, Integer, String, DateTime, Float, ForeignKey, Text, Boolean
from sqlalchemy.orm import relationship
from datetime import datetime
from .database import Base


class Camera(Base):
    """Camera model."""
    __tablename__ = "cameras"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True, nullable=False)
    location = Column(String, nullable=True)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    detections = relationship("Detection", back_populates="camera", cascade="all, delete-orphan")
    annotations = relationship("Annotation", back_populates="camera", cascade="all, delete-orphan")


class Detection(Base):
    """Detection log model."""
    __tablename__ = "detections"

    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(Integer, ForeignKey("cameras.id"), nullable=False)
    image_path = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

    # Detection metrics
    total_people = Column(Integer, default=0)
    people_in_zone = Column(Integer, default=0)
    unique_ids = Column(Text, nullable=True)  # JSON string of unique track IDs

    # Relationship
    camera = relationship("Camera", back_populates="detections")


class Annotation(Base):
    """Annotation model for labeled images."""
    __tablename__ = "annotations"

    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(Integer, ForeignKey("cameras.id"), nullable=False)
    image_path = Column(String, nullable=False)
    json_path = Column(String, nullable=False)
    is_labeled = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationship
    camera = relationship("Camera", back_populates="annotations")


class TrainingJob(Base):
    """Training job tracking model."""
    __tablename__ = "training_jobs"

    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(Integer, ForeignKey("cameras.id"), nullable=False)
    status = Column(String, default="pending")  # pending, running, completed, failed
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    epochs = Column(Integer, default=50)
    batch_size = Column(Integer, default=16)

    # Metrics
    final_map = Column(Float, nullable=True)
    final_precision = Column(Float, nullable=True)
    final_recall = Column(Float, nullable=True)

    # Paths
    weights_path = Column(String, nullable=True)
    log_path = Column(String, nullable=True)
