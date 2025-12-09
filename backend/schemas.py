"""Pydantic schemas for request/response validation."""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


# Camera Schemas
class CameraCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    location: Optional[str] = None
    description: Optional[str] = None


class CameraResponse(BaseModel):
    id: int
    name: str
    location: Optional[str]
    description: Optional[str]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class CameraStats(BaseModel):
    camera_id: int
    camera_name: str
    labeled_count: int
    unlabeled_count: int
    total_count: int


# Zone Configuration Schemas
class ZoneConfig(BaseModel):
    camera_id: int
    points: List[List[float]]  # [[x1, y1], [x2, y2], ...]
    threshold: float = Field(default=0.3, ge=0.0, le=1.0)


class ZoneConfigResponse(BaseModel):
    camera_id: int
    points: List[List[float]]
    threshold: float


# Annotation Schemas
class BoundingBox(BaseModel):
    label: str = "person"
    points: List[List[float]]  # [[x1, y1], [x2, y2]]
    shape_type: str = "rectangle"
    confidence: Optional[float] = None


class AnnotationData(BaseModel):
    shapes: List[BoundingBox]
    imagePath: str
    imageHeight: int
    imageWidth: int


class AnnotationApproval(BaseModel):
    camera_id: int
    image_name: str
    annotations: AnnotationData


# Inference Schemas
class InferenceRequest(BaseModel):
    camera_id: int
    image_names: Optional[List[str]] = None  # If None, process all unlabeled
    confidence: float = Field(default=0.25, ge=0.0, le=1.0)


class InferenceResult(BaseModel):
    image_name: str
    detections: List[BoundingBox]
    image_width: int
    image_height: int


class BatchInferenceResponse(BaseModel):
    camera_id: int
    results: List[InferenceResult]
    total_images: int


# Training Schemas
class TrainingRequest(BaseModel):
    camera_id: int
    epochs: int = Field(default=50, ge=1, le=500)
    batch_size: int = Field(default=16, ge=1, le=64)
    image_size: int = Field(default=640, ge=320, le=1280)


class TrainingStatus(BaseModel):
    job_id: int
    camera_id: int
    status: str
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    progress: Optional[Dict[str, Any]] = None


# Dashboard Schemas
class DetectionLog(BaseModel):
    id: int
    camera_id: int
    image_path: str
    timestamp: datetime
    total_people: int
    people_in_zone: int
    unique_ids: Optional[str]

    class Config:
        from_attributes = True


class DashboardStats(BaseModel):
    camera_id: int
    camera_name: str
    total_unique_people: int
    current_people_in_zone: int
    crossing_conversion_rate: float
    time_series: List[Dict[str, Any]]


class DetectionLogFilter(BaseModel):
    camera_id: Optional[int] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    limit: int = Field(default=100, ge=1, le=1000)
    offset: int = Field(default=0, ge=0)
