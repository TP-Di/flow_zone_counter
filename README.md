# People Counter - Advanced Computer Vision Web Application

A complete end-to-end computer vision web application for people counting, tracking, and analytics across multiple fixed cameras. Built with FastAPI, React, and YOLOv8 with advanced features including Re-ID tracking, zone-based analytics, and automated model training.

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18-blue.svg)](https://reactjs.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple.svg)](https://github.com/ultralytics/ultralytics)

---

## Table of Contents

- [Features](#features)
- [Technology Stack](#technology-stack)
- [Project Architecture](#project-architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage Guide](#usage-guide)
- [Advanced Features](#advanced-features)
- [API Reference](#api-reference)
- [Database Schema](#database-schema)
- [Model Training](#model-training)
- [Performance Optimization](#performance-optimization)
- [Metrics & Analytics](#metrics--analytics)
- [Troubleshooting](#troubleshooting)

---

## Features

### 🎥 Camera Management
- **Multi-camera support** with independent configurations
- **Batch image upload** via drag-and-drop interface
- **Automatic dataset organization** (labeled/unlabeled directories)
- **Camera statistics tracking** (image counts, annotations)
- **Metadata storage** (name, location, description, timestamps)
- **RESTful CRUD operations** for camera lifecycle management

### 🎯 Advanced Zone Configuration
- **Interactive polygon zone drawing** on sample images
- **Arbitrary polygon shapes** (not limited to rectangles)
- **Dual detection method:**
  - Centroid-based detection (point-in-polygon)
  - Area overlap percentage (configurable threshold)
- **Per-camera zone persistence** in JSON configuration
- **Configurable overlap threshold** (0-100%, default 30%)
- **Visual zone preview** with transparency overlay

### ✏️ Intelligent Annotation Tool
- **Automated YOLO inference** on unlabeled images
- **Interactive bounding box editor** with canvas manipulation
- **Approve/deny/edit/delete** detection workflow
- **Confidence score display** for each detection
- **Batch annotation processing**
- **LabelMe JSON format** compatibility
- **Automatic format conversion** (LabelMe → YOLO)
- **One-click model retraining** with background job tracking
- **Real-time training progress monitoring**

### 📊 Dashboard & Analytics
- **Real-time people counting** across all cameras
- **Unique person tracking** with Re-ID capabilities
- **Zone crossing analytics** and conversion rates
- **Time-series visualization** with interactive charts
- **Date range filtering** for historical analysis
- **Detection log browser** with pagination
- **CSV export** for external analysis
- **Crossing conversion rate metrics** (people in zone / total people)

### 🤖 Dual Person Tracking System
- **DeepSORT tracking** (production mode):
  - Appearance-based Re-ID with MobileNet embeddings
  - Kalman filter for motion prediction
  - Track ID persistence across frames
  - Handles occlusions and re-entries
- **Simple IOU Tracker** (lightweight fallback):
  - Intersection-over-Union matching
  - Low computational overhead
  - Fast processing for edge devices
- **Configurable tracker selection** via environment variable
- **Track history maintenance** for unique counting

### 🧠 Model Training & Fine-tuning
- **Background training jobs** (non-blocking API)
- **Automatic dataset preparation** with train/val split (80/20)
- **YOLOv8n base model** for speed optimization
- **Configurable hyperparameters:**
  - Epochs (default: 50)
  - Batch size (default: 16)
  - Image size (default: 640x640)
  - Confidence threshold (default: 0.25)
- **Training metrics extraction** (mAP, Precision, Recall)
- **Automatic weight saving** with versioning
- **Job status tracking** in database

---

## Technology Stack

### Backend
| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Framework** | FastAPI 0.104.1 | High-performance async API |
| **Runtime** | Python 3.11+ | Core application logic |
| **ML Framework** | PyTorch 2.1.1 + torchvision 0.16.1 | Deep learning inference |
| **Object Detection** | YOLOv8n (Ultralytics 8.0.228) | Person detection |
| **Person Tracking** | DeepSORT 1.3.2 | Multi-object tracking with Re-ID |
| **Computer Vision** | OpenCV 4.8.1.78 | Image processing |
| **Geometry** | Shapely 2.0.2 | Polygon operations |
| **Database** | SQLAlchemy 2.0.23 + aiosqlite 0.19.0 | ORM and async DB |
| **Validation** | Pydantic 2.5.0 | Schema validation |
| **Server** | Uvicorn 0.24.0 | ASGI web server |

### Frontend
| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Framework** | React 18 | UI components |
| **Styling** | Tailwind CSS | Utility-first CSS |
| **Charts** | Recharts | Data visualization |
| **Canvas** | React-Konva | Interactive zone drawing |
| **File Upload** | React-Dropzone | Drag-and-drop uploads |
| **Routing** | React Router | Client-side navigation |

### Deployment
| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Container** | Docker with CUDA 11.8 | GPU-accelerated runtime |
| **Orchestration** | Docker Compose | Multi-container management |
| **Base Image** | nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 | GPU support |
| **Web Server** | Nginx (in Docker) | Static file serving |

### Database
- **Development:** SQLite 3 (embedded, single file)
- **Production:** PostgreSQL support (configurable via DATABASE_URL)
- **ORM:** SQLAlchemy with async support

---

## Project Architecture

### Directory Structure

```
end_to_end_2_0/
├── backend/                          # FastAPI Backend (2,423 LOC)
│   ├── main.py                      # Application entry point, CORS, routing
│   ├── database.py                  # SQLAlchemy setup, session factory
│   ├── models.py                    # Database models (4 tables)
│   ├── schemas.py                   # Pydantic request/response schemas
│   │
│   ├── ml/                          # Machine Learning Modules
│   │   ├── __init__.py
│   │   ├── yolo_inference.py       # YOLODetector wrapper class
│   │   ├── tracking.py             # PersonTracker (DeepSORT) + SimpleIOUTracker
│   │   ├── zone_logic.py           # ZoneDetector + ZoneManager
│   │   └── training.py             # YOLOTrainer with dataset conversion
│   │
│   └── routers/                     # API Endpoints (5 modules)
│       ├── __init__.py
│       ├── cameras.py              # Camera CRUD, upload, statistics
│       ├── zones.py                # Zone configuration endpoints
│       ├── inference.py            # YOLO inference, annotation approval
│       ├── training.py             # Background training job management
│       └── dashboard.py            # Analytics aggregation, CSV export
│
├── frontend/                         # React Application
│   ├── public/
│   │   └── index.html
│   ├── src/
│   │   ├── index.js                # Entry point
│   │   ├── App.js                  # Main application component
│   │   ├── pages/
│   │   │   ├── CameraManager.jsx   # Camera creation and management
│   │   │   ├── ZoneDrawer.jsx      # Interactive polygon zone drawing
│   │   │   ├── AnnotationTool.jsx  # Annotation interface with YOLO
│   │   │   └── Dashboard.jsx       # Analytics and visualization
│   │   └── services/
│   │       └── api.js              # Axios-based API client
│   ├── package.json
│   └── tailwind.config.js
│
├── cameras/                         # Camera Dataset Storage
│   ├── .gitkeep
│   └── {camera_name}/
│       ├── labeled/                # Annotated images + LabelMe JSON
│       │   ├── image1.jpg
│       │   ├── image1.json
│       │   └── ...
│       └── unlabeled/              # Raw images awaiting annotation
│           ├── image1.jpg
│           └── ...
│
├── models/                          # Trained Model Weights
│   ├── .gitkeep
│   ├── camera{N}_weights.pt        # Fine-tuned YOLO weights
│   └── camera{N}/                  # Training artifacts (runs/)
│
├── config/                          # Configuration Files
│   ├── .gitkeep
│   └── camera_zones.json           # Zone polygon definitions
│
├── scripts/                         # Utility Scripts
│   ├── convert_labelme_to_yolo.py # Standalone annotation converter
│   ├── start_local.sh              # Linux/macOS startup script
│   └── start_local.bat             # Windows startup script
│
├── examples/                        # Example Files
│   └── sample_annotation.json      # LabelMe format example
│
├── Dockerfile                       # Multi-stage Docker build
├── docker-compose.yml              # Container orchestration
├── requirements.txt                # Python dependencies (34 packages)
├── .env.example                    # Environment variable template
├── .env                            # Active environment configuration
├── .gitignore                      # Git ignore rules
├── .dockerignore                   # Docker build exclusions
├── people_counter.db               # SQLite database (auto-created)
└── README.md                        # This file
```

### Component Interaction Flow

```
┌─────────────┐       HTTP/REST        ┌──────────────┐
│   React     │ ◄────────────────────► │   FastAPI    │
│  Frontend   │    JSON Payloads       │   Backend    │
└─────────────┘                        └──────┬───────┘
                                              │
                    ┌─────────────────────────┼─────────────────┐
                    │                         │                 │
              ┌─────▼─────┐          ┌───────▼────────┐  ┌─────▼──────┐
              │ SQLAlchemy│          │  YOLOv8 Engine │  │ File System│
              │    ORM    │          │   (PyTorch)    │  │  (Images)  │
              └─────┬─────┘          └───────┬────────┘  └────────────┘
                    │                        │
              ┌─────▼─────┐          ┌───────▼────────┐
              │  SQLite/  │          │   DeepSORT     │
              │ PostgreSQL│          │    Tracker     │
              └───────────┘          └────────────────┘
```

---

## Installation

### Prerequisites

- **Docker** with NVIDIA Container Toolkit (nvidia-docker2)
- **CUDA-compatible GPU** (CUDA 11.8+)
- **8GB+ GPU memory** (recommended)
- **Python 3.11+** (for local development)
- **Node.js 18+** (for frontend development)

### Quick Start with Docker (Recommended)

1. **Clone the repository**
```bash
git clone <repository-url>
cd end_to_end_2_0
```

2. **Configure environment variables**
```bash
cp .env.example .env
# Edit .env with your preferred settings (see Configuration section)
```

3. **Build and run with Docker Compose**
```bash
docker-compose up --build
```

4. **Access the application**
- **Frontend:** http://localhost:8000
- **API Documentation:** http://localhost:8000/docs
- **Alternative API Docs:** http://localhost:8000/redoc
- **Health Check:** http://localhost:8000/health

### Local Development Setup

#### Backend Setup

1. **Create virtual environment**
```bash
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download YOLOv8n base model** (automatic on first run)
```bash
# Model automatically downloaded to ~/.cache/torch/hub/ultralytics/
```

4. **Run backend server**
```bash
python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

#### Frontend Setup

1. **Install Node dependencies**
```bash
cd frontend
npm install
```

2. **Configure API endpoint** (if different from default)
```bash
# Edit frontend/src/services/api.js
# Change baseURL to your backend URL
```

3. **Run development server**
```bash
npm start
```

Frontend will be available at http://localhost:3000 (development mode).

---

## Configuration

### Environment Variables (.env)

The `.env` file controls all runtime configurations. Copy `.env.example` to `.env` and customize:

```bash
# ===== Database Configuration =====
# SQLite (default for development)
DATABASE_URL=sqlite:///./people_counter.db

# PostgreSQL (recommended for production)
# DATABASE_URL=postgresql://user:password@localhost:5432/people_counter

# ===== CUDA Configuration =====
# GPU device selection (0 for first GPU, -1 for CPU)
CUDA_DEVICE=0

# ===== API Server Configuration =====
API_HOST=0.0.0.0
API_PORT=8000

# ===== YOLO Model Configuration =====
# Confidence threshold for detections (0.0 - 1.0)
MODEL_CONFIDENCE=0.25

# IOU threshold for non-max suppression (0.0 - 1.0)
MODEL_IOU_THRESHOLD=0.45

# ===== Tracking Configuration =====
# Tracker type: 'deepsort' or 'simple_iou'
TRACKER_TYPE=deepsort

# DeepSORT max age (frames to keep lost tracks)
DEEPSORT_MAX_AGE=30

# DeepSORT minimum hits (detections before track confirmation)
DEEPSORT_MIN_HITS=3

# DeepSORT IOU threshold for matching
DEEPSORT_IOU_THRESHOLD=0.3

# ===== Training Configuration =====
# Default training epochs
TRAINING_EPOCHS=50

# Default batch size
TRAINING_BATCH_SIZE=16

# Training image size (square)
TRAINING_IMAGE_SIZE=640

# Train/validation split ratio
TRAINING_VAL_SPLIT=0.2

# ===== Zone Configuration =====
# Default zone overlap threshold (0.0 - 1.0)
DEFAULT_ZONE_THRESHOLD=0.3

# ===== CORS Configuration =====
# Allowed origins (comma-separated). Use * for development only!
CORS_ORIGINS=*

# ===== File Upload Configuration =====
# Maximum file upload size (bytes)
MAX_UPLOAD_SIZE=10485760  # 10MB

# Allowed image extensions
ALLOWED_EXTENSIONS=jpg,jpeg,png,bmp
```

### Camera Zone Configuration (config/camera_zones.json)

Automatically managed via the API, but manual editing is supported:

```json
{
  "1": {
    "points": [
      [100, 200],
      [500, 200],
      [500, 600],
      [100, 600]
    ],
    "threshold": 0.3
  },
  "2": {
    "points": [
      [150, 150],
      [450, 100],
      [600, 500],
      [200, 550]
    ],
    "threshold": 0.5
  }
}
```

**Field Descriptions:**
- `points`: Array of [x, y] coordinates defining polygon vertices (minimum 3 points)
- `threshold`: Overlap percentage (0.0-1.0) required for zone detection

---

## Usage Guide

### 1. Camera Management

#### Creating a Camera

1. Navigate to **Camera Management** page
2. Click **Create Camera** button
3. Fill in camera details:
   - **Name:** Unique identifier (e.g., "Entrance Camera", "C9")
   - **Location:** Physical location (e.g., "Main Entrance", "Floor 2")
   - **Description:** Optional notes
4. Click **Create**

#### Uploading Images

1. Select a camera from the list
2. Click **Upload Images** button
3. Drag and drop images or click to browse
4. Supported formats: JPG, JPEG, PNG, BMP
5. Images automatically placed in `cameras/{camera_name}/unlabeled/`

#### Viewing Statistics

- **Total Images:** Combined labeled + unlabeled count
- **Labeled:** Images with approved annotations
- **Unlabeled:** Images awaiting annotation

### 2. Zone Configuration

#### Drawing a Zone

1. Navigate to **Zone Configuration** page
2. Select a camera from dropdown
3. Click **Start Drawing** button
4. Click on the sample image to add polygon vertices
5. **Minimum 3 points** required
6. Click **Finish Zone** when complete
7. Adjust **Threshold Percentage** slider (default: 30%)
8. Click **Save Zone**

#### Zone Detection Logic

A person is counted in the zone if **either** condition is met:

1. **Centroid Method:** Person's bounding box center point is inside the polygon
2. **Overlap Method:** Bounding box overlap with zone ≥ threshold percentage

**Example:**
```
Zone threshold: 30%
Person bounding box: 1000 pixels²
Overlap area: 350 pixels²
Overlap percentage: 35%
Result: Person counted in zone ✓
```

#### Modifying Zones

- **To edit:** Delete existing zone and redraw
- **To delete:** Use the delete button or API endpoint
- **To view:** Zone overlays displayed on sample images

### 3. Annotation Tool

#### Automated Annotation Workflow

1. Navigate to **Annotation Tool** page
2. Select a camera with unlabeled images
3. Click **Start Analysis**
4. YOLO inference runs on current image
5. Review detected bounding boxes:
   - **Green boxes:** Approved detections
   - **Red boxes:** Denied (false positives)
   - **Confidence scores:** Displayed on each box

#### Annotation Actions

| Action | Button | Description |
|--------|--------|-------------|
| **Approve** | ✓ | Keep detection as valid |
| **Deny** | ✗ | Mark as false positive |
| **Delete** | 🗑️ | Remove bounding box |
| **Edit** | Drag corners | Adjust bounding box size |
| **Add** | Draw new box | Manually add missed detections |

#### Saving Annotations

1. Review all detections
2. Click **Approve & Save**
3. Annotation saved in LabelMe JSON format
4. Image moved to `cameras/{camera_name}/labeled/`
5. Corresponding `.json` file created

#### Model Training

1. Annotate **minimum 20-30 images** per camera
2. Click **Train Model** button
3. Training job starts in background
4. Monitor progress in **Training Jobs** section
5. Status updates: `pending` → `running` → `completed`
6. Trained weights saved to `models/camera{N}_weights.pt`

### 4. Dashboard & Analytics

#### Viewing Metrics

1. Navigate to **Dashboard** page
2. Select a camera from dropdown
3. Set date range filters (optional)
4. View key metrics:
   - **Total People:** Unique individuals detected
   - **People in Zone:** Count inside defined zone
   - **Crossing Rate:** (People in zone / Total people) × 100%

#### Time-Series Chart

- **X-Axis:** Timestamp
- **Y-Axis:** People count
- **Blue Line:** Total people per detection
- **Green Line:** People in zone per detection
- **Hover:** View exact values

#### Detection Logs

- **Paginated table** with all detections
- **Columns:** Timestamp, Total People, People in Zone, Image Path
- **Sorting:** Click column headers
- **Filtering:** Use date range selector

#### Exporting Data

1. Click **Export to CSV** button
2. Downloads CSV file: `camera_{id}_detections_{date}.csv`
3. **Columns:**
   - `timestamp`
   - `total_people`
   - `people_in_zone`
   - `crossing_rate`
   - `image_path`

---

## Advanced Features

### 🔍 Hidden Metrics & Analytics

#### Unique Person Tracking

**How it works:**
- Each person assigned a unique **Track ID** by DeepSORT
- Track IDs persist across frames (within video sequence)
- Appearance features (MobileNet embeddings) enable Re-ID
- Track history stored in `unique_ids` JSON field

**Access via API:**
```python
GET /api/dashboard/stats/{camera_id}
```

**Response includes:**
```json
{
  "unique_people_count": 15,
  "detections": [
    {
      "timestamp": "2024-11-16T10:30:00",
      "unique_ids": [1, 3, 5, 7]  // Track IDs in this frame
    }
  ]
}
```

#### Crossing Conversion Rate

**Formula:**
```
Crossing Rate = (People in Zone / Total People) × 100%
```

**Use Cases:**
- Retail: Conversion from foot traffic to store entry
- Security: Monitor restricted area access
- Analytics: Zone popularity metrics

**Calculation Example:**
```
Image 1: 10 total people, 3 in zone → 30% crossing rate
Image 2: 15 total people, 8 in zone → 53% crossing rate
Average crossing rate: (30% + 53%) / 2 = 41.5%
```

### 🎯 Dual Tracker System

#### DeepSORT Tracker (Production)

**Advantages:**
- Appearance-based Re-ID with deep learning
- Handles occlusions and re-entries
- Kalman filter for motion prediction
- High accuracy for crowded scenes

**Configuration:**
```bash
TRACKER_TYPE=deepsort
DEEPSORT_MAX_AGE=30        # Keep lost tracks for 30 frames
DEEPSORT_MIN_HITS=3        # Require 3 detections before confirmation
DEEPSORT_IOU_THRESHOLD=0.3 # IOU threshold for matching
```

**Performance:**
- **Speed:** ~15-20 FPS (GPU)
- **Accuracy:** High (with embeddings)
- **Memory:** Moderate (stores embeddings)

#### Simple IOU Tracker (Fallback)

**Advantages:**
- Lightweight (no deep learning)
- Fast processing (CPU-friendly)
- Low memory footprint
- Good for non-crowded scenes

**Configuration:**
```bash
TRACKER_TYPE=simple_iou
```

**Performance:**
- **Speed:** ~50-60 FPS (CPU)
- **Accuracy:** Moderate (geometry-based only)
- **Memory:** Low

**When to use:**
- Edge devices with limited resources
- Non-crowded scenes
- Real-time requirements without GPU

### 🧠 Advanced Training Features

#### Automatic Dataset Splitting

Training script automatically splits labeled data:
- **80% training set**
- **20% validation set**
- Random shuffle before split
- Maintains class distribution

#### Training Metrics Extraction

After training, these metrics are automatically extracted and stored:

| Metric | Description | Database Field |
|--------|-------------|----------------|
| **mAP50** | Mean Average Precision at IoU=0.5 | `final_map` |
| **Precision** | True Positives / (TP + FP) | `final_precision` |
| **Recall** | True Positives / (TP + FN) | `final_recall` |

**Accessing metrics:**
```python
GET /api/training/status/{job_id}

Response:
{
  "status": "completed",
  "final_map": 0.87,
  "final_precision": 0.92,
  "final_recall": 0.89,
  "weights_path": "models/camera1_weights.pt"
}
```

#### Incremental Training

- Models are fine-tuned from **YOLOv8n base**
- Each training job builds on base model (not previous training)
- To enable incremental training, modify `training.py`:

```python
# Line 267 in backend/ml/training.py
model = YOLO('models/camera1_weights.pt')  # Load previous weights
```

### 📐 Advanced Zone Detection

#### Polygon Point-in-Polygon Algorithm

Uses Shapely's `contains()` method implementing ray-casting algorithm:
- Cast ray from point to infinity
- Count polygon edge intersections
- Odd count → inside, Even count → outside
- **Time complexity:** O(n) where n = polygon vertices

#### Area Overlap Calculation

```python
# Pseudo-code from zone_logic.py
bbox_polygon = Polygon(bbox_coords)
zone_polygon = Polygon(zone_coords)

intersection = bbox_polygon.intersection(zone_polygon).area
bbox_area = bbox_polygon.area

overlap_percentage = intersection / bbox_area
is_in_zone = overlap_percentage >= threshold
```

**Optimization:**
- Bounding box pre-check before expensive intersection
- Polygon caching to avoid re-creation

### 🔒 Security Considerations

#### Current Status

- **CORS:** Set to `allow_origins=["*"]` (development only)
- **Authentication:** Not implemented
- **File validation:** Extension-based (not content-based)
- **SQL injection:** Protected by SQLAlchemy ORM
- **XSS:** React's built-in escaping

#### Hardening for Production

1. **Restrict CORS origins:**
```python
# backend/main.py
origins = [
    "https://yourdomain.com",
    "https://api.yourdomain.com"
]
```

2. **Add API authentication:**
```bash
pip install python-jose[cryptography] passlib[bcrypt]
```

3. **Implement rate limiting:**
```bash
pip install slowapi
```

4. **Enable HTTPS:**
```bash
# Use reverse proxy (Nginx) with SSL certificates
```

5. **Validate file content:**
```python
from PIL import Image
# Verify image integrity before processing
```

---

## API Reference

### Base URL
```
http://localhost:8000
```

### Cameras API

#### Create Camera
```http
POST /api/cameras/create
Content-Type: application/json

{
  "name": "string",
  "location": "string",
  "description": "string (optional)"
}

Response 200:
{
  "id": 1,
  "name": "Entrance Camera",
  "location": "Main Building",
  "description": "",
  "created_at": "2024-11-16T10:00:00",
  "updated_at": "2024-11-16T10:00:00"
}
```

#### List Cameras
```http
GET /api/cameras/list

Response 200:
[
  {
    "id": 1,
    "name": "Entrance Camera",
    "location": "Main Building",
    "description": "",
    "created_at": "2024-11-16T10:00:00",
    "updated_at": "2024-11-16T10:00:00"
  }
]
```

#### Get Camera Details
```http
GET /api/cameras/{camera_id}

Response 200:
{
  "id": 1,
  "name": "Entrance Camera",
  "location": "Main Building",
  "description": "",
  "created_at": "2024-11-16T10:00:00",
  "updated_at": "2024-11-16T10:00:00"
}
```

#### Get Camera Statistics
```http
GET /api/cameras/{camera_id}/stats

Response 200:
{
  "labeled_count": 151,
  "unlabeled_count": 523,
  "total_count": 674
}
```

#### Upload Images
```http
POST /api/cameras/{camera_id}/upload
Content-Type: multipart/form-data

files: [File, File, ...]

Response 200:
{
  "uploaded": 10,
  "failed": 0,
  "message": "Successfully uploaded 10 images"
}
```

#### Delete Camera
```http
DELETE /api/cameras/{camera_id}

Response 200:
{
  "message": "Camera deleted successfully"
}
```

### Zones API

#### Set Zone Configuration
```http
POST /api/zones/set
Content-Type: application/json

{
  "camera_id": 1,
  "points": [[100, 200], [500, 200], [500, 600], [100, 600]],
  "threshold": 0.3
}

Response 200:
{
  "message": "Zone configuration saved",
  "camera_id": 1
}
```

#### Get Zone Configuration
```http
GET /api/zones/{camera_id}

Response 200:
{
  "points": [[100, 200], [500, 200], [500, 600], [100, 600]],
  "threshold": 0.3
}

Response 404:
{
  "detail": "No zone configured for this camera"
}
```

#### Delete Zone
```http
DELETE /api/zones/{camera_id}

Response 200:
{
  "message": "Zone deleted successfully"
}
```

#### Get Sample Image
```http
GET /api/zones/{camera_id}/sample

Response 200:
{
  "image_url": "/cameras/camera_name/unlabeled/image1.jpg",
  "width": 1920,
  "height": 1080
}
```

### Inference API

#### Run YOLO Inference
```http
POST /api/inference/run
Content-Type: application/json

{
  "camera_id": 1,
  "image_path": "cameras/C9/unlabeled/image1.jpg",
  "confidence": 0.25  // Optional, defaults to MODEL_CONFIDENCE
}

Response 200:
{
  "detections": [
    {
      "bbox": [100, 200, 300, 500],  // [x1, y1, x2, y2]
      "confidence": 0.89,
      "class": "person",
      "track_id": 3
    }
  ],
  "image_width": 1920,
  "image_height": 1080,
  "image_path": "cameras/C9/unlabeled/image1.jpg"
}
```

#### Approve Annotation
```http
POST /api/inference/approve
Content-Type: application/json

{
  "camera_id": 1,
  "image_path": "cameras/C9/unlabeled/image1.jpg",
  "detections": [
    {
      "bbox": [100, 200, 300, 500],
      "confidence": 0.89
    }
  ]
}

Response 200:
{
  "message": "Annotation saved successfully",
  "json_path": "cameras/C9/labeled/image1.json",
  "image_path": "cameras/C9/labeled/image1.jpg"
}
```

#### Get Unlabeled Images
```http
GET /api/inference/unlabeled/{camera_id}

Response 200:
{
  "unlabeled_images": [
    "cameras/C9/unlabeled/image1.jpg",
    "cameras/C9/unlabeled/image2.jpg"
  ],
  "count": 2
}
```

### Training API

#### Start Training Job
```http
POST /api/training/start
Content-Type: application/json

{
  "camera_id": 1,
  "epochs": 50,        // Optional, defaults to TRAINING_EPOCHS
  "batch_size": 16,    // Optional, defaults to TRAINING_BATCH_SIZE
  "image_size": 640    // Optional, defaults to TRAINING_IMAGE_SIZE
}

Response 200:
{
  "job_id": 1,
  "status": "pending",
  "message": "Training job started"
}
```

#### Get Training Job Status
```http
GET /api/training/status/{job_id}

Response 200:
{
  "id": 1,
  "camera_id": 1,
  "status": "completed",  // pending | running | completed | failed
  "started_at": "2024-11-16T10:00:00",
  "completed_at": "2024-11-16T11:30:00",
  "epochs": 50,
  "batch_size": 16,
  "final_map": 0.87,
  "final_precision": 0.92,
  "final_recall": 0.89,
  "weights_path": "models/camera1_weights.pt",
  "log_path": "models/camera1/training.log"
}
```

#### List Training Jobs for Camera
```http
GET /api/training/camera/{camera_id}/jobs

Response 200:
[
  {
    "id": 1,
    "status": "completed",
    "started_at": "2024-11-16T10:00:00",
    "completed_at": "2024-11-16T11:30:00",
    "final_map": 0.87
  }
]
```

### Dashboard API

#### Get Camera Statistics
```http
GET /api/dashboard/stats/{camera_id}?start_date=2024-11-01&end_date=2024-11-30

Response 200:
{
  "unique_people": 1523,
  "people_in_zone": 487,
  "crossing_rate": 31.97,  // percentage
  "total_detections": 2341
}
```

#### Get Detection Logs
```http
POST /api/dashboard/logs
Content-Type: application/json

{
  "camera_id": 1,
  "start_date": "2024-11-01T00:00:00",  // Optional
  "end_date": "2024-11-30T23:59:59",    // Optional
  "limit": 100,                          // Optional, default 100
  "offset": 0                            // Optional, default 0
}

Response 200:
{
  "logs": [
    {
      "id": 1,
      "camera_id": 1,
      "timestamp": "2024-11-16T10:30:00",
      "total_people": 5,
      "people_in_zone": 2,
      "unique_ids": [1, 3, 5, 7, 9],
      "image_path": "cameras/C9/labeled/image1.jpg"
    }
  ],
  "total": 2341,
  "limit": 100,
  "offset": 0
}
```

#### Export Detections to CSV
```http
GET /api/dashboard/export/{camera_id}?start_date=2024-11-01&end_date=2024-11-30

Response 200:
Content-Type: text/csv
Content-Disposition: attachment; filename="camera_1_detections_2024-11-16.csv"

timestamp,total_people,people_in_zone,crossing_rate,image_path
2024-11-16T10:30:00,5,2,40.0,cameras/C9/labeled/image1.jpg
...
```

### Health Check

```http
GET /health

Response 200:
{
  "status": "healthy",
  "timestamp": "2024-11-16T10:00:00",
  "version": "1.0.0"
}
```

---

## Database Schema

### Entity-Relationship Diagram

```
┌─────────────────────┐
│      cameras        │
├─────────────────────┤
│ id (PK)             │
│ name                │
│ location            │
│ description         │
│ created_at          │
│ updated_at          │
└──────────┬──────────┘
           │
           │ 1:N
           │
     ┌─────┴──────┬──────────────┬─────────────┐
     │            │              │             │
┌────▼─────┐ ┌───▼──────┐ ┌─────▼──────┐ ┌───▼──────────┐
│detections│ │annotations│ │training_jobs│ │ zone_configs │
└──────────┘ └───────────┘ └────────────┘ └──────────────┘
```

### Table Schemas

#### cameras
```sql
CREATE TABLE cameras (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name VARCHAR(255) NOT NULL UNIQUE,
    location VARCHAR(255),
    description TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

#### detections
```sql
CREATE TABLE detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id INTEGER NOT NULL,
    image_path VARCHAR(512) NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    total_people INTEGER DEFAULT 0,
    people_in_zone INTEGER DEFAULT 0,
    unique_ids JSON,  -- Array of track IDs: [1, 3, 5, 7]
    FOREIGN KEY (camera_id) REFERENCES cameras(id) ON DELETE CASCADE
);

CREATE INDEX idx_detections_camera_id ON detections(camera_id);
CREATE INDEX idx_detections_timestamp ON detections(timestamp);
```

#### annotations
```sql
CREATE TABLE annotations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id INTEGER NOT NULL,
    image_path VARCHAR(512) NOT NULL UNIQUE,
    json_path VARCHAR(512) NOT NULL,
    is_labeled BOOLEAN DEFAULT TRUE,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (camera_id) REFERENCES cameras(id) ON DELETE CASCADE
);

CREATE INDEX idx_annotations_camera_id ON annotations(camera_id);
CREATE INDEX idx_annotations_is_labeled ON annotations(is_labeled);
```

#### training_jobs
```sql
CREATE TABLE training_jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id INTEGER NOT NULL,
    status VARCHAR(50) DEFAULT 'pending',  -- pending | running | completed | failed
    started_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    completed_at DATETIME,
    epochs INTEGER DEFAULT 50,
    batch_size INTEGER DEFAULT 16,
    final_map FLOAT,        -- Mean Average Precision at IoU=0.5
    final_precision FLOAT,  -- Precision metric
    final_recall FLOAT,     -- Recall metric
    weights_path VARCHAR(512),
    log_path VARCHAR(512),
    FOREIGN KEY (camera_id) REFERENCES cameras(id) ON DELETE CASCADE
);

CREATE INDEX idx_training_jobs_camera_id ON training_jobs(camera_id);
CREATE INDEX idx_training_jobs_status ON training_jobs(status);
```

### Sample Queries

#### Get total people detected per day
```sql
SELECT
    DATE(timestamp) as date,
    SUM(total_people) as total_people,
    SUM(people_in_zone) as people_in_zone,
    CAST(SUM(people_in_zone) AS FLOAT) / NULLIF(SUM(total_people), 0) * 100 as crossing_rate
FROM detections
WHERE camera_id = 1
GROUP BY DATE(timestamp)
ORDER BY date DESC;
```

#### Get unique track IDs across all detections
```sql
-- SQLite doesn't natively support JSON array operations
-- Use application-level aggregation in Python
SELECT unique_ids FROM detections WHERE camera_id = 1;
```

#### Get training job history with metrics
```sql
SELECT
    id,
    status,
    started_at,
    completed_at,
    ROUND((julianday(completed_at) - julianday(started_at)) * 24, 2) as duration_hours,
    epochs,
    batch_size,
    final_map,
    final_precision,
    final_recall
FROM training_jobs
WHERE camera_id = 1 AND status = 'completed'
ORDER BY started_at DESC;
```

---

## Model Training

### Training Pipeline

```
┌──────────────────┐
│ Labeled Images   │
│   + JSON files   │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ LabelMe → YOLO   │
│   Conversion     │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Dataset Split    │
│  80% train       │
│  20% validation  │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ YOLOv8 Training  │
│  - 50 epochs     │
│  - Batch 16      │
│  - Image 640px   │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Metric Extract   │
│  - mAP50         │
│  - Precision     │
│  - Recall        │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Save Weights     │
│ camera_N.pt      │
└──────────────────┘
```

### Training Best Practices

#### Dataset Requirements

| Scenario | Minimum Images | Recommended Images | Notes |
|----------|---------------|-------------------|-------|
| **Simple scenes** | 20-30 | 50-100 | Single angle, consistent lighting |
| **Complex scenes** | 50-100 | 200-500 | Multiple angles, varying conditions |
| **Crowded scenes** | 100-200 | 500-1000 | Heavy occlusions, high density |

#### Annotation Quality Checklist

- [ ] **Tight bounding boxes:** Minimal background inclusion
- [ ] **Full visibility:** Include partially visible persons
- [ ] **Consistent labeling:** Same standards across all images
- [ ] **Difficult cases:** Include edge cases (occlusions, small persons)
- [ ] **Lighting variations:** Day/night, bright/dark conditions
- [ ] **Pose diversity:** Standing, sitting, walking, etc.

#### Hyperparameter Tuning

**Default Configuration (YOLOv8n):**
```python
epochs = 50
batch_size = 16
image_size = 640
learning_rate = 0.01 (auto)
optimizer = 'SGD'
augmentation = True (auto)
```

**Small Dataset (<50 images):**
```python
epochs = 100  # More epochs for limited data
batch_size = 8   # Smaller batch for stability
image_size = 640
```

**Large Dataset (>500 images):**
```python
epochs = 50
batch_size = 32  # Larger batch for efficiency
image_size = 640
```

**High accuracy requirement:**
```python
epochs = 100
batch_size = 16
image_size = 1280  # Larger image size
model = 'yolov8s'  # Use larger model
```

#### Training Monitoring

Monitor these metrics during training:

1. **mAP50** (Mean Average Precision at IoU=0.5)
   - Target: >0.80 for good performance
   - >0.90 for excellent performance

2. **Precision** (Quality of detections)
   - Target: >0.85
   - Low precision → too many false positives

3. **Recall** (Completeness of detections)
   - Target: >0.80
   - Low recall → missing true positives

4. **Training/Validation Loss**
   - Should decrease steadily
   - Large gap → overfitting (reduce epochs or add data)

### Data Augmentation

YOLOv8 automatically applies these augmentations:
- Random scaling (0.5x - 1.5x)
- Random cropping
- Horizontal flip (50% probability)
- HSV color jittering
- Mosaic augmentation (4 images combined)

Disable augmentation for testing:
```python
# Modify backend/ml/training.py line 275
results = model.train(
    data=data_yaml_path,
    augment=False  # Disable augmentation
)
```

---

## Performance Optimization

### Inference Speed Benchmarks

| Configuration | Hardware | FPS | Latency | Notes |
|--------------|----------|-----|---------|-------|
| YOLOv8n + DeepSORT | RTX 3080 | 45-50 | ~20ms | Production recommended |
| YOLOv8n + Simple IOU | RTX 3080 | 80-90 | ~11ms | No Re-ID features |
| YOLOv8n + DeepSORT | CPU (i7-12700) | 8-10 | ~100ms | Development only |
| YOLOv8s + DeepSORT | RTX 3080 | 25-30 | ~33ms | Higher accuracy |
| YOLOv8n INT8 | RTX 3080 | 70-80 | ~12ms | Quantized model |

### Optimization Strategies

#### 1. Model Selection

```python
# backend/ml/yolo_inference.py

# Fast (FPS priority)
model = YOLO('yolov8n.pt')  # 3.2M parameters

# Balanced
model = YOLO('yolov8s.pt')  # 11.2M parameters

# Accurate (accuracy priority)
model = YOLO('yolov8m.pt')  # 25.9M parameters
```

#### 2. Batch Processing

Process multiple images in parallel:

```python
import concurrent.futures
from backend.ml.yolo_inference import YOLODetector

detector = YOLODetector()

def process_image(image_path):
    return detector.detect(image_path)

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(process_image, image_paths))
```

#### 3. Model Quantization

Convert to INT8 for 2-3x speedup:

```python
from ultralytics import YOLO

model = YOLO('models/camera1_weights.pt')
model.export(format='engine', int8=True)  # TensorRT INT8
```

#### 4. Image Preprocessing

Optimize image loading:

```python
import cv2

# Use resize instead of padding for speed
image = cv2.imread(image_path)
image = cv2.resize(image, (640, 640))  # Faster than padding
```

#### 5. Database Optimization

**Add indexes:**
```sql
CREATE INDEX idx_detections_camera_timestamp
ON detections(camera_id, timestamp);
```

**Use connection pooling:**
```python
# backend/database.py
from sqlalchemy.pool import StaticPool

engine = create_engine(
    DATABASE_URL,
    poolclass=StaticPool,
    pool_size=10,
    max_overflow=20
)
```

**Switch to PostgreSQL for production:**
```bash
# .env
DATABASE_URL=postgresql://user:password@localhost/people_counter
```

#### 6. Caching Strategies

**Cache detector instance:**
```python
# backend/routers/inference.py
from functools import lru_cache

@lru_cache(maxsize=1)
def get_detector():
    return YOLODetector()
```

**Cache zone configurations:**
```python
# backend/ml/zone_logic.py
class ZoneManager:
    def __init__(self):
        self._cache = {}

    def get_zone(self, camera_id):
        if camera_id not in self._cache:
            self._cache[camera_id] = self._load_from_file(camera_id)
        return self._cache[camera_id]
```

### Memory Management

**GPU Memory Optimization:**
```python
import torch

# Clear GPU cache periodically
torch.cuda.empty_cache()

# Use mixed precision (FP16)
model = YOLO('yolov8n.pt')
model.to('cuda').half()  # FP16 inference
```

**Limit batch size to prevent OOM:**
```python
# Adjust based on GPU memory
MAX_BATCH_SIZE = 32  # For 8GB GPU
MAX_BATCH_SIZE = 64  # For 16GB GPU
```

---

## Metrics & Analytics

### Available Metrics

#### Real-time Detection Metrics

| Metric | Description | Calculation | Use Case |
|--------|-------------|-------------|----------|
| **Total People** | All persons detected in frame | Count of bounding boxes | Foot traffic analysis |
| **People in Zone** | Persons inside defined zone | Count via zone logic | Zone activity monitoring |
| **Crossing Rate** | Zone entry percentage | (People in zone / Total) × 100% | Conversion tracking |
| **Unique IDs** | Track IDs in current frame | Set of active track IDs | Individual tracking |

#### Aggregated Analytics

| Metric | Description | API Endpoint | Database Query |
|--------|-------------|--------------|----------------|
| **Total Unique People** | Distinct track IDs over period | `/api/dashboard/stats/{camera_id}` | `SELECT COUNT(DISTINCT track_id)` |
| **Average Crossing Rate** | Mean zone entry rate | `/api/dashboard/stats/{camera_id}` | `AVG(people_in_zone / total_people)` |
| **Peak Hours** | Time periods with highest traffic | Custom query | `GROUP BY HOUR(timestamp)` |
| **Daily Trends** | People count by day | Custom query | `GROUP BY DATE(timestamp)` |

#### Model Performance Metrics

| Metric | Description | Stored In | Good Value |
|--------|-------------|-----------|------------|
| **mAP50** | Detection accuracy at IoU=0.5 | `training_jobs.final_map` | >0.80 |
| **Precision** | Detection quality | `training_jobs.final_precision` | >0.85 |
| **Recall** | Detection completeness | `training_jobs.final_recall` | >0.80 |
| **F1 Score** | Harmonic mean of P&R | Calculated | >0.80 |

**F1 Score Calculation:**
```python
f1_score = 2 * (precision * recall) / (precision + recall)
```

### Custom Analytics Queries

#### Hourly Traffic Distribution

```python
GET /api/dashboard/logs
# Post-process in client:

import pandas as pd

df = pd.DataFrame(logs)
df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
hourly_stats = df.groupby('hour').agg({
    'total_people': 'sum',
    'people_in_zone': 'sum'
}).reset_index()
```

#### Peak Detection Times

```sql
SELECT
    HOUR(timestamp) as hour,
    SUM(total_people) as total,
    AVG(people_in_zone) as avg_in_zone
FROM detections
WHERE camera_id = 1
GROUP BY HOUR(timestamp)
ORDER BY total DESC
LIMIT 5;
```

#### Weekly Trends

```sql
SELECT
    strftime('%W', timestamp) as week,
    SUM(total_people) as total_people,
    SUM(people_in_zone) as people_in_zone,
    COUNT(*) as detection_count
FROM detections
WHERE camera_id = 1
GROUP BY week
ORDER BY week DESC;
```

### Visualization Examples

#### Time-Series Chart (Recharts)

```javascript
import { LineChart, Line, XAxis, YAxis, Tooltip, Legend } from 'recharts';

<LineChart data={detections}>
  <XAxis dataKey="timestamp" />
  <YAxis />
  <Tooltip />
  <Legend />
  <Line type="monotone" dataKey="total_people" stroke="#3B82F6" />
  <Line type="monotone" dataKey="people_in_zone" stroke="#10B981" />
</LineChart>
```

#### Crossing Rate Heatmap

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Create hour x day matrix
pivot = df.pivot_table(
    values='crossing_rate',
    index=df['timestamp'].dt.hour,
    columns=df['timestamp'].dt.dayofweek,
    aggfunc='mean'
)

sns.heatmap(pivot, cmap='YlOrRd', annot=True)
plt.xlabel('Day of Week')
plt.ylabel('Hour of Day')
plt.title('Crossing Rate Heatmap')
```

---

## Troubleshooting

### Common Issues

#### 1. CUDA Not Available

**Symptoms:**
```
RuntimeError: CUDA not available
```

**Solutions:**
```bash
# Check CUDA installation
nvidia-smi

# Verify PyTorch CUDA support
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**Fallback to CPU:**
```bash
# .env
CUDA_DEVICE=-1  # Use CPU
```

#### 2. Out of Memory (OOM)

**Symptoms:**
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**Solutions:**
```bash
# Reduce batch size in .env
TRAINING_BATCH_SIZE=8  # Instead of 16

# Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"

# Use smaller model
# Edit backend/ml/yolo_inference.py
model = YOLO('yolov8n.pt')  # Instead of yolov8s or yolov8m
```

#### 3. Database Locked

**Symptoms:**
```
sqlite3.OperationalError: database is locked
```

**Solutions:**
```bash
# Increase timeout
# Edit backend/database.py
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False, "timeout": 30}
)

# Or switch to PostgreSQL
DATABASE_URL=postgresql://user:pass@localhost/people_counter
```

#### 4. Training Job Stuck in "Running"

**Symptoms:**
- Job status never changes to "completed"
- No weights file generated

**Solutions:**
```bash
# Check training logs
cat models/camera1/training.log

# Verify labeled images exist
ls cameras/camera1/labeled/*.json | wc -l  # Should be >20

# Manually reset job status
# Use SQLite browser or:
sqlite3 people_counter.db "UPDATE training_jobs SET status='failed' WHERE id=1;"
```

#### 5. Zone Detection Not Working

**Symptoms:**
- All detections show 0 people in zone
- Zone doesn't appear on dashboard

**Solutions:**
```bash
# Verify zone configuration exists
cat config/camera_zones.json

# Check zone points are valid
# Points should be in image coordinates (pixels)

# Verify zone threshold is reasonable
# Default: 0.3 (30%)
# If too high, no detections will match

# Check zone logic in backend/ml/zone_logic.py:87
# Ensure both centroid and overlap checks are enabled
```

#### 6. Inference Returns No Detections

**Symptoms:**
- API returns empty detections array
- Dashboard shows 0 people

**Solutions:**
```bash
# Lower confidence threshold
# .env
MODEL_CONFIDENCE=0.15  # Instead of 0.25

# Verify YOLO model loaded
python -c "from backend.ml.yolo_inference import YOLODetector; d = YOLODetector(); print(d.model)"

# Check image quality
# Images should be:
# - Well-lit
# - In focus
# - High enough resolution (>640px)

# Test with sample image
python -m backend.ml.yolo_inference
```

#### 7. Frontend Not Loading

**Symptoms:**
- Blank page at http://localhost:8000
- 404 errors in console

**Solutions:**
```bash
# Rebuild frontend
cd frontend
npm install
npm run build

# Verify static files exist
ls frontend/build/static/

# Check backend static file serving
# In backend/main.py, verify:
app.mount("/", StaticFiles(directory="frontend/build", html=True))
```

#### 8. Port Already in Use

**Symptoms:**
```
OSError: [Errno 98] Address already in use
```

**Solutions:**
```bash
# Find process using port 8000
lsof -i :8000  # Linux/macOS
netstat -ano | findstr :8000  # Windows

# Kill process
kill -9 <PID>  # Linux/macOS
taskkill /PID <PID> /F  # Windows

# Or change port
# docker-compose.yml
ports:
  - "8001:8000"
```

#### 9. Image Upload Fails

**Symptoms:**
- 413 Request Entity Too Large
- Upload times out

**Solutions:**
```bash
# Increase max upload size
# .env
MAX_UPLOAD_SIZE=52428800  # 50MB

# Configure Nginx if using reverse proxy
# nginx.conf
client_max_body_size 50M;

# Upload images in smaller batches
# <100 images per upload
```

#### 10. Model Training Fails

**Symptoms:**
```
ValueError: Dataset must contain at least 20 images
```

**Solutions:**
```bash
# Verify labeled images count
ls cameras/camera1/labeled/*.json | wc -l

# Ensure JSON files are valid LabelMe format
cat cameras/camera1/labeled/image1.json | python -m json.tool

# Check for missing image files
# Each .json file needs corresponding .jpg file

# Verify annotations contain "person" class
grep -r '"label": "person"' cameras/camera1/labeled/
```

### Debug Mode

Enable detailed logging:

```python
# backend/main.py (add at top)
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Now all API calls will log detailed info
```

### Performance Profiling

Profile inference speed:

```python
import time
from backend.ml.yolo_inference import YOLODetector

detector = YOLODetector()

start = time.time()
results = detector.detect("path/to/image.jpg")
end = time.time()

print(f"Inference time: {(end - start) * 1000:.2f} ms")
print(f"FPS: {1 / (end - start):.2f}")
```

### Health Checks

Monitor application health:

```bash
# Basic health check
curl http://localhost:8000/health

# Check GPU utilization
nvidia-smi

# Check disk space
df -h

# Check database size
du -h people_counter.db

# Monitor logs
tail -f models/camera1/training.log
```

---

## Deployment

### Production Checklist

- [ ] **Security:**
  - [ ] Restrict CORS origins
  - [ ] Add API authentication
  - [ ] Enable HTTPS
  - [ ] Validate file uploads (content, not just extension)
  - [ ] Set up firewall rules

- [ ] **Database:**
  - [ ] Migrate to PostgreSQL
  - [ ] Set up automated backups
  - [ ] Add indexes on query-heavy columns
  - [ ] Configure connection pooling

- [ ] **Performance:**
  - [ ] Enable model caching
  - [ ] Set up reverse proxy (Nginx)
  - [ ] Configure rate limiting
  - [ ] Optimize static file serving

- [ ] **Monitoring:**
  - [ ] Set up logging (ELK stack or similar)
  - [ ] Configure alerts (Prometheus + Grafana)
  - [ ] Track API metrics
  - [ ] Monitor GPU utilization

- [ ] **Backup:**
  - [ ] Database backups (daily)
  - [ ] Model weights backups
  - [ ] Configuration backups
  - [ ] Camera images backup (optional)

### Docker Production Build

```dockerfile
# Dockerfile (optimized for production)
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/
COPY models/ ./models/
COPY config/ ./config/

# Run as non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

### Environment Variables (Production)

```bash
# .env.production
DATABASE_URL=postgresql://user:secure_password@postgres:5432/people_counter
CUDA_DEVICE=0
API_HOST=0.0.0.0
API_PORT=8000
MODEL_CONFIDENCE=0.25
TRACKER_TYPE=deepsort
CORS_ORIGINS=https://yourdomain.com,https://www.yourdomain.com
MAX_UPLOAD_SIZE=52428800
```

---

## Contributing

We welcome contributions! Please follow these guidelines:

### Development Workflow

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes**
   - Follow PEP 8 style guide for Python
   - Use ESLint for JavaScript
   - Add docstrings to all functions
4. **Test your changes**
   ```bash
   pytest tests/  # Run tests
   ```
5. **Commit with descriptive messages**
   ```bash
   git commit -m "Add: Feature description"
   ```
6. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```
7. **Create a Pull Request**

### Code Style

**Python:**
```python
def function_name(param1: str, param2: int) -> dict:
    """
    Brief description.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value
    """
    pass
```

**JavaScript:**
```javascript
/**
 * Brief description
 * @param {string} param1 - Description
 * @returns {Object} Description
 */
function functionName(param1) {
    // Implementation
}
```

### Testing

Add tests for new features:

```python
# tests/test_zone_logic.py
import pytest
from backend.ml.zone_logic import ZoneDetector

def test_point_in_polygon():
    detector = ZoneDetector([[0, 0], [100, 0], [100, 100], [0, 100]])
    assert detector.is_point_in_zone((50, 50)) == True
    assert detector.is_point_in_zone((150, 150)) == False
```

---

## License

MIT License

Copyright (c) 2024 People Counter Project

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

## Acknowledgments

- **Ultralytics** for the excellent YOLOv8 implementation
- **DeepSORT** tracking algorithm by Nicolai Wojke
- **FastAPI** framework by Sebastián Ramírez
- **React** library by Meta
- **PyTorch** framework by Meta AI
- The open-source computer vision community

---

## Support & Contact

### Getting Help

1. **Documentation:** Read this README thoroughly
2. **API Docs:** Check http://localhost:8000/docs for interactive API documentation
3. **GitHub Issues:** Create an issue with detailed description
4. **Discussions:** Use GitHub Discussions for questions

### Reporting Bugs

When reporting bugs, please include:
- OS and version
- Python version
- CUDA version (if applicable)
- Error messages and stack traces
- Steps to reproduce
- Screenshots (if UI-related)

### Feature Requests

We welcome feature requests! Please:
- Check existing issues first
- Describe the use case
- Explain expected behavior
- Provide examples if possible

---

## Roadmap

### Planned Features

- [ ] **Multi-camera aggregation:** Combined analytics across cameras
- [ ] **Real-time video streaming:** WebRTC support for live feeds
- [ ] **Advanced Re-ID:** Cross-camera person tracking
- [ ] **Heatmap generation:** Spatial density visualization
- [ ] **Anomaly detection:** Unusual crowd patterns
- [ ] **API authentication:** JWT token-based auth
- [ ] **User management:** Role-based access control
- [ ] **Export formats:** Support for JSON, XML, Parquet
- [ ] **Mobile app:** React Native companion app
- [ ] **Cloud deployment:** AWS/GCP/Azure templates

### Version History

- **v1.0.0** (2025-11-16): Initial release
  - Core features: Camera management, zone configuration, annotation, training
  - YOLOv8n + DeepSORT integration
  - React dashboard with analytics
  - Docker deployment support

---

**Built with ❤️ by the People Counter Team**

For more information, visit our [GitHub repository](#).
