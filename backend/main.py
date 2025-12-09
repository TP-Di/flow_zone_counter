"""Main FastAPI application."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
import os

from .database import init_db
from .routers import (
    cameras_router,
    zones_router,
    inference_router,
    training_router,
    dashboard_router
)

# Create FastAPI app
app = FastAPI(
    title="People Counter API",
    description="Computer vision web application for people counting and tracking",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(cameras_router)
app.include_router(zones_router)
app.include_router(inference_router)
app.include_router(training_router)
app.include_router(dashboard_router)

# Mount static files for serving images
cameras_dir = Path("cameras")
cameras_dir.mkdir(exist_ok=True)
app.mount("/cameras", StaticFiles(directory=str(cameras_dir)), name="cameras")

# Mount frontend static files
frontend_build_dir = Path("frontend/build")
if frontend_build_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_build_dir / "static")), name="static")


@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    # Initialize database
    init_db()

    # Create required directories
    for directory in ["cameras", "models", "config", "temp"]:
        Path(directory).mkdir(exist_ok=True)

    print("People Counter API started successfully!")
    print("API Docs available at: http://localhost:8000/docs")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    print("People Counter API shutting down...")


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "database": "connected",
        "ml_backend": "available"
    }


@app.get("/api")
def api_root():
    """API root endpoint."""
    return {
        "message": "People Counter API",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "running"
    }


@app.get("/{full_path:path}")
async def serve_frontend(full_path: str):
    """Serve React frontend for all non-API routes."""
    frontend_build_dir = Path("frontend/build")

    # Check if frontend build exists
    if not frontend_build_dir.exists():
        return {
            "message": "People Counter API",
            "version": "1.0.0",
            "docs": "/docs",
            "status": "running",
            "note": "Frontend not available"
        }

    # Serve index.html for all routes (React Router will handle routing)
    index_file = frontend_build_dir / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))

    return {"error": "Frontend not found"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
