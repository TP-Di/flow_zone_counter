"""API routers."""
from .cameras import router as cameras_router
from .zones import router as zones_router
from .inference import router as inference_router
from .training import router as training_router
from .dashboard import router as dashboard_router

__all__ = [
    'cameras_router',
    'zones_router',
    'inference_router',
    'training_router',
    'dashboard_router'
]
