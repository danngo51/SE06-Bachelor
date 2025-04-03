from fastapi import APIRouter
from Backend.api.api import api  # Import API submodule
from Backend.api.predict import predict  # Import Predict submodule


router = APIRouter()

# Register API routes from `api.py`
router.include_router(api.router, prefix="/api", tags=["API"])
router.include_router(predict.router , prefix="/predict", tags=["Predict"])
