from fastapi import APIRouter
from api.routes.api import api as apiRouter  # Import API submodule
from api.routes.predict import predict as predictRouter  # Import Predict submodule


router = APIRouter()

# Register API routes from `api.py`
router.include_router(apiRouter, prefix="/api", tags=["API"])
router.include_router(predictRouter, prefix="/predict", tags=["Predict"])
