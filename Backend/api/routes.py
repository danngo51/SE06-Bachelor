from fastapi import APIRouter
from api import api  # Import API submodule

router = APIRouter()

# Register API routes from `api.py`
router.include_router(api.router, prefix="/api", tags=["API"])
