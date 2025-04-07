from fastapi import APIRouter

api = APIRouter()

@api.get("/status")
def get_status():
    return {"status": "API is live!"}

