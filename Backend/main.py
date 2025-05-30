from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from core.config import settings
from api.router import router


# FastAPI run command:
# uvicorn main:app --host 0.0.0.0 --port 8000 --reload


app = FastAPI(**settings.model_dump())

# Enable CORS (for frontend communication)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router)

@app.get("/")
def home():
    return {"message": "FastAPI Backend is Running!"}
