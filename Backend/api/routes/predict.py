from fastapi import APIRouter, Depends, Query
from typing import List, Optional
from core.di_container import get_prediction_service
from interfaces.PredictionServiceInterface import IPredictionService
from model.prediction import PredictionRequest, FrontendPredictionResponse

predict = APIRouter()

@predict.get("/status")
def get_status(service: IPredictionService = Depends(get_prediction_service)):
    return service.status()

@predict.post("/predict", response_model=FrontendPredictionResponse)
def predict_endpoint(
    request: PredictionRequest,
    service: IPredictionService = Depends(get_prediction_service)
):
    # Get the prediction using the internal format
    internal_response = service.predict(request)
    
    # Convert to frontend format and return
    return internal_response.to_frontend_format()

