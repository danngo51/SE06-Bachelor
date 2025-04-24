from fastapi import APIRouter, Depends
from core.di_container import get_prediction_service
from interfaces.PredictionServiceInterface import IPredictionService

predict = APIRouter()

@predict.get("/status")
def get_status(service: IPredictionService = Depends(get_prediction_service)):
    return service.status()

@predict.get("/test")
def get_test(service: IPredictionService = Depends(get_prediction_service)):
    return service.test()

