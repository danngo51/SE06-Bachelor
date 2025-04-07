from services.prediction.PredictionService import PredictionService
from interfaces.PredictionServiceInterface import IPredictionService

def get_prediction_service() -> PredictionService:
    return PredictionService()