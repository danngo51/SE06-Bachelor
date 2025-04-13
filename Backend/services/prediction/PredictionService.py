from interfaces.PredictionServiceInterface import IPredictionService
from typing import Dict

from ml_models.hybrid_model import test


class PredictionService(IPredictionService):
    def status(self) -> Dict:
        return {"status": "Predict running"}
    
    def test(self) -> Dict:
        result = test()
        return {"Models": result}
