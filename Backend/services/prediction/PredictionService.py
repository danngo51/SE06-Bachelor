from interfaces.PredictionServiceInterface import IPredictionService
from typing import Dict


class PredictionService(IPredictionService):
    def status(self) -> Dict:
        return {"status": "Predict running"}
    

    
