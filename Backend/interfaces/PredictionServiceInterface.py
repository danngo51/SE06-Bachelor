from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from models.prediction import PredictionRequest, PredictionResponse

class IPredictionService(ABC):
    @abstractmethod
    def status(self) -> Dict:
        pass
        
    @abstractmethod
    def predict(self, request: PredictionRequest) -> PredictionResponse:
        """
        Generate predictions using the hybrid model
        
        Args:
            request: PredictionRequest with country_codes and date
            test_mode: Whether to generate mock data instead of using the real model
            
        Returns:
            PredictionResponse with prediction data for all requested countries
        """
        pass

    @abstractmethod
    def predict_test(self, request: PredictionRequest) -> PredictionResponse:
        """
        Generate predictions using the hybrid model
        
        Args:
            request: PredictionRequest with country_codes and date
            test_mode: Whether to generate mock data instead of using the real model
            
        Returns:
            PredictionResponse with prediction data for all requested countries
        """
        pass