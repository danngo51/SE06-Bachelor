from interfaces.PredictionServiceInterface import IPredictionService
from typing import Dict, List, Optional
from models.prediction import PredictionRequest, PredictionResponse, CountryPredictionData, HourlyPredictionData, HybridModelOutput
import datetime
import random
import math
import ml_models.hybrid_model as hybrid_model

class PredictionService(IPredictionService):
    def status(self) -> Dict:
        return {"status": "Predict running"}
    
    def predict(self, request: PredictionRequest) -> PredictionResponse:
        """
        Generate predictions using the test method in hybrid model
        
        Args:
            request: PredictionRequest with country_codes and date
            test_mode: Parameter retained for interface compatibility but not used
            
        Returns:
            PredictionResponse with prediction data for all requested countries
        """
        # Use current date if none provided
        if not request.date:
            today = datetime.datetime.now()
            prediction_date = today.strftime("%Y-%m-%d")
        else:
            prediction_date = request.date
        
        # Create a list to hold data for each country
        country_predictions = []
        
        # For each requested country, get predictions
        for country_code in request.country_codes:
            # Use the test_predict method from hybrid_model
            model_output: HybridModelOutput = hybrid_model.test_predict(country_code, prediction_date)
            country_predictions.append(
                self._process_model_results(model_output, country_code, prediction_date)
            )
        
        # Return structured response
        return PredictionResponse(predictions=country_predictions)
    
    def _process_model_results(self, model_output: HybridModelOutput, country_code: str, prediction_date: str) -> CountryPredictionData:
        """Process model results into the expected format for a country"""
        # Extract predictions from model results - only use the model_prediction for the frontend
        model_predictions = model_output.model_prediction
        
        # Create hourly data dictionary
        hourly_data = {}
        
        # Ensure we have enough prediction points (24 hours)
        num_hours = min(24, len(model_predictions))
        
        for hour in range(num_hours):
            # Get model prediction for this hour
            model_value = model_predictions[hour] if hour < len(model_predictions) else 0
            
            # For demo purposes, we generate a simulated "actual" price
            # In a real system, this might come from historical data or be None for future dates
            actual_price = model_value * (1 + ((random.random() * 0.1) - 0.05))
            
            # Store data for this hour with the model result
            hourly_data[str(hour)] = HourlyPredictionData(
                prediction_model=model_value,
                actual_price=round(actual_price, 2)
            )
        
        # Return formatted prediction data for this country
        return CountryPredictionData(
            hourly_data=hourly_data,
            country_code=country_code,
            prediction_date=prediction_date
        )
