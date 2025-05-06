from interfaces.PredictionServiceInterface import IPredictionService
from typing import Dict, List, Optional
from models.prediction import PredictionRequest, PredictionResponse, CountryPredictionData, HourlyPredictionData, HybridModelOutput
import datetime
import random
import math
import ml_models.hybrid_model as hybrid_model
import pandas as pd
import os
import pathlib
from ml_models.Normalize.Denormalization import denormalize_column

class PredictionService(IPredictionService):
    def status(self) -> Dict:
        return {"status": "Predict running"}
    
    def predict(self, request: PredictionRequest) -> PredictionResponse:
        """
        Generate predictions using the hybrid model
        
        Args:
            request: PredictionRequest with country_codes and date
            
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
            # Use the hybrid model to get predictions
            model_output: HybridModelOutput = hybrid_model.predict_with_hybrid_model(country_code, prediction_date)
            country_predictions.append(
                self._process_model_results(model_output, country_code, prediction_date)
            )
        
        # Return structured response
        return PredictionResponse(predictions=country_predictions)
    

    def predict_test(self, request: PredictionRequest) -> PredictionResponse:
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
        # Extract all predictions from model output
        informer_predictions = model_output.informer_prediction
        gru_predictions = model_output.gru_prediction
        model_predictions = model_output.model_prediction
        
        # Create hourly data dictionary
        hourly_data = {}
        
        # Ensure we have enough prediction points (24 hours)
        num_hours = min(24, len(model_predictions))
        
        for hour in range(num_hours):
            # Get predictions for this hour from all models
            informer_value = informer_predictions[hour] if hour < len(informer_predictions) else 0
            gru_value = gru_predictions[hour] if hour < len(gru_predictions) else 0
            model_value = model_predictions[hour] if hour < len(model_predictions) else 0
            
            # For demo purposes, we generate a simulated "actual" price
            # In a real system, this might come from historical data or be None for future dates
            actual_price = model_value * (1 + ((random.random() * 0.1) - 0.05))
            
            # Store data for this hour with the model result and individual model predictions
            hourly_data[str(hour)] = HourlyPredictionData(
                prediction_model=model_value,
                actual_price=round(actual_price, 2),
                informer_prediction=round(informer_value, 2),
                gru_prediction=round(gru_value, 2),
                model_prediction=round(model_value, 2)
            )
        
        # Return formatted prediction data for this country
        return CountryPredictionData(
            hourly_data=hourly_data,
            country_code=country_code,
            prediction_date=prediction_date
        )
    
    def _denormalize_predictions(self, model_output: HybridModelOutput, country_code: str) -> HybridModelOutput:
        """
        Denormalizes prediction data in a HybridModelOutput object
        
        Args:
            model_output: HybridModelOutput with normalized prediction values
            country_code: Country code to help determine the correct minmax file
            
        Returns:
            HybridModelOutput with denormalized prediction values
        """
        try:
            # Find minmax file for the specific country or use a default one
            normalize_dir = pathlib.Path(__file__).parent.parent.parent / "ml_models" / "Normalize" / "output"
            
            # Try country-specific file first
            minmax_file = normalize_dir / f"minmax.csv"
            
            # Fall back to default minmax file if country-specific one doesn't exist
            if not os.path.exists(minmax_file):
                possible_files = [f for f in os.listdir(normalize_dir) if f.endswith('-minmax.csv')]
                if possible_files:
                    minmax_file = normalize_dir / possible_files[0]
                    print(f"[INFO] Using default minmax file: {minmax_file}")
                else:
                    print(f"[WARNING] No minmax file found for denormalization. Returning original predictions.")
                    return model_output
            
            # Convert prediction lists to DataFrame for denormalization
            df_informer = pd.DataFrame({"Price[Currency/MWh]": model_output.informer_prediction})
            df_gru = pd.DataFrame({"Price[Currency/MWh]": model_output.gru_prediction})
            df_model = pd.DataFrame({"Price[Currency/MWh]": model_output.model_prediction})
            
            # Denormalize each model's predictions
            denorm_informer = denormalize_column(df_informer, minmax_file, "Price[Currency/MWh]")
            denorm_gru = denormalize_column(df_gru, minmax_file, "Price[Currency/MWh]")
            denorm_model = denormalize_column(df_model, minmax_file, "Price[Currency/MWh]")
            
            # Create a new HybridModelOutput with denormalized predictions
            return HybridModelOutput(
                informer_prediction=denorm_informer["Price[Currency/MWh]"].tolist(),
                gru_prediction=denorm_gru["Price[Currency/MWh]"].tolist(),
                model_prediction=denorm_model["Price[Currency/MWh]"].tolist()
            )
            
        except Exception as e:
            print(f"[ERROR] Error during denormalization: {str(e)}")
            # Return original predictions if denormalization fails
            return model_output
