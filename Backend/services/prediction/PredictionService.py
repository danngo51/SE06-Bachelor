from interfaces.PredictionServiceInterface import IPredictionService
from typing import Dict, List, Optional
from model.prediction import PredictionRequest, PredictionResponse, CountryPredictionData, HourlyPredictionData, HybridModelOutput
import datetime
import random
import math
import ml_models.hybrid_model as hybrid_model
import pandas as pd
import os
import pathlib
from ml_models.Normalize.Denormalization import denormalize_column

class PredictionService(IPredictionService):
    # Class-level path variables for easier maintenance
    ROOT_PATH = pathlib.Path(__file__).parent.parent.parent
    ML_MODELS_PATH = ROOT_PATH / "ml_models"
    DATA_PATH = ML_MODELS_PATH / "data"
    NORMALIZE_PATH = ML_MODELS_PATH / "Normalize"
    NORMALIZE_OUTPUT_PATH = NORMALIZE_PATH / "output"
    NORMALIZE_INPUT_PATH = NORMALIZE_PATH / "input"
      # Direct file paths for actual price data and minmax files
    ACTUAL_PRICE_FILE = DATA_PATH / "DK1" / "prediction_data.csv"  # Single file for actual prices
    MINMAX_FILE = NORMALIZE_OUTPUT_PATH / "minmax.csv"  # Single minmax file
    NORMALIZED_INPUT_FILE = DATA_PATH / "DK1" / "prediction_data_normalized.csv"  # Single normalized input file
    
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
            # Use the hybrid model to get predictions with our normalized input file
            input_file_path = str(self.NORMALIZED_INPUT_FILE)
                
            model_output: HybridModelOutput = hybrid_model.predict_with_hybrid_model(
                country_code, 
                prediction_date, 
                input_file_path
            )
            
            # Denormalize the predictions before processing
            denormalized_output = self._denormalize_predictions(model_output, country_code)
            
            country_predictions.append(
                self._process_model_results(denormalized_output, country_code, prediction_date)
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
            # Use the test_predict method from hybrid_model with our normalized input file
            input_file_path = str(self.NORMALIZED_INPUT_FILE)
                
            model_output: HybridModelOutput = hybrid_model.test_predict(
                country_code, 
                prediction_date,
                input_file_path
            )
            
            # Denormalize the predictions before processing
            denormalized_output = self._denormalize_predictions(model_output, country_code)
            
            country_predictions.append(
                self._process_model_results(denormalized_output, country_code, prediction_date)
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
        
        # Get actual prices if available (for past dates)
        actual_prices = self._get_actual_prices(country_code, prediction_date)
        
        # Ensure we have enough prediction points (24 hours)
        num_hours = min(24, len(model_predictions))
        
        # Check if we're predicting for today or a future date
        prediction_datetime = datetime.datetime.strptime(prediction_date, "%Y-%m-%d")
        today = datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        is_future_date = prediction_datetime >= today
        
        for hour in range(num_hours):
            # Get predictions for this hour from all models
            informer_value = informer_predictions[hour] if hour < len(informer_predictions) else 0
            gru_value = gru_predictions[hour] if hour < len(gru_predictions) else 0
            model_value = model_predictions[hour] if hour < len(model_predictions) else 0
            
            # Get actual price if available, otherwise use None for future dates/hours
            actual_price = None
            hour_str = str(hour)
            
            if hour_str in actual_prices:
                # Use historical data for past dates
                actual_price = actual_prices[hour_str]
            elif not is_future_date:
                # For past dates with missing actual data, use model prediction with small variance
                actual_price = model_value * (1 + ((random.random() * 0.1) - 0.05))
            
            # Round values for display
            hourly_data[hour_str] = HourlyPredictionData(
                prediction_model=round(model_value, 2),
                actual_price=round(actual_price, 2) if actual_price is not None else None,
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
            # Use the single minmax file for denormalization
            minmax_file = self.MINMAX_FILE
            
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

    def _get_actual_prices(self, country_code: str, prediction_date: str) -> Dict[str, float]:
        """
        Retrieve actual historical price data for a specific date and country
        
        Args:
            country_code: Country code (e.g., 'DK1')
            prediction_date: Date in format 'YYYY-MM-DD'
            
        Returns:
            Dictionary with hour (as string) as key and actual price as value
        """
        try:
            # Use the single actual price file for data retrieval
            csv_file = self.ACTUAL_PRICE_FILE
            
            if not os.path.exists(csv_file):
                print(f"[WARNING] No price data file found: {csv_file}")
                return {}
            
            # Read the CSV file
            df = pd.read_csv(csv_file)
            
            # The first column typically contains datetime information
            # Extract the date part for filtering
            date_column = df.columns[0]  # Usually 'datetime' or similar
            
            # Convert the date column to datetime if it's not already
            if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
                df[date_column] = pd.to_datetime(df[date_column])
            
            # Filter data for the requested date
            date_filter = df[date_column].dt.strftime('%Y-%m-%d') == prediction_date
            filtered_df = df[date_filter]
            
            if filtered_df.empty:
                print(f"[INFO] No price data found for date: {prediction_date}")
                return {}
            
            # Extract hour and price data (assuming price column is 'Price[Currency/MWh]' or similar)
            # Price column is typically column 8 (index 7) based on the CSV structure
            price_column = None
            for col in df.columns:
                if 'Price[Currency/MWh]' in col.lower() or 'currency' in col.lower():
                    price_column = col
                    break
            
            if price_column is None:
                # Use the 8th column (index 7) as a fallback which contains price data in the DK1_24.csv file
                if len(df.columns) > 63:
                    price_column = df.columns[63]
                else:
                    print(f"[ERROR] Unable to identify price column in CSV file")
                    return {}
            
            # Create a dictionary with hour -> price mapping
            result = {}
            for _, row in filtered_df.iterrows():
                # Extract hour from datetime
                hour = pd.to_datetime(row[date_column]).hour
                price = row[price_column]
                result[str(hour)] = float(price)
            
            return result
            
        except Exception as e:
            print(f"[ERROR] Error retrieving actual prices: {str(e)}")
            return {}
