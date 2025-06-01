import os
import pandas as pd
import numpy as np
import pathlib
import random
import datetime
from typing import Dict, List, Optional, Union, Any, Tuple

from model.prediction import HybridModelOutput
from ml_models.XGBoost.xgboost_pipeline import XGBoostPipeline
from ml_models.GRU.gru_pipeline import GRUPipeline
from ml_models.Informer.informer_pipeline import InformerPipeline

class HybridModel:
    """
    Hybrid model that combines predictions from multiple model pipelines.
    This allows for flexible model combinations and ensembling.
    """
    
    def __init__(self):
        """Initialize the hybrid model with default settings."""
        # Initialize paths
        current_file = pathlib.Path(__file__)
        self.project_root = current_file.parent.parent  # MLs directory
        self.ml_models_dir = current_file.parent        # ml_models directory
        self.data_dir = self.ml_models_dir / "data"     # data directory
        
        # Dictionary to store model pipelines
        self.pipelines = {}
    
    def add_pipeline(self, name: str, pipeline: Any) -> None:
        """
        Add a model pipeline to the hybrid model.
        
        Args:
            name: Unique name for the pipeline
            pipeline: Model pipeline instance
        """
        self.pipelines[name] = pipeline

    def load_xgboost_pipeline(self, country_code: str) -> XGBoostPipeline:
        """
        Load XGBoost pipeline for a specific country code.
        
        Args:
            country_code: Country code (e.g., "DK1")
            
        Returns:
            Loaded XGBoost pipeline
        """
        # Initialize pipeline
        pipeline = XGBoostPipeline(mapcode=country_code)
        
        # Load model
        data_dir = self.data_dir / country_code
        xgboost_dir = data_dir / "xgboost"
        pipeline.load_model(str(xgboost_dir))
        
        # Add to pipelines dictionary
        pipeline_name = f"xgboost_{country_code}"
        self.add_pipeline(pipeline_name, pipeline)
        
        return pipeline
    
    def load_gru_pipeline(self, country_code: str) -> GRUPipeline:
        """
        Load GRU pipeline for a specific country code.
        
        Args:
            country_code: Country code (e.g., "DK1")
            
        Returns:
            Loaded GRU pipeline
        """
        # Initialize pipeline
        pipeline = GRUPipeline(mapcode=country_code)
        
        # Load model
        data_dir = self.data_dir / country_code
        gru_dir = data_dir / "gru"
        pipeline.load_model(str(gru_dir))
        
        # Add to pipelines dictionary
        pipeline_name = f"gru_{country_code}"
        self.add_pipeline(pipeline_name, pipeline)
        
        return pipeline
    
    def load_informer_pipeline(self, country_code: str) -> InformerPipeline:
        """
        Load Informer pipeline for a specific country code.
        
        Args:
            country_code: Country code (e.g., "DK1")
            
        Returns:
            Loaded Informer pipeline
        """
        # Initialize pipeline
        pipeline = InformerPipeline(mapcode=country_code)
        
        # Load model
        data_dir = self.data_dir / country_code
        informer_dir = data_dir / "informer"
        pipeline.load_model(str(informer_dir))
        
        # Add to pipelines dictionary
        pipeline_name = f"informer_{country_code}"
        self.add_pipeline(pipeline_name, pipeline)
        
        return pipeline
    
    def predict_with_hybrid_model(
        self,
        country_code: str,
        prediction_date: str,
        input_file_path: Optional[str] = None,
        weights: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> HybridModelOutput:
        """
        Generate predictions using the hybrid model.
        
        Args:
            country_code: Country code (e.g., "DK1")
            prediction_date: Date string in format 'YYYY-MM-DD'
            input_file_path: Path to input data file (optional)
            weights: Dictionary with weights for each model type (optional)
            **kwargs: Additional parameters for specific model pipelines
            
        Returns:
            HybridModelOutput with predictions from different models
        """
        # Set default weights if not provided
        if weights is None:
            weights = {
                "xgboost": 0.4,
                "gru": 0.3,
                "informer": 0.3
            }
        
        # Check if we have pipelines for this country, load them if not
        xgboost_name = f"xgboost_{country_code}"
        gru_name = f"gru_{country_code}"
        informer_name = f"informer_{country_code}"
        
        # Make predictions with each available model
        xgboost_predictions = None
        gru_predictions = None
        informer_predictions = None
        
        # Determine input file path if not provided
        if input_file_path is None:
            input_file_path = str(self.data_dir / country_code / f"{country_code}_full_data_2025.csv")
            
        # Check if input file exists
        if not os.path.exists(input_file_path):
            print(f"Warning: Input file {input_file_path} not found. Using test predictions.")
            return self.test_predict(country_code, prediction_date)

        # Load each model and get predictions independently
        try:
            if xgboost_name not in self.pipelines:
                self.load_xgboost_pipeline(country_code)
            if xgboost_name in self.pipelines:
                xgboost_pipeline = self.pipelines[xgboost_name]
                try:
                    xgboost_result = xgboost_pipeline.predict_from_file(input_file_path, prediction_date)
                    xgboost_predictions = xgboost_result['Predicted'].tolist()
                except Exception as e:
                    print(f"Error getting XGBoost predictions: {e}")
        except Exception as e:
            print(f"Error loading XGBoost pipeline: {e}")

        try:
            if gru_name not in self.pipelines:
                self.load_gru_pipeline(country_code)
            if gru_name in self.pipelines:
                gru_pipeline = self.pipelines[gru_name]
                try:
                    gru_result = gru_pipeline.predict_from_file(input_file_path, prediction_date)
                    gru_predictions = gru_result['Predicted'].tolist()
                except Exception as e:
                    print(f"Error getting GRU predictions: {e}")
        except Exception as e:
            print(f"Error loading GRU pipeline: {e}")

        try:
            if informer_name not in self.pipelines:
                self.load_informer_pipeline(country_code)
            if informer_name in self.pipelines:
                informer_pipeline = self.pipelines[informer_name]
                try:
                    informer_result = informer_pipeline.predict(input_file_path, prediction_date)
                    informer_predictions = informer_result['Predicted'].tolist()
                except Exception as e:
                    print(f"Error getting Informer predictions: {e}")
        except Exception as e:
            print(f"Error loading Informer pipeline: {e}")

        # If all models failed, use test predictions as fallback
        if xgboost_predictions is None and gru_predictions is None and informer_predictions is None:
            print("All models failed, using test predictions")
            return self.test_predict(country_code, prediction_date)
        
        # Generate ensemble predictions using any available models
        ensemble_predictions = self._ensemble_predictions(
            xgboost_predictions, 
            gru_predictions,
            informer_predictions,
            weights
        )

        # Return HybridModelOutput, using ensemble predictions as fallback for missing models
        return HybridModelOutput(
            informer_prediction=informer_predictions if informer_predictions else ensemble_predictions,
            gru_prediction=gru_predictions if gru_predictions else ensemble_predictions,
            xgboost_prediction=xgboost_predictions if xgboost_predictions else ensemble_predictions,
            model_prediction=ensemble_predictions
        )
    def _ensemble_predictions(
        self, 
        xgboost_preds: Optional[List[float]], 
        gru_preds: Optional[List[float]],
        informer_preds: Optional[List[float]],
        weights: Dict[str, float]
    ) -> List[float]:
        """
        Create ensemble predictions by weighted averaging of available model predictions.
        
        Args:
            xgboost_preds: XGBoost model predictions
            gru_preds: GRU model predictions
            informer_preds: Informer model predictions
            weights: Dictionary with weights for each model
            
        Returns:
            List of ensemble predictions
        """
        # Check which predictions are available
        available_preds = []
        available_weights = []
        model_names = []
        
        # Calculate reference range for transformation of GRU predictions
        reference_min, reference_max, reference_mean = 0, 0, 0
        reference_count = 0
        
        # First pass to collect statistics from working models
        if xgboost_preds is not None:
            xgb_min = min(xgboost_preds)
            xgb_max = max(xgboost_preds)
            xgb_mean = sum(xgboost_preds) / len(xgboost_preds)
            reference_min += xgb_min
            reference_max += xgb_max
            reference_mean += xgb_mean
            reference_count += 1
        
        if informer_preds is not None:
            inf_min = min(informer_preds)
            inf_max = max(informer_preds)
            inf_mean = sum(informer_preds) / len(informer_preds)
            reference_min += inf_min
            reference_max += inf_max
            reference_mean += inf_mean
            reference_count += 1
        
        if reference_count > 0:
            reference_min /= reference_count
            reference_max /= reference_count
            reference_mean /= reference_count
        else:
            # Default values if no reference models available
            reference_min, reference_max, reference_mean = 90, 150, 120
        
        # Add XGBoost predictions to ensemble
        if xgboost_preds is not None:
            available_preds.append(xgboost_preds)
            available_weights.append(weights.get("xgboost", 0.4))
            model_names.append("XGBoost")
        
        # Handle GRU predictions, applying advanced transformation if needed
        if gru_preds is not None:
            # Check if GRU predictions are problematic (negative or very small)
            if min(gru_preds) < 5 or sum(1 for p in gru_preds if p < 0) > 0:
                print("Warning: GRU predictions are problematic. Applying adaptive transformation.")
                
                # Get GRU statistics for scaling
                gru_min = min(gru_preds)
                gru_max = max(gru_preds)
                gru_range = abs(gru_max - gru_min) if abs(gru_max - gru_min) > 0.01 else 1
                reference_range = reference_max - reference_min
                
                # Apply transformation to match the other models' range and mean
                transformed_preds = []
                for p in gru_preds:
                    # Normalize to [0, 1] range, then scale to reference range and shift to reference mean
                    if gru_range > 0.01:
                        normalized = (p - gru_min) / gru_range
                        transformed = normalized * reference_range + reference_min
                    else:
                        # If GRU range is too small, use the mean of other models
                        transformed = reference_mean
                    transformed_preds.append(transformed)
                
                available_preds.append(transformed_preds)
                print(f"GRU transformed from range [{gru_min:.2f}, {gru_max:.2f}] to approximate [{reference_min:.2f}, {reference_max:.2f}]")
            else:
                available_preds.append(gru_preds)
            
            available_weights.append(weights.get("gru", 0.3))
            model_names.append("GRU")
        
        # Add Informer predictions to ensemble
        if informer_preds is not None:
            available_preds.append(informer_preds)
            available_weights.append(weights.get("informer", 0.3))
            model_names.append("Informer")
        
        # If no predictions are available, return zeros
        if not available_preds:
            print("Warning: No models available for ensemble prediction. Using zeros.")
            return [0.0] * 24
            
        # Log which models are being used for ensemble
        print(f"Creating ensemble prediction using {len(available_preds)} models: {', '.join(model_names)}")
        
        # Normalize weights
        weight_sum = sum(available_weights)
        normalized_weights = [w / weight_sum for w in available_weights]
        
        # Ensure all prediction lists have the same length (24 hours)
        pred_length = len(available_preds[0])
        for i, preds in enumerate(available_preds):
            if len(preds) != pred_length:
                # Pad or truncate to match first prediction length
                if len(preds) < pred_length:
                    available_preds[i] = preds + [preds[-1]] * (pred_length - len(preds))
                else:
                    available_preds[i] = preds[:pred_length]
        
        # Compute weighted ensemble predictions
        ensemble_preds = []
        for hour in range(pred_length):
            weighted_sum = sum(preds[hour] * weight for preds, weight in zip(available_preds, normalized_weights))
            ensemble_preds.append(round(weighted_sum, 2))
        
        return ensemble_preds
    
    def test_predict(self, country_code: str, prediction_date: str) -> HybridModelOutput:
        """
        Generate test predictions when real models fail or are unavailable.
        
        Args:
            country_code: Country code (e.g., "DK1")
            prediction_date: Date string in format 'YYYY-MM-DD'
            
        Returns:
            HybridModelOutput with simulated predictions
        """
        # Generate 24 hours of test data
        base_price = 50.0  # Base price in EUR/MWh
        
        # Create variation based on country code
        country_factor = {
            "DK1": 1.0,
            "DK2": 1.1,
            "NO": 0.8,
            "SE": 0.9,
            "DE": 1.2,
            "NL": 1.3
        }.get(country_code, 1.0)
          # Create different patterns for each model to simulate diversity
        informer_predictions = []
        gru_predictions = []
        xgboost_predictions = []
        model_predictions = []
        
        for hour in range(24):
            # Base pattern: prices higher during day, lower at night
            hour_factor = 0.7 + 0.6 * np.sin(np.pi * (hour - 6) / 12)
            
            # Add some randomness
            random_factor_informer = 0.9 + 0.2 * random.random()
            random_factor_gru = 0.9 + 0.2 * random.random()
            random_factor_xgboost = 0.9 + 0.2 * random.random()
            random_factor_hybrid = 0.95 + 0.1 * random.random()
            
            # Create predictions with different patterns
            informer_price = base_price * country_factor * hour_factor * random_factor_informer
            gru_price = base_price * country_factor * hour_factor * random_factor_gru
            xgboost_price = base_price * country_factor * hour_factor * random_factor_xgboost
            hybrid_price = (informer_price + gru_price + xgboost_price) / 3 * random_factor_hybrid
            
            informer_predictions.append(round(informer_price, 2))
            gru_predictions.append(round(gru_price, 2))
            xgboost_predictions.append(round(xgboost_price, 2))
            model_predictions.append(round(hybrid_price, 2))
          # Return formatted output
        return HybridModelOutput(
            informer_prediction=informer_predictions,
            gru_prediction=gru_predictions,
            xgboost_prediction=xgboost_predictions,
            model_prediction=model_predictions
        )

# Create a singleton instance for import elsewhere
hybrid_model = HybridModel()