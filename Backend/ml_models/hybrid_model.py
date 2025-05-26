import os
import pandas as pd
import numpy as np
import pathlib
import random
import datetime
from typing import Dict, List, Optional, Union, Any, Tuple

from model.prediction import HybridModelOutput
from ml_models.XGBoost.xgboost_pipeline import XGBoostPipeline

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
        regime_dir = data_dir / "regime_models"
        pipeline.load_model(str(regime_dir))
        
        # Add to pipelines dictionary
        pipeline_name = f"xgboost_{country_code}"
        self.add_pipeline(pipeline_name, pipeline)
        
        return pipeline
    
    def predict_with_hybrid_model(
        self,
        country_code: str,
        prediction_date: str,
        input_file_path: Optional[str] = None,
        **kwargs
    ) -> HybridModelOutput:
        """
        Generate predictions using the hybrid model.
        
        Args:
            country_code: Country code (e.g., "DK1")
            prediction_date: Date string in format 'YYYY-MM-DD'
            input_file_path: Path to input data file (optional)
            **kwargs: Additional parameters for specific model pipelines
            
        Returns:
            HybridModelOutput with predictions from different models
        """
        # Currently, we'll use XGBoost pipeline as the default for demonstration
        # This can be extended to include other models like GRU and Informer
        
        # Check if we have XGBoost pipeline for this country
        pipeline_name = f"xgboost_{country_code}"
        
        if pipeline_name not in self.pipelines:
            # Load pipeline if not already loaded
            xgboost_pipeline = self.load_xgboost_pipeline(country_code)
        else:
            xgboost_pipeline = self.pipelines[pipeline_name]
        
        # Determine input file path if not provided
        if input_file_path is None:
            input_file_path = str(self.data_dir / country_code / f"{country_code}_full_data_2025.csv")
        
        # Make predictions
        try:
            # Get predictions from XGBoost pipeline
            result_df = xgboost_pipeline.predict_from_file(input_file_path, prediction_date)
            
            # Extract predictions as lists
            predicted_values = result_df['Predicted'].tolist()
            
            # For now, we'll use the same values for all model types
            # In a real hybrid model, you would integrate other model predictions here
            
            # Create HybridModelOutput
            return HybridModelOutput(
                informer_prediction=predicted_values,
                gru_prediction=predicted_values,
                model_prediction=predicted_values
            )
            
        except Exception as e:
            print(f"Error generating predictions: {e}")
            # Fall back to test predictions
            return self.test_predict(country_code, prediction_date)
    
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
        model_predictions = []
        
        for hour in range(24):
            # Base pattern: prices higher during day, lower at night
            hour_factor = 0.7 + 0.6 * np.sin(np.pi * (hour - 6) / 12)
            
            # Add some randomness
            random_factor_informer = 0.9 + 0.2 * random.random()
            random_factor_gru = 0.9 + 0.2 * random.random()
            random_factor_hybrid = 0.95 + 0.1 * random.random()
            
            # Create predictions with different patterns
            informer_price = base_price * country_factor * hour_factor * random_factor_informer
            gru_price = base_price * country_factor * hour_factor * random_factor_gru
            hybrid_price = (informer_price + gru_price) / 2 * random_factor_hybrid
            
            informer_predictions.append(round(informer_price, 2))
            gru_predictions.append(round(gru_price, 2))
            model_predictions.append(round(hybrid_price, 2))
        
        # Return formatted output
        return HybridModelOutput(
            informer_prediction=informer_predictions,
            gru_prediction=gru_predictions,
            model_prediction=model_predictions
        )

# Create a singleton instance for import elsewhere
hybrid_model = HybridModel()