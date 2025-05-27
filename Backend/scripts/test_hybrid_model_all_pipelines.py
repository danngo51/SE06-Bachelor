"""
Test script for the hybrid model with all three pipeline types.
"""

import os
import sys
import pathlib
import pandas as pd
import argparse
from datetime import datetime
from typing import Dict, List, Optional

# Add project root to path
project_root = pathlib.Path(__file__).parent.parent  # Go up one level from scripts folder
sys.path.append(str(project_root))

from ml_models.Hybrid_Model import hybrid_model
from ml_models.XGBoost import XGBoostPipeline
from ml_models.GRU import GRUPipeline
from ml_models.Informer import InformerPipeline
from model.prediction import HybridModelOutput

def test_hybrid_model_with_all_pipelines(
    country_code: str = "DK1",
    date_str: Optional[str] = None,
    weights: Optional[Dict[str, float]] = None
) -> HybridModelOutput:
    """
    Test the hybrid model with all available pipeline types.
    
    Args:
        country_code: Country code to test
        date_str: Specific date to predict (YYYY-MM-DD)
        weights: Weights for each model type (optional)
        
    Returns:
        HybridModelOutput with predictions
    """
    # Set default date to today if not provided
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")
    
    # Set default weights if not provided
    if weights is None:
        weights = {
            "xgboost": 0.4,
            "gru": 0.3,
            "informer": 0.3
        }
    
    print(f"Testing hybrid model for {country_code} on {date_str}")
    print(f"Using weights: {weights}")
    
    # Load all model pipelines
    try:
        # Load XGBoost pipeline
        xgboost_pipeline = hybrid_model.load_xgboost_pipeline(country_code)
        print("✅ XGBoost pipeline loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load XGBoost pipeline: {e}")
    
    try:
        # Load GRU pipeline
        gru_pipeline = hybrid_model.load_gru_pipeline(country_code)
        print("✅ GRU pipeline loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load GRU pipeline: {e}")
    
    try:
        # Load Informer pipeline
        informer_pipeline = hybrid_model.load_informer_pipeline(country_code)
        print("✅ Informer pipeline loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load Informer pipeline: {e}")
    
    # Get predictions from hybrid model
    predictions = hybrid_model.predict_with_hybrid_model(
        country_code=country_code,
        prediction_date=date_str,
        weights=weights
    )
    
    # Print predictions
    print("\nPredictions:")
    for hour in range(min(24, len(predictions.model_prediction))):        print(f"Hour {hour:02d}: "
              f"XGBoost: {predictions.xgboost_prediction[hour]:.2f} €/MWh, "
              f"GRU: {predictions.gru_prediction[hour]:.2f} €/MWh, "
              f"Informer: {predictions.informer_prediction[hour]:.2f} €/MWh, "
              f"Ensemble: {predictions.model_prediction[hour]:.2f} €/MWh")
    
    return predictions

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the hybrid model with all pipelines.")
    parser.add_argument("--country", type=str, default="DK1", help="Country code to test")
    parser.add_argument("--date", type=str, help="Date to predict (YYYY-MM-DD)")
    parser.add_argument("--xgboost-weight", type=float, default=0.4, help="Weight for XGBoost model")
    parser.add_argument("--gru-weight", type=float, default=0.3, help="Weight for GRU model")
    parser.add_argument("--informer-weight", type=float, default=0.3, help="Weight for Informer model")
    
    args = parser.parse_args()
    
    # Set weights
    weights = {
        "xgboost": args.xgboost_weight,
        "gru": args.gru_weight,
        "informer": args.informer_weight
    }
    
    # Normalize weights
    weight_sum = sum(weights.values())
    for k in weights:
        weights[k] = weights[k] / weight_sum
    
    print("Testing Hybrid Model with All Pipeline Types")
    print("===========================================")
    
    # Run test
    test_hybrid_model_with_all_pipelines(args.country, args.date, weights)
