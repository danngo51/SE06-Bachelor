"""
Test script for the hybrid model and pipeline structure.
"""

import contextlib
from io import StringIO
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

# Import hybrid model and prediction service
from ml_models.Hybrid_Model import hybrid_model
from services.prediction.PredictionService import PredictionService
from model.prediction import PredictionRequest


def test_hybrid_model_direct(
    country_code: str = "DK1",
    prediction_date: Optional[str] = None,
    weights: Optional[Dict[str, float]] = None
):
    """
    Test direct prediction with the hybrid model.

    Args:
        country_code: Country code to test
        prediction_date: Specific date to predict (YYYY-MM-DD)
        weights: Model weights dictionary
    """
    # Set default date to today if not provided
    if prediction_date is None:
        prediction_date = datetime.now().strftime("%Y-%m-%d")

    # Set default weights if not provided
    if weights is None:
        weights = {
            "xgboost": 0.4,
            "gru": 0.3,
            "informer": 0.3
        }

    print(f"\nTesting direct hybrid model prediction for {country_code} on {prediction_date}...")
    print(f"Using weights: {weights}")

    try:
        # Get predictions
        model_output = hybrid_model.predict_with_hybrid_model(
            country_code=country_code,
            prediction_date=prediction_date,
            weights=weights
        )

        # Print results
        print("\nPredictions:")
        for hour in range(min(24, len(model_output.model_prediction))):
            print(f"Hour {hour:02d}: "
                  f"XGBoost: {model_output.xgboost_prediction[hour]:.2f} €/MWh, "
                  f"GRU: {model_output.gru_prediction[hour]:.2f} €/MWh, "
                  f"Informer: {model_output.informer_prediction[hour]:.2f} €/MWh, "
                  f"Ensemble: {model_output.model_prediction[hour]:.2f} €/MWh")

        print("Prediction successful!")
    except Exception as e:
        print(f"Error in direct prediction: {e}")


def test_prediction_service(
    country_codes: List[str] = ["DK1"],
    prediction_date: Optional[str] = "2025-03-01",
    weights: Optional[Dict[str, float]] = None
):
    """
    Test prediction through the service layer.

    Args:
        country_codes: List of country codes to test
        prediction_date: Specific date to predict (YYYY-MM-DD)
        weights: Model weights dictionary
    """
    # Set default date to today if not provided
    if prediction_date is None:
        prediction_date = "2025-03-01"  # Default date for testing
    
    # Set default weights if not provided
    if weights is None:
        weights = {
            "xgboost": 0.6,
            "gru": 0.4,
            "informer": 0.0
        }

    print(f"\nTesting prediction service for {', '.join(country_codes)} on {prediction_date}...")
    print(f"Using weights: {weights}")
    
    try:
        # Create request with weights included
        request = PredictionRequest(
            country_codes=country_codes,
            date=prediction_date,
            weights=weights
        )
        
        # Create service and get predictions
        service = PredictionService()
        response = service.predict(request)

        # Print results
        print(f"Received predictions for {len(response.predictions)} countries")
        
        for country_data in response.predictions:
            print(f"\nCountry: {country_data.country_code}")
            print(f"Date: {country_data.prediction_date}")
            print(f"Hours with predictions: {len(country_data.hourly_data)}")
            for hour in range(min(24, len(country_data.hourly_data))):
                hour_str = str(hour)
                if hour_str in country_data.hourly_data:
                    hour_data = country_data.hourly_data[hour_str]
                    print(f"  Hour {hour}: "
                          f"GRU={hour_data.gru:.2f}, "
                          f"XGBoost={hour_data.xgboost:.2f}, "
                          f"Model={hour_data.model:.2f}, "
                          f"Actual={hour_data.actual_price if hour_data.actual_price else 'N/A'}")
              # Model performance comparison vs actual
            print(f"\nModel Performance vs Actual for {country_data.country_code}:")
            for hour in range(min(24, len(country_data.hourly_data))):
                hour_str = str(hour)
                if hour_str in country_data.hourly_data:
                    hour_data = country_data.hourly_data[hour_str]
                    if hour_data.actual_price is not None:
                        actual = hour_data.actual_price
                        
                        # Calculate differences and percentages
                        gru_diff = hour_data.gru - actual
                        xgboost_diff = hour_data.xgboost - actual
                        model_diff = hour_data.model - actual
                        
                        gru_pct = (gru_diff / actual) * 100 if actual != 0 else 0
                        xgboost_pct = (xgboost_diff / actual) * 100 if actual != 0 else 0
                        model_pct = (model_diff / actual) * 100 if actual != 0 else 0
                        
                        # Format with + for positive, - for negative
                        def format_diff_pct(diff, pct):
                            diff_sign = "+" if diff >= 0 else ""
                            pct_sign = "+" if pct >= 0 else ""
                            return f"{diff_sign}{diff:.1f}/{pct_sign}{pct:.1f}%"
                        
                        print(f"  Hour {hour}: "
                              f"GRU: {format_diff_pct(gru_diff, gru_pct)}, "
                              f"XGBoost: {format_diff_pct(xgboost_diff, xgboost_pct)}, "
                              f"Model: {format_diff_pct(model_diff, model_pct)}")
                    else:
                        print(f"  Hour {hour}: No actual data available for comparison")

        print("Prediction service test successful!")
    except Exception as e:
        print(f"Error in prediction service: {e}")

def run_with_optional_logging(action_fn, args, label: str = "test"):
    """
    Runs the given action function with optional output logging.

    Args:
        action_fn: Function to run (no arguments).
        args: Parsed argparse arguments.
        label: Descriptive label for filename generation.
    """
    results_dir = pathlib.Path(__file__).parent / "test-results"
    results_dir.mkdir(exist_ok=True)

    buffer = StringIO()
    writer = contextlib.redirect_stdout(buffer) if args.print else contextlib.nullcontext()

    with writer:
        action_fn()

    if args.print:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        country_str = "_".join(args.countries)
        date_str = args.date if args.date else "today"
        filename = f"{timestamp}_{label}_{country_str}_{date_str}.txt"
        file_path = results_dir / filename
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(buffer.getvalue())
        print(f"\n📄 Output saved to {file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the hybrid price prediction model.")
    parser.add_argument("--countries", "-c", type=str, nargs="+", default=["DK1"], 
                        help="One or more country codes to test")
    parser.add_argument("--date", "-d", type=str, 
                        help="Date to predict (YYYY-MM-DD)")
    parser.add_argument("--xgboost-weight", "-x", type=float, default=0.6, 
                        help="Weight for XGBoost model")
    parser.add_argument("--gru-weight", "-g", type=float, default=0.4, 
                        help="Weight for GRU model")
    parser.add_argument("--informer-weight", "-i", type=float, default=0.0, 
                        help="Weight for Informer model")
    parser.add_argument("--hybrid", "-y", action="store_true", 
                        help="Use direct hybrid interface instead of service layer")
    parser.add_argument("--print", "-p", action="store_true", 
                        help="Save printed output to a text file in test-results")

    args = parser.parse_args()

    # Normalize weights
    weights = {
        "xgboost": args.xgboost_weight,
        "gru": args.gru_weight,
        "informer": args.informer_weight
    }
    total_weight = sum(weights.values())
    for key in weights:
        weights[key] /= total_weight

    # Define what to run inside the logging wrapper
    def run_test():
        print("Testing Hybrid Model")
        print("===================")
        
        if args.hybrid:
            test_hybrid_model_direct(args.countries[0], args.date, weights)
            if len(args.countries) > 1:
                print("\nNote: When using direct pipeline interface, only the first country is tested.")
        else:
            print(f"Testing prediction service for countries: {', '.join(args.countries)}")
            test_prediction_service(args.countries, args.date, weights)
        
        print("\nUsage examples:")
        print("  python -m scripts.test_hybrid_model -c DK1 DK2 -d 2024-05-24")
        print("  python -m scripts.test_hybrid_model --countries DK1 SE1 --date 2024-05-24 --hybrid")
        print("  python -m scripts.test_hybrid_model -c DK1 -x 0.5 -g 0.25 -i 0.25 -y")
        print("  python -m scripts.test_hybrid_model --country DK1 -xgboost-weight 0.5 -gru-weight 0.25 -informer-weight 0.25 -hybrid")

    # Run with or without output logging
    run_with_optional_logging(run_test, args, label="hybrid-test")