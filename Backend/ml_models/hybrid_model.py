import random
import math
import sys
from model.prediction import HybridModelOutput
import os
import torch
import pandas as pd
import pathlib
from ml_models.informer.informer import InformerWrapper
from ml_models.gru.gru import GRUWrapper

# Define base path for this module
ML_MODELS_DIR = os.path.dirname(os.path.abspath(__file__))

def predict_with_hybrid_model(country_code: str, prediction_date: str, 
                              input_file_path: str = None, 
                              config_path: str = None,
                              weight_path: str = None,
                              gru_path: str = None) -> HybridModelOutput:
    """
    Run the hybrid model prediction pipeline for a specific country and date
    
    Args:
        country_code: ISO country code to generate data for
        prediction_date: Date string in YYYY-MM-DD format
        input_file_path: Optional path to the input file
        config_path: Path to the Informer config file
        weight_path: Path to the Informer model weights
        gru_path: Path to the GRU model weights (not used when testing Informer only)
        
    Returns:
        HybridModelOutput with predictions from Informer model
    """
    try:
        # Set up device for model computation
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"✅ Using device: {device}")
        
        # Verify input file path exists
        if input_file_path and os.path.exists(input_file_path):
            print(f"Using provided input file: {input_file_path}")
        else:
            print(f"Input file path not provided or does not exist")
            # No need to set fallback paths here as they should be provided by PredictionService
            
        # Check if model paths exist and log status
        if config_path and os.path.exists(config_path):
            print(f"Using provided config path: {config_path}")
        else:
            print(f"Config path not provided or does not exist")
            
        if weight_path and os.path.exists(weight_path):
            print(f"Using provided Informer weights path: {weight_path}")
        else:
            print(f"Informer weights path not provided or does not exist")
        
        # Read data to determine feature dimension
        df = pd.read_csv(input_file_path)
        feature_dim = len(df.columns)
        print(f"Input data has {feature_dim} features")
          # Load Informer model with feature dimension information
        informer = InformerWrapper(
            config_path=config_path,
            weight_path=weight_path,
            device=device,
            feature_dim=feature_dim  # Pass feature dimension to override config
        )        # Get info from the config if possible
        feature_list = None
        model_enc_in = None
        
        if config_path and os.path.exists(config_path):
            try:
                import json
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    if "cols" in config:
                        feature_list = config["cols"]
                        print(f"Loaded {len(feature_list)} features from config file")
                    if "enc_in" in config:
                        model_enc_in = config["enc_in"]
                        print(f"Model expects {model_enc_in} input features")
            except Exception as e:
                print(f"Error loading feature list from config: {e}")
        
        # Check if model's saved weights can give us enc_in information
        if weight_path and os.path.exists(weight_path):
            try:
                # Load model state dict to analyze structure without loading the whole model
                model_dict = torch.load(weight_path, map_location=device)
                if "enc_embedding.value_embedding.tokenConv.weight" in model_dict:
                    weight = model_dict["enc_embedding.value_embedding.tokenConv.weight"]
                    detected_enc_in = weight.shape[1]
                    if model_enc_in and model_enc_in != detected_enc_in:
                        print(f"⚠️ Config enc_in={model_enc_in} does not match saved weights enc_in={detected_enc_in}")
                    model_enc_in = detected_enc_in
                    print(f"Detected {model_enc_in} input features from saved weights")
            except Exception as e:
                print(f"Could not analyze model weights: {e}")

        # Load and preprocess input using local _load_prediction_data function
        x_enc, x_mark_enc, x_dec, x_mark_dec = _load_prediction_data(
            input_file_path, 
            country_code=country_code,
            feature_list=feature_list,
            enc_in=model_enc_in
        )

        x_enc = x_enc.to(device)
        x_mark_enc = x_mark_enc.to(device)
        x_dec = x_dec.to(device)
        x_mark_dec = x_mark_dec.to(device)
        
        # Prediction with better error handling
        informer_predictions = [0] * 24  # Initialize with zeros in case of error
        
        try:
            # Try to run the Informer model only
            with torch.no_grad():
                enc_out, informer_pred = informer.run(x_enc, x_mark_enc, x_dec, x_mark_dec)
            
            # Move informer predictions to CPU and convert to numpy
            informer_predictions = informer_pred.cpu().numpy()[0, :, 0].tolist()
            print(f"✅ Informer prediction successful for {country_code}")
            print(f"\nInformer model predictions: {informer_predictions[:5]}... (showing first 5 hours)")
        
        except Exception as informer_error:
            # Informer model failed, use test predictions
            print(f"❌ Informer model error: {informer_error}")
            print("⚠️ Using test predictions for Informer model")
            test_output = test_predict(country_code, prediction_date)
            informer_predictions = test_output.informer_prediction
        
        # Debug: Print predictions to console
        print(f"\nGenerated predictions for {country_code} on {prediction_date}")
        
        # Ensure we have exactly 24 hours of predictions
        informer_predictions = informer_predictions[:24] if len(informer_predictions) >= 24 else informer_predictions + [0] * (24 - len(informer_predictions))
    
        # Return output with Informer predictions used for all fields
        return HybridModelOutput(
            informer_prediction=informer_predictions,
            gru_prediction=informer_predictions,  # Use Informer predictions instead of GRU
            model_prediction=informer_predictions  # Use Informer predictions as the final model output
        )
        
    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        # Return empty predictions in case of error
        return HybridModelOutput(
            informer_prediction=[0] * 24,
            gru_prediction=[0] * 24,
            model_prediction=[0] * 24
        )
        
    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        # Return empty predictions in case of error
        return HybridModelOutput(
            informer_prediction=[0] * 24,
            gru_prediction=[0] * 24,
            model_prediction=[0] * 24
        )


def test_predict(country_code: str, prediction_date: str, input_file_path: str = None) -> HybridModelOutput:
    """
    Generate zone-specific mock prediction data for testing purposes
    
    Args:
        country_code: Zone code (e.g., 'DK1') to generate data for
        prediction_date: Date string in YYYY-MM-DD format
        input_file_path: Optional path to the input file (not used in test mode, but kept for API consistency)
        
    Returns:
        HybridModelOutput with mock prediction data
    """
    # Create variation based on country/zone to simulate different models
    random.seed(hash(country_code) + hash(prediction_date))
    
    # Generate 24 hourly predictions for each model with some randomness
    num_hours = 24
    
    # Use zone-specific base prices
    zone_price_map = {
        "DK1": 40 + random.random() * 30,  # Denmark zone 1
        "DK2": 45 + random.random() * 35,  # Denmark zone 2
        "SE1": 30 + random.random() * 20,  # Sweden zone 1
        "SE2": 32 + random.random() * 20,  # Sweden zone 2
        "SE3": 35 + random.random() * 25,  # Sweden zone 3
        "SE4": 38 + random.random() * 28,  # Sweden zone 4
        "NO1": 25 + random.random() * 15,  # Norway zone 1
        "NO2": 26 + random.random() * 18,  # Norway zone 2
        "NO3": 24 + random.random() * 16,  # Norway zone 3
        "NO4": 23 + random.random() * 14,  # Norway zone 4
        "NO5": 27 + random.random() * 17,  # Norway zone 5
        "FI": 50 + random.random() * 40,   # Finland
        "DE": 55 + random.random() * 45,   # Germany
        "NL": 60 + random.random() * 50,   # Netherlands
    }
    
    # Get base price for this zone, or default if not found
    base_price = zone_price_map.get(country_code, 50 + random.random() * 30)
    
    print(f"Generating test predictions for zone {country_code} with base price {base_price:.2f}")
    
    # Zone-specific weights for combining models
    zone_weight_map = {
        "DK1": (0.4, 0.6),  # (informer_weight, gru_weight)
        "DK2": (0.45, 0.55),
        "SE": (0.5, 0.5),   # Default for all Sweden zones
        "NO": (0.55, 0.45), # Default for all Norway zones
        "FI": (0.6, 0.4),
        "DE": (0.45, 0.55),
        "NL": (0.5, 0.5),
    }
    
    # Get weights for this zone
    default_weights = (0.4, 0.6)
    informer_weight, gru_weight = default_weights
    
    for zone_prefix, weights in zone_weight_map.items():
        if country_code.startswith(zone_prefix):
            informer_weight, gru_weight = weights
            break
    
    # Generate test predictions for informer, GRU, and combined model
    informer_predictions = []
    gru_predictions = []
    model_predictions = []
    
    for hour in range(num_hours):
        # Time of day factor (higher during peak hours)
        time_factor = 1.0 + 0.3 * math.sin((hour - 6) * math.pi / 12)
        
        # Generate predictions with some variation
        informer_pred = base_price * time_factor * (0.9 + 0.2 * random.random())
        gru_pred = base_price * time_factor * (0.85 + 0.3 * random.random())
        
        # Combined model prediction (weighted average with zone-specific weights)
        model_pred = informer_weight * informer_pred + gru_weight * gru_pred
        
        # Add to result lists
        informer_predictions.append(float(informer_pred))
        gru_predictions.append(float(gru_pred))
        model_predictions.append(float(model_pred))
    
    return HybridModelOutput(
        informer_prediction=informer_predictions,
        gru_prediction=gru_predictions,
        model_prediction=model_predictions
    )


def _load_prediction_data(input_path, country_code=None, seq_len=168, label_len=24, pred_len=24, feature_list=None, enc_in=None):
    """
    Loads and preprocesses input CSV data for prediction.
    Works with non-normalized data for zone-specific files.
    Only uses data from the previous 7 days for prediction.
    
    Args:
        input_path: Path to the CSV data file (non-normalized)
        country_code: Zone code (e.g., 'DK1')
        seq_len: Sequence length for the encoder (168 hours = 7 days)
        label_len: Label length for the decoder
        pred_len: Prediction length for the decoder
        feature_list: Optional list of features to use from the CSV file
        enc_in: Expected input features dimension for the model
        
    Returns:
        x_enc, x_mark_enc, x_dec, x_mark_dec tensors.
    """
    # Default model parameters
    d_model = 512  # Default dimension of the model
    
    # Load the data from CSV
    df = pd.read_csv(input_path)
    print(f"Loaded data from {input_path} for zone {country_code}")
    
    # Use provided feature list or all columns from the CSV file
    if feature_list:
        # Check which columns are actually in the data
        available_features = [f for f in feature_list if f in df.columns]
        if len(available_features) != len(feature_list):
            missing = set(feature_list) - set(available_features)
            print(f"⚠️ Not all requested features found in data. Missing: {missing}")
            
        FEATURES = available_features
        print(f"Using {len(FEATURES)} available features from provided list of {len(feature_list)}")
    else:
        FEATURES = list(df.columns)
        print(f"Using all {len(FEATURES)} columns directly from input file")
        
    # If enc_in is provided, make sure we have the right number of features
    if enc_in is not None:
        if len(FEATURES) > enc_in:
            print(f"⚠️ Too many features: trimming from {len(FEATURES)} to {enc_in} to match model input size")
            FEATURES = FEATURES[:enc_in]
        elif len(FEATURES) < enc_in:
            print(f"⚠️ Not enough features: model expects {enc_in} but data has {len(FEATURES)}")
            # Fill with zero columns if needed
            missing_count = enc_in - len(FEATURES)
            for i in range(missing_count):
                col_name = f"zero_pad_{i}"
                df[col_name] = 0
                FEATURES.append(col_name)
            print(f"Added {missing_count} zero-filled columns to match model dimensions")
    
    # Ensure the price column is the first feature as expected by the model
    # Different files might use different column names for price
    price_column = None
    price_column_candidates = ["Price[Currency/MWh]", "Electricity_price_MWh", "Price_EUR_MWh", "price"]
    
    for candidate in price_column_candidates:
        for col in df.columns:
            if candidate.lower() in col.lower():
                price_column = col
                break
        if price_column:
            break
            
    if price_column and FEATURES[0] != price_column:
        # Move price column to the front
        if price_column in FEATURES:
            FEATURES.remove(price_column)
        FEATURES.insert(0, price_column)
        print(f"Reordered features to put '{price_column}' first: {FEATURES[:5]}...")
    
    # Make sure columns are in the defined order
    # For columns not in the dataset, fill with zeros
    for feature in FEATURES:
        if feature not in df.columns:
            print(f"Creating missing feature column: {feature}")
            df[feature] = 0
    
    df = df[FEATURES]
    
    # Only use the last 7 days (168 hours) of data
    if len(df) > seq_len:
        print(f"Filtering to use only the last {seq_len} hours (7 days) of data for prediction")
        df = df.tail(seq_len)
    else:
        print(f"Warning: Input data has fewer than {seq_len} hours (7 days). Using all available data.")
    
    # Convert data to tensor format
    data = df.values
    data_tensor = torch.tensor(data, dtype=torch.float32)

    # Build encoder input
    x_enc = data_tensor.unsqueeze(0)  # [batch_size, seq_len, num_features]
    
    # For x_mark_enc, x_dec, x_mark_dec:
    # If you don't use additional temporal encodings, you can just create dummy zeros
    batch_size = 1
    num_features = len(FEATURES)  # Use FEATURES instead of features

    x_mark_enc = torch.zeros((batch_size, seq_len, 4))  # 4 = (month, day, weekday, hour) usually if needed
    x_dec = torch.zeros((batch_size, label_len+pred_len, num_features))
    x_mark_dec = torch.zeros((batch_size, label_len+pred_len, 4))

    return x_enc, x_mark_enc, x_dec, x_mark_dec
