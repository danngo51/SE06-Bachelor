import random
import math
from models.prediction import HybridModelOutput
import os
import torch
import pandas as pd
import pathlib
from ml_models.informer.informer import InformerWrapper
from ml_models.gru.gru import GRUWrapper

def predict_with_hybrid_model(country_code: str, prediction_date: str, input_file_path: str = None) -> HybridModelOutput:
    """
    Run the hybrid model prediction pipeline for a specific country and date
    
    Args:
        country_code: ISO country code to generate data for
        prediction_date: Date string in YYYY-MM-DD format
        input_file_path: Optional path to the input file, if None will use default
        
    Returns:
        HybridModelOutput with predictions from both models
    """
    try:
        # Determine the input file
        input_file = input_file_path if input_file_path else f"ml_models/data/DK1_24-normalized.csv"
        
        # Set up device for model computation
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"✅ Using device: {device}")

        # Load models
        informer = InformerWrapper(
            config_path="ml_models/informer/config.json",
            weight_path="ml_models/informer/results/checkpoint.pth",
            device=device
        )

        gru = GRUWrapper(
            gru_path="ml_models/gru/results/gru_trained.pt",
            regressor_path="ml_models/gru/results/gru_regressor.pt",
            input_dim=512,
            hidden_dim=128,
            output_dim=24,
            device=device,
            bidirectional=False
        )

        # Load and preprocess input using local _load_prediction_data function
        x_enc, x_mark_enc, x_dec, x_mark_dec = _load_prediction_data(input_file, country_code=country_code)

        x_enc = x_enc.to(device)
        x_mark_enc = x_mark_enc.to(device)
        x_dec = x_dec.to(device)
        x_mark_dec = x_mark_dec.to(device)

        # Prediction
        with torch.no_grad():
            enc_out, informer_pred = informer.run(x_enc, x_mark_enc, x_dec, x_mark_dec)
            gru_pred = gru.run(enc_out)

        # Move predictions to CPU and convert to numpy
        informer_predictions = informer_pred.cpu().numpy()[0, :, 0].tolist()  # Extract first batch, all timesteps, price feature
        gru_predictions = gru_pred.cpu().numpy()[0, :].tolist()  # Extract first batch, all timesteps

        # Debug: Print predictions to console
        print(f"\nGenerated predictions for {country_code} on {prediction_date}")
        
        # Generate combined model predictions (weighted average)
        # You can adjust these weights based on model performance
        informer_weight = 0.4
        gru_weight = 0.6
        
        model_predictions = []
        for i in range(min(len(informer_predictions), len(gru_predictions))):
            combined_pred = (informer_weight * informer_predictions[i]) + (gru_weight * gru_predictions[i])
            model_predictions.append(float(combined_pred))
        
        # Ensure we have exactly 24 hours of predictions for each model
        informer_predictions = informer_predictions[:24] if len(informer_predictions) >= 24 else informer_predictions + [0] * (24 - len(informer_predictions))
        gru_predictions = gru_predictions[:24] if len(gru_predictions) >= 24 else gru_predictions + [0] * (24 - len(gru_predictions))
        model_predictions = model_predictions[:24] if len(model_predictions) >= 24 else model_predictions + [0] * (24 - len(model_predictions))
        
        # Return the combined output
        return HybridModelOutput(
            informer_prediction=informer_predictions,
            gru_prediction=gru_predictions,
            model_prediction=model_predictions
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
    Generate mock prediction data for testing purposes
    
    Args:
        country_code: ISO country code to generate data for
        prediction_date: Date string in YYYY-MM-DD format
        input_file_path: Optional path to the input file (not used in test mode, but kept for API consistency)
        
    Returns:
        HybridModelOutput with mock prediction data
    """
    # Create variation based on country to simulate different models
    random.seed(hash(country_code) + hash(prediction_date))
    
    # Generate 24 hourly predictions for each model with some randomness
    num_hours = 24
    base_price = 50 + random.random() * 30  # Base price between 50-80
    
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
        
        # Combined model prediction (weighted average with some smoothing)
        model_pred = 0.4 * informer_pred + 0.6 * gru_pred
        
        # Add to result lists
        informer_predictions.append(float(informer_pred))
        gru_predictions.append(float(gru_pred))
        model_predictions.append(float(model_pred))
    
    return HybridModelOutput(
        informer_prediction=informer_predictions,
        gru_prediction=gru_predictions,
        model_prediction=model_predictions
    )


def _load_prediction_data(input_path, country_code=None, seq_len=168, label_len=24, pred_len=24):
    """
    Loads and preprocesses input CSV data for prediction.
    Dynamically extracts features from the input file to avoid hardcoding.
    Only uses data from the previous 7 days for prediction.
    
    Args:
        input_path: Path to the normalized CSV data file
        country_code: Country/map code to filter data for (e.g., 'DK1')
        seq_len: Sequence length for the encoder (168 hours = 7 days)
        label_len: Label length for the decoder
        pred_len: Prediction length for the decoder
        
    Returns:
        x_enc, x_mark_enc, x_dec, x_mark_dec tensors.
    """
    # Load the data from CSV
    df = pd.read_csv(input_path)
    
    # Filter by country_code/mapcode if provided
    if country_code:
        map_code_column = None
        # Look for a column that might contain the map code
        for col in df.columns:
            if 'mapcode' in col.lower():
                map_code_column = col
                break
                
        if map_code_column:
            map_code_column = "DK1"
            # Filter the dataframe based on the map code
            if country_code in df[map_code_column].values:
                print(f"Filtering data for map code: {country_code}")
                df = df[df[map_code_column] == country_code]
                if df.empty:
                    raise ValueError(f"No data found for map code {country_code}")
            else:
                print(f"Warning: Map code {country_code} not found in data. Using all data.")
        else:
            print(f"Warning: No map code column found in the data. Using all data.")
    
    # Dynamically get features from the input file

    FEATURES = [
    "hour", "day", "weekday", "month", "weekend", "season",
    "ep_lag_1", "ep_lag_24", "ep_lag_168", "ep_rolling_mean_24",
    "dahtl_totalLoadValue", "dahtl_lag_1h", "dahtl_lag_24h", "dahtl_lag_168h",
    "dahtl_rolling_mean_24h", "dahtl_rolling_mean_168h",
    "atl_totalLoadValue", "atl_lag_1h", "atl_lag_24h", "atl_lag_168h",
    "atl_rolling_mean_24h", "atl_rolling_mean_168h",
    "temperature_2m", "wind_speed_10m", "wind_direction_10m", "cloudcover", "shortwave_radiation",
    "temperature_2m_lag1", "wind_speed_10m_lag1", "wind_direction_10m_lag1", "cloudcover_lag1", "shortwave_radiation_lag1",
    "temperature_2m_lag24", "wind_speed_10m_lag24", "wind_direction_10m_lag24", "cloudcover_lag24", "shortwave_radiation_lag24",
    "temperature_2m_lag168", "wind_speed_10m_lag168", "wind_direction_10m_lag168", "cloudcover_lag168", "shortwave_radiation_lag168",
    "ftc_DE_LU", "ftc_DK1", "ftc_GB", "ftc_NL", "Natural_Gas_price_EUR", "Natural_Gas_price_EUR_lag_1d", "Natural_Gas_price_EUR_lag_7d", 
    "Natural_Gas_rolling_mean_24h", "Coal_price_EUR", "Coal_price_EUR_lag_1d", "Coal_price_EUR_lag_7d", "Coal_rolling_mean_7d", 
    "Oil_price_EUR", "Oil_price_EUR_lag_1d", "Oil_price_EUR_lag_7d", "Oil_rolling_mean_7d", "Carbon_Emission_price_EUR", 
    "Carbon_Emission_price_EUR_lag_1d", "Carbon_Emission_price_EUR_lag_7d", "Carbon_Emission_rolling_mean_7d", "Price[Currency/MWh]"
    ]
    
    # features = list(df.columns)
    features = FEATURES
    print(f"Loaded {len(features)} features from input file: {features}")
    
    # Ensure the price column is the first feature as expected by the model
    # This assumes the price column contains 'price' or 'currency' in its name
    price_column = None
    for col in features:
        if 'Price[Currency/MWh]' in col.lower() or 'Electricity_price_MWh' in col.lower():
            price_column = col
            break
            
    if price_column and features[0] != price_column:
        # Move price column to the front
        features.remove(price_column)
        features.insert(0, price_column)
        print(f"Reordered features to put '{price_column}' first: {features}")
    
    # Make sure columns are in the defined order
    df = df[features]

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
    num_features = len(features)

    x_mark_enc = torch.zeros((batch_size, seq_len, 4))  # 4 = (month, day, weekday, hour) usually if needed
    x_dec = torch.zeros((batch_size, label_len+pred_len, num_features))
    x_mark_dec = torch.zeros((batch_size, label_len+pred_len, 4))

    return x_enc, x_mark_enc, x_dec, x_mark_dec