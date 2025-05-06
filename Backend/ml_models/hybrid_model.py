import random
import math
from models.prediction import HybridModelOutput
import os
import torch
import pandas as pd
import pathlib
from ml_models.informer.informer import InformerWrapper
from ml_models.gru.gru import GRUWrapper
from ml_models.preprocessing import load_prediction_data


def predict_with_hybrid_model(country_code: str, prediction_date: str) -> HybridModelOutput:
    """
    Run the hybrid model prediction pipeline for a specific country and date
    
    Args:
        country_code: ISO country code to generate data for
        prediction_date: Date string in YYYY-MM-DD format
        
    Returns:
        HybridModelOutput with predictions from both models
    """
    try:
        # Determine the input file
        input_file = f"ml_models/data/DK1_24-normalized.csv"
        
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

        # Load and preprocess input
        x_enc, x_mark_enc, x_dec, x_mark_dec = load_prediction_data(input_file)

        x_enc = x_enc.to(device)
        x_mark_enc = x_mark_enc.to(device)
        x_dec = x_dec.to(device)
        x_mark_dec = x_mark_dec.to(device)

        # Prediction
        with torch.no_grad():
            enc_out, informer_pred = informer.run(x_enc, x_mark_enc, x_dec, x_mark_dec)
            gru_pred = gru.run(enc_out)

        # Move predictions to CPU
        informer_predictions = informer_pred.cpu().numpy()
        gru_predictions = gru_pred.cpu().numpy()

        # Print predictions to console
        # print(f"\n📈 {country_code} - {prediction_date} Predictions:")
        # print(f"Informer shape: {informer_predictions.shape}, GRU shape: {gru_predictions.shape}")

        # Extract the predictions and convert to lists
        informer_pred_list = informer_predictions[0].tolist()
        gru_pred_list = gru_predictions[0].tolist()
        
        # Calculate model predictions (weighted average: 60% informer, 40% gru)
        model_pred_list = [(0.6 * i + 0.4 * g) for i, g in zip(informer_pred_list, gru_pred_list)]
        
        # Round all predictions to 2 decimal places
        informer_pred_list = [round(x, 2) for x in informer_pred_list]
        gru_pred_list = [round(x, 2) for x in gru_pred_list]
        model_pred_list = [round(x, 2) for x in model_pred_list]

        # Optional: Save results to CSV
        """
        output_dir = pathlib.Path(__file__).parent.parent / "data" / "output"
        os.makedirs(output_dir, exist_ok=True)
        output_path = output_dir / f"{country_code}-{prediction_date}-predictions.csv"
        
        # Combine predictions
        df = pd.DataFrame({
            **{f"informer_pred_{i}": [informer_pred_list[i]] for i in range(len(informer_pred_list))},
            **{f"gru_pred_{i}": [gru_pred_list[i]] for i in range(len(gru_pred_list))},
            **{f"model_pred_{i}": [model_pred_list[i]] for i in range(len(model_pred_list))}
        })

        df.to_csv(output_path, index=False)
        print(f"✅ Predictions saved to: {output_path}")
        """
        
        # Return structured model output
        return HybridModelOutput(
            informer_prediction=informer_pred_list,
            gru_prediction=gru_pred_list,
            model_prediction=model_pred_list
        )
    
    except Exception as e:
        print(f"❌ Error in prediction: {str(e)}")
        # Fall back to test prediction in case of error
        return test_predict(country_code, prediction_date)


def test_predict(country_code: str, prediction_date: str) -> HybridModelOutput:
    """
    Generate mock prediction data for testing purposes
    
    Args:
        country_code: ISO country code to generate data for
        prediction_date: Date string in YYYY-MM-DD format
            
    Returns:
        HybridModelOutput with mock prediction data in the format that will be used by the real model
    """
    # Create variation based on country
    country_factor = 0
    for char in country_code:
        country_factor += ord(char)
    country_factor = country_factor % 10
    
    # Parse date for additional variation
    try:
        date_parts = prediction_date.split('-')
        day_factor = int(date_parts[2]) * 0.5  # Day affects pattern
    except:
        day_factor = 5.0  # Default
    
    # Generate 24 hours of mock data for each model
    informer_pred = []
    gru_pred = []
    model_pred = []  # This will be the combined/final prediction
    
    for hour in range(24):
        # Create realistic price patterns with some randomness
        base_price = 45 + math.sin(hour / 3) * 15 + country_factor
        hour_pattern = 0.7 if (hour < 7 or hour > 19) else 1.3  # Lower at night, higher during day
        
        # Different models have slightly different predictions
        informer_price = base_price * hour_pattern + (random.random() * 5) + day_factor
        gru_price = informer_price * (1 + ((random.random() * 0.2) - 0.1))
        
        # Model prediction is a weighted average of the two models (informer and gru)
        # Could be adjusted with different weights based on model performance
        model_price = (informer_price * 0.6) + (gru_price * 0.4)
        
        informer_pred.append(round(informer_price, 2))
        gru_pred.append(round(gru_price, 2))
        model_pred.append(round(model_price, 2))
    
    # Return as a structured model instead of a dictionary
    return HybridModelOutput(
        informer_prediction=informer_pred,
        gru_prediction=gru_pred,
        model_prediction=model_pred
    )



