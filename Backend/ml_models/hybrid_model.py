import random
import math
import datetime
# Lazy import these only when needed
# from ml_models.informer.informer import InformerWrapper
# from ml_models.gru.gru import GRUWrapper
from models.prediction import HybridModelOutput

def run_pipeline():
    # Import the modules only when this function is called
    from ml_models.preprocessing import load_csv
    import torch
    from ml_models.informer.informer import InformerWrapper
    from ml_models.gru.gru import GRUWrapper
    
    config_path = "ml_models/informer/config.json"
    weight_path = "ml_models/informer/informer_trained.pt"
    data_path = "ml_models/informer/test_input.csv"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    informer = InformerWrapper(config_path, weight_path, device=device)
    gru = GRUWrapper(input_dim=512, hidden_dim=128, device=device)

    x_enc, x_mark_enc, x_dec, x_mark_dec = load_csv(data_path)
    x_enc, x_mark_enc, x_dec, x_mark_dec = (
        x_enc.to(device), x_mark_enc.to(device), x_dec.to(device), x_mark_dec.to(device)
    )

    embedding, prediction = informer.run(x_enc, x_mark_enc, x_dec, x_mark_dec)
    gru_out = gru.run(embedding)

    return {
        "informer_embedding": embedding.squeeze().cpu().numpy().tolist(),
        "informer_prediction": prediction.squeeze().cpu().numpy().tolist(),
        "gru_output": gru_out.squeeze().cpu().numpy().tolist()
    }

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
    xgboost_pred = []
    model_pred = []  # This will be the combined/final prediction
    
    for hour in range(24):
        # Create realistic price patterns with some randomness
        base_price = 45 + math.sin(hour / 3) * 15 + country_factor
        hour_pattern = 0.7 if (hour < 7 or hour > 19) else 1.3  # Lower at night, higher during day
        
        # Different models have slightly different predictions
        informer_price = base_price * hour_pattern + (random.random() * 5) + day_factor
        gru_price = informer_price * (1 + ((random.random() * 0.2) - 0.1))
        xgboost_price = base_price * hour_pattern + (random.random() * 8) - 2 + day_factor
        
        # Model prediction is a weighted average of the three models
        # Could be adjusted with different weights based on model performance
        model_price = (informer_price * 0.5) + (gru_price * 0.3) + (xgboost_price * 0.2)
        
        informer_pred.append(round(informer_price, 2))
        gru_pred.append(round(gru_price, 2))
        xgboost_pred.append(round(xgboost_price, 2))
        model_pred.append(round(model_price, 2))
    
    # Return as a structured model instead of a dictionary
    return HybridModelOutput(
        informer_prediction=informer_pred,
        gru_prediction=gru_pred,
        xgboost_prediction=xgboost_pred,
        model_prediction=model_pred
    )



