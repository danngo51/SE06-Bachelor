#!/usr/bin/env python
"""
Script to test the trained GRU model on its own
This script loads the trained GRU model and generates predictions directly.

Usage:
python test_gru_only.py --zone DK1
"""

import os
import sys
import torch
import pandas as pd
import numpy as np
import json
import argparse
from pathlib import Path
from datetime import datetime

# Add the current directory to the path to import local modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # Backend directory
sys.path.insert(0, parent_dir)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test trained GRU model for a specific zone")
    parser.add_argument('--zone', type=str, required=True, help="Zone code (e.g., DK1, SE1)")
    parser.add_argument('--date', type=str, default=None, help="Prediction date (YYYY-MM-DD)")
    
    args = parser.parse_args()
    zone_code = args.zone
    prediction_date = args.date or datetime.now().strftime("%Y-%m-%d")
    
    # Define paths
    base_path = Path(current_dir)
    data_file = base_path / "data" / zone_code / "prediction_data.csv"
    config_file = base_path / "informer" / zone_code / "config.json"
    model_file = base_path / "gru" / zone_code / "results" / "gru_trained.pt"
    
    # Check if required files exist
    if not data_file.exists():
        print(f"❌ Error: Prediction data file not found at {data_file}")
        return False
    
    if not config_file.exists():
        print(f"❌ Error: Config file not found at {config_file}")
        return False
        
    if not model_file.exists():
        print(f"❌ Error: GRU model file not found at {model_file}")
        return False

    # Import required modules
    try:
        # Add GRU model path
        sys.path.append(str(base_path / "gru" / "gruModel"))
        from ml_models.gru.gruModel.gruModel import GRUModel
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please make sure the module ml_models.gru.gruModel.gruModel is available")
        return False
    
    # Load config
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    print(f"✅ Loaded config from {config_file}")
    
    # Set parameters
    input_dim = len(config.get("cols", [])) or 63  # Use number of features from config or default
    hidden_dim = 128
    output_dim = 24  # 24 hours prediction
    seq_len = config.get("seq_len", 168)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✅ Using device: {device}")
    
    # Load the data
    print(f"⏳ Loading prediction data from {data_file}...")
    df = pd.read_csv(data_file)
    
    # Get features from config
    cols = config.get("cols", [])
    if not cols:
        print("⚠️ No columns found in config, using all columns from data file")
        cols = list(df.columns)
    
    # Ensure all columns exist in the dataframe
    use_cols = [col for col in cols if col in df.columns]
    if len(use_cols) != len(cols):
        print(f"⚠️ Some columns from config not found in data: {set(cols) - set(use_cols)}")
    
    # Prepare input data - use the last seq_len rows
    data = df[use_cols].values
    if len(data) > seq_len:
        data = data[-seq_len:]
    elif len(data) < seq_len:
        print(f"⚠️ Warning: Prediction data has only {len(data)} rows, but {seq_len} are expected")
        # Pad with zeros if necessary
        padding = np.zeros((seq_len - len(data), len(use_cols)))
        data = np.vstack([padding, data])
    
    # Load the GRU model
    print(f"⏳ Loading GRU model...")
    gru = GRUModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim
    ).to(device)
    
    # Load the trained weights
    gru.load_state_dict(torch.load(model_file, map_location=device))
    gru.eval()
    print(f"✅ Loaded GRU model from {model_file}")
    
    # Convert data to tensor and reshape for batch
    x = torch.tensor(data, dtype=torch.float32).unsqueeze(0).to(device)  # Add batch dimension
    
    # Generate predictions
    print(f"⏳ Generating predictions for zone {zone_code} on {prediction_date}...")
    with torch.no_grad():
        predictions = gru(x)
        
    # Convert predictions to numpy array
    predictions = predictions.cpu().numpy().squeeze()
    
    # Print results
    print("\n📊 GRU Model Prediction Results:")
    print(f"Zone: {zone_code}")
    print(f"Date: {prediction_date}")
    print(f"Hourly predictions:")
    
    # Display hour by hour predictions
    for i, pred in enumerate(predictions):
        hour = i % 24
        print(f"Hour {hour:02d}: {pred:.2f}")
    
    print("\n✅ Prediction complete!")
    return True

if __name__ == "__main__":
    success = main()
    if success:
        sys.exit(0)
    else:
        sys.exit(1)
