#!/usr/bin/env python
"""
Test script specifically for the Informer model in DK1
This script bypasses the usual abstraction to directly test the model
"""
import os
import sys
import torch
import pandas as pd
import json
from pathlib import Path

# Add the Backend directory to the path
current_file_path = Path(os.path.abspath(__file__))
backend_path = current_file_path.parent.parent
sys.path.insert(0, str(backend_path))

from ml_models.informer.informer import InformerWrapper
import argparse

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Direct test of the Informer model")
    parser.add_argument('--zone', type=str, default="DK1", help="Zone code (e.g., DK1, SE1)")
    parser.add_argument('--date', type=str, default="2025-01-05", help="Prediction date (YYYY-MM-DD)")
    
    args = parser.parse_args()
    print(f"🔮 Testing Informer model directly for zone {args.zone} on date {args.date}")
    
    try:
        # Set up device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Set up paths
        ml_models_dir = current_file_path.parent
        data_dir = ml_models_dir / "data"
        informer_dir = ml_models_dir / "informer"
        
        # Find the input file
        input_file_path = str(data_dir / args.zone / "prediction_data.csv")
        if not os.path.exists(input_file_path):
            input_file_path = str(data_dir / "DK1" / "prediction_data.csv")
            print(f"⚠️ Using fallback input data from DK1")
        
        # Find the config file and load it
        config_path = str(informer_dir / args.zone / "config.json")
        with open(config_path, 'r') as f:
            config = json.load(f)
            print(f"Loaded config from {config_path}")
            print(f"Config enc_in: {config.get('enc_in')}")
            print(f"Config columns: {len(config.get('cols', []))}")
          # Find the weights file
        weight_path = str(informer_dir / args.zone / "results" / "checkpoint.pth")
        
        # Analyze model weights to determine exact dimensions
        print(f"Analyzing model weights from {weight_path}")
        model_state = torch.load(weight_path, map_location=device)
        
        # Extract encoder input dimension from weights
        enc_in = None
        if "enc_embedding.value_embedding.tokenConv.weight" in model_state:
            enc_weight = model_state["enc_embedding.value_embedding.tokenConv.weight"]
            enc_in = enc_weight.shape[1]
            print(f"Detected enc_in={enc_in} from model weights")
        elif "enc_in" in config:
            enc_in = config["enc_in"]
            print(f"Using enc_in={enc_in} from config")
        else:
            enc_in = 56  # Default for DK1
            print(f"Using default enc_in={enc_in}")
        
        # Load the data
        df = pd.read_csv(input_file_path)
        print(f"Data shape: {df.shape}")
        
        # Make sure we're using exactly the columns specified in the config
        # Get all the columns specified in the config that are also in the data
        if "cols" in config:
            available_columns = [col for col in config["cols"] if col in df.columns]
            print(f"Found {len(available_columns)} of {len(config['cols'])} columns in the data")
            
            # If there are missing columns, create them with zeros
            for col in config["cols"]:
                if col not in df.columns:
                    df[col] = 0
                    print(f"Created missing column: {col}")
            
            # Select only the columns specified in the config and in the right order
            df = df[config["cols"]]
            print(f"Data shape after column selection: {df.shape}")
        
        # Take only the first enc_in columns to match model's expected feature count
        if df.shape[1] != enc_in:
            print(f"⚠️ Feature count mismatch: data has {df.shape[1]} columns but model expects {enc_in}")
            if df.shape[1] > enc_in:
                # Trim extra columns
                df = df.iloc[:, :enc_in]
                print(f"Trimmed data to first {enc_in} columns")
            else:
                # Add padding columns
                for i in range(df.shape[1], enc_in):
                    col_name = f"pad_{i}"
                    df[col_name] = 0
                print(f"Added {enc_in - df.shape[1]} padding columns")
            
            print(f"Data shape after adjustment: {df.shape}")
        
        # Take only the last 168 hours of data
        seq_len = config.get("seq_len", 168)
        if len(df) > seq_len:
            df = df.tail(seq_len)
            print(f"Using the last {seq_len} hours of data")
          # Convert data to tensors
        data = df.values
        data_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0)  # [batch_size, seq_len, num_features]
        print(f"Data tensor shape: {data_tensor.shape}")
        
        # Create informer model
        informer = InformerWrapper(
            config_path=config_path,
            weight_path=weight_path,
            device=device
        )
        
        # Create dummy time features and decoder inputs
        batch_size = 1
        label_len = config.get("label_len", 48)
        pred_len = config.get("pred_len", 24)
        num_features = data_tensor.shape[2]  # Use the actual features in the input tensor
        
        x_mark_enc = torch.zeros((batch_size, seq_len, 4), device=device)  # 4 time features
        
        # Make sure x_dec has the same number of features as x_enc
        x_dec = torch.zeros((batch_size, label_len+pred_len, num_features), device=device)
        x_mark_dec = torch.zeros((batch_size, label_len+pred_len, 4), device=device)
        
        print(f"Encoder input shape: {data_tensor.shape}")
        print(f"Decoder input shape: {x_dec.shape}")
        
        # Run the model
        with torch.no_grad():
            enc_out, informer_pred = informer.run(
                data_tensor.to(device), 
                x_mark_enc, 
                x_dec, 
                x_mark_dec
            )
        
        # Get the predictions
        predictions = informer_pred.cpu().numpy()[0, :, 0].tolist()
        
        # Print predictions
        print("\n📊 Prediction Results:")
        print(f"✅ All 24 hours: {predictions}")
        
        print("\n🏁 Test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
