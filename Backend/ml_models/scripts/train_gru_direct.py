#!/usr/bin/env python
"""
GRU training script for direct training without using the Informer embeddings
This script trains the GRU model directly on the input data without first passing it through the Informer encoder.

Usage:
python train_gru_direct.py --zone DK1
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import json
import argparse
import random
from pathlib import Path

# Add the current directory to the path to import local modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # Backend directory
sys.path.insert(0, parent_dir)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train GRU model directly for a specific zone")
    parser.add_argument('--zone', type=str, required=True, help="Zone code (e.g., DK1, SE1)")
    parser.add_argument('--epochs', type=int, default=15, help="Number of epochs to train for")
    parser.add_argument('--batch-size', type=int, default=64, help="Batch size for training")
    
    args = parser.parse_args()
    zone_code = args.zone
    epochs = args.epochs
    batch_size = args.batch_size
    
    # Define paths
    base_path = Path(current_dir)
    data_file = base_path / "data" / zone_code / "training_data.csv"
    informer_config = base_path / "informer" / zone_code / "config.json"
    
    # Check if required files exist
    if not data_file.exists():
        print(f"❌ Error: Training data file not found at {data_file}")
        return False
    
    if not informer_config.exists():
        print(f"❌ Error: Informer config not found at {informer_config}")
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
    
    # Load Informer config
    with open(informer_config, 'r') as f:
        config = json.load(f)
    
    print(f"✅ Loaded Informer config from {informer_config}")
    
    # Set training parameters
    input_dim = len(config.get("cols", [])) or 63  # Use number of features from config or default
    hidden_dim = 128
    output_dim = 24  # 24 hours prediction
    bidirectional = False
    learning_rate = 0.001
    seq_len = config.get("seq_len", 168)
    label_len = config.get("label_len", 24)
    pred_len = config.get("pred_len", 24)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✅ Using device: {device}")
    
    # Load the GRU model
    print(f"⏳ Creating GRU model...")
    gru = GRUModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        bidirectional=bidirectional
    ).to(device)
    
    # Define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(gru.parameters(), lr=learning_rate)
    
    # Data loading and preprocessing
    print(f"⏳ Loading and preprocessing data from {data_file}...")
    df = pd.read_csv(data_file)
    
    # Create results directory
    results_dir = base_path / "gru" / zone_code / "results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Define helper function to create batches for GRU training
    def get_data_batches():
        # Get configurations for the features
        cols = config.get("cols", [])
        if not cols:
            print("⚠️ No columns found in config, using all columns from data file")
            cols = list(df.columns)
        
        target_col = config.get("target", "Price[Currency/MWh]")
        
        # Ensure all columns exist in the dataframe
        use_cols = [col for col in cols if col in df.columns]
        if len(use_cols) != len(cols):
            print(f"⚠️ Some columns from config not found in data: {set(cols) - set(use_cols)}")
        
        # Prepare data
        data = df[use_cols].values
        
        # Create batches
        num_samples = len(df) - (seq_len + pred_len)
        for i in range(0, num_samples - batch_size, batch_size):
            batch_x = []
            batch_y = []
            
            for j in range(batch_size):
                start_idx = i + j
                end_idx = start_idx + seq_len
                target_idx = end_idx
                target_end_idx = target_idx + pred_len
                
                x = data[start_idx:end_idx]
                y = df[target_col].values[target_idx:target_end_idx]
                
                batch_x.append(x)
                batch_y.append(y)
            
            # Convert to tensors
            batch_x = torch.tensor(batch_x, dtype=torch.float32)
            batch_y = torch.tensor(batch_y, dtype=torch.float32)
            
            yield batch_x, batch_y
    
    # Training loop
    print(f"⏳ Starting GRU training for {epochs} epochs...")
    best_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    for epoch in range(epochs):
        total_loss = 0
        batch_count = 0
        
        for x_enc, y in get_data_batches():
            # Move data to device
            x_enc = x_enc.to(device)
            y = y.to(device)
            
            # Forward pass through GRU
            optimizer.zero_grad()
            pred = gru(x_enc)  # Direct input to GRU
            
            # Compute loss
            loss = criterion(pred, y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
            
            if batch_count % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_count}, Loss: {loss.item():.6f}")
        
        # Calculate average loss for the epoch
        avg_loss = total_loss / (batch_count if batch_count > 0 else 1)
        print(f"Epoch {epoch+1}/{epochs} completed, Avg Loss: {avg_loss:.6f}")
        
        # Save best model and early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            model_path = results_dir / "gru_trained.pt"
            torch.save(gru.state_dict(), model_path)
            print(f"✅ Saved improved model to {model_path}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"⏹️ Early stopping triggered after {epoch+1} epochs")
                break
    
    print(f"✅ GRU training completed for zone {zone_code} with final loss: {best_loss:.6f}")
    print(f"✅ Model saved to: {results_dir / 'gru_trained.pt'}")
    
    # Let's also create a test to verify the model works
    print(f"\n📊 Testing the trained GRU model...")
    
    # Load the trained model
    gru.load_state_dict(torch.load(results_dir / "gru_trained.pt"))
    gru.eval()
    
    # Generate some test predictions
    with torch.no_grad():
        # Get the first batch for testing
        for x_batch, _ in get_data_batches():
            x_batch = x_batch.to(device)
            preds = gru(x_batch)
            
            # Only look at first sample
            sample_pred = preds[0].cpu().numpy()
            print(f"Sample prediction (first 5 hours): {sample_pred[:5]}")
            break
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        sys.exit(0)
    else:
        sys.exit(1)
