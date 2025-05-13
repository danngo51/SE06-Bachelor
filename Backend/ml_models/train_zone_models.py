#!/usr/bin/env python
"""
Utility script for training zone-specific models
This script helps with training models for each zone independently.
"""
import os
import sys
import argparse
from pathlib import Path

def train_informer_for_zone(zone_code, epochs=10, batch_size=32):
    """
    Train an Informer model for a specific zone
    
    Args:
        zone_code: The zone code (e.g., 'DK1', 'SE1')
        epochs: Number of epochs to train for
        batch_size: Batch size for training
    """
    # Base path for ML models
    base_path = Path(os.path.dirname(os.path.abspath(__file__)))
    zone_config_path = base_path / "informer" / zone_code / "config.json"
    
    if not zone_config_path.exists():
        print(f"❌ Error: Config file not found at {zone_config_path}")
        return False
    
    # Import informer train script
    informer_path = str(base_path / "informer" / "informerModel")
    sys.path.insert(0, informer_path)
    
    try:
        from exp.exp_informer import Exp_Informer
        import torch        # Load the zone's config file to get parameters
        import json
        try:
            with open(zone_config_path, 'r') as f:
                config = json.load(f)
        except Exception as e:
            print(f"❌ Error loading config from {zone_config_path}: {e}")
            return False
            
        # Set training parameters with minimal required parameters
        args = argparse.Namespace(
            model='informer',
            data='custom',
            root_path=config.get("root_path", "./data/"),
            data_path=config.get("data_path", f"{zone_code}_19-23.csv"),
            target=config.get("target", "Price[Currency/MWh]"),
            cols=config.get("cols", []),
            enc_in=config.get("enc_in", 63),
            dec_in=config.get("dec_in", 63),
            c_out=config.get("c_out", 1),
            seq_len=config.get("seq_len", 168),
            label_len=config.get("label_len", 24),
            pred_len=config.get("pred_len", 24),
            d_model=config.get("d_model", 512),
            # Additional parameters needed for training but not stored in simplified config
            features='MS',
            e_layers=2,
            d_layers=1,
            n_heads=8,
            d_ff=2048,
            dropout=0.05,
            attn='prob',
            embed='timeF',
            activation='gelu',
            output_attention=False,
            distil=True,
            mix=True,
            freq='h',
            train_epochs=epochs,
            batch_size=batch_size,
            patience=3,
            learning_rate=0.0001,
            des='zone_training',
            itr=1,
            train=True,
            resume=False,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            use_amp=False,
            checkpoints=str(base_path / "informer" / zone_code / "results")
        )
        
        # Create experiment
        exp = Exp_Informer(args)
        
        print(f"✅ Training Informer model for zone {zone_code}...")
        print(f"✅ Using device: {args.device}")
        
        # Train model
        print(f"⏳ Starting training for {args.train_epochs} epochs...")
        exp.train(use_strict_loading=True)
        
        print(f"✅ Training completed. Model saved to {args.checkpoints}")
        return True
    
    except Exception as e:
        print(f"❌ Error training Informer model for zone {zone_code}: {e}")
        return False

def train_gru_for_zone(zone_code, epochs=10, batch_size=32):
    """
    Train a GRU model for a specific zone using embeddings from a trained Informer model
    
    Args:
        zone_code: The zone code (e.g., 'DK1', 'SE1')
        epochs: Number of epochs to train for
        batch_size: Batch size for training
    """
    # Base path for ML models
    base_path = Path(os.path.dirname(os.path.abspath(__file__)))
    data_file = base_path / "data" / zone_code / "training_data.csv"
    
    if not data_file.exists():
        print(f"❌ Error: Training data file not found at {data_file}")
        return False
    
    # Check if Informer checkpoint exists
    informer_checkpoint = base_path / "informer" / zone_code / "results" / "checkpoint.pth"
    informer_config = base_path / "informer" / zone_code / "config.json"
    
    if not informer_checkpoint.exists():
        print(f"❌ Error: Informer checkpoint not found at {informer_checkpoint}")
        return False
    
    if not informer_config.exists():
        print(f"❌ Error: Informer config not found at {informer_config}")
        return False
    
    # Import needed packages
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
        import pandas as pd
        import numpy as np
        import json        
        import sys
        # Add paths for imports
        sys.path.append(str(base_path.parent))  # Add Backend directory
        sys.path.append(str(base_path / "gru" / "gruModel"))  # Add GRU model directory
        
        # Import required modules
        from gru.gruModel.gruModel import GRUModel
        from informer.informer import InformerWrapper
        
        # Load Informer config to get the same parameters
        with open(informer_config, 'r') as f:
            config = json.load(f)
        
        print(f"✅ Loaded Informer config from {informer_config}")
        
        # Set training parameters
        input_dim = 512  # This should match the Informer encoder output
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
        
        # Load the Informer model (pre-trained)
        print(f"⏳ Loading pre-trained Informer model for zone {zone_code}...")
        informer = InformerWrapper(
            config_path=str(informer_config),
            weight_path=str(informer_checkpoint),
            device=device
        )
        informer.model.eval()  # Set to evaluation mode since we don't train it
        print(f"✅ Loaded Informer model from {informer_checkpoint}")
        
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
            
            # Prepare data for Informer
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
                
                # Create time features (simplified for now)
                batch_x_mark = torch.zeros((batch_size, seq_len, 4))
                batch_y_mark = torch.zeros((batch_size, pred_len, 4))
                batch_dec = torch.zeros((batch_size, pred_len, len(use_cols)))
                
                yield batch_x, batch_y, batch_x_mark, batch_y_mark, batch_dec
        
        # Training loop
        print(f"⏳ Starting GRU training for {epochs} epochs...")
        best_loss = float('inf')
        patience = 5
        patience_counter = 0
        
        for epoch in range(epochs):
            total_loss = 0
            batch_count = 0
            
            for x_enc, y, x_mark_enc, y_mark, x_dec in get_data_batches():
                # Move data to device
                x_enc = x_enc.to(device)
                x_mark_enc = x_mark_enc.to(device)
                x_dec = x_dec.to(device)
                y = y.to(device)
                
                # Get encoder output from Informer (frozen)
                with torch.no_grad():
                    enc_out = informer.encode(x_enc, x_mark_enc)
                
                # Forward pass through GRU
                optimizer.zero_grad()
                pred = gru(enc_out)
                
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
        return True
    
    except Exception as e:
        print(f"❌ Error training GRU model for zone {zone_code}: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train zone-specific models")
    parser.add_argument('--zone', type=str, required=True, help="Zone code (e.g., DK1, SE1)")
    parser.add_argument('--model', type=str, default="both", choices=["informer", "gru", "both"], 
                        help="Which model to train (informer, gru, or both)")
    parser.add_argument('--epochs', type=int, default=10, help="Number of epochs to train for")
    parser.add_argument('--batch-size', type=int, default=32, help="Batch size for training")
    
    args = parser.parse_args()
    
    # Check if zone exists
    base_path = Path(os.path.dirname(os.path.abspath(__file__)))
    data_file = base_path / "data" / args.zone / "training_data.csv"
    
    if not data_file.exists():
        print(f"❌ Error: Training data file not found at {data_file}")
        return
    
    print(f"🚀 Starting training for zone {args.zone}")
    
    if args.model in ["informer", "both"]:
        print(f"\n📊 Training Informer model for zone {args.zone}...")
        success = train_informer_for_zone(args.zone, args.epochs, args.batch_size)
        if success:
            print(f"✅ Informer training completed for zone {args.zone}")
    
    if args.model in ["gru", "both"]:
        print(f"\n📊 Setting up GRU model for zone {args.zone}...")
        success = train_gru_for_zone(args.zone, args.epochs, args.batch_size)
        if success:
            print(f"✅ GRU setup completed for zone {args.zone}")
    
    print(f"\n🏁 All training tasks completed for zone {args.zone}")

if __name__ == "__main__":
    main()
