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
        import torch
          # Set training parameters
        args = argparse.Namespace(
            model='informer',
            data='custom',
            root_path=str(base_path / "data"),
            data_path=f"{zone_code}_19-23.csv",  # Use historical data (2019-2023) for training
            features='MS',  # Main and Selected features
            target='Price[Currency/MWh]',  # Target column
            enc_in=63,      # Encoder input size
            dec_in=63,      # Decoder input size
            c_out=1,        # Output size
            d_model=512,    # Model dimension
            n_heads=8,      # MultiHead Attention heads
            e_layers=2,     # Encoder layers
            d_layers=1,     # Decoder layers
            d_ff=2048,      # Dimension of FCN
            dropout=0.05,   # Dropout
            attn='prob',    # Attention type
            embed='timeF',  # Time features encoding
            activation='gelu',
            output_attention=True,
            distil=True,    # Distilling
            mix=True,       # Mix attention
            freq='h',       # Frequency of time features
            train_epochs=epochs,
            batch_size=batch_size,
            patience=3,     # Early stopping patience
            learning_rate=0.0001,
            des='train',    # Experiment description
            itr=1,          # Iteration
            train=True,     # Training or not
            resume=False,   # Resume training or not
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            seq_len=168,    # Input sequence length (7 days)
            label_len=24,   # Start token length for decoder
            pred_len=24,    # Prediction length (1 day)
            use_amp=False,  # Mixed precision
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
    Train a GRU model for a specific zone
    
    Args:
        zone_code: The zone code (e.g., 'DK1', 'SE1')
        epochs: Number of epochs to train for
        batch_size: Batch size for training
    """
    # Base path for ML models
    base_path = Path(os.path.dirname(os.path.abspath(__file__)))
    data_file = base_path / "data" / f"{zone_code}_24.csv"
    
    if not data_file.exists():
        print(f"❌ Error: Data file not found at {data_file}")
        return False
    
    # Import needed packages
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
        import pandas as pd
        import numpy as np
        from ml_models.gru.gruModel.gruModel import GRUModel
        
        # Set training parameters
        input_dim = 512  # This should match the Informer encoder output
        hidden_dim = 128
        output_dim = 24  # 24 hours prediction
        bidirectional = False
        learning_rate = 0.001
        
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"✅ Using device: {device}")
        
        # Load the GRU model
        model = GRUModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            bidirectional=bidirectional
        ).to(device)
        
        # Define loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # This is where you'd load your dataset and create embeddings
        # For now, we'll create a placeholder for the embeddings
        # In a real scenario, you'd run data through the Informer encoder first
        
        print(f"⚠️ Note: For actual training, you need to generate embeddings from Informer first")
        print(f"⚠️ This script is a template that needs to be adapted to your specific data pipeline")
        
        # Create results directory
        results_dir = base_path / "gru" / zone_code / "results"
        results_dir.mkdir(exist_ok=True)
        
        # Save the model
        model_path = results_dir / "gru_trained.pt"
        torch.save(model.state_dict(), model_path)
        print(f"✅ Saved initial GRU model to {model_path}")
        print(f"⚠️ This is just a placeholder - actual training requires Informer embeddings")
        
        return True
    
    except Exception as e:
        print(f"❌ Error setting up GRU model for zone {zone_code}: {e}")
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
    data_file = base_path / "data" / f"{args.zone}_24.csv"
    
    if not data_file.exists():
        print(f"❌ Error: Data file not found at {data_file}")
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
