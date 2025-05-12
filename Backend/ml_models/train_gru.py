#!/usr/bin/env python
"""
Example script for training the GRU model for a specific zone,
assuming the Informer model has already been trained for that zone.
"""
import os
from pathlib import Path

def train_gru_for_single_zone():
    """Train the GRU model for a specific zone"""
    # Import the training function
    import sys
    base_path = Path(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, str(base_path.parent))
    
    from ml_models.train_zone_models import train_gru_for_zone
    
    # Set the zone and training parameters
    zone_code = "DK1"  # Change this to the zone you want to train
    epochs = 15
    batch_size = 64
    
    print(f"🚀 Starting GRU training for zone {zone_code}")
    print(f"Parameters: Epochs = {epochs}, Batch Size = {batch_size}")
    
    # Train the GRU model
    success = train_gru_for_zone(zone_code, epochs, batch_size)
    
    if success:
        print(f"\n✅ Successfully trained GRU model for zone {zone_code}")
        print(f"The model weights are saved at: ml_models/gru/{zone_code}/results/gru_trained.pt")
    else:
        print(f"\n❌ Failed to train GRU model for zone {zone_code}")

if __name__ == "__main__":
    train_gru_for_single_zone()
