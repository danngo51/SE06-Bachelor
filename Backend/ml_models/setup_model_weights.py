#!/usr/bin/env python
"""
Move model weights to zone-specific directories
"""
import os
import shutil
from pathlib import Path

def setup_model_weights():
    # Base paths
    base_path = Path(os.path.dirname(os.path.abspath(__file__)))
    informer_path = base_path / "informer"
    gru_path = base_path / "gru"
    
    print("🔍 Finding zone directories...")
    # Find all zone directories
    excluded_dirs = ["informerModel", "results", "__pycache__"]
    zone_dirs = [d for d in informer_path.iterdir() 
              if d.is_dir() and d.name not in excluded_dirs]
    zone_names = [d.name for d in zone_dirs]
    
    print(f"✅ Found {len(zone_names)} zones: {', '.join(zone_names)}")
    
    # Source files
    informer_checkpoint = informer_path / "results" / "checkpoint.pth"
    gru_checkpoint = gru_path / "results" / "gru_trained.pt"
    
    if not informer_checkpoint.exists():
        print(f"⚠️ Informer checkpoint not found at {informer_checkpoint}")
    else:
        print(f"✅ Found Informer checkpoint: {informer_checkpoint}")
    
    if not gru_checkpoint.exists():
        print(f"⚠️ GRU checkpoint not found at {gru_checkpoint}")
    else:
        print(f"✅ Found GRU checkpoint: {gru_checkpoint}")
    
    # For each zone, copy the model weights to the zone's results directory
    for zone in zone_names:
        print(f"\n🔄 Setting up model weights for zone {zone}...")
        
        # Setup Informer model weights
        informer_results_dir = informer_path / zone / "results"
        if not informer_results_dir.exists():
            os.makedirs(informer_results_dir)
            print(f"✅ Created Informer results directory for {zone}")
        
        # Copy Informer checkpoint
        if informer_checkpoint.exists():
            informer_dest = informer_results_dir / "checkpoint.pth"
            shutil.copy2(informer_checkpoint, informer_dest)
            print(f"✅ Copied Informer checkpoint to {informer_dest}")
        
        # Setup GRU model weights
        gru_zone_dir = gru_path / zone
        if not gru_zone_dir.exists():
            os.makedirs(gru_zone_dir)
            print(f"✅ Created GRU directory for {zone}")
            
        gru_results_dir = gru_zone_dir / "results"
        if not gru_results_dir.exists():
            os.makedirs(gru_results_dir)
            print(f"✅ Created GRU results directory for {zone}")
        
        # Copy GRU checkpoint
        if gru_checkpoint.exists():
            gru_dest = gru_results_dir / "gru_trained.pt"
            shutil.copy2(gru_checkpoint, gru_dest)
            print(f"✅ Copied GRU checkpoint to {gru_dest}")
    
    print("\n✅ Model weights setup complete!")
    print("\nNOTE: All zones are currently using the same model weights.")
    print("You may want to train zone-specific models later.")

if __name__ == "__main__":
    setup_model_weights()
