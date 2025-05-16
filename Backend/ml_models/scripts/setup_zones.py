#!/usr/bin/env python
"""
Script to set up zone-specific folders and configuration files for the ML models.
This helps with the restructuring of the project to support zone-specific models.
"""
import os
import json
import shutil
from pathlib import Path

# Define the zones we want to support
ZONES = [
    "DK1", "DK2",               # Denmark zones
    "SE1", "SE2", "SE3", "SE4", # Sweden zones
    "NO1", "NO2", "NO3", "NO4", "NO5", # Norway zones
    "FI",                       # Finland
    "DE_LU", "NL"               # Germany+Luxembourg, Netherlands
]

def setup_zone_structure():
    """Set up the folder structure for zone-specific models and configs"""
    
    # Base path for ML models
    base_path = Path(os.path.dirname(os.path.abspath(__file__)))
    print(f"Setting up zone structure in: {base_path}")
    
    # Create zone-specific folders for each model type
    for model_type in ["informer", "gru"]:
        model_path = base_path / model_type
        
        # Skip if the base model folder doesn't exist
        if not model_path.exists():
            print(f"Warning: {model_path} doesn't exist, skipping...")
            continue
        
        # Find default config.json for Informer
        default_config = None
        if model_type == "informer" and (model_path / "config.json").exists():
            with open(model_path / "config.json", "r") as f:
                default_config = json.load(f)
                print(f"Loaded default config from {model_path / 'config.json'}")

        # Create directories for each zone
        for zone in ZONES:
            zone_dir = model_path / zone
            zone_dir.mkdir(exist_ok=True)
            print(f"Created: {zone_dir}")
            
            # Create results directory
            results_dir = zone_dir / "results"
            results_dir.mkdir(exist_ok=True)
            
            # For Informer, copy config.json with zone-specific path
            if model_type == "informer" and default_config:
                zone_config = default_config.copy()
                # Update config for zone-specific data path
                zone_config["data_path"] = f"{zone}_24.csv"  # Non-normalized
                
                with open(zone_dir / "config.json", "w") as f:
                    json.dump(zone_config, f, indent=2)
                print(f"Created zone-specific config: {zone_dir / 'config.json'}")    # Set up data directory for zone-specific data files
    data_dir = base_path / "data"
    if data_dir.exists():
        print(f"Setting up data directory: {data_dir}")
        
        # Find existing data files
        data_files = list(data_dir.glob("*.csv"))
        print(f"Found {len(data_files)} data files")
        
        # Create placeholder for missing zone data files
        for zone in ZONES:
            # Create training data placeholder (2019-2023)
            training_data_file = data_dir / f"{zone}_19-23.csv"
            if not training_data_file.exists():
                print(f"Creating placeholder for training data: {training_data_file}")
                with open(training_data_file, "w") as f:
                    f.write(f"# Placeholder for {zone} training data (2019-2023)\n")
                    f.write("# Replace this file with actual historical zone data\n")
            
            # Create prediction/validation data placeholder (2024)
            prediction_data_file = data_dir / f"{zone}_24.csv"
            if not prediction_data_file.exists():
                print(f"Creating placeholder for prediction data: {prediction_data_file}")
                with open(prediction_data_file, "w") as f:
                    f.write(f"# Placeholder for {zone} prediction data (2024)\n")
                    f.write("# Replace this file with actual 2024 zone data\n")

def main():
    """Main function"""
    print("Setting up zone-specific structure for ML models...")
    setup_zone_structure()
    print("\nDone! The project is now set up for zone-specific models.")
    print("\nNext steps:")
    print("1. Add zone-specific data files to ml_models/data/")
    print("2. Train zone-specific models and place the model files in their respective folders")
    print("   - Informer models go in: ml_models/informer/<ZONE>/results/")
    print("   - GRU models go in: ml_models/gru/<ZONE>/results/")
    print("\nThe hybrid model will now use zone-specific models if available,")
    print("or fall back to default models if zone-specific ones don't exist.")

if __name__ == "__main__":
    main()
