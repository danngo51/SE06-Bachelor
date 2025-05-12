#!/usr/bin/env python
"""
Reorganize the data folder by creating zone-specific subfolders
and moving the corresponding files into them.
"""
import os
import json
import shutil
from pathlib import Path

def reorganize_data_folders():
    """Create zone-specific subfolders and move data files accordingly."""
    # Get base paths
    base_path = Path(os.path.dirname(os.path.abspath(__file__)))
    data_path = base_path / "data"
    informer_path = base_path / "informer"
    
    print("🔍 Checking zone directories in informer folder...")
    
    # Find all zone directories (excluding non-zone directories)
    excluded_dirs = ["informerModel", "results", "__pycache__"]
    zone_dirs = [d for d in informer_path.iterdir() 
              if d.is_dir() and d.name not in excluded_dirs]
    
    zone_names = [d.name for d in zone_dirs]
    print(f"✅ Found {len(zone_names)} zones: {', '.join(zone_names)}")
    
    # Create directories for each zone in the data folder
    for zone in zone_names:
        zone_data_dir = data_path / zone
        if not zone_data_dir.exists():
            os.makedirs(zone_data_dir)
            print(f"✅ Created zone data directory: {zone_data_dir}")
        else:
            print(f"ℹ️ Zone data directory already exists: {zone_data_dir}")
    
    print("\n🔄 Moving zone-specific files to their folders...")
    # Move files to zone-specific folders
    for zone in zone_names:
        # Training data file (19-23)
        training_file = data_path / f"{zone}_19-23.csv"
        if training_file.exists():
            dest_file = data_path / zone / "training_data.csv"
            shutil.copy2(training_file, dest_file)
            print(f"✅ Copied {training_file.name} to {dest_file}")
        
        # Prediction data file (24)
        prediction_file = data_path / f"{zone}_24.csv"
        if prediction_file.exists():
            dest_file = data_path / zone / "prediction_data.csv"
            shutil.copy2(prediction_file, dest_file)
            print(f"✅ Copied {prediction_file.name} to {dest_file}")
        
        # Normalized prediction data if exists
        normalized_file = data_path / f"{zone}_24-normalized.csv"
        if normalized_file.exists():
            dest_file = data_path / zone / "prediction_data_normalized.csv"
            shutil.copy2(normalized_file, dest_file)
            print(f"✅ Copied {normalized_file.name} to {dest_file}")
    
    print("\n🔄 Updating config files to use new data paths...")
    # Update config files to point to the new data paths
    for zone in zone_names:
        zone_config_path = informer_path / zone / "config.json"
        if zone_config_path.exists():
            try:
                with open(zone_config_path, 'r') as f:
                    config = json.load(f)
                
                # Update root_path and data_path
                original_root_path = config.get("root_path", "./data/")
                original_data_path = config.get("data_path", f"{zone}_19-23.csv")
                
                config["root_path"] = f"./data/{zone}/"
                config["data_path"] = "training_data.csv"
                
                with open(zone_config_path, 'w') as f:
                    json.dump(config, f, indent=2)
                
                print(f"✅ Updated config for {zone}: {original_root_path + original_data_path} -> {config['root_path'] + config['data_path']}")
            except Exception as e:
                print(f"❌ Error updating config for {zone}: {e}")
    
    # Update main config as an example
    main_config_path = informer_path / "config.json"
    if main_config_path.exists():
        try:
            with open(main_config_path, 'r') as f:
                config = json.load(f)
            
            # Use DK1 as default example
            config["root_path"] = "./data/DK1/"
            config["data_path"] = "training_data.csv"
            
            with open(main_config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            print(f"✅ Updated main config to use DK1 data path")
        except Exception as e:
            print(f"❌ Error updating main config: {e}")
    
    print("\n🔄 Updating hybrid_model.py to use new data paths...")
    
    try:
        # Update hybrid_model.py to use the new file paths
        hybrid_model_path = base_path / "hybrid_model.py"
        if hybrid_model_path.exists():
            with open(hybrid_model_path, 'r') as f:
                hybrid_model_code = f.read()
            
            # Path to update in the hybrid_model.py file
            path1 = 'input_file = os.path.join(ml_models_dir, "data", f"{country_code}_24.csv")'
            path2 = 'input_file = os.path.join(ml_models_dir, "data", "DK1_24.csv")'
            
            new_path1 = 'input_file = os.path.join(ml_models_dir, "data", country_code, "prediction_data.csv")'
            new_path2 = 'input_file = os.path.join(ml_models_dir, "data", "DK1", "prediction_data.csv")'
            
            # Replace paths
            if path1 in hybrid_model_code:
                hybrid_model_code = hybrid_model_code.replace(path1, new_path1)
                print(f"✅ Updated prediction path in hybrid_model.py")
            else:
                print(f"⚠️ Could not find prediction path in hybrid_model.py")
                
            if path2 in hybrid_model_code:
                hybrid_model_code = hybrid_model_code.replace(path2, new_path2)
                print(f"✅ Updated fallback path in hybrid_model.py")
            else:
                print(f"⚠️ Could not find fallback path in hybrid_model.py")
            
            # Write back the updated file
            with open(hybrid_model_path, 'w') as f:
                f.write(hybrid_model_code)
                
        # Also update PredictionService.py
        prediction_service_path = base_path.parent / "services" / "prediction" / "PredictionService.py"
        if prediction_service_path.exists():
            print("✅ Found PredictionService.py, updating file paths...")
            
            with open(prediction_service_path, 'r') as f:
                service_code = f.read()
            
            # Update paths in PredictionService
            service_path1 = 'ACTUAL_PRICE_FILE = DATA_PATH / "DK1_24.csv"'
            service_path2 = 'NORMALIZED_INPUT_FILE = DATA_PATH / "DK1_24-normalized.csv"'
            
            new_service_path1 = 'ACTUAL_PRICE_FILE = DATA_PATH / "DK1" / "prediction_data.csv"'
            new_service_path2 = 'NORMALIZED_INPUT_FILE = DATA_PATH / "DK1" / "prediction_data_normalized.csv"'
            
            if service_path1 in service_code:
                service_code = service_code.replace(service_path1, new_service_path1)
                print(f"✅ Updated actual price file path in PredictionService.py")
            else:
                print(f"⚠️ Could not find actual price path in PredictionService.py")
                
            if service_path2 in service_code:
                service_code = service_code.replace(service_path2, new_service_path2)
                print(f"✅ Updated normalized input path in PredictionService.py")
            else:
                print(f"⚠️ Could not find normalized input path in PredictionService.py")
            
            with open(prediction_service_path, 'w') as f:
                f.write(service_code)
    except Exception as e:
        print(f"❌ Error updating files: {e}")
    
    # Create a README in the data folder to explain the structure
    readme_content = """# Data Folder Structure

This folder contains zone-specific data organized in subfolders:

## Structure
Each zone has its own subfolder (e.g., DK1, SE1) containing:
- `training_data.csv` - Historical data from 2019-2023 used for training models
- `prediction_data.csv` - Recent data from 2024 used for predictions
- `prediction_data_normalized.csv` - Normalized version of the prediction data (if available)

## Usage
The config files have been updated to point to these new locations.
"""
    
    with open(data_path / "README.md", 'w') as f:
        f.write(readme_content)
    
    print("\n✅ Created README.md in the data folder")
    print("\n🎉 Data folder reorganization complete!")
    print("\nNOTE: The original files have been kept in the main data folder.")
    print("Once you've verified everything works correctly, you may want to delete them to save space.")

if __name__ == "__main__":
    reorganize_data_folders()
