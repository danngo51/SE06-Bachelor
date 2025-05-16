#!/usr/bin/env python
"""
Script to update zone-specific config files to use the correct data files (19-23 for training)
"""
import os
import json
from pathlib import Path

def update_zone_configs():
    """Update zone-specific config files to use the correct training data files"""
    # Base path for ML models
    base_path = Path(os.path.dirname(os.path.abspath(__file__)))
    informer_path = base_path / "informer"
    
    print(f"Looking for zone directories in {informer_path}")
    
    # Find all zone directories
    zone_dirs = [d for d in informer_path.iterdir() if d.is_dir() and d.name != "informerModel" and d.name != "results"]
    print(f"Found {len(zone_dirs)} zone directories")
    
    for zone_dir in zone_dirs:
        config_path = zone_dir / "config.json"
        if config_path.exists():
            print(f"\nUpdating config for zone {zone_dir.name}")
            try:
                with open(config_path, "r") as f:
                    config = json.load(f)
                
                # Update data path to use training data
                if "data_path" in config:
                    old_path = config["data_path"]
                    config["data_path"] = f"{zone_dir.name}_19-23.csv"
                    print(f"  Updated data_path: {old_path} -> {config['data_path']}")
                
                # Add comment about data split
                if "root_path" in config:
                    config["root_path"] = "./data/"
                    print(f"  Set root_path to: {config['root_path']}")
                
                # Write updated config
                with open(config_path, "w") as f:
                    json.dump(config, f, indent=2)
                print(f"  Updated config file: {config_path}")
                
            except Exception as e:
                print(f"  Error updating config for {zone_dir.name}: {e}")

def main():
    """Main function"""
    print("Updating zone-specific config files...")
    update_zone_configs()
    print("\nDone! Updated config files to use training data (19-23)")

if __name__ == "__main__":
    main()
