#!/usr/bin/env python
"""
Utility script to check the zone-specific configuration and model files.
Verifies that all zones have proper config files and model checkpoints.
"""
import os
import json
from pathlib import Path
import argparse
from colorama import init, Fore, Style

# Initialize colorama for colored console output
init()

def check_zone_configs():
    """Check zone-specific configuration and model files."""
    # Base path for ML models
    base_path = Path(os.path.dirname(os.path.abspath(__file__)))
    informer_path = base_path / "informer"
    gru_path = base_path / "gru"
    data_path = base_path / "data"
    
    print(f"{Style.BRIGHT}Checking zone configurations...{Style.RESET_ALL}")
      # Find all zone directories under informer (excluding system directories)
    zone_dirs = [d for d in informer_path.iterdir() 
                if d.is_dir() and d.name not in ["informerModel", "results", "__pycache__"]]
    
    if not zone_dirs:
        print(f"{Fore.RED}❌ No zone directories found in {informer_path}{Style.RESET_ALL}")
        return
    
    print(f"{Fore.BLUE}Found {len(zone_dirs)} zone directories{Style.RESET_ALL}")
    
    all_ok = True
    
    # Check each zone
    for zone_dir in sorted(zone_dirs, key=lambda x: x.name):
        zone_name = zone_dir.name
        print(f"\n{Style.BRIGHT}Checking zone: {Fore.CYAN}{zone_name}{Style.RESET_ALL}")
        
        # Check informer config
        config_path = zone_dir / "config.json"
        config_ok = check_config_file(config_path, zone_name)
        
        # Check informer model file
        informer_model_path = zone_dir / "results" / "checkpoint.pth"
        informer_model_ok = check_file(informer_model_path, "Informer model")
        
        # Check gru model directory and files
        gru_zone_path = gru_path / zone_name
        gru_dir_ok = check_directory(gru_zone_path, "GRU directory")
        
        gru_model_ok = False
        gru_regressor_ok = False
        
        if gru_dir_ok:
            gru_model_path = gru_zone_path / "results" / "gru_trained.pt"
            gru_regressor_path = gru_zone_path / "results" / "gru_regressor.pt"
            gru_model_ok = check_file(gru_model_path, "GRU model")
            gru_regressor_ok = check_file(gru_regressor_path, "GRU regressor")
        
        # Check data files
        train_data_path = data_path / f"{zone_name}_19-23.csv"
        pred_data_path = data_path / f"{zone_name}_24.csv"
        train_data_ok = check_file(train_data_path, "Training data")
        pred_data_ok = check_file(pred_data_path, "Prediction data")
        
        # Determine overall status for this zone
        zone_ok = all([config_ok, informer_model_ok, gru_dir_ok, 
                      gru_model_ok, gru_regressor_ok, 
                      train_data_ok, pred_data_ok])
        
        if zone_ok:
            print(f"{Fore.GREEN}✅ Zone {zone_name} is correctly configured{Style.RESET_ALL}")
        else:
            print(f"{Fore.YELLOW}⚠️ Zone {zone_name} has configuration issues{Style.RESET_ALL}")
            all_ok = False
    
    # Print overall summary
    print(f"\n{Style.BRIGHT}Summary:{Style.RESET_ALL}")
    if all_ok:
        print(f"{Fore.GREEN}✅ All zones are correctly configured{Style.RESET_ALL}")
    else:
        print(f"{Fore.YELLOW}⚠️ Some zones have configuration issues - see details above{Style.RESET_ALL}")

def check_config_file(path, zone_name):
    """Check if a config file exists and has the correct structure."""
    if not path.exists():
        print(f"{Fore.RED}❌ Config file missing: {path}{Style.RESET_ALL}")
        return False
    
    try:
        with open(path, 'r') as f:
            config = json.load(f)
        
        # Check required fields
        required_fields = [
            "root_path", "data_path", "target", "cols", 
            "enc_in", "dec_in", "c_out", 
            "seq_len", "label_len", "pred_len", "d_model"
        ]
        
        missing_fields = [field for field in required_fields if field not in config]
        
        if missing_fields:
            print(f"{Fore.YELLOW}⚠️ Config {path} is missing these fields: {', '.join(missing_fields)}{Style.RESET_ALL}")
            return False
        
        # Check data path refers to the correct zone
        if zone_name not in config['data_path']:
            print(f"{Fore.YELLOW}⚠️ Config {path} data_path does not match zone: {config['data_path']}{Style.RESET_ALL}")
            return False
        
        print(f"{Fore.GREEN}✅ Config file valid: {path}{Style.RESET_ALL}")
        return True
    
    except Exception as e:
        print(f"{Fore.RED}❌ Error reading config {path}: {str(e)}{Style.RESET_ALL}")
        return False

def check_file(path, description):
    """Check if a file exists."""
    if not path.exists():
        print(f"{Fore.RED}❌ {description} missing: {path}{Style.RESET_ALL}")
        return False
    
    print(f"{Fore.GREEN}✅ {description} found: {path}{Style.RESET_ALL}")
    return True

def check_directory(path, description):
    """Check if a directory exists."""
    if not path.exists() or not path.is_dir():
        print(f"{Fore.RED}❌ {description} missing: {path}{Style.RESET_ALL}")
        return False
    
    results_dir = path / "results"
    if not results_dir.exists() or not results_dir.is_dir():
        print(f"{Fore.YELLOW}⚠️ Results directory missing: {results_dir}{Style.RESET_ALL}")
        return False
    
    print(f"{Fore.GREEN}✅ {description} found: {path}{Style.RESET_ALL}")
    return True

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Check zone-specific configuration files")
    args = parser.parse_args()
    
    check_zone_configs()

if __name__ == "__main__":
    main()
