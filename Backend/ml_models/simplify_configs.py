import os
import json
import glob

def simplify_config(config):
    """Keep only the essential parameters in the config file."""
    essential_fields = [
        "root_path", "data_path", "target", "cols",
        "enc_in", "dec_in", "c_out", 
        "seq_len", "label_len", "pred_len", "d_model"
    ]
    
    simplified = {}
    for field in essential_fields:
        if field in config:
            simplified[field] = config[field]
    
    return simplified

def process_configs():
    """Process all config.json files in zone directories."""
    # Get the base directory where this script is located
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Find all config.json files in the informer directory and its subdirectories
    config_files = glob.glob(os.path.join(base_dir, "informer", "*", "config.json"))
    config_files.append(os.path.join(base_dir, "informer", "config.json"))
    
    for config_file in config_files:
        print(f"Processing: {config_file}")
        try:
            # Read the current config
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            # Simplify it
            simplified_config = simplify_config(config)
            
            # Write back
            with open(config_file, 'w') as f:
                json.dump(simplified_config, f, indent=2)
                
            print(f"✅ Updated: {config_file}")
        except Exception as e:
            print(f"❌ Error processing {config_file}: {e}")

if __name__ == "__main__":
    process_configs()
    print("Config simplification completed.")
