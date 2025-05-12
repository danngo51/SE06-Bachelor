#!/usr/bin/env python
"""
Combined script to set up the new zone-specific structure with separate training and prediction datasets
"""
import os
import sys
import subprocess
from pathlib import Path

def run_setup_script(script_name, description):
    """Run a setup script and print its output"""
    print(f"\n{'='*80}")
    print(f"Running {script_name}: {description}")
    print(f"{'='*80}")
    
    script_path = Path(os.path.dirname(os.path.abspath(__file__))) / script_name
    if not script_path.exists():
        print(f"❌ Error: Script {script_path} not found")
        return False
    
    # Add backend directory to PYTHONPATH to help with imports
    env = os.environ.copy()
    backend_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent
    if 'PYTHONPATH' in env:
        env['PYTHONPATH'] = f"{str(backend_dir)}{os.pathsep}{env['PYTHONPATH']}"
    else:
        env['PYTHONPATH'] = str(backend_dir)
    
    result = subprocess.run(['python', str(script_path)], capture_output=True, text=True, env=env)
    print(result.stdout)
    
    if result.returncode != 0:
        print(f"❌ Error running {script_name}: {result.stderr}")
        return False
    
    return True

def main():
    """Main function to run all setup scripts"""
    print("🚀 Setting up new zone-specific structure with separate training and prediction datasets")
    
    # Run each setup script in sequence
    scripts = [
        ("setup_zones.py", "Setting up zone-specific directory structure"),
        ("update_configs.py", "Updating zone configs to use training data"),
        ("convert_data_to_zones.py", "Converting and splitting data into training/prediction sets")
    ]
    
    for script_name, description in scripts:
        success = run_setup_script(script_name, description)
        if not success:
            print(f"❌ Setup stopped due to error in {script_name}")
            return
    
    print("\n✅ All setup scripts completed successfully!")
    print("\n📋 Next steps:")
    print("1. Check that your data files were correctly split into training and prediction sets")
    print("   - Training files: *_19-23.csv (2019-2023 data for training)")
    print("   - Prediction files: *_24.csv (2024 data for validation/prediction)")
    print("2. Train your zone-specific models using:")
    print("   python train_zone_models.py --zone <ZONE_CODE> --model both")
    print("3. Test your zone-specific models using:")
    print("   python test_hybrid_model.py --zone <ZONE_CODE>")
    
    # List the available zones
    base_path = Path(os.path.dirname(os.path.abspath(__file__)))
    zone_paths = [d.name for d in (base_path / "informer").iterdir() 
                  if d.is_dir() and d.name != "informerModel" and d.name != "results"]
    
    print("\n🌍 Available zones:")
    for zone in sorted(zone_paths):
        print(f"  - {zone}")

if __name__ == "__main__":
    main()
