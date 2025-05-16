#!/usr/bin/env python
"""
Clean up original data files after verifying that the new structure works.
This script should only be run after confirming that the new data structure
is working correctly for all zones.
"""
import os
import glob
from pathlib import Path

def clean_up_original_files():
    """Remove the original zone-specific files from the data folder."""
    # Get base paths
    base_path = Path(os.path.dirname(os.path.abspath(__file__)))
    data_path = base_path / "data"
    
    # Ask for confirmation
    print("⚠️ WARNING: This script will remove all original zone-specific files from the data folder.")
    print("Before running this, make sure that you've verified the new folder structure works correctly.")
    confirm = input("Are you sure you want to proceed? (yes/no): ")
    
    if confirm.lower() != "yes":
        print("❌ Operation cancelled.")
        return
    
    # Get all zone-specific files using patterns
    training_files = list(data_path.glob("*_19-23.csv"))
    prediction_files = list(data_path.glob("*_24.csv"))
    normalized_files = list(data_path.glob("*_24-normalized.csv"))
    
    all_files = training_files + prediction_files + normalized_files
    
    print(f"🔍 Found {len(all_files)} files to delete:")
    for file in all_files:
        print(f"  - {file}")
    
    # Ask for final confirmation
    confirm = input("Delete these files? (yes/no): ")
    
    if confirm.lower() != "yes":
        print("❌ Operation cancelled.")
        return
    
    # Delete files
    for file in all_files:
        try:
            os.remove(file)
            print(f"✅ Deleted {file}")
        except Exception as e:
            print(f"❌ Error deleting {file}: {e}")
    
    print("\n✅ Clean-up complete!")

if __name__ == "__main__":
    clean_up_original_files()
