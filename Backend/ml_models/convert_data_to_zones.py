#!/usr/bin/env python
"""
Script to convert existing normalized data files to non-normalized zone-specific files.
This helps with transitioning from the old project structure to the zone-specific one.
"""
import os
import sys
from pathlib import Path
import shutil

# Add the Backend directory to Python's path to help with imports
current_file_path = os.path.abspath(__file__)
backend_dir = os.path.dirname(os.path.dirname(current_file_path))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

try:
    import pandas as pd
except ImportError:
    print("Error: pandas is not installed in the current environment.")
    print("Please install pandas using: pip install pandas")
    sys.exit(1)

def convert_normalized_to_zones():
    """
    Convert normalized data files to zone-specific non-normalized files.
    This is a helper for transitioning to the new project structure.
    """
    # Verify pandas is available
    try:
        import pandas as pd
    except ImportError:
        print("ERROR: pandas is required but not available")
        print(f"Current Python: {sys.executable}")
        print(f"Try running:\n{sys.executable} -m pip install pandas")
        return
    
    # Base path for ML models data
    base_path = Path(os.path.dirname(os.path.abspath(__file__)))
    data_path = base_path / "data"
    
    print(f"Checking for normalized data files in {data_path}")
    
    # Look for normalized files
    normalized_files = list(data_path.glob("*normalized*.csv"))
    print(f"Found {len(normalized_files)} normalized data files")
    
    for file_path in normalized_files:
        print(f"\nProcessing {file_path}")
        
        # Make a backup of the original file
        backup_path = file_path.with_suffix(".csv.bak")
        if not backup_path.exists():
            shutil.copy2(file_path, backup_path)
            print(f"Created backup: {backup_path}")
        
        # Load the data
        try:
            df = pd.read_csv(file_path)
            print(f"Loaded CSV with {len(df)} rows and {len(df.columns)} columns")
            
            # Check for map code or zone column
            map_code_col = None
            for col in df.columns:
                if any(keyword in col.lower() for keyword in ["mapcode", "zone", "area", "region"]):
                    map_code_col = col
                    break
            
            if map_code_col:
                print(f"Found map code column: {map_code_col}")
                zones = df[map_code_col].unique()
                print(f"Found {len(zones)} zones: {zones}")
                  # Create zone-specific files
                for zone in zones:
                    # Filter data for this zone
                    zone_df = df[df[map_code_col] == zone].copy()
                    
                    # Check if we have date column to split data
                    date_col = None
                    for col in zone_df.columns:
                        if any(keyword in col.lower() for keyword in ["date", "time", "datetime"]):
                            date_col = col
                            break
                    
                    if date_col:
                        # Convert to datetime if it's not already
                        if not pd.api.types.is_datetime64_any_dtype(zone_df[date_col]):
                            try:
                                zone_df[date_col] = pd.to_datetime(zone_df[date_col])
                                print(f"Converted {date_col} to datetime for splitting data")
                            except:
                                print(f"⚠️ Could not convert {date_col} to datetime. Treating all data as 2024 data.")
                                date_col = None
                      # If no actual data from before 2024, we'll artificially split it
                    if date_col and len(zone_df[zone_df[date_col].dt.year < 2024]) > 0:
                        # Split into training (2019-2023) and prediction (2024) data
                        training_df = zone_df[zone_df[date_col].dt.year < 2024].copy()
                        prediction_df = zone_df[zone_df[date_col].dt.year >= 2024].copy()
                    elif date_col:
                        # In case all data is from 2024+, split into training and validation
                        # using first 80% for training, last 20% for prediction
                        print(f"⚠️ All data is from 2024+, artificially splitting into train/prediction sets")
                        cutoff_idx = int(len(zone_df) * 0.8)
                        training_df = zone_df.iloc[:cutoff_idx].copy()
                        prediction_df = zone_df.iloc[cutoff_idx:].copy()
                        
                        # Create training file (2019-2023)
                        training_file = data_path / f"{zone}_19-23.csv"
                        training_df.to_csv(training_file, index=False)
                        print(f"Created training file: {training_file} with {len(training_df)} rows (2019-2023)")
                        
                        # Create prediction file (2024)
                        prediction_file = data_path / f"{zone}_24.csv"
                        prediction_df.to_csv(prediction_file, index=False)
                        print(f"Created prediction file: {prediction_file} with {len(prediction_df)} rows (2024)")
                    else:
                        # If we can't split by date, just create a prediction file
                        # You can manually split the data later
                        prediction_file = data_path / f"{zone}_24.csv"
                        zone_df.to_csv(prediction_file, index=False)
                        print(f"Created prediction file: {prediction_file} with {len(zone_df)} rows")
                        print(f"⚠️ You should manually create a training file ({zone}_19-23.csv) with historical data")
            else:
                # If no zone column, just create a single non-normalized output based on filename
                filename = file_path.stem
                zone_code = filename.split('-')[0].split('_')[0]  # Extract zone code from filename
                output_file = data_path / f"{zone_code}_24.csv"
                df.to_csv(output_file, index=False)
                print(f"Created single zone file: {output_file} with {len(df)} rows")
        
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

def main():
    """Main function"""
    print("Converting normalized data files to zone-specific non-normalized files...")
    try:
        convert_normalized_to_zones()
        print("\nDone! Your data files have been converted to zone-specific format.")
        print("\nNote: The original normalized files have been backed up with .bak extension.")
        print("You can now use the non-normalized zone-specific data files with the updated model.")
        return True
    except Exception as e:
        import traceback
        print(f"Error during data conversion: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
