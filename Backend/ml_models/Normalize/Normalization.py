import pandas as pd
import os
import pathlib
import sys
from typing import List, Optional, Tuple, Dict, Any, Union


# Default keywords to normalize
DEFAULT_KEYWORDS = [
    "ep", "dahtl", "atl", "temperature", "wind", "cloudcover", "shortwave",
    "ftc_", "Output", "Capacity", "Utilization", "Natural_Gas_", 
    "Coal_", "Oil_", "Carbon_Emission_", "Price[Currency/MWh]"
]


def get_data_dir() -> pathlib.Path:
    """Get the path to the centralized data directory"""
    current_file = pathlib.Path(__file__).resolve()
    data_dir = current_file.parent.parent / "Normalize"
    
    # Create required directories
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(data_dir / "input", exist_ok=True)
    os.makedirs(data_dir / "output", exist_ok=True)
    
    return data_dir


def normalize_dataset(
    input_csv: str,
    output_csv: Optional[str] = None,
    minmax_output_csv: Optional[str] = None,
    buffer_percentage: float = 0.5,
    keywords_to_normalize: List[str] = None
) -> Tuple[Optional[pd.DataFrame], Optional[List[Dict[str, Any]]]]:
    """
    Normalizes a dataset with zero-preservation and buffering.
    
    Args:
        input_csv: Path to the input CSV file
        output_csv: Path to save the normalized dataset. If None, doesn't save.
        minmax_output_csv: Path to save min/max values. If None, doesn't save.
        buffer_percentage: Buffer percentage to add to min/max (default: 0.5 or 50%)
        keywords_to_normalize: List of keywords to identify columns to normalize
    
    Returns:
        tuple: (normalized_dataframe, minmax_records) or (None, None) if error
    """
    print("[INFO] Starting normalization with zero-preservation and buffering...")
    
    if keywords_to_normalize is None:
        keywords_to_normalize = DEFAULT_KEYWORDS
    
    # Load dataset
    try:
        df = pd.read_csv(input_csv)
        print(f"[INFO] Loaded dataset: {input_csv} with shape {df.shape}")
    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {e}")
        return None, None
    
    # Determine which columns to normalize
    normalize_cols = [col for col in df.columns if any(keyword in col for keyword in keywords_to_normalize)]
    print(f"[INFO] Found {len(normalize_cols)} columns to normalize")
    
    # Normalize columns and collect minmax values
    minmax_records = []
    for col in normalize_cols:
        if col not in df.columns:
            continue
            
        col_values = df[col]
        
        # Skip constant columns or columns with only zeros
        if col_values.nunique() <= 1:
            print(f"[WARN] Skipping column '{col}' (constant values)")
            continue
            
        zero_mask = (col_values == 0)
        non_zero_values = col_values[~zero_mask]
        
        if non_zero_values.empty:
            print(f"[WARN] Skipping column '{col}' (all zeros)")
            continue
        
        # Calculate min/max with buffer
        orig_min = non_zero_values.min()
        orig_max = non_zero_values.max()
        orig_range = orig_max - orig_min
        
        buffer = orig_range * buffer_percentage
        buffered_min = orig_min - buffer
        buffered_max = orig_max + buffer
        buffered_range = buffered_max - buffered_min
        
        # Record minmax values
        minmax_records.append({
            "Column": col,
            "Min": buffered_min,
            "Max": buffered_max,
            "Scale": buffered_range
        })
        
        # Determine feature range based on values
        feature_range = (-1, 1) if orig_min < 0 else (0, 1)
        
        # Scale non-zero values
        scaled_non_zero = (non_zero_values - buffered_min) / buffered_range
        if feature_range == (-1, 1):
            scaled_non_zero = scaled_non_zero * 2 - 1
        
        # Apply scaling back to original column
        new_col = col_values.copy()
        new_col.loc[~zero_mask] = scaled_non_zero
        df[col] = new_col
    
    # Final cleanup of dataframe
    df.fillna(0, inplace=True)
    
    # Move Price column to the end if present
    if "Price[Currency/MWh]" in df.columns:
        price_col = df.pop("Price[Currency/MWh]")
        df["Price[Currency/MWh]"] = price_col
    
    # Save outputs if paths provided
    if output_csv:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"[INFO] Normalized dataset saved to: {output_csv}")
    
    if minmax_output_csv and minmax_records:
        os.makedirs(os.path.dirname(minmax_output_csv), exist_ok=True)
        pd.DataFrame(minmax_records).to_csv(minmax_output_csv, index=False)
        print(f"[INFO] Min/max values saved to: {minmax_output_csv}")
    
    return df, minmax_records


def normalize_with_existing_minmax(
    input_csv: str,
    minmax_csv: str,
    output_csv: Optional[str] = None,
    keywords_to_normalize: Optional[List[str]] = None
) -> Optional[pd.DataFrame]:
    """
    Normalizes a dataset using existing min/max values from a CSV file.
    
    Args:
        input_csv: Path to the input CSV file to normalize
        minmax_csv: Path to the CSV file containing min/max values
        output_csv: Path to save the normalized dataset. If None, doesn't save.
        keywords_to_normalize: List of columns to normalize.
            If None, uses all columns in minmax_csv.
    
    Returns:
        pd.DataFrame: Normalized dataframe or None if error
    """
    print("[INFO] Starting normalization with existing min/max values...")
    
    # Load files
    try:
        df = pd.read_csv(input_csv)
        minmax_df = pd.read_csv(minmax_csv)
        print(f"[INFO] Loaded dataset: {input_csv} and minmax values: {minmax_csv}")
    except Exception as e:
        print(f"[ERROR] Failed to load data: {e}")
        return None
    
    # Determine columns to normalize
    if keywords_to_normalize is None:
        normalize_cols = minmax_df["Column"].tolist()
    else:
        normalize_cols = [
            col for col in df.columns 
            if any(keyword in col for keyword in keywords_to_normalize)
            and col in minmax_df["Column"].values
        ]
    
    print(f"[INFO] Normalizing {len(normalize_cols)} columns")
    
    # Apply normalization to each column
    for col in normalize_cols:
        if col not in df.columns:
            continue
            
        # Get minmax values
        minmax_row = minmax_df[minmax_df["Column"] == col]
        if minmax_row.empty:
            print(f"[WARN] No min/max values found for '{col}'")
            continue
            
        min_val = minmax_row["Min"].values[0]
        max_val = minmax_row["Max"].values[0]
        
        # Process column values
        col_values = df[col]
        zero_mask = (col_values == 0)
        non_zero_values = col_values[~zero_mask]
        
        if non_zero_values.empty:
            print(f"[WARN] Column '{col}' has only zeros")
            continue
        
        # Check for valid range
        range_val = max_val - min_val
        if range_val == 0:
            print(f"[WARN] Min and max are identical for '{col}'")
            continue
            
        # Apply scaling
        feature_range = (-1, 1) if min_val < 0 else (0, 1)
        scaled_non_zero = (non_zero_values - min_val) / range_val
        if feature_range == (-1, 1):
            scaled_non_zero = scaled_non_zero * 2 - 1
        
        # Update column
        new_col = col_values.copy()
        new_col.loc[~zero_mask] = scaled_non_zero
        df[col] = new_col
    
    # Final cleanup
    df.fillna(0, inplace=True)
    
    # Save if output path provided
    if output_csv:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"[INFO] Normalized dataset saved to: {output_csv}")
    
    return df


class Normalizer:
    """Class for normalizing datasets with consistent directory structure"""
    
    def __init__(self):
        self.data_dir = get_data_dir()
    
    def normalize(self, 
                 input_folder: str = "input",
                 output_folder: str = "output",
                 input_filename: str = "DK1-24.csv",
                 output_filename: str = None,
                 minmax_filename: str = None) -> Tuple[Optional[pd.DataFrame], Optional[List[Dict[str, Any]]]]:
        """
        Normalizes data using calculated minmax values
        
        Args:
            input_folder: Name of input folder within data directory
            output_folder: Name of output folder within data directory
            input_filename: Name of input CSV file
            output_filename: Name of output CSV file (defaults to input name with -normalized)
            minmax_filename: Name of minmax values CSV file (defaults to input name with -minmax)
        
        Returns:
            tuple: (normalized_dataframe, minmax_records) or (None, None) if error
        """
        # Set default filenames if not provided
        if output_filename is None:
            output_filename = input_filename.replace(".csv", "-normalized.csv")
        if minmax_filename is None:
            minmax_filename = input_filename.replace(".csv", "-minmax.csv")
            
        # Set up file paths
        input_dir = self.data_dir / input_folder
        output_dir = self.data_dir / output_folder
        input_csv = input_dir / input_filename
        output_csv = output_dir / output_filename
        minmax_csv = output_dir / minmax_filename
        
        # Ensure directories exist
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        
        # Check if input file exists
        if not os.path.exists(input_csv):
            print(f"[ERROR] Input file not found: {input_csv}")
            return None, None
        
        # Call the normalization function
        return normalize_dataset(
            input_csv=str(input_csv),
            output_csv=str(output_csv),
            minmax_output_csv=str(minmax_csv)
        )
    
    def normalize_with_existing(self,
                              input_folder: str = "input",
                              output_folder: str = "output",
                              input_filename: str = "DK1-24.csv",
                              output_filename: str = None,
                              minmax_filename: str = "DK1-24-minmax.csv",
                              keywords_to_normalize: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
        """
        Normalizes data using existing minmax values
        
        Args:
            input_folder: Name of input folder within data directory
            output_folder: Name of output folder within data directory
            input_filename: Name of input CSV file
            output_filename: Name of output CSV file (defaults to input name with -normalized)
            minmax_filename: Name of minmax values CSV file to use
            keywords_to_normalize: List of columns to normalize
        
        Returns:
            pd.DataFrame: Normalized dataframe or None if error
        """
        # Set default output filename if not provided
        if output_filename is None:
            output_filename = input_filename.replace(".csv", "-normalized.csv")
            
        # Set up file paths
        input_dir = self.data_dir / input_folder
        output_dir = self.data_dir / output_folder
        input_csv = input_dir / input_filename
        output_csv = output_dir / output_filename
        minmax_csv = output_dir / minmax_filename
        
        # Ensure directories exist
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        
        # Check if required files exist
        if not os.path.exists(input_csv):
            print(f"[ERROR] Input file not found: {input_csv}")
            return None
            
        if not os.path.exists(minmax_csv):
            print(f"[ERROR] Minmax file not found: {minmax_csv}")
            return None
        
        # Call normalization with existing minmax
        return normalize_with_existing_minmax(
            input_csv=str(input_csv),
            minmax_csv=str(minmax_csv),
            output_csv=str(output_csv),
            keywords_to_normalize=keywords_to_normalize
        )


def parse_args():
    """Parse command line arguments"""
    args = sys.argv[1:]
    params = {
        'use_existing': False,
        'minmax_file': None,
        'input_filename': "DK1-24.csv"  # default
    }
    
    # Check for --use-existing-minmax flag
    if "--existing-minmax" in args:
        params['use_existing'] = True
        idx = args.index("--existing-minmax")
        if idx + 1 < len(args):
            params['minmax_file'] = args[idx + 1]
            args = args[:idx] + args[idx+2:]  # Remove flag and value
        else:
            print("[ERROR] --existing-minmax requires a filename argument")
            sys.exit(1)
    
    # Get input filename (first remaining argument)
    if args:
        params['input_filename'] = args[0]
    
    return params


def main():
    """Main function for command line usage"""
    # Show available input files
    data_dir = get_data_dir()
    input_dir = data_dir / "input"
    
    print(f"Available files in {input_dir}:")
    try:
        files = os.listdir(input_dir)
        for file in files:
            print(f"  - {file}")
        if not files:
            print("  No files found")
    except Exception as e:
        print(f"  Error listing files: {e}")
    
    # Print usage instructions
    print("\nUsage:")
    print("  python Normalization.py [input_file.csv] [options]")
    print("Options:")
    print("  --existing-minmax MINMAX_FILE  Use existing minmax values")
    
    # Parse arguments
    params = parse_args()
    input_filename = params['input_filename']
    output_filename = input_filename.replace(".csv", "-normalized.csv")
    
    print(f"Using input file: {input_filename}")
    
    # Run normalization
    normalizer = Normalizer()
    
    if params['use_existing']:
        minmax_filename = params['minmax_file']
        print(f"Using existing minmax values from: {minmax_filename}")
        
        df = normalizer.normalize_with_existing(
            input_filename=input_filename,
            output_filename=output_filename,
            minmax_filename=minmax_filename
        )
        
        if df is not None:
            print(f"[SUCCESS] Normalization with existing minmax completed. Shape: {df.shape}")
            print(f"[INFO] Normalized data saved to: {data_dir}/output/{output_filename}")
    else:
        minmax_filename = input_filename.replace(".csv", "-minmax.csv") 
        df, _ = normalizer.normalize(
            input_filename=input_filename,
            output_filename=output_filename,
            minmax_filename=minmax_filename
        )
        
        if df is not None:
            print(f"[SUCCESS] Normalization completed. Shape: {df.shape}")
            print(f"[INFO] Normalized data saved to: {data_dir}/output/{output_filename}")
            print(f"[INFO] Minmax values saved to: {data_dir}/output/{minmax_filename}")


if __name__ == "__main__":
    main()
