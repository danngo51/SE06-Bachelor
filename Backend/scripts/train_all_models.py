"""
Script to train all model types (XGBoost, GRU, Informer) for multiple regions.
"""

import os
import sys
import pathlib
from typing import List

# Add project root to path
project_root = pathlib.Path(__file__).parent.parent  # Go up one level from scripts folder
sys.path.append(str(project_root))

from train_models import train_models

def train_all_models_for_countries(country_codes: List[str]):
    """Train models for a list of countries"""
    for country_code in country_codes:
        print(f"\nTraining models for {country_code}...")
        success = train_models(country_code)
        
        if success:
            print(f"✅ Training completed successfully for {country_code}")
        else:
            print(f"⚠️ Some models failed to train for {country_code}")

if __name__ == "__main__":
    # Example: train models for DK1 and DK2
    train_all_models_for_countries(["DK1", "DK2"])
