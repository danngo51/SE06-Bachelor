#!/usr/bin/env python3
"""
Script to train and save models for testing purposes.
Includes XGBoost, GRU, and Informer models.
"""

import os
import sys
import pathlib
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Add project root to path
project_root = pathlib.Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import the model classes
from ml_models.XGBoost.XGBoost_model import XGBoostRegimeModel
from ml_models.Informer.Informer_model import InformerModelTrainer
from ml_models.GRU.GRU_model import GRUModelTrainer

def train_xgboost(country_code):
    """Train and save XGBoost models for a specific country"""
    model = XGBoostRegimeModel(mapcode=country_code)
    training_file = f"{country_code}_full_data_2018_2024.csv"
    print(f"Training file: {model.data_dir / training_file}")
    
    if not os.path.exists(model.data_dir / training_file):
        print(f"Training file not found: {model.data_dir / training_file}")
        return False
    
    print(f"Training XGBoost models for {country_code}...")
    
    xgb_params = {
        'objective': 'reg:squarederror',
        'learning_rate': 0.1,
        'max_depth': 3,
        'n_estimators': 50,
        'random_state': 42
    }
    
    results = model.train(training_file=training_file, xgb_params=xgb_params)
    
    if results:
        print(f"✅ XGBoost training completed successfully for {country_code}")
        print(f"Normal regime - RMSE: {results['normal_rmse']:.2f}, R²: {results['normal_r2']:.4f}")
        print(f"Spike regime - RMSE: {results['spike_rmse']:.2f}, R²: {results['spike_r2']:.4f}")
        return True
    else:
        print(f"❌ XGBoost training failed for {country_code}")
        return False

def train_informer(country_code):
    """Train and save Informer model for a specific country"""
    trainer = InformerModelTrainer(
        mapcode=country_code,
        seq_len=168,        # 7 days of hourly data
        label_len=48,       # 2 days overlap for attention
        pred_len=24,        # 1 day ahead prediction
        batch_size=32,
        learning_rate=1e-4,
        epochs=50,          # Reduced for testing
        early_stop_patience=10
    )
    
    training_file = f"{country_code}_full_data_2018_2024.csv"
    data_path = trainer.data_dir / training_file
    print(f"Training file: {data_path}")
    
    if not os.path.exists(data_path):
        print(f"Training file not found: {data_path}")
        return False
    
    os.makedirs(trainer.informer_dir, exist_ok=True)
    
    print(f"Training Informer model for {country_code}...")
    print(f"Using device: {trainer.device}")
    
    try:
        results = trainer.train(data_path=data_path)
        
        if results:
            print(f"✅ Informer training completed successfully for {country_code}")
            print(f"MSE: {results['metrics']['mse']:.2f}, MAE: {results['metrics']['mae']:.2f}, R²: {results['metrics']['r2']:.4f}")
            print(f"Model saved to {results['model_path']}")
            return True
        else:
            print(f"❌ Informer training failed for {country_code}")
            return False
    except Exception as e:
        print(f"❌ Informer training failed with error: {str(e)}")
        return False

def train_gru(country_code):
    """Train and save GRU model for a specific country"""
    trainer = GRUModelTrainer(
        mapcode=country_code,
        seq_len=168,        # 7 days of hourly data
        pred_len=24         # 1 day ahead prediction
    )
    
    training_file = f"{country_code}_full_data_2018_2024.csv"
    data_path = trainer.data_dir / training_file
    print(f"Training file: {data_path}")
    
    if not os.path.exists(data_path):
        print(f"Training file not found: {data_path}")
        return False
    
    os.makedirs(trainer.gru_dir, exist_ok=True)
    
    print(f"Training GRU model for {country_code}...")
    print(f"Using device: {trainer.device}")
    
    try:
        # Set reduced parameters for testing
        trainer.num_epochs = 200  # Reduced for testing
        trainer.patience = 10    # Early stopping patience
        
        train_losses, val_losses = trainer.train(
            train_file=str(data_path),
            save_model=True
        )
        
        print(f"✅ GRU training completed successfully for {country_code}")
        print(f"Final training loss: {train_losses[-1]:.6f}")
        if val_losses:
            print(f"Final validation loss: {val_losses[-1]:.6f}")
        print(f"Model saved to {trainer.gru_dir}")
        return True
        
    except Exception as e:
        print(f"❌ GRU training failed with error: {str(e)}")
        return False

def train_models(country_code, model_type="all"):
    """Train and save models for a specific country
    
    Args:
        country_code (str): Country or market code (e.g., DK1)
        model_type (str): Type of model to train (all, xgboost, gru, informer)
        
    Returns:
        bool: True if training was successful, False otherwise
    """
    success = True
    
    if model_type.lower() in ["all", "xgboost"]:
        xgb_success = train_xgboost(country_code)
        success = success and xgb_success
    
    if model_type.lower() in ["all", "gru"]:
        gru_success = train_gru(country_code)
        success = success and gru_success
    
    if model_type.lower() in ["all", "informer"]:
        informer_success = train_informer(country_code)
        success = success and informer_success
    
    return success


if __name__ == "__main__":
    import argparse
    
    # Set up command line argument parser
    parser = argparse.ArgumentParser(description="Train ML models for electricity price forecasting")
    parser.add_argument("--country", "-c", nargs="+", default=["DK1"], 
                        help="One or more country/market codes (e.g., DK1 DK2 SE1)")
    parser.add_argument("--model", "-m", nargs="+", default=["all"],
                        help="One or more model types to train: 'all', 'xgboost', 'gru', 'informer'")
    
    args = parser.parse_args()
    
    # Process multiple countries and models
    overall_success = True
    for country_code in args.country:
        for model_type in args.model:
            print(f"\n===== Training {model_type} models for {country_code} =====")
            success = train_models(country_code, model_type)
            
            if success:
                print(f"✅ Training completed successfully for {country_code} - {model_type}")
            else:
                print(f"❌ Training failed for {country_code} - {model_type}")
                overall_success = False
    
    print("\n===== Overall Training Summary =====")
    if overall_success:
        print("✅ All model training tasks completed successfully")
    else:
        print("❌ Some model training tasks failed - check logs above for details")