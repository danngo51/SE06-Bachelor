import os
import pandas as pd
import numpy as np
import joblib
import pathlib
from typing import Dict, Any, Optional, Union, List, Tuple

from ml_models.base_pipeline import ModelPipeline

class XGBoostPipeline(ModelPipeline):
    """
    Pipeline for the XGBoost regime-based model.
    Handles preprocessing, model loading, and prediction.
    """
    
    def __init__(self, mapcode: str = "DK1"):
        """
        Initialize the XGBoost pipeline.
        
        Args:
            mapcode: String code for the market area (e.g., "DK1")
        """
        self.mapcode = mapcode
        self.threshold = None
        self.model_normal = None
        self.model_spike = None
        
        # Set up paths
        current_file = pathlib.Path(__file__)
        self.project_root = current_file.parent.parent  # ml_models directory
        self.data_dir = self.project_root / "data" / self.mapcode
        self.regime_dir = self.data_dir / "regime_models"
    
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess data for XGBoost model input.
        
        Args:
            data: Raw input data
            
        Returns:
            Preprocessed data ready for model input
        """
        # Create a copy to avoid modifying the original data
        df = data.copy()
        
        # Cyclical encoding for hour
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # Rolling statistics
        df['price_roll_3h'] = df['Electricity_price_MWh'].shift(1).rolling(3, min_periods=1).mean()
        df['load_roll_6h'] = df['DAHTL_TotalLoadValue'].shift(1).rolling(6, min_periods=1).mean()
        
        # Interaction term
        df['gas_x_load'] = df['Natural_Gas_price_EUR'] * df['DAHTL_TotalLoadValue']
        
        # One-hour lag (needed for regime split)
        df['EP_lag_1'] = df['Electricity_price_MWh'].shift(1)
        
        # Drop rows with NaN values
        return df.dropna()
    
    def predict(self, data: pd.DataFrame) -> pd.Series:
        """
        Make predictions using the regime-based model.
        
        Args:
            data: Preprocessed input data
            
        Returns:
            Series with predictions
        """
        if self.model_normal is None or self.model_spike is None or self.threshold is None:
            raise ValueError("Models not loaded. Call load_model() first.")
        
        # Determine regime for each sample
        is_spike = data['EP_lag_1'] > self.threshold
        
        # Initialize prediction array with NaNs
        predictions = pd.Series(index=data.index, dtype=float)
        
        # Apply appropriate model based on regime
        if not is_spike.empty and is_spike.any():
            predictions.loc[is_spike] = self.model_spike.predict(data.loc[is_spike])
        
        if not is_spike.empty and (~is_spike).any():
            predictions.loc[~is_spike] = self.model_normal.predict(data.loc[~is_spike])
        
        return predictions
    
    def load_model(self, model_path: Optional[str] = None) -> None:
        """
        Load regime models from disk.
        
        Args:
            model_path: Path to saved model directory (optional, uses default path if None)
        """
        # Use provided path or default
        regime_dir = pathlib.Path(model_path) if model_path else self.regime_dir
        
        try:
            normal_model_path = str(regime_dir / "model_normal.pkl")
            spike_model_path = str(regime_dir / "model_spike.pkl")
            
            # Check if model files exist
            if not os.path.exists(normal_model_path):
                raise FileNotFoundError(f"Normal model file not found: {normal_model_path}")
            
            if not os.path.exists(spike_model_path):
                raise FileNotFoundError(f"Spike model file not found: {spike_model_path}")
            
            # Load the models
            self.model_normal = joblib.load(normal_model_path)
            self.model_spike = joblib.load(spike_model_path)
            
            # Load or set threshold
            threshold_path = str(regime_dir / "threshold.txt")
            if os.path.exists(threshold_path):
                with open(threshold_path, 'r') as f:
                    self.threshold = float(f.read().strip())
            else:
                # Default to 90th percentile if threshold file doesn't exist
                # Try to load training data
                training_data_path = self.data_dir / f"{self.mapcode}_full_data_2018_2024.csv"
                if os.path.exists(training_data_path):
                    dataset = pd.read_csv(training_data_path)
                    self.threshold = dataset['Electricity_price_MWh'].quantile(0.90)
                else:
                    # Fallback value if no data is available
                    self.threshold = 100.0
                    print(f"Warning: Using default threshold value of {self.threshold}")
            
            print(f"Models loaded successfully. Threshold: {self.threshold:.2f}")
            return True
            
        except Exception as e:
            print(f"Error loading models: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_type": "XGBoost Regime-Based",
            "mapcode": self.mapcode,
            "threshold": self.threshold,
            "normal_model_loaded": self.model_normal is not None,
            "spike_model_loaded": self.model_spike is not None
        }
    
    def predict_from_file(self, file_path: str, date_str: Optional[str] = None) -> pd.DataFrame:
        """
        Load data from file and make predictions for a specific date.
        
        Args:
            file_path: Path to CSV file with prediction data
            date_str: Date string in format 'YYYY-MM-DD' (optional)
            
        Returns:
            DataFrame with predictions
        """
        # Load data
        df = pd.read_csv(file_path, parse_dates=['date'])
        
        # Preprocess data
        df = self.preprocess(df)
        
        # Filter by date if provided
        if date_str:
            df = df.loc[df['date'].dt.strftime('%Y-%m-%d') == date_str].copy()
            
            if df.empty:
                raise ValueError(f"No data found for date: {date_str}")
        
        # Prepare feature matrix
        features = [c for c in df.columns if c not in ['date', 'Electricity_price_MWh']]
        X = df[features]
        
        # Make predictions
        df['Predicted'] = self.predict(X)
        df['True'] = df['Electricity_price_MWh']
        df['Pct_of_True'] = df['Predicted'] / df['True'] * 100
        
        return df[['date', 'hour', 'True', 'Predicted', 'Pct_of_True']]
