import os
import pandas as pd
import numpy as np
import joblib
import pathlib
from typing import Dict, Any, Optional, Union, List, Tuple

from interfaces.ModelPipelineInterface import IModelPipeline
from ml_models.XGBoost.XGBoost_model import XGBoostRegimeModel

class XGBoostPipeline(IModelPipeline):
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
        self.model_trainer = None  # For training interface
        
        # Set up paths
        current_file = pathlib.Path(__file__)
        self.project_root = current_file.parent.parent  # ml_models directory
        self.data_dir = self.project_root / "data" / self.mapcode
        self.xgboost_dir = self.data_dir / "xgboost"
    
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess data for XGBoost model input.
        Creates essential features required for prediction.
        
        Args:
            data: Raw input data
            
        Returns:
            Preprocessed data ready for model input
        """
        # Create a copy to avoid modifying the original data
        df = data.copy()
        
        # Check for essential columns
        if 'hour' not in df.columns:
            raise ValueError("Required column 'hour' not found in data - needed for cyclical encoding")
        
        # Create cyclical time features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # Create price features if possible, needed for regime split
        if 'Electricity_price_MWh' in df.columns:
            df['price_roll_3h'] = df['Electricity_price_MWh'].shift(1).rolling(3, min_periods=1).mean()
            df['EP_lag_1'] = df['Electricity_price_MWh'].shift(1)
            
        # Create load features if possible
        load_columns = ['DAHTL_TotalLoadValue', 'TotalLoadValue', 'Load_Value', 'Total_Load', 'DALoadValue', 'SystemLoad']
        load_col = next((col for col in load_columns if col in df.columns), None)
        if load_col:
            df['load_roll_6h'] = df[load_col].shift(1).rolling(6, min_periods=1).mean()
          # Create gas-load interaction only if both components are available
        gas_columns = ['Natural_Gas_price_EUR', 'Gas_Price']
        gas_col = next((col for col in gas_columns if col in df.columns), None)
        if gas_col and load_col:
            print(f"Creating gas-load interaction using {gas_col} and {load_col}")
            df['gas_x_load'] = df[gas_col] * df[load_col]        # Add required fossil columns and FTC columns
        required_fossil_cols = [
            'Fossil_Hard_coal_Capacity',
            'Fossil_Hard_coal_Output',
            'Fossil_Hard_coal_Utilization'
        ]
        
        for col in required_fossil_cols:
            if col not in df.columns:
                print(f"Warning: Adding missing column {col} with zeros")
                df[col] = 0

        # Handle Flow Transfer Capacity (FTC) columns
        ftc_cols = [col for col in df.columns if col.startswith('ftc_')]
        if not ftc_cols:
            print(f"Warning: No FTC columns found in data for {self.mapcode}")
            # Add essential FTC columns with zeros based on country
            if self.mapcode in ["DK1", "DK2"]:
                df['ftc_DE_LU'] = 0
        
        # Drop any rows with NaN values
        before_rows = len(df)
        df = df.dropna()
        after_rows = len(df)
        
        if before_rows > after_rows:
            print(f"Dropped {before_rows - after_rows} rows with NaN values")
            
        return df
    
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
        xgboost_dir = pathlib.Path(model_path) if model_path else self.xgboost_dir
        
        try:
            normal_model_path = str(xgboost_dir / "model_normal.pkl")
            spike_model_path = str(xgboost_dir / "model_spike.pkl")
            
            # Check if model files exist
            if not os.path.exists(normal_model_path):
                raise FileNotFoundError(f"Normal model file not found: {normal_model_path}")
            
            if not os.path.exists(spike_model_path):
                raise FileNotFoundError(f"Spike model file not found: {spike_model_path}")
            
            # Load the models
            self.model_normal = joblib.load(normal_model_path)
            self.model_spike = joblib.load(spike_model_path)
            
            # Load or set threshold
            threshold_path = str(xgboost_dir / "threshold.txt")
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
        # df = self.preprocess(df)
        
        # Filter by date if provided
        if date_str:
            df = df.loc[df['date'].dt.strftime('%Y-%m-%d') == date_str].copy()
            
            if df.empty:
                raise ValueError(f"No data found for date: {date_str}")
                
        # Simply use all columns except date and target variable, matching training code
        base_features = [c for c in df.columns if c not in ['date', 'Electricity_price_MWh']]
        
        X = df[base_features]
        
        # Make predictions
        df['Predicted'] = self.predict(X)
        df['True'] = df['Electricity_price_MWh']
        df['Pct_of_True'] = df['Predicted'] / df['True'] * 100
        
        return df[['date', 'hour', 'True', 'Predicted', 'Pct_of_True']]
    
    def train_model(self, train_file: str, xgb_params: Optional[Dict] = None) -> bool:
        """
        Train the XGBoost model using the specified data file.
        
        Args:
            train_file: Path to training data CSV file
            xgb_params: XGBoost parameters (optional)
            
        Returns:
            True if training completed successfully, False otherwise
        """
        try:
            # Initialize model trainer if not already done
            if self.model_trainer is None:
                self.model_trainer = XGBoostRegimeModel(mapcode=self.mapcode)
            
            # Train the model
            results = self.model_trainer.train(
                training_file=train_file,
                xgb_params=xgb_params
            )
            
            if results:
                print(f"XGBoost model training completed for {self.mapcode}")
                # Load the trained models into pipeline
                self.load_model()
                return True
            else:
                print(f"XGBoost model training failed for {self.mapcode}")
                return False
                
        except Exception as e:
            print(f"Error training XGBoost model: {e}")
            return False
    
    def evaluate_model(self, test_file: str) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            test_file: Path to test data CSV file
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model_normal is None or self.model_spike is None:
            raise ValueError("Models not loaded. Call load_model() first.")
        
        try:
            # Use the model trainer for evaluation
            if self.model_trainer is None:
                self.model_trainer = XGBoostRegimeModel(mapcode=self.mapcode)
                self.model_trainer.load_models()
            
            # Make predictions on test file
            results = self.model_trainer.predict(
                forecast_file=test_file
            )
            
            # Calculate metrics
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            
            true_values = results['True']
            predictions = results['Predicted']
            
            mse = mean_squared_error(true_values, predictions)
            mae = mean_absolute_error(true_values, predictions)
            r2 = r2_score(true_values, predictions)
            
            return {
                "mse": float(mse),
                "mae": float(mae),
                "r2": float(r2),
                "rmse": float(np.sqrt(mse))
            }
            
        except Exception as e:
            print(f"Error evaluating XGBoost model: {e}")
            return {}
