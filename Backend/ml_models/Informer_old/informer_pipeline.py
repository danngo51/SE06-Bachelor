import os
import pandas as pd
import numpy as np
import torch
import pathlib
from typing import Dict, Any, Optional, Union, List, Tuple

from ml_models.base_pipeline import ModelPipeline

class InformerPipeline(ModelPipeline):
    """
    Pipeline for the Informer model.
    Handles preprocessing, model loading, and prediction.
    
    The Informer is an efficient transformer-based model for long sequence
    time-series forecasting, which outperforms standard transformers with
    lower time complexity and memory usage.
    """
    
    def __init__(self, mapcode: str = "DK1", seq_len: int = 96, pred_len: int = 24):
        """
        Initialize the Informer pipeline.
        
        Args:
            mapcode: String code for the market area (e.g., "DK1")
            seq_len: Length of input sequence
            pred_len: Length of prediction sequence (usually 24 for day-ahead forecasting)
        """
        self.mapcode = mapcode
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.model = None
        self.scaler = None
        
        # Set up paths
        current_file = pathlib.Path(__file__)
        self.project_root = current_file.parent.parent  # ml_models directory
        self.data_dir = self.project_root / "data" / self.mapcode
        self.informer_dir = self.data_dir / "informer"
    
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess data for Informer model input.
        
        Args:
            data: Raw input data
            
        Returns:
            Preprocessed data ready for model input
        """
        # Create a copy to avoid modifying the original data
        df = data.copy()
        
        # Ensure date column is datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            
            # Extract temporal features
            df['day_of_week'] = df['date'].dt.dayofweek
            df['month'] = df['date'].dt.month
            df['day'] = df['date'].dt.day
            
            # Cyclical encoding
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Calculate rolling statistics
        df['price_roll_24h'] = df['Electricity_price_MWh'].shift(1).rolling(24, min_periods=1).mean()
        df['price_roll_7d'] = df['Electricity_price_MWh'].shift(1).rolling(168, min_periods=1).mean()
        df['price_std_24h'] = df['Electricity_price_MWh'].shift(1).rolling(24, min_periods=1).std()
        
        # Load-related features
        df['load_roll_24h'] = df['DAHTL_TotalLoadValue'].shift(1).rolling(24, min_periods=1).mean()
        
        # Create lagged features - Informer works well with longer context
        for lag in [1, 24, 48, 72, 168]:  # 1h, 1d, 2d, 3d, 7d
            df[f'price_lag_{lag}'] = df['Electricity_price_MWh'].shift(lag)
        
        # Drop rows with NaN values
        return df.dropna()
    
    def _prepare_informer_data(self, data: pd.DataFrame) -> torch.Tensor:
        """
        Prepare data in the format required by Informer model.
        
        Args:
            data: Preprocessed input data
            
        Returns:
            Tensor with data in Informer format
        """
        # Select all features except target and date
        feature_cols = [col for col in data.columns if col not in ['date', 'Electricity_price_MWh']]
        
        # Normalize data
        if self.scaler is not None:
            scaled_data = self.scaler.transform(data[feature_cols])
        else:
            # If no scaler is available, just standardize with mean/std
            scaled_data = (data[feature_cols] - data[feature_cols].mean()) / data[feature_cols].std()
            scaled_data = scaled_data.values
        
        # Reshape data for Informer: [batch_size, seq_len, num_features]
        # For inference, we usually use a batch size of 1
        batch_data = torch.tensor(scaled_data[-self.seq_len:], dtype=torch.float32).unsqueeze(0)
        
        return batch_data
    
    def predict(self, data: pd.DataFrame) -> pd.Series:
        """
        Make predictions using the Informer model.
        
        Args:
            data: Preprocessed input data
            
        Returns:
            Series with predictions
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Prepare feature data
        informer_tensor = self._prepare_informer_data(data)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Make predictions
        with torch.no_grad():
            predictions = self.model(informer_tensor)
        
        # Convert to numpy and scale back if needed
        predictions_np = predictions.squeeze().numpy()
        
        # Create index for predictions (usually the next 24 hours after the input data)
        if 'date' in data.columns and 'hour' in data.columns:
            last_date = data['date'].iloc[-1]
            last_hour = data['hour'].iloc[-1]
            
            # Create prediction dates and hours
            pred_dates = []
            pred_hours = []
            
            for i in range(self.pred_len):
                next_hour = (last_hour + i + 1) % 24
                days_to_add = (last_hour + i + 1) // 24
                next_date = last_date + pd.Timedelta(days=days_to_add)
                
                pred_dates.append(next_date)
                pred_hours.append(next_hour)
            
            # Create a DataFrame for predictions
            pred_df = pd.DataFrame({
                'date': pred_dates,
                'hour': pred_hours,
                'Predicted': predictions_np
            })
            
            return pred_df.set_index(['date', 'hour'])['Predicted']
        else:
            # If no date/hour information, just return as a series
            return pd.Series(predictions_np)
    
    def load_model(self, model_path: Optional[str] = None) -> None:
        """
        Load Informer model from disk.
        
        Args:
            model_path: Path to saved model directory (optional, uses default path if None)
        """
        # Use provided path or default
        informer_dir = pathlib.Path(model_path) if model_path else self.informer_dir
        
        try:
            model_path = str(informer_dir / "informer_model.pt")
            scaler_path = str(informer_dir / "scaler.pkl")
            
            # Check if model file exists
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Informer model file not found: {model_path}")
            
            # Load the model
            self.model = torch.load(model_path)
            
            # Load scaler if available
            if os.path.exists(scaler_path):
                import joblib
                self.scaler = joblib.load(scaler_path)
            
            print(f"Informer model loaded successfully from {model_path}")
            return True
            
        except Exception as e:
            print(f"Error loading Informer model: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_type": "Informer",
            "mapcode": self.mapcode,
            "seq_len": self.seq_len,
            "pred_len": self.pred_len,
            "model_loaded": self.model is not None,
            "scaler_loaded": self.scaler is not None
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
        
        # For Informer, we need historical data to make predictions
        # If date_str is provided, we need data up to that date
        if date_str:
            # Find the position of the target date
            target_date = pd.to_datetime(date_str)
            target_indices = df.index[df['date'] == target_date]
            
            if len(target_indices) == 0:
                raise ValueError(f"No data found for date: {date_str}")
            
            # Get historical data up to the target date
            historical_data = df.loc[:target_indices[-1]]
            
            # Make predictions
            predictions = self.predict(historical_data)
            
            # Create result DataFrame
            result_df = pd.DataFrame(predictions).reset_index()
            result_df = result_df.rename(columns={0: 'Predicted'})
            
            # Filter to just the target date if needed
            if 'date' in result_df.columns:
                result_df = result_df[result_df['date'] == target_date]
            
            # Add True values if available
            true_values = df.loc[df['date'] == target_date, 'Electricity_price_MWh']
            if not true_values.empty:
                result_df['True'] = true_values.values
                result_df['Pct_of_True'] = result_df['Predicted'] / result_df['True'] * 100
            
            return result_df
        else:
            # Make predictions using all available data
            predictions = self.predict(df)
            
            # Create result DataFrame
            result_df = pd.DataFrame(predictions).reset_index()
            result_df = result_df.rename(columns={0: 'Predicted'})
            
            return result_df
