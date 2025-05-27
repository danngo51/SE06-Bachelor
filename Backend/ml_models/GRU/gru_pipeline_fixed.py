"""
An updated GRU pipeline that handles different file formats
"""
import os
import pandas as pd
import numpy as np
import torch
import pathlib
import joblib
from typing import Dict, Any, Optional, Union, List, Tuple
from sklearn.preprocessing import StandardScaler

from ml_models.base_pipeline import ModelPipeline
from ml_models.GRU.GRU_model import GRUModel

class GRUPipeline(ModelPipeline):
    """
    Fixed GRU Pipeline with support for multiple file formats
    """
    
    def __init__(self, mapcode: str = "DK1", seq_len: int = 168, pred_len: int = 24):
        self.mapcode = mapcode
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.model = None
        self.scaler = None
        self.scaler_features = None
        self.scaler_target = None
        self.feature_cols = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Set up paths
        current_file = pathlib.Path(__file__)
        self.project_root = current_file.parent.parent  # ml_models directory
        self.data_dir = self.project_root / "data" / self.mapcode
        self.gru_dir = self.data_dir / "gru"
        
        # Model configuration
        self.input_dim = None
        self.hidden_dim = 128
        self.num_layers = 2
        self.bidirectional = True
        self.dropout = 0.2
        
    def load_model(self, model_path: Optional[str] = None) -> bool:
        """Load trained GRU model from disk."""
        try:
            # Use provided path or default
            model_dir = pathlib.Path(model_path) if model_path else self.gru_dir
            print(f"Looking for GRU model in {model_dir}")
            
            # List all files in the directory to aid debugging
            try:
                print(f"Files in {model_dir}:")
                for file in os.listdir(model_dir):
                    print(f"  - {file}")
            except Exception as e:
                print(f"Could not list directory contents: {e}")
            
            # Check for model file extensions
            model_file = model_dir / "gru_model.pt"
            if not model_file.exists():
                model_file = model_dir / "gru_model.pth"
                if not model_file.exists():
                    model_file = model_dir / "gru_model_complete.pth"
                    if not model_file.exists():
                        print(f"No GRU model file found in {model_dir}")
                        return False
            
            print(f"Using model file: {model_file}")
            
            # Load model configuration
            config_file = model_dir / "model_config.pkl"
            if config_file.exists():
                config = joblib.load(config_file)
                self.input_dim = config.get('input_dim', 64)
                self.hidden_dim = config.get('hidden_dim', self.hidden_dim)
                self.num_layers = config.get('num_layers', self.num_layers)
                self.bidirectional = config.get('bidirectional', self.bidirectional)
                self.dropout = config.get('dropout', self.dropout)
            # Initialize model
            if self.input_dim is None:
                self.input_dim = 64  # Default fallback
                
            self.model = GRUModel(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                output_dim=self.pred_len,
                num_layers=self.num_layers,
                bidirectional=self.bidirectional
                # Note: GRUModel doesn't accept dropout parameter
            )
            
            # Load model weights
            if str(model_file).endswith("complete.pth"):
                self.model = torch.load(model_file, map_location=self.device)
            else:
                state_dict = torch.load(model_file, map_location=self.device)
                self.model.load_state_dict(state_dict)
            
            self.model.to(self.device)
            self.model.eval()
            
            # Load scalers
            scaler_file = model_dir / "scaler.pkl"
            if scaler_file.exists():
                self.scaler = joblib.load(scaler_file)
            else:
                features_scaler_file = model_dir / "scaler_features.pkl"
                target_scaler_file = model_dir / "scaler_target.pkl"
                if features_scaler_file.exists() and target_scaler_file.exists():
                    print(f"Found separate feature and target scalers")
                    self.scaler_features = joblib.load(str(features_scaler_file))
                    self.scaler_target = joblib.load(str(target_scaler_file))
                    self.scaler = self.scaler_features  # Use features scaler as primary
            
            # Load feature columns
            features_file = model_dir / "feature_columns.pkl"
            if features_file.exists():
                self.feature_cols = joblib.load(features_file)
            
            print(f"GRU model loaded successfully from {model_dir}")
            return True
            
        except Exception as e:
            print(f"Error loading GRU model: {e}")
            import traceback
            traceback.print_exc()
            return False

    def get_model_info(self):
        """Get basic model information"""
        return {
            "model_type": "GRU (Gated Recurrent Unit)",
            "mapcode": self.mapcode,
            "seq_len": self.seq_len,
            "pred_len": self.pred_len,
            "model_loaded": self.model is not None,
            "scaler_loaded": self.scaler is not None
        }
        
    def preprocess(self, data: pd.DataFrame) -> np.ndarray:
        """
        Preprocess data for GRU model prediction.
        
        Args:
            data: DataFrame with input features
            
        Returns:
            Preprocessed data as numpy array
        """
        try:
            # Make a copy to avoid modifying the original
            df = data.copy()
            
            # Check if we have the expected columns
            price_col = 'Electricity_price_MWh'
            if price_col not in df.columns:
                print(f"Warning: {price_col} not found in input data. Using zeros.")
                df[price_col] = 0.0
            
            # Feature engineering - similar to what was used in training
            # Add time features - check if hour/date columns already exist
            if 'hour' not in df.columns and isinstance(df.index, pd.DatetimeIndex):
                df['hour'] = df.index.hour
                df['day'] = df.index.day
                df['month'] = df.index.month
                df['day_of_week'] = df.index.dayofweek
                df['weekend'] = df.index.dayofweek.isin([5, 6]).astype(int)
            elif 'hour' not in df.columns and 'date' in df.columns:
                # If we have a 'date' column but not datetime index
                if not pd.api.types.is_datetime64_any_dtype(df['date']):
                    df['date'] = pd.to_datetime(df['date'])
                df['hour'] = df['date'].dt.hour
                df['day'] = df['date'].dt.day
                df['month'] = df['date'].dt.month
                df['day_of_week'] = df['date'].dt.dayofweek
                df['weekend'] = df['date'].dt.dayofweek.isin([5, 6]).astype(int)
            
            # Add lag features if sufficient history is available
            if len(df) > 24:
                df['price_lag_24'] = df[price_col].shift(24)
                df['price_lag_48'] = df[price_col].shift(48)
                df['price_lag_168'] = df[price_col].shift(168)  # 1 week
            
            # Add rolling statistics if sufficient history
            if len(df) > 24:
                df['price_rolling_mean_24h'] = df[price_col].rolling(24).mean()
                df['price_rolling_std_24h'] = df[price_col].rolling(24).std()
            
            # Replace NaN values with 0 for lag and rolling features
            df = df.fillna(0)
            
            # Check if we have feature columns from training; if not, use all available
            if self.feature_cols is not None:
                # Check which features are available
                available_cols = [col for col in self.feature_cols if col in df.columns]
                missing_cols = [col for col in self.feature_cols if col not in df.columns]
                
                if missing_cols:
                    print(f"Warning: Missing {len(missing_cols)} features: {missing_cols[:5]}...")
                    # Add missing columns with zeros
                    for col in missing_cols:
                        df[col] = 0
                
                # Use same feature columns as in training
                features = df[self.feature_cols].values
                print(f"Using {len(self.feature_cols)} features from training")
            else:
                # If no feature columns specified, use all numeric columns except the target
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if price_col in numeric_cols:
                    numeric_cols.remove(price_col)
                features = df[numeric_cols].values
                print(f"Using {len(numeric_cols)} features based on input data")
            
            # Scale features
            if self.scaler_features is not None:
                features = self.scaler_features.transform(features)
            elif self.scaler is not None:
                features = self.scaler.transform(features)
            
            # Reshape for GRU [batch, seq_len, features]
            batch_size = 1  # For prediction we use batch_size=1
            seq_len = min(self.seq_len, features.shape[0])  # Use available history up to seq_len
            
            # If we don't have enough history, pad with zeros
            if seq_len < self.seq_len:
                print(f"Warning: Not enough history. Using {seq_len}/{self.seq_len} timesteps.")
                padding = np.zeros((self.seq_len - seq_len, features.shape[1]))
                features = np.vstack([padding, features[-seq_len:]])
            else:
                features = features[-self.seq_len:]  # Take the last seq_len timesteps
                
            # Reshape to [batch, seq_len, features]
            reshaped = features.reshape(batch_size, self.seq_len, features.shape[1])
            return reshaped
            
        except Exception as e:
            print(f"Error in GRU preprocessing: {e}")
            import traceback
            traceback.print_exc()
            raise e

    def predict(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Generate predictions using trained GRU model.
        
        Args:
            data: Preprocessed input data [batch, seq_len, features]
            
        Returns:
            Predictions [batch, pred_len]
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        try:
            # Preprocess if input is DataFrame
            if isinstance(data, pd.DataFrame):
                data = self.preprocess(data)
            
            # Ensure numpy array is torch tensor on correct device
            if isinstance(data, np.ndarray):
                data = torch.FloatTensor(data).to(self.device)
            
            # Generate predictions
            self.model.eval()
            with torch.no_grad():
                predictions = self.model(data).cpu().numpy()
            
            # Inverse transform predictions if we have a target scaler
            if self.scaler_target is not None:
                # Reshape to 2D for inverse_transform
                orig_shape = predictions.shape
                predictions = self.scaler_target.inverse_transform(predictions.reshape(-1, 1)).reshape(orig_shape)
            
            return predictions
            
        except Exception as e:
            print(f"Error in GRU prediction: {e}")
            import traceback
            traceback.print_exc()
            raise e
    
    def predict_from_file(self, file_path: str, date_str: Optional[str] = None) -> pd.DataFrame:
        """
        Load data from file and make predictions for a specific date.
        
        Args:
            file_path: Path to CSV file with prediction data
            date_str: Date string in format 'YYYY-MM-DD' (optional)
            
        Returns:
            DataFrame with predictions
        """
        try:
            # Load data
            df = pd.read_csv(file_path, parse_dates=['date'])
            
            # Filter by date if provided
            if date_str:
                df = df.loc[df['date'].dt.strftime('%Y-%m-%d') == date_str].copy()
                
                if df.empty:
                    raise ValueError(f"No data found for date: {date_str}")
            
            # Ensure we have hourly data
            if 'hour' not in df.columns and 'date' in df.columns:
                df['hour'] = df['date'].dt.hour
            
            # Preprocess data
            preprocessed_data = self.preprocess(df)
            
            # Make predictions
            predictions = self.predict(preprocessed_data)
            
        except Exception as e:
            print(f"Error in predict_from_file: {e}")
            import traceback
            traceback.print_exc()
            raise e
        
        # Format predictions into DataFrame
        # For GRU the predictions shape is [batch_size, pred_len]
        if isinstance(predictions, np.ndarray):
            if predictions.ndim == 2:
                predictions = predictions.flatten()
            
            # Create a DataFrame with date, hour, and predictions
            result_df = pd.DataFrame({
                'date': df['date'].values,
                'hour': df['hour'].values,
                'Predicted': predictions
            })
            
            # Add true values if available
            if 'Electricity_price_MWh' in df.columns:
                result_df['True'] = df['Electricity_price_MWh'].values
                result_df['Pct_of_True'] = result_df['Predicted'] / result_df['True'] * 100
        else:
            # Fallback for unexpected prediction format
            result_df = pd.DataFrame({
                'date': df['date'].values,
                'hour': df['hour'].values,
                'Predicted': predictions
            })
        
        return result_df
