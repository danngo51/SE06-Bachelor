from typing import Optional, Union, Dict, List
import pandas as pd
import numpy as np
import torch
import pathlib
import joblib
from dateutil.parser import parse

from interfaces.ModelPipelineInterface import IModelPipeline
from ml_models.GRU.GRU_model import ElectricityGRU

class GRUPipeline(IModelPipeline):
    """
    Pipeline for GRU model inference
    """
    def __init__(self, mapcode: str = "DK1", seq_len: int = 168, pred_len: int = 24):
        """
        Initialize GRU pipeline
        
        Args:
            mapcode: Region code (DK1, DK2, etc.)
            seq_len: Length of input sequence
            pred_len: Length of prediction sequence
        """
        self.mapcode = mapcode
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.model = None
        self.scaler_features = None
        self.scaler_target = None
        self.feature_cols = None
        self.price_shift = 0
        self.model_config = None
        
        # Set paths
        current_file = pathlib.Path(__file__)
        self.project_root = current_file.parent.parent
        self.data_dir = self.project_root / "data" / self.mapcode
        self.model_dir = self.data_dir / "gru"
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def load_model(self, model_path: Optional[str] = None) -> None:
        """
        Load a pretrained GRU model
        
        Args:
            model_path: Path to model directory (optional)
        """
        try:
            # Get model directory
            model_dir = pathlib.Path(model_path) if model_path else self.model_dir
            print(f"Loading model from {model_dir}")
            
            # Check if directory exists
            if not model_dir.exists():
                raise FileNotFoundError(f"Model directory {model_dir} does not exist")
            
            # Load model configuration
            config_path = model_dir / "model_config.pkl"
            if not config_path.exists():
                raise FileNotFoundError(f"Model config not found at {config_path}")
            
            self.model_config = joblib.load(config_path)
            print(f"Loaded model configuration: {self.model_config}")
            
            # Extract model parameters
            input_dim = self.model_config.get('input_dim')
            hidden_dim = self.model_config.get('hidden_dim')
            num_layers = self.model_config.get('num_layers')
            output_dim = self.model_config.get('output_dim', self.pred_len)
            dropout = self.model_config.get('dropout', 0.2)
            self.price_shift = self.model_config.get('price_shift', 0)
            
            # Load feature columns
            feature_cols_path = model_dir / "feature_columns.pkl"
            if not feature_cols_path.exists():
                raise FileNotFoundError(f"Feature columns not found at {feature_cols_path}")
            
            self.feature_cols = joblib.load(feature_cols_path)
            print(f"Loaded {len(self.feature_cols)} feature columns")
            
            # Load scalers
            feature_scaler_path = model_dir / "feature_scaler.pkl"
            if not feature_scaler_path.exists():
                raise FileNotFoundError(f"Feature scaler not found at {feature_scaler_path}")
            
            self.scaler_features = joblib.load(feature_scaler_path)
            
            target_scaler_path = model_dir / "target_scaler.pkl"
            if not target_scaler_path.exists():
                raise FileNotFoundError(f"Target scaler not found at {target_scaler_path}")
            
            self.scaler_target = joblib.load(target_scaler_path)
            
            # Initialize model
            self.model = ElectricityGRU(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                output_dim=output_dim,
                dropout=dropout
            )
            
            # Load weights - try both filenames
            model_files = ["best_model.pth", "gru_model.pth"]
            loaded = False
            
            for filename in model_files:
                model_weights_path = model_dir / filename
                if model_weights_path.exists():
                    self.model.load_state_dict(torch.load(model_weights_path, map_location=self.device))
                    print(f"Loaded model weights from {model_weights_path}")
                    loaded = True
                    break
            
            if not loaded:
                raise FileNotFoundError(f"Model weights not found in {model_dir}")
            
            # Move model to device and set to evaluation mode
            self.model.to(self.device)
            self.model.eval()
            
            print("Model loaded successfully and ready for prediction")
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Failed to load model: {str(e)}")
    
    def preprocess(self, data: pd.DataFrame) -> np.ndarray:
        """
        Preprocess data for GRU model
        
        Args:
            data: Input DataFrame with features
        
        Returns:
            Preprocessed numpy array
        """
        # Make a copy to avoid modifying the original
        df = data.copy()
        print(f"Preprocessing input data of shape {df.shape}")
        
        # Process date column if present
        if 'date' in df.columns:
            date_col = pd.to_datetime(df['date'])
            
            # Extract time features if not already present
            if 'hour' not in df.columns:
                df['hour'] = date_col.dt.hour
                df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
                df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            
            if 'day_of_week' not in df.columns:
                df['day_of_week'] = date_col.dt.dayofweek
                df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
                df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            
            if 'month_sin' not in df.columns:
                df['month_sin'] = np.sin(2 * np.pi * date_col.dt.month / 12)
                df['month_cos'] = np.cos(2 * np.pi * date_col.dt.month / 12)
        
        # Process price features if available
        if 'Electricity_price_MWh' in df.columns:
            # Add lag features (only if not already present)
            for lag in [1, 2, 3, 24, 48]:
                lag_col = f'price_lag_{lag}'
                if lag_col not in df.columns:
                    df[lag_col] = df['Electricity_price_MWh'].shift(lag)
            
            # Add rolling statistics
            for window in [24, 48]:
                mean_col = f'price_roll_mean_{window}h'
                std_col = f'price_roll_std_{window}h'
                
                if mean_col not in df.columns:
                    df[mean_col] = df['Electricity_price_MWh'].rolling(window).mean()
                
                if std_col not in df.columns:
                    df[std_col] = df['Electricity_price_MWh'].rolling(window).std()
        
        # Ensure all required feature columns are present
        if self.feature_cols is not None:
            missing_cols = [col for col in self.feature_cols if col not in df.columns]
            if missing_cols:
                print(f"Warning: {len(missing_cols)} missing feature columns will be filled with 0")
                for col in missing_cols:
                    df[col] = 0
            
            # Select only the required features in the right order
            features = df[self.feature_cols].values
        else:
            raise ValueError("Feature columns not defined. Call load_model() first.")
        
        # Apply scaling
        if self.scaler_features is not None:
            features = self.scaler_features.transform(features)
        else:
            raise ValueError("Feature scaler not defined. Call load_model() first.")
        
        # Handle sequence length (get the last seq_len values)
        if len(features) < self.seq_len:
            # Pad with zeros if not enough data
            print(f"Warning: Input length ({len(features)}) < sequence length ({self.seq_len})")
            padding = np.zeros((self.seq_len - len(features), features.shape[1]))
            features = np.vstack([padding, features])
        else:
            # Take last seq_len values
            features = features[-self.seq_len:]
        
        # Reshape for model input [batch_size=1, seq_len, features]
        processed_data = features.reshape(1, self.seq_len, -1)
        print(f"Preprocessed data shape: {processed_data.shape}")
        
        return processed_data
    
    def predict(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Make predictions using the loaded GRU model
        
        Args:
            data: Input data as DataFrame or preprocessed numpy array
        
        Returns:
            Predictions as numpy array
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        try:
            # Preprocess data if needed
            if isinstance(data, pd.DataFrame):
                data = self.preprocess(data)
            
            # Convert to tensor
            data_tensor = torch.FloatTensor(data).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                self.model.eval()
                predictions = self.model(data_tensor)
                predictions = predictions.cpu().numpy()
            
            # Invert scaling
            if self.scaler_target is not None:
                predictions = predictions.reshape(-1, 1)
                predictions = self.scaler_target.inverse_transform(predictions).flatten()
                
                # Invert log transform
                predictions = np.expm1(predictions)
                
                # Invert price shifting if needed
                if self.price_shift > 0:
                    predictions -= self.price_shift
            
            print(f"Generated {len(predictions)} predictions")
            return predictions
        
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Prediction failed: {str(e)}")
    
    def predict_from_file(self, file_path: str, date_str: Optional[str] = None) -> pd.DataFrame:
        """
        Make predictions from a CSV file
        
        Args:
            file_path: Path to CSV file
            date_str: Optional date string to filter data
        
        Returns:
            DataFrame with predictions
        """
        try:
            # Load data
            df = pd.read_csv(file_path)
            
            # Convert date column if present
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            
            # Filter by date if specified
            if date_str:
                target_date = parse(date_str).date()
                df = df[df['date'].dt.date == target_date]
                
                if len(df) == 0:
                    raise ValueError(f"No data found for date {target_date}")
            
            # Make predictions
            predictions = self.predict(df)
            
            # Create result DataFrame
            result_df = pd.DataFrame()
            
            # Add date column if present in input
            if 'date' in df.columns:
                # Take the last pred_len dates or pad with new dates if needed
                dates = df['date'].iloc[-self.pred_len:].copy()
                
                # If we don't have enough dates, generate future dates
                if len(dates) < self.pred_len:
                    last_date = dates.iloc[-1] if len(dates) > 0 else df['date'].iloc[-1]
                    future_dates = pd.date_range(
                        start=last_date + pd.Timedelta(hours=1), 
                        periods=self.pred_len - len(dates), 
                        freq='H'
                    )
                    dates = pd.concat([dates, pd.Series(future_dates)])
                
                result_df['date'] = dates.values[:self.pred_len]
            
            # Add predictions
            result_df['Predicted'] = predictions
            
            # Add actuals if available (from the last pred_len rows)
            if 'Electricity_price_MWh' in df.columns:
                actuals = df['Electricity_price_MWh'].iloc[-self.pred_len:].values
                if len(actuals) == self.pred_len:
                    result_df['True'] = actuals
            
            return result_df
        
        except Exception as e:
            print(f"Error in predict_from_file: {str(e)}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"File prediction failed: {str(e)}")
    
    def get_model_info(self) -> Dict[str, Union[str, int]]:
        """Get information about the loaded model"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        return {
            'model_type': 'GRU',
            'mapcode': self.mapcode,
            'sequence_length': self.seq_len,
            'prediction_length': self.pred_len,
            'hidden_dimension': self.model_config.get('hidden_dim') if self.model_config else None,
            'num_layers': self.model_config.get('num_layers') if self.model_config else None,
            'num_features': len(self.feature_cols) if self.feature_cols else None
        }