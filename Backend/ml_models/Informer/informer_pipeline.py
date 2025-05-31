# filepath: d:\Uni\6. semester\MLs\ml_models\Informer\informer_pipeline.py
import os
import pandas as pd
import numpy as np
import torch
import pathlib
import joblib
from typing import Dict, Any, Optional, Union, List, Tuple

from interfaces.ModelPipelineInterface import IModelPipeline
from ml_models.Informer.Informer_model import Informer, TimeSeriesDataset, InformerModelTrainer

class InformerPipeline(IModelPipeline):
    """
    Pipeline for the Informer model.
    Handles preprocessing, model loading, and prediction.
    
    The Informer is an efficient transformer-based model for long sequence
    time-series forecasting, which outperforms standard transformers with
    lower time complexity and memory usage.
    """
    
    def __init__(self, mapcode: str = "DK1", seq_len: int = 168, label_len: int = 48, pred_len: int = 24):
        """
        Initialize the Informer pipeline.
        
        Args:
            mapcode: String code for the market area (e.g., "DK1")
            seq_len: Length of input sequence
            label_len: Length of label sequence (overlap with decoder)
            pred_len: Length of prediction sequence (usually 24 for day-ahead forecasting)
        """
        self.mapcode = mapcode
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.model = None
        self.scaler = None
        self.training_feature_cols = None
        self.model_trainer = None  # For training interface
        self.device = torch.device("cuda" if torch.cuda.is_available() else 
                                   "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else 
                                   "cpu")
        
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
        
        # Calculate rolling statistics (if price column exists)
        if 'Electricity_price_MWh' in df.columns:
            df['price_roll_24h'] = df['Electricity_price_MWh'].shift(1).rolling(24, min_periods=1).mean()
            df['price_roll_7d'] = df['Electricity_price_MWh'].shift(1).rolling(168, min_periods=1).mean()
            df['price_std_24h'] = df['Electricity_price_MWh'].shift(1).rolling(24, min_periods=1).std()
        
        # Load-related features
        load_columns = ['DAHTL_TotalLoadValue', 'TotalLoadValue', 'Load_Value', 'Total_Load', 'DALoadValue', 'SystemLoad']
        load_col = next((col for col in load_columns if col in df.columns), None)
        if load_col:
            df['load_roll_24h'] = df[load_col].shift(1).rolling(24, min_periods=1).mean()
        
        # Create lagged features - Informer works well with longer context
        if 'Electricity_price_MWh' in df.columns:
            for lag in [1, 24, 48, 72, 168]:  # 1h, 1d, 2d, 3d, 7d
                df[f'price_lag_{lag}'] = df['Electricity_price_MWh'].shift(lag)
        
        # Drop rows with NaN values
        return df.dropna()
    
    def _prepare_informer_data(self, data: pd.DataFrame) -> Tuple[torch.Tensor, List[str]]:
        """
        Prepare data in the format required by Informer model.
        
        Args:
            data: Preprocessed input data
            
        Returns:
            Tuple of (tensor with data in Informer format, feature column names)
        """
        # Use saved feature columns if available, otherwise use all features except target and date
        if hasattr(self, 'training_feature_cols') and self.training_feature_cols is not None:
            # Use the exact same features as during training
            feature_cols = self.training_feature_cols
            # Check if all training features are available in the current data
            missing_cols = [col for col in feature_cols if col not in data.columns]
            if missing_cols:
                print(f"Warning: Missing features from training: {missing_cols}")
                # Use only the features that are available
                feature_cols = [col for col in feature_cols if col in data.columns]
        else:
            # Fallback to all features except target and date
            feature_cols = [col for col in data.columns if col not in ['date', 'Electricity_price_MWh']]
        if self.scaler is None:
            print("Warning: No scaler found. Using standard scaling.")
            # If no scaler is available, just standardize with mean/std
            features = data[feature_cols].values
            features_mean = np.mean(features, axis=0)
            features_std = np.std(features, axis=0)
            features_std[features_std == 0] = 1  # Avoid division by zero
            scaled_data = (features - features_mean) / features_std
        else:
            # Use the loaded scaler with only the training features
            if 'Electricity_price_MWh' in data.columns:
                # Include target for scaling, then remove it
                all_cols = feature_cols + ['Electricity_price_MWh']
                # Check if all required columns exist
                missing_cols = [col for col in all_cols if col not in data.columns]
                if missing_cols:
                    print(f"Warning: Missing columns for scaling: {missing_cols}")                    # Use only available columns
                    available_cols = [col for col in all_cols if col in data.columns]
                    # Convert to numpy array to avoid feature names warning
                    scaled_data = self.scaler.transform(data[available_cols].values)
                    if 'Electricity_price_MWh' in available_cols:
                        scaled_data = scaled_data[:, :-1]  # Remove target column
                else:
                    # Convert to numpy array to avoid feature names warning
                    scaled_data = self.scaler.transform(data[all_cols].values)[:, :-1]
            else:
                # Check if all feature columns exist
                missing_cols = [col for col in feature_cols if col not in data.columns]
                if missing_cols:
                    print(f"Warning: Missing feature columns: {missing_cols}")
                    # Use only available columns
                    available_cols = [col for col in feature_cols if col in data.columns]
                    # Convert to numpy array to avoid feature names warning
                    scaled_data = self.scaler.transform(data[available_cols].values)
                else:
                    # Convert to numpy array to avoid feature names warning
                    scaled_data = self.scaler.transform(data[feature_cols].values)
            
        # Ensure we have enough data for the sequence length
        if len(scaled_data) < self.seq_len:
            # Pad with zeros if needed
            pad_amount = self.seq_len - len(scaled_data)
            padding = np.zeros((pad_amount, scaled_data.shape[1]))
            scaled_data = np.vstack((padding, scaled_data))
            print(f"Warning: Input data shorter than seq_len ({len(data)} < {self.seq_len}). Padded with zeros.")
        
        # Use the last seq_len points
        seq_data = scaled_data[-self.seq_len:]
        
        # Reshape data for Informer: [batch_size, seq_len, num_features]
        # For inference, we usually use a batch size of 1
        batch_data = torch.tensor(seq_data, dtype=torch.float32).unsqueeze(0)
        
        return batch_data, feature_cols
    
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
        informer_tensor, feature_cols = self._prepare_informer_data(data)
        
        # Set model to evaluation mode
        self.model.eval()        # Create decoder input (for first prediction step)
        # For the Informer model, we need a decoder input sequence that contains
        # label_len historical values and pred_len future values (filled with zeros)
        
        # Always create decoder input with correct size: [batch_size, label_len + pred_len]
        dec_inp = torch.zeros((1, self.label_len + self.pred_len), device=self.device)
        
        # If price data exists, use it for the decoder input (historical part)
        if 'Electricity_price_MWh' in data.columns and len(data) >= self.label_len and self.label_len > 0:
            target_data = data['Electricity_price_MWh'].values[-self.label_len:]
            
            # Standardize the target data if we have a scaler
            if self.scaler is not None:
                # Extract the mean and std of the target column (last column)
                target_mean, target_std = self.scaler.mean_[-1], np.sqrt(self.scaler.var_[-1])
                target_data = (target_data - target_mean) / target_std
            
            dec_inp[0, :self.label_len] = torch.tensor(target_data, dtype=torch.float32)
        
        # Make predictions
        with torch.no_grad():            # Move data to the correct device
            informer_tensor = informer_tensor.to(self.device)
            dec_inp = dec_inp.to(self.device)
            
            # Forward pass
            predictions = self.model(informer_tensor, dec_inp)     
            # Get only the predictions part (not the label part)
            # The model output should be [batch_size, label_len + pred_len]            # We only want the last pred_len values
            if predictions.dim() == 2:
                if predictions.shape[1] == (self.label_len + self.pred_len):
                    # For Informer, the prediction sequence is at the BEGINNING, not the end
                    # The model outputs [prediction_sequence, label_sequence] not [label, prediction]
                    predictions = predictions[:, :self.pred_len].cpu().numpy().flatten()
                        
                elif predictions.shape[1] == self.pred_len:
                    # Model already returns only predictions
                    predictions = predictions.cpu().numpy().flatten()
                    
                elif predictions.shape[1] >= self.pred_len:
                    # Take the last pred_len values
                    predictions = predictions[:, -self.pred_len:].cpu().numpy().flatten()
                else:
                    # Unexpected shape, take what we can get
                    predictions = predictions.cpu().numpy().flatten()
            elif predictions.dim() == 3:
                # Sometimes model outputs [batch, seq, 1] - flatten and get last pred_len
                predictions = predictions.cpu().numpy().flatten()
                if len(predictions) >= self.pred_len:
                    predictions = predictions[-self.pred_len:]
                else:
                    # Pad if necessary
                    last_val = predictions[-1] if len(predictions) > 0 else 0.0
                    padding = [last_val] * (self.pred_len - len(predictions))
                    predictions = np.concatenate([predictions, padding])
            else:
                # Fallback: if shape is unexpected, try to extract what we can
                predictions = predictions.cpu().numpy().flatten()
                print(f"Unexpected model output shape. Flattened to: {len(predictions)} values")
                if len(predictions) > self.pred_len:
                    predictions = predictions[-self.pred_len:]
                elif len(predictions) < self.pred_len:
                    # If we get fewer predictions than expected, pad with the last value
                    last_val = predictions[-1] if len(predictions) > 0 else 0.0
                    padding = [last_val] * (self.pred_len - len(predictions))
                    predictions = np.concatenate([predictions, padding])
            
        if self.scaler is not None:
            # Extract the mean and std of the target column (last column)
            target_mean, target_std = self.scaler.mean_[-1], np.sqrt(self.scaler.var_[-1])
            predictions = predictions * target_std + target_mean
        else:
            print("No scaler available for inverse transform")
          # Create index for predictions (usually the next 24 hours after the input data)
        if 'date' in data.columns and 'hour' in data.columns:
            last_date = data['date'].iloc[-1]
            last_hour = data['hour'].iloc[-1]
            
            # Extract just the date part to avoid time component issues
            last_date_only = pd.to_datetime(last_date.date())
            
            # Create prediction dates and hours
            pred_dates = []
            pred_hours = []
            for i in range(self.pred_len):
                next_hour = (last_hour + i + 1) % 24
                days_to_add = (last_hour + i + 1) // 24
                next_date = last_date_only + pd.Timedelta(days=days_to_add) + pd.Timedelta(hours=next_hour)
                
                pred_dates.append(next_date)
                pred_hours.append(next_hour)
            
            # Create a DataFrame for predictions
            pred_df = pd.DataFrame({
                'date': pred_dates,
                'hour': pred_hours,
                'Predicted': predictions
            })
            
            return pred_df.set_index(['date', 'hour'])['Predicted']
        else:
            # If no date/hour information, just return as a series
            return pd.Series(predictions, name='Predicted')
    
    def load_model(self, model_path: Optional[str] = None) -> bool:
        """
        Load Informer model from disk.
        
        Args:
            model_path: Path to saved model directory (optional, uses default path if None)
            
        Returns:
            True if successful, False otherwise
        """        # Use provided path or default
        informer_dir = pathlib.Path(model_path) if model_path else self.informer_dir
        
        try:
            model_path = str(informer_dir / "informer_model.pt")
            scaler_path = str(informer_dir / "scaler.pkl")
            feature_cols_path = str(informer_dir / "feature_columns.pkl")
            
            # Check if model file exists
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Informer model file not found: {model_path}")
            
            # Load the model with weights_only=False for compatibility
            self.model = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # Setup model for inference
            self.model.eval()            # Load scaler if available
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                # Fix the StandardScaler feature names warning
                if hasattr(self.scaler, 'feature_names_in_'):
                    if self.scaler.feature_names_in_ is None:
                        # Set feature names after loading training features
                        pass  # Will be set later when feature columns are loaded
                # Note: Older scalers may not have feature_names_in_ attribute
            
            # Load feature columns if available
            if os.path.exists(feature_cols_path):
                self.training_feature_cols = joblib.load(feature_cols_path)
                # Set feature names on scaler if not already set
                if (hasattr(self, 'scaler') and self.scaler is not None and 
                    hasattr(self.scaler, 'feature_names_in_') and 
                    self.scaler.feature_names_in_ is None):
                    feature_names = self.training_feature_cols + ['Electricity_price_MWh']
                    self.scaler.feature_names_in_ = np.array(feature_names)
                    print(f"Set feature names on scaler: {len(feature_names)} features")
            else:
                self.training_feature_cols = None
                print("Warning: Feature columns not found. Using all available features.")
            
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
            "label_len": self.label_len,
            "pred_len": self.pred_len,
            "model_loaded": self.model is not None,
            "scaler_loaded": self.scaler is not None,
            "device": str(self.device)
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
        if date_str:            # Find the position of the target date
            target_date = pd.to_datetime(date_str)
            target_indices = df.index[df['date'] == target_date]
            
            if len(target_indices) == 0:
                raise ValueError(f"No data found for date: {date_str}")
            
            # Get historical data up to (but not including) the target date
            # We want to predict FOR the target date, not AFTER it
            historical_data = df.loc[:target_indices[0]-1]
            
            # Make predictions
            predictions = self.predict(historical_data)
            
            # Ensure we have exactly pred_len predictions
            if isinstance(predictions, pd.Series):
                if len(predictions) != self.pred_len:
                    print(f"Warning: Expected {self.pred_len} predictions, got {len(predictions)}")
                    if len(predictions) < self.pred_len:                        # Pad with last value
                        last_val = predictions.iloc[-1] if len(predictions) > 0 else 0.0
                        padding_series = pd.Series([last_val] * (self.pred_len - len(predictions)))
                        predictions = pd.concat([predictions, padding_series], ignore_index=True)
                    else:
                        # Truncate to expected length
                        predictions = predictions.iloc[:self.pred_len]
            
            
            if isinstance(predictions, pd.Series) and hasattr(predictions.index, 'levels'):
                # Multi-index case (date, hour)
                result_df = predictions.reset_index()
                result_df = result_df.rename(columns={'Predicted': 'Predicted'})
            elif isinstance(predictions, pd.Series):
                # Simple series case
                result_df = pd.DataFrame({'Predicted': predictions.values})                # Add date and hour information
                if 'date' in df.columns and 'hour' in df.columns:
                    last_date = historical_data['date'].iloc[-1]
                    last_hour = historical_data['hour'].iloc[-1]
                    
                    # Extract just the date part to avoid time component issues
                    last_date_only = pd.to_datetime(last_date.date())
                    
                    pred_dates = []
                    pred_hours = []
                    for i in range(len(predictions)):
                        next_hour = (last_hour + i + 1) % 24
                        days_to_add = (last_hour + i + 1) // 24
                        next_date = last_date_only + pd.Timedelta(days=days_to_add) + pd.Timedelta(hours=next_hour)
                        pred_dates.append(next_date)
                        pred_hours.append(next_hour)
                    
                    result_df['date'] = pred_dates
                    result_df['hour'] = pred_hours
            else:
                # Fallback: just predictions array
                print("DEBUG: Using fallback case")
                result_df = pd.DataFrame({'Predicted': predictions})            # DON'T filter predictions - they are for future dates after target_date
            # The Informer predicts the next 24 hours, which may span multiple dates
              # Add True values if available and ensure length matches
            # Use date-only comparison to get all hours of the target date
            true_values = df.loc[df['date'].dt.date == target_date.date(), 'Electricity_price_MWh']
            if not true_values.empty and len(true_values) == len(result_df):
                result_df['True'] = true_values.values
                result_df['Pct_of_True'] = result_df['Predicted'] / result_df['True'] * 100
            elif not true_values.empty:
                print(f"Warning: Length mismatch - predictions: {len(result_df)}, true values: {len(true_values)}")
                # Don't truncate predictions - they are for future hours after target_date
                # Only add true values where they exist (usually just the first hour)
                if len(true_values) > 0:
                    # Add NaN columns for True and Pct_of_True 
                    result_df['True'] = float('nan')
                    result_df['Pct_of_True'] = float('nan')
                    # Fill in the first few values where we have true data
                    min_len = min(len(result_df), len(true_values))
                    result_df.loc[:min_len-1, 'True'] = true_values.values[:min_len]
                    result_df.loc[:min_len-1, 'Pct_of_True'] = (
                        result_df.loc[:min_len-1, 'Predicted'] / result_df.loc[:min_len-1, 'True'] * 100
                    )
            
            return result_df
        else:
            # Make predictions using all available data
            predictions = self.predict(df)
            
            # Create result DataFrame
            result_df = pd.DataFrame(predictions).reset_index()
            result_df = result_df.rename(columns={0: 'Predicted'})
            
            return result_df
    
    def train_model(self, data_path: str, **training_kwargs) -> bool:
        """
        Train the Informer model using the specified data file.
        
        Args:
            data_path: Path to training data CSV file
            **training_kwargs: Additional training parameters
            
        Returns:
            True if training completed successfully, False otherwise
        """
        try:
            # Initialize model trainer if not already done
            if self.model_trainer is None:
                self.model_trainer = InformerModelTrainer(
                    mapcode=self.mapcode,
                    seq_len=self.seq_len,
                    label_len=self.label_len,
                    pred_len=self.pred_len,
                    **training_kwargs
                )
            
            # Train the model
            results = self.model_trainer.train(data_path=data_path)
            
            if results:
                print(f"Informer model training completed for {self.mapcode}")
                # Load the trained model into pipeline
                self.load_model()
                return True
            else:
                print(f"Informer model training failed for {self.mapcode}")
                return False
                
        except Exception as e:
            print(f"Error training Informer model: {e}")
            return False
    
    def evaluate_model(self, test_file: str) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            test_file: Path to test data CSV file
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        try:
            # Load and preprocess test data
            df = pd.read_csv(test_file, parse_dates=['date'])
            df = self.preprocess(df)
            
            # Prepare data for prediction
            sequences, feature_cols = self._prepare_informer_data(df)
            
            if len(sequences) == 0:
                raise ValueError("No valid sequences found in test data")
            
            # Make predictions
            predictions = self.predict(df)
            
            # Get true values (assuming target column exists in test data)
            if 'Electricity_price_MWh' in df.columns:
                # Align true values with predictions
                n_predictions = len(predictions)
                true_values = df['Electricity_price_MWh'].tail(n_predictions)
                
                # Calculate metrics
                from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                
                mse = mean_squared_error(true_values, predictions)
                mae = mean_absolute_error(true_values, predictions)
                r2 = r2_score(true_values, predictions)
                
                return {
                    "mse": float(mse),
                    "mae": float(mae),
                    "r2": float(r2),
                    "rmse": float(np.sqrt(mse))
                }
            else:
                print("Warning: No target column found in test data. Cannot compute evaluation metrics.")
                return {}
                
        except Exception as e:
            print(f"Error evaluating Informer model: {e}")
            return {}
