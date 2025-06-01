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
    def __init__(self, mapcode: str = "DK1", seq_len: int = 168, label_len: int = 48, pred_len: int = 24):
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
    
    def _prepare_informer_data(self, data: pd.DataFrame) -> Tuple[torch.Tensor, List[str]]:
        if hasattr(self, 'training_feature_cols') and self.training_feature_cols is not None:
            feature_cols = self.training_feature_cols
            missing_cols = [col for col in feature_cols if col not in data.columns]
            if missing_cols:
                print(f"Warning: Missing features from training: {missing_cols}")
                feature_cols = [col for col in feature_cols if col in data.columns]
        else:
            feature_cols = [col for col in data.columns if col not in ['date', 'Electricity_price_MWh']]
        if self.scaler is None:
            print("Warning: No scaler found. Using standard scaling.")
            features = data[feature_cols].values
            features_mean = np.mean(features, axis=0)
            features_std = np.std(features, axis=0)
            features_std[features_std == 0] = 1  # Avoid division by zero
            scaled_data = (features - features_mean) / features_std
        else:
            if 'Electricity_price_MWh' in data.columns:
                # Include target for scaling, then remove it
                all_cols = feature_cols + ['Electricity_price_MWh']
                missing_cols = [col for col in all_cols if col not in data.columns]
                if missing_cols:
                    print(f"Warning: Missing columns for scaling: {missing_cols}")
                    available_cols = [col for col in all_cols if col in data.columns]
                    scaled_data = self.scaler.transform(data[available_cols].values)
                    if 'Electricity_price_MWh' in available_cols:
                        scaled_data = scaled_data[:, :-1]  # Remove target column
                else:
                    scaled_data = self.scaler.transform(data[all_cols].values)[:, :-1]
            else:
                missing_cols = [col for col in feature_cols if col not in data.columns]
                if missing_cols:
                    print(f"Warning: Missing feature columns: {missing_cols}")
                    available_cols = [col for col in feature_cols if col in data.columns]
                    scaled_data = self.scaler.transform(data[available_cols].values)
                else:
                    scaled_data = self.scaler.transform(data[feature_cols].values)
            
        if len(scaled_data) < self.seq_len:
            # Pad with zeros if needed
            pad_amount = self.seq_len - len(scaled_data)
            padding = np.zeros((pad_amount, scaled_data.shape[1]))
            scaled_data = np.vstack((padding, scaled_data))
            print(f"Warning: Input data shorter than seq_len ({len(data)} < {self.seq_len}). Padded with zeros.")
        
        seq_data = scaled_data[-self.seq_len:]
        
        batch_data = torch.tensor(seq_data, dtype=torch.float32).unsqueeze(0)
        
        return batch_data, feature_cols
    
    def predict(self, data: pd.DataFrame) -> pd.Series:
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        informer_tensor, feature_cols = self._prepare_informer_data(data)
        
        self.model.eval()

        dec_inp = torch.zeros((1, self.label_len + self.pred_len), device=self.device)
        
        if 'Electricity_price_MWh' in data.columns and len(data) >= self.label_len and self.label_len > 0:
            target_data = data['Electricity_price_MWh'].values[-self.label_len:]
            
            if self.scaler is not None:
                target_mean, target_std = self.scaler.mean_[-1], np.sqrt(self.scaler.var_[-1])
                target_data = (target_data - target_mean) / target_std
            
            dec_inp[0, :self.label_len] = torch.tensor(target_data, dtype=torch.float32)
        
        # Make predictions
        with torch.no_grad():
            informer_tensor = informer_tensor.to(self.device)
            dec_inp = dec_inp.to(self.device)
            
            # Forward pass
            predictions = self.model(informer_tensor, dec_inp)     

            if predictions.dim() == 2:
                if predictions.shape[1] == (self.label_len + self.pred_len):

                    predictions = predictions[:, :self.pred_len].cpu().numpy().flatten()
                        
                elif predictions.shape[1] == self.pred_len:
                    predictions = predictions.cpu().numpy().flatten()
                    
                elif predictions.shape[1] >= self.pred_len:
                    predictions = predictions[:, -self.pred_len:].cpu().numpy().flatten()
                else:
                    predictions = predictions.cpu().numpy().flatten()
            elif predictions.dim() == 3:
                predictions = predictions.cpu().numpy().flatten()
                if len(predictions) >= self.pred_len:
                    predictions = predictions[-self.pred_len:]
                else:
                    last_val = predictions[-1] if len(predictions) > 0 else 0.0
                    padding = [last_val] * (self.pred_len - len(predictions))
                    predictions = np.concatenate([predictions, padding])
            else:
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
            target_mean, target_std = self.scaler.mean_[-1], np.sqrt(self.scaler.var_[-1])
            predictions = predictions * target_std + target_mean
        else:
            print("No scaler available for inverse transform")
        if 'date' in data.columns and 'hour' in data.columns:
            last_date = data['date'].iloc[-1]
            last_hour = data['hour'].iloc[-1]
            
            last_date_only = pd.to_datetime(last_date.date())
            
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
            return pd.Series(predictions, name='Predicted')
    
    def load_model(self, model_path: Optional[str] = None) -> bool:
        informer_dir = pathlib.Path(model_path) if model_path else self.informer_dir
        
        try:
            model_path = str(informer_dir / "informer_model.pt")
            scaler_path = str(informer_dir / "scaler.pkl")
            feature_cols_path = str(informer_dir / "feature_columns.pkl")
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Informer model file not found: {model_path}")
            
            self.model = torch.load(model_path, map_location=self.device, weights_only=False)
            
            self.model.eval()
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                if hasattr(self.scaler, 'feature_names_in_'):
                    if self.scaler.feature_names_in_ is None:
                        pass 
            
            if os.path.exists(feature_cols_path):
                self.training_feature_cols = joblib.load(feature_cols_path)
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
    
    def predict_from_file(self, file_path: str, date_str: Optional[str] = None) -> pd.DataFrame:
        # Load data
        df = pd.read_csv(file_path, parse_dates=['date'])
    
        
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

    
    