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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set up paths
        current_file = pathlib.Path(__file__)
        self.project_root = current_file.parent.parent  # ml_models directory
        self.data_dir = self.project_root / "data" / self.mapcode
        self.informer_dir = self.data_dir / "informer"



    def load_model(self, model_path: Optional[str]) -> None:
        informer_dir = pathlib.Path(model_path) if model_path else self.informer_dir

        scaler_path = informer_dir / "scaler.pkl"
        feature_columns_path = informer_dir / "feature_columns.pkl"
        model_weights_path = informer_dir / "best_informer.pt"
        
        if not scaler_path.exists() or not feature_columns_path.exists() or not model_weights_path.exists():
            raise FileNotFoundError(f"Model files not found in {informer_dir}")
        
        self.scaler = joblib.load(scaler_path)
        self.feature_cols = joblib.load(feature_columns_path)
        
        # Print information about loaded model
        print(f"Loaded model from {informer_dir}")
        print(f"Model expects {len(self.feature_cols)} features: {self.feature_cols}")
        
        # Debug the scaler
        print(f"Scaler expecting {self.scaler.n_features_in_} features")
        
        # Check for mismatch
        if self.scaler.n_features_in_ != len(self.feature_cols) + 1:  # +1 for target
            print(f"WARNING: Scaler expects {self.scaler.n_features_in_} features but model uses {len(self.feature_cols)}")
            raise ValueError(
                f"Scaler and feature count mismatch. Please retrain the model. "
                f"Scaler expects {self.scaler.n_features_in_} features, model has {len(self.feature_cols)}"
            )
        
        # Store target statistics for denormalization
        self.target_mean = self.scaler.mean_[-1]  # Last element is target
        self.target_std = np.sqrt(self.scaler.var_[-1])
        print(f"Target statistics for denormalization: mean={self.target_mean:.4f}, std={self.target_std:.4f}")
        
        # Create model
        self.model = Informer(input_dim=len(self.feature_cols),
                       seq_len=self.seq_len,
                       label_len=self.label_len,
                       pred_len=self.pred_len).to(self.device)
        self.model.load_state_dict(torch.load(model_weights_path, map_location=self.device))
        self.model.eval()

    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        
        # Check for missing features
        missing_features = [col for col in self.feature_cols if col not in data.columns]
        if missing_features:
            print(f"Adding missing features: {missing_features}")
            for feature in missing_features:
                data[feature] = 0.0
    
        # Ensure we use features in the exact order the model expects
        data = data[self.feature_cols].astype(np.float32)
    
        # Apply scaling - add a dummy target column with zeros
        try:
            # Create input with dummy target column (will be ignored in prediction)
            data_with_dummy_target = np.hstack([data.values, np.zeros((len(data), 1))])
            scaled_data = self.scaler.transform(data_with_dummy_target)[:, :-1]  # Remove dummy target
            data = pd.DataFrame(scaled_data, columns=self.feature_cols, index=data.index)
        except ValueError as e:
            print(f"Error during scaling: {e}")
            print(f"Data shape: {data.shape}, Feature count: {len(self.feature_cols)}")
            print(f"Scaler n_features_in_: {self.scaler.n_features_in_}")
            raise
    
        return data

    def predict(self, data: pd.DataFrame) -> List[float]:
        preprocessed = self.preprocess(data)
        if len(preprocessed) < self.seq_len:
            raise ValueError(f"Input data must have at least {self.seq_len} rows for prediction.")
        
        # Convert to correct dtype and format
        last_seq = preprocessed[-self.seq_len:].values
        enc_x = torch.from_numpy(last_seq).float().unsqueeze(0).to(self.device)  # Ensure float

        # Prepare decoder input with correct dtype
        dec_y = torch.zeros((1, self.label_len + self.pred_len), dtype=torch.float32, device=self.device)
        if self.label_len > 0 and len(preprocessed) >= self.label_len:
            label_input = preprocessed.iloc[-self.label_len:][self.feature_cols[-1]].values
            dec_y[0, :self.label_len] = torch.from_numpy(label_input).float()  # Ensure float

        with torch.no_grad():
            output = self.model(enc_x, dec_y)
        preds = output[0, -self.pred_len:].cpu().numpy()

        # De-normalize using the original target statistics
        if hasattr(self, 'target_mean') and hasattr(self, 'target_std'):
            # Use saved statistics from original scaler
            denorm_preds = preds * self.target_std + self.target_mean
            print(f"Using saved target statistics for denormalization: mean={self.target_mean:.4f}, std={self.target_std:.4f}")
        else:
            # Fallback to current scaler (though this might be problematic if it was recreated)
            denorm_preds = preds * np.sqrt(self.scaler.var_[-1]) + self.scaler.mean_[-1]
        
        # Debug final predictions
        print(f"Final predictions after denormalization: {denorm_preds}")
        print(f"Final prediction stats - min: {denorm_preds.min():.2f}, max: {denorm_preds.max():.2f}, mean: {denorm_preds.mean():.2f}")
        
        return denorm_preds.tolist()

    def predict_from_file(self, file_path: str, date_str: Optional[str] = None) -> pd.DataFrame:
        # Load all data from file
        df = pd.read_csv(file_path, parse_dates=['date'])
        
        # Get the full dataset index for the target date
        target_indices = None
        if date_str:
            target_date_data = df[df['date'].dt.strftime('%Y-%m-%d') == date_str]
            if target_date_data.empty:
                raise ValueError(f"No data found for date: {date_str}")
            
            # Store indices of rows with the target date
            target_indices = target_date_data.index.tolist()
            
            # Find the index of the first row with the target date
            first_target_idx = min(target_indices)
            
            # Include enough historical data for prediction
            historical_idx = max(0, first_target_idx - self.seq_len)
            
            # Get historical + target data
            df = df.iloc[historical_idx:max(target_indices) + 1]
            
            print(f"Using {len(df)} rows for prediction ({self.seq_len} historical, {len(target_indices)} target)")
        
        # Rest of validation and preprocessing
        if 'Electricity_price_MWh' not in df.columns:
            raise ValueError("Column 'Electricity_price_MWh' not found in input file.")

        if 'hour' not in df.columns:
            df['hour'] = df['date'].dt.hour
        
        # Check and prepare features
        # Keep only relevant features in correct order
        X = df[self.feature_cols]
        
        # Make prediction using all available data
        preds = self.predict(X)
        
        # Return only results for target date
        if target_indices:
            result_df = df.loc[target_indices].copy()
            result_df['Predicted'] = preds
            result_df['True'] = result_df['Electricity_price_MWh']
        else:
            # If no specific date, use the last pred_len rows
            result_df = df.tail(self.pred_len).copy()
            result_df['Predicted'] = preds
            result_df['True'] = result_df['Electricity_price_MWh']
            
        result_df['Pct_of_True'] = result_df['Predicted'] / result_df['True'] * 100
        
        return result_df[['date', 'hour', 'True', 'Predicted', 'Pct_of_True']]