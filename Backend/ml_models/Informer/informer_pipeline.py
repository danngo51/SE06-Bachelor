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
        print(f"Scaler mean shape: {self.scaler.mean_.shape}")
        print(f"Scaler variance shape: {self.scaler.var_.shape}")
        
        # Recreate scaler if there's a mismatch
        if self.scaler.n_features_in_ != len(self.feature_cols):
            print("WARNING: Scaler and feature columns count mismatch. Recreating scaler...")
            from sklearn.preprocessing import StandardScaler
            dummy_data = np.zeros((1, len(self.feature_cols)))
            self.scaler = StandardScaler()
            self.scaler.fit(dummy_data)
            print(f"New scaler created with {self.scaler.n_features_in_} features")

        self.model = Informer(input_dim=len(self.feature_cols),
                            seq_len=self.seq_len,
                            label_len=self.label_len,
                            pred_len=self.pred_len).to(self.device)
        self.model.load_state_dict(torch.load(model_weights_path, map_location=self.device))

        self.model.eval()

    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        
        # Debug: Print expected vs actual features
        print(f"Expected features ({len(self.feature_cols)}): {self.feature_cols}")
        print(f"Input data features ({len(data.columns)}): {data.columns.tolist()}")
        print(f"Scaler expects {self.scaler.n_features_in_} features")
        
        # Check for missing features
        missing_features = [col for col in self.feature_cols if col not in data.columns]
        if missing_features:
            print(f"Adding missing features: {missing_features}")
            for feature in missing_features:
                data[feature] = 0.0
    
        # Handle scaler feature mismatch
        if self.scaler.n_features_in_ > len(self.feature_cols):
            print(f"WARNING: Scaler expects {self.scaler.n_features_in_} features but model uses {len(self.feature_cols)}")
            # Add dummy feature to match scaler expectations
            extra_features = self.scaler.n_features_in_ - len(self.feature_cols)
            for i in range(extra_features):
                dummy_col = f"dummy_feature_{i}"
                print(f"Adding dummy feature: {dummy_col}")
                data[dummy_col] = 0.0
                self.feature_cols.append(dummy_col)
    
        # Ensure we use features in the exact order the model expects
        data = data[self.feature_cols].astype(np.float32)
    
        # Apply scaling
        try:
            scaled_data = self.scaler.transform(data)
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
        last_seq = preprocessed[-self.seq_len:].values
        enc_x = torch.from_numpy(last_seq).unsqueeze(0).to(self.device)

        # Prepare decoder input
        dec_y = torch.zeros((1, self.label_len + self.pred_len), device=self.device)
        if self.label_len > 0 and len(preprocessed) >= self.label_len:
            dec_y[0, :self.label_len] = torch.from_numpy(
                preprocessed.iloc[-self.label_len:][self.feature_cols[-1]].values
            )

        with torch.no_grad():
            output = self.model(enc_x, dec_y)
        preds = output[0, -self.pred_len:].cpu().numpy()

        # De-normalize
        preds = preds * np.sqrt(self.scaler.var_[-1]) + self.scaler.mean_[-1]
        return preds.tolist()

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