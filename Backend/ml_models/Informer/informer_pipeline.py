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

    def load_model(self, model_path: Optional[str] = None) -> bool:
        informer_dir = pathlib.Path(model_path) if model_path else self.informer_dir
        
        try:
            model_path = str(informer_dir / "best_informer.pt")
            scaler_path = str(informer_dir / "scaler.pkl")
            features_file = self.data_dir / "feature" / "features.csv"
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Informer model file not found: {model_path}")
            
            # Reinitialize the model architecture
            self.model = Informer(
                input_dim=10,  # Number of features from features.csv
                seq_len=self.seq_len,
                label_len=self.label_len,
                pred_len=self.pred_len
            ).to(self.device)
            
            # Load the state dictionary
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
            
            # Dynamically load features from features.csv
            if os.path.exists(features_file):
                features_df = pd.read_csv(features_file)
                self.training_feature_cols = features_df['Feature'].tolist()
            else:
                raise FileNotFoundError(f"Features file not found: {features_file}")
            
            print(f"Informer model loaded successfully from {model_path}")
            return True
        except Exception as e:
            print(f"Error loading Informer model: {e}")
            return False

    def _prepare_informer_data(self, data: pd.DataFrame) -> Tuple[torch.Tensor, List[str]]:
        if self.training_feature_cols is None:
            raise ValueError("Training feature columns are not defined. Ensure the model is trained first.")

        feature_cols = [col for col in self.training_feature_cols if col in data.columns]
        if not feature_cols:
            raise ValueError("No matching feature columns found in the historical data.")

        features = data[feature_cols].values.astype(np.float32)
        if self.scaler is not None:
            features = self.scaler.transform(features)

        informer_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        return informer_tensor, feature_cols

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

        with torch.no_grad():
            informer_tensor = informer_tensor.to(self.device)
            dec_inp = dec_inp.to(self.device)
            predictions = self.model(informer_tensor, dec_inp).cpu().numpy().flatten()

        if self.scaler is not None:
            target_mean, target_std = self.scaler.mean_[-1], np.sqrt(self.scaler.var_[-1])
            predictions = predictions * target_std + target_mean

        prediction_dates = pd.date_range(start=data['date'].iloc[-1], periods=self.pred_len, freq='H')
        return pd.Series(predictions, index=prediction_dates, name='Predicted')

    def predict_from_file(self, file_path: str, date_str: Optional[str] = None) -> pd.DataFrame:
        df = pd.read_csv(file_path, parse_dates=['date'])
        if date_str:
            prediction_date = pd.to_datetime(date_str)
            historical_data = df[df['date'] < prediction_date].tail(self.seq_len)
            if len(historical_data) < self.seq_len:
                raise ValueError(f"Insufficient data for prediction. Expected at least {self.seq_len} rows.")
            predictions = self.predict(historical_data)
            return predictions.reset_index().rename(columns={'index': 'date'})
        else:
            raise ValueError("Prediction date must be specified.")