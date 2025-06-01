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



    def load_model(self, model_path: str) -> None:
        base_dir = pathlib.Path(model_path).parent
        self.scaler = joblib.load(base_dir / "scaler.pkl")
        self.feature_cols = joblib.load(base_dir / "feature_columns.pkl")
        self.model = Informer(input_dim=len(self.feature_cols),
                              seq_len=self.seq_len,
                              label_len=self.label_len,
                              pred_len=self.pred_len).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        data = data[self.feature_cols].astype(np.float32)
        data = pd.DataFrame(self.scaler.transform(data), columns=self.feature_cols, index=data.index)
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
        df = pd.read_csv(file_path, parse_dates=['date'])

        if date_str:
            df = df[df['date'].dt.strftime('%Y-%m-%d') == date_str]
            if df.empty:
                raise ValueError(f"No data found for date: {date_str}")

        if 'Electricity_price_MWh' not in df.columns:
            raise ValueError("Column 'Electricity_price_MWh' not found in input file.")

        if 'hour' not in df.columns:
            df['hour'] = df['date'].dt.hour

        base_features = [c for c in df.columns if c not in ['date', 'hour', 'Electricity_price_MWh']]
        if not all(f in df.columns for f in self.feature_cols):
            raise ValueError("Missing one or more required feature columns for prediction.")

        # Keep only relevant features in correct order
        X = df[self.feature_cols]

        preds = self.predict(X)
        df = df.tail(self.pred_len).copy()
        df['Predicted'] = preds
        df['True'] = df['Electricity_price_MWh'].values[-self.pred_len:]
        df['Pct_of_True'] = df['Predicted'] / df['True'] * 100

        return df[['date', 'hour', 'True', 'Predicted', 'Pct_of_True']]