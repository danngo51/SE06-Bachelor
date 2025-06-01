import os
import pandas as pd
import numpy as np
import torch
import pathlib
import joblib
from typing import Optional, Union, Dict
from sklearn.preprocessing import StandardScaler

from interfaces.ModelPipelineInterface import IModelPipeline
from ml_models.GRU.GRU_model import GRUModel

class GRUPipeline(IModelPipeline):
    def __init__(self, mapcode: str = "DK1", seq_len: int = 168, pred_len: int = 24):
        self.mapcode = mapcode
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.model = None
        self.scaler_features = None
        self.scaler_target = None
        self.feature_cols = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        current_file = pathlib.Path(__file__)
        self.project_root = current_file.parent.parent
        self.data_dir = self.project_root / "data" / self.mapcode
        self.gru_dir = self.data_dir / "gru"

    def preprocess(self, data: pd.DataFrame) -> np.ndarray:
        df = data.copy()
        if 'hour' not in df.columns and 'date' in df.columns:
            date_dt = pd.to_datetime(df['date'])
            df['hour'] = date_dt.dt.hour
            # Add cyclical encoding for time features
            df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
            df['day_of_week'] = date_dt.dt.dayofweek
            df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week']/7)
            df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week']/7)
            df['month_sin'] = np.sin(2 * np.pi * date_dt.dt.month/12)
            df['month_cos'] = np.cos(2 * np.pi * date_dt.dt.month/12)

        if 'Electricity_price_MWh' in df.columns:
            # Add more sophisticated lag features
            for lag in [1, 2, 3, 24, 25, 26, 48, 72, 96, 168]:
                df[f'price_lag_{lag}'] = df['Electricity_price_MWh'].shift(lag)
            
            # Add rolling statistics
            for window in [6, 12, 24, 48, 72]:
                df[f'price_roll_mean_{window}h'] = df['Electricity_price_MWh'].rolling(window).mean()
                df[f'price_roll_std_{window}h'] = df['Electricity_price_MWh'].rolling(window).std()
                df[f'price_roll_min_{window}h'] = df['Electricity_price_MWh'].rolling(window).min()
                df[f'price_roll_max_{window}h'] = df['Electricity_price_MWh'].rolling(window).max()
            
            # Add price momentum features
            df['price_diff_1h'] = df['Electricity_price_MWh'].diff()
            df['price_diff_24h'] = df['Electricity_price_MWh'].diff(24)
            
            # Add volatility measure
            df['price_volatility'] = df['Electricity_price_MWh'].rolling(24).std() / df['Electricity_price_MWh'].rolling(24).mean()

        df = df.fillna(0)
        features = df[self.feature_cols].values if self.feature_cols else df.select_dtypes(include=[np.number]).values

        if self.scaler_features:
            features = self.scaler_features.transform(features)

        seq_len = min(self.seq_len, len(features))
        if seq_len < self.seq_len:
            padding = np.zeros((self.seq_len - seq_len, features.shape[1]))
            features = np.vstack([padding, features[-seq_len:]])
        else:
            features = features[-self.seq_len:]

        return features.reshape(1, self.seq_len, features.shape[1])

    def predict(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        if isinstance(data, pd.DataFrame):
            data = self.preprocess(data)

        data_tensor = torch.FloatTensor(data).to(self.device)
        with torch.no_grad():
            predictions = self.model(data_tensor).cpu().numpy()

        if self.scaler_target:
            predictions = self.scaler_target.inverse_transform(predictions.flatten().reshape(-1, 1)).flatten()

        return predictions

    def load_model(self, model_path: Optional[str] = None) -> None:
        model_dir = pathlib.Path(model_path) if model_path else self.gru_dir
        model_file = model_dir / "gru_model.pth"
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")

        self.model = GRUModel(
            input_dim=64,  # Default input dimension
            hidden_dim=128,
            num_layers=2,
            output_dim=self.pred_len,
            bidirectional=True
        )
        self.model.load_state_dict(torch.load(model_file, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        scaler_features_path = model_dir / "scaler_features.pkl"
        scaler_target_path = model_dir / "scaler_target.pkl"
        feature_cols_path = model_dir / "feature_columns.pkl"

        if scaler_features_path.exists():
            self.scaler_features = joblib.load(scaler_features_path)
        if scaler_target_path.exists():
            self.scaler_target = joblib.load(scaler_target_path)
        if feature_cols_path.exists():
            self.feature_cols = joblib.load(feature_cols_path)

    def predict_from_file(self, file_path: str, date_str: Optional[str] = None) -> pd.DataFrame:
        df = pd.read_csv(file_path, parse_dates=['date'])
        if date_str:
            df = df[df['date'].dt.strftime('%Y-%m-%d') == date_str]
            if df.empty:
                raise ValueError(f"No data found for date: {date_str}")

        preprocessed_data = self.preprocess(df)
        predictions = self.predict(preprocessed_data)

        result_df = pd.DataFrame({
            'date': df['date'].values,
            'hour': df['hour'].values,
            'Predicted': predictions
        })

        if 'Electricity_price_MWh' in df.columns:
            result_df['True'] = df['Electricity_price_MWh'].values
            result_df['Pct_of_True'] = result_df['Predicted'] / result_df['True'] * 100

        return result_df

    def get_model_info(self) -> Dict[str, Union[str, int]]:
        return {
            "mapcode": self.mapcode,
            "seq_len": self.seq_len,
            "pred_len": self.pred_len,
            "device": str(self.device),
            "model_dir": str(self.gru_dir)
        }