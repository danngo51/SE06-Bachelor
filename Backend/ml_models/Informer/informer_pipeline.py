import os
import pandas as pd
import numpy as np
import torch
import pathlib
import joblib
from typing import Dict, List, Optional, Union, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from interfaces.ModelPipelineInterface import IModelPipeline
from ml_models.Informer.Informer_model import InformerModelTrainer

class InformerPipeline(IModelPipeline):
    def __init__(self, mapcode: str = "DK1", seq_len: int = 168, label_len: int = 48, pred_len: int = 24):
        self.mapcode = mapcode
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.model = None
        self.scaler = None
        self.training_feature_cols = None
        self.model_trainer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else 
                                   "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else 
                                   "cpu")

        current_file = pathlib.Path(__file__)
        self.project_root = current_file.parent.parent
        self.data_dir = self.project_root / "data" / self.mapcode
        self.informer_dir = self.data_dir / "informer"

    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()

        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df['day_of_week'] = df['date'].dt.dayofweek
            df['month'] = df['date'].dt.month
            df['day'] = df['date'].dt.day

            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        if 'Electricity_price_MWh' in df.columns:
            df['price_roll_24h'] = df['Electricity_price_MWh'].shift(1).rolling(24, min_periods=1).mean()
            df['price_roll_7d'] = df['Electricity_price_MWh'].shift(1).rolling(168, min_periods=1).mean()
            df['price_std_24h'] = df['Electricity_price_MWh'].shift(1).rolling(24, min_periods=1).std()

        load_columns = ['DAHTL_TotalLoadValue', 'TotalLoadValue', 'Load_Value', 'Total_Load', 'DALoadValue', 'SystemLoad']
        load_col = next((col for col in load_columns if col in df.columns), None)
        if load_col:
            df['load_roll_24h'] = df[load_col].shift(1).rolling(24, min_periods=1).mean()

        if 'Electricity_price_MWh' in df.columns:
            for lag in [1, 24, 48, 72, 168]:
                df[f'price_lag_{lag}'] = df['Electricity_price_MWh'].shift(lag)

        return df.dropna()

    def predict(self, data: pd.DataFrame, prediction_date: str) -> pd.Series:
        if self.model is None or self.scaler is None:
            self.load_model()

        prediction_date = pd.to_datetime(prediction_date)
        if prediction_date not in data['date'].values:
            raise ValueError(f"Prediction date {prediction_date} not found in the input data.")

        historical_data = data[data['date'] < prediction_date].tail(self.seq_len)

        if len(historical_data) < self.seq_len:
            raise ValueError(f"Insufficient data for prediction. Expected at least {self.seq_len} rows.")

        informer_tensor, feature_cols = self._prepare_informer_data(historical_data)
        self.model.eval()

        dec_inp = torch.zeros((1, self.label_len + self.pred_len), device=self.device)
        if 'Electricity_price_MWh' in historical_data.columns and len(historical_data) >= self.label_len and self.label_len > 0:
            target_data = historical_data['Electricity_price_MWh'].values[-self.label_len:]
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

        # Create a Series for the next 24 hours
        prediction_dates = pd.date_range(start=prediction_date, periods=self.pred_len, freq='H')
        return pd.Series(predictions, index=prediction_dates, name='Predicted')

    def load_model(self, model_path: Optional[str] = None) -> bool:
        informer_dir = pathlib.Path(model_path) if model_path else self.informer_dir

        try:
            model_path = str(informer_dir / "best_informer.pt")
            scaler_path = str(informer_dir / "scaler.pkl")

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Informer model file not found: {model_path}")

            self.model = torch.load(model_path, map_location=self.device)
            self.model.eval()

            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)

            return True
        except Exception as e:
            print(f"Error loading Informer model: {e}")
            return False

    def train_model(self, data_path: str, **training_kwargs) -> bool:
        try:
            if self.model_trainer is None:
                self.model_trainer = InformerModelTrainer(
                    mapcode=self.mapcode,
                    seq_len=self.seq_len,
                    label_len=self.label_len,
                    pred_len=self.pred_len,
                    **training_kwargs
                )

            results = self.model_trainer.train(data_path=data_path)
            if results:
                self.load_model()
                return True
            return False
        except Exception as e:
            print(f"Error training Informer model: {e}")
            return False

    def evaluate_model(self, test_file: str) -> Dict[str, float]:
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        try:
            df = pd.read_csv(test_file, parse_dates=['date'])
            df = self.preprocess(df)

            predictions = self.predict(df)
            true_values = df['Electricity_price_MWh'].tail(len(predictions))

            mse = mean_squared_error(true_values, predictions)
            mae = mean_absolute_error(true_values, predictions)
            r2 = r2_score(true_values, predictions)

            return {
                "mse": mse,
                "mae": mae,
                "r2": r2,
                "rmse": np.sqrt(mse)
            }
        except Exception as e:
            print(f"Error evaluating Informer model: {e}")
            return {}
        
    def _prepare_informer_data(self, historical_data: pd.DataFrame) -> Tuple[torch.Tensor, List[str]]:
        if self.training_feature_cols is None:
            raise ValueError("Training feature columns are not defined. Ensure the model is trained first.")

        feature_cols = [col for col in self.training_feature_cols if col in historical_data.columns]
        if not feature_cols:
            raise ValueError("No matching feature columns found in the historical data.")

        # Extract features and scale them
        features = historical_data[feature_cols].values.astype(np.float32)
        if self.scaler is not None:
            features = self.scaler.transform(features)

        # Convert features to PyTorch tensor
        informer_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

        return informer_tensor, feature_cols