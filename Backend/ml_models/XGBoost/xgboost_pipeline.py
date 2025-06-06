import os
import pandas as pd
import numpy as np
import joblib
import pathlib
from typing import Dict, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from interfaces.ModelPipelineInterface import IModelPipeline
from ml_models.XGBoost.XGBoost_model import XGBoostRegimeModel

class XGBoostPipeline(IModelPipeline):
    def __init__(self, mapcode: str = "DK1"):
        self.mapcode = mapcode
        self.threshold = None
        self.model_normal = None
        self.model_spike = None
        self.model_trainer = None

        current_file = pathlib.Path(__file__)
        self.project_root = current_file.parent.parent
        self.data_dir = self.project_root / "data" / self.mapcode
        self.xgboost_dir = self.data_dir / "xgboost"

    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        pass

    def predict(self, data: pd.DataFrame) -> pd.Series:
        if self.model_normal is None or self.model_spike is None or self.threshold is None:
            raise ValueError("Models not loaded. Call load_model() first.")

        is_spike = data['EP_lag_1'] > self.threshold
        predictions = pd.Series(index=data.index, dtype=float)

        if is_spike.any():
            predictions.loc[is_spike] = self.model_spike.predict(data.loc[is_spike])
        if (~is_spike).any():
            predictions.loc[~is_spike] = self.model_normal.predict(data.loc[~is_spike])

        return predictions

    def load_model(self, model_path: Optional[str] = None) -> None:
        xgboost_dir = pathlib.Path(model_path) if model_path else self.xgboost_dir

        normal_model_path = xgboost_dir / "model_normal.pkl"
        spike_model_path = xgboost_dir / "model_spike.pkl"
        threshold_path = xgboost_dir / "threshold.txt"

        if not normal_model_path.exists() or not spike_model_path.exists():
            raise FileNotFoundError("Model files not found")

        self.model_normal = joblib.load(normal_model_path)
        self.model_spike = joblib.load(spike_model_path)

        if threshold_path.exists():
            with open(threshold_path, 'r') as f:
                self.threshold = float(f.read().strip())
        else:
            training_data_path = self.data_dir / f"{self.mapcode}_full_data_2018_2024.csv"
            if training_data_path.exists():
                dataset = pd.read_csv(training_data_path)
                self.threshold = dataset['Electricity_price_MWh'].quantile(0.90)
            else:
                self.threshold = 100.0

    def predict_from_file(self, file_path: str, date_str: Optional[str] = None) -> pd.DataFrame:
        df = pd.read_csv(file_path, parse_dates=['date'])

        if date_str:
            df = df[df['date'].dt.strftime('%Y-%m-%d') == date_str]
            if df.empty:
                raise ValueError(f"No data found for date: {date_str}")

        base_features = [c for c in df.columns if c not in ['date', 'Electricity_price_MWh']]
        X = df[base_features]

        df['Predicted'] = self.predict(X)
        df['True'] = df['Electricity_price_MWh']
        df['Pct_of_True'] = df['Predicted'] / df['True'] * 100

        return df[['date', 'hour', 'True', 'Predicted', 'Pct_of_True']]

    def train_model(self, train_file: str, xgb_params: Optional[Dict] = None) -> bool:
        if self.model_trainer is None:
            self.model_trainer = XGBoostRegimeModel(mapcode=self.mapcode)

        results = self.model_trainer.train(training_file=train_file, xgb_params=xgb_params)
        if results:
            self.load_model()
            return True
        return False

    def evaluate_model(self, test_file: str) -> Dict[str, float]:
        if self.model_normal is None or self.model_spike is None:
            raise ValueError("Models not loaded. Call load_model() first.")

        results = self.model_trainer.predict(forecast_file=test_file)
        true_values = results['True']
        predictions = results['Predicted']

        mse = mean_squared_error(true_values, predictions)
        mae = mean_absolute_error(true_values, predictions)
        r2 = r2_score(true_values, predictions)

        return {
            "mse": mse,
            "mae": mae,
            "r2": r2,
            "rmse": np.sqrt(mse)
        }