from typing import Optional
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
from ml_models.GRU.GRU_model import GRUModel
from interfaces.ModelPipelineInterface import IModelPipeline

class GRUPipeline(IModelPipeline):
    def __init__(self, input_size, hidden_size, num_layers, output_size, model_path: str):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.seq_len = 168
        self.model = GRUModel(input_size, hidden_size, num_layers, output_size)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        self.scaler = StandardScaler()

    def load_model(self, model_path: Optional[str] = None) -> None:
        if model_path is None:
            raise ValueError("Model path must be provided.")
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def preprocess(self, df: pd.DataFrame) -> torch.Tensor:
        values = self.scaler.fit_transform(df.values)
        sequences = []
        for i in range(len(values) - self.seq_len + 1):
            seq = values[i:i + self.seq_len]
            sequences.append(seq)
        return torch.tensor(sequences, dtype=torch.float32)

    def predict(self, input_tensor: torch.Tensor) -> pd.Series:
        input_tensor = input_tensor.to(self.device)
        with torch.no_grad():
            prediction = self.model(input_tensor)
        return pd.Series(prediction.cpu().numpy().flatten())

    def predict_from_file(self, file_path: str, date_str: Optional[str] = None) -> pd.DataFrame:
        df = pd.read_csv(file_path, parse_dates=['date'])
        if not date_str:
            raise ValueError("Prediction date must be specified.")

        prediction_date = pd.to_datetime(date_str)
        historical_data = df[df['date'] < prediction_date].tail(self.seq_len)

        if len(historical_data) < self.seq_len:
            raise ValueError(
                f"Insufficient data for prediction. Expected at least {self.seq_len} rows, got {len(historical_data)}."
            )

        # Extract only feature columns
        feature_data = historical_data.drop(columns=['date'])
        input_tensor = self.preprocess(feature_data).unsqueeze(0)

        prediction = self.predict(input_tensor)
        return pd.DataFrame({
            'date': [prediction_date],
            'prediction': [prediction.iloc[0]]
        })
