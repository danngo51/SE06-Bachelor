from typing import Optional
import torch
import pandas as pd
import os
import pathlib
from sklearn.preprocessing import StandardScaler
import joblib
from ml_models.GRU.GRU_model import GRUModel
from interfaces.ModelPipelineInterface import IModelPipeline

class GRUPipeline(IModelPipeline):
    def __init__(self, mapcode="DK1", model_path: Optional[str] = None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.seq_len = 168
        self.pred_len = 24
        self.mapcode = mapcode
        
        # Set default paths if not provided
        if model_path is None:
            current_file = pathlib.Path(__file__)
            project_root = current_file.parent.parent
            data_dir = project_root / "data" / self.mapcode
            gru_dir = data_dir / "gru"
            model_path = str(gru_dir / "gru_model.pt")
            scaler_path = str(gru_dir / "scaler.pkl")
            
            # Check if model exists
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found at {model_path}. Please train the model first.")
        else:
            scaler_path = os.path.join(os.path.dirname(model_path), "scaler.pkl")
        
        # Try to load scaler if exists
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
        else:
            self.scaler = StandardScaler()
        
        # Load model
        # GRU parameters - we'll determine these from the saved model if possible
        input_size = 4  # Default input size (adjust based on your data features)
        hidden_size = 64
        num_layers = 2
        output_size = self.pred_len
        
        self.model = GRUModel(input_size, hidden_size, num_layers, output_size)
        
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f"Model loaded successfully from {model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {model_path}: {str(e)}")
            
        self.model.to(self.device)
        self.model.eval()

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
        print("predict_from_file - gru pipe")
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
