from typing import Optional
import numpy as np
import torch
import pandas as pd
import os
import pathlib
from sklearn.preprocessing import StandardScaler
import joblib
from ml_models.GRU.GRU_model import GRUModel
from interfaces.ModelPipelineInterface import IModelPipeline
import tempfile

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
            model_path = str(gru_dir / "gru_model.pth")
            scaler_path = str(gru_dir / "scaler.pkl")
            
            # Only check if file exists, don't try to access directory
            if not os.path.isfile(model_path):
                raise FileNotFoundError(f"Model not found at {model_path}. Please train the model first.")
        else:
            scaler_path = os.path.join(os.path.dirname(model_path), "scaler.pkl")
        
        # Store paths for reference but don't try to write to them
        self.model_path = model_path
        self.scaler_path = scaler_path
        
        # Try to load scaler if exists - directly from file without accessing directory
        if os.path.isfile(scaler_path):
            try:
                self.scaler = joblib.load(scaler_path)
                print(f"Scaler loaded successfully from {scaler_path}")
            except Exception as e:
                print(f"Error loading scaler: {e}. Using default StandardScaler.")
                self.scaler = StandardScaler()
        else:
            print(f"Scaler not found at {scaler_path}. Using default StandardScaler.")
            self.scaler = StandardScaler()
        
        # Load model without attempting to access directory structure
        try:
            # Load directly from file path
            self.load_model(model_path)
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise

    def load_model(self, model_path: str) -> None:
        # Load the state_dict directly from file
        try:
            # Direct file load without directory checks
            state_dict = torch.load(model_path, map_location=self.device)
            
            # Determine input_size from the weight dimensions of the first GRU layer
            input_size = state_dict['gru.weight_ih_l0'].shape[1]
            hidden_size = state_dict['gru.weight_ih_l0'].shape[0] // 3
            num_layers = sum(1 for key in state_dict if 'weight_ih_l' in key)
            output_size = self.pred_len
            
            print(f"Detected model parameters - input_size: {input_size}, hidden_size: {hidden_size}, num_layers: {num_layers}")
            
            # Initialize model with the correct dimensions
            self.model = GRUModel(input_size, hidden_size, num_layers, output_size)
            
            # Load the weights
            self.model.load_state_dict(state_dict)
            print(f"Model loaded successfully from {model_path}")
                
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            # Report exception with file path, not directory
            model_filename = os.path.basename(model_path)
            raise RuntimeError(f"Failed to load model file {model_filename}: {str(e)}")

    def preprocess(self, df: pd.DataFrame) -> torch.Tensor:
        # Transform the input data using the saved scaler
        try:
            # Make a copy to avoid modifying the original DataFrame
            df_copy = df.copy()
            
            # Check if scaler is fitted
            if hasattr(self.scaler, 'mean_'):
                values = self.scaler.transform(df_copy.values)
            else:
                print("Warning: Scaler not fitted. Using fit_transform instead.")
                values = self.scaler.fit_transform(df_copy.values)
            
            input_size = self.model.gru.input_size if hasattr(self.model, 'gru') else self.model.gru.weight_ih_l0.shape[1]
            
            # If the input data has fewer features than the model expects, pad with zeros
            if values.shape[1] < input_size:
                pad_width = ((0, 0), (0, input_size - values.shape[1]))
                values = np.pad(values, pad_width, mode='constant', constant_values=0)
            # If it has more features than expected, truncate
            elif values.shape[1] > input_size:
                values = values[:, :input_size]
            
            sequences = []
            for i in range(len(values) - self.seq_len + 1):
                seq = values[i:i + self.seq_len]
                sequences.append(seq)
            
            return torch.tensor(np.array(sequences), dtype=torch.float32)
        except Exception as e:
            print(f"Error in preprocessing: {str(e)}")
            raise

    def predict(self, input_tensor: torch.Tensor) -> pd.Series:
        try:
            input_tensor = input_tensor.to(self.device)
            with torch.no_grad():
                prediction = self.model(input_tensor)
            return pd.Series(prediction.cpu().numpy().flatten())
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            raise

    def predict_from_file(self, file_path: str, date_str: Optional[str] = None) -> pd.DataFrame:
        try:
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
                'Predicted': [prediction.iloc[0]]  # Changed from 'prediction' to 'Predicted' to match other models
            })
        except Exception as e:
            print(f"Error in predict_from_file: {str(e)}")
            raise
