def gru_test():
    return "This is a test from the GRU model."

import os
import torch
import torch.nn as nn
from ml_models.gru.gruModel.gruModel import GRUModel

class GRUWrapper:
    def __init__(self, gru_path, regressor_path=None, input_dim=512, hidden_dim=128, output_dim=24, device="cpu", bidirectional=False):
        """
        Initialize the GRU model wrapper with more resilience to missing files
        
        Args:
            gru_path: Path to the GRU model weights
            regressor_path: Path to the regressor model weights (not used anymore)
            input_dim: Dimension of the input features (should match Informer's d_model)
            hidden_dim: Hidden dimension of the GRU
            output_dim: Output dimension (prediction hours, typically 24)
            device: Device to run the model on
            bidirectional: Whether to use bidirectional GRU
        """
        self.device = device
        self.output_dim = output_dim

        # Initialize GRU model with default parameters
        self.gru = GRUModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            bidirectional=bidirectional
        ).to(device)

        try:
            if gru_path and os.path.exists(gru_path):
                self.gru.load_state_dict(torch.load(gru_path, map_location=device))
                print(f"✅ Loaded GRU weights from {gru_path}")
            else:
                print(f"⚠️ GRU weights not found at {gru_path}. Using untrained model.")
        except Exception as e:
            print(f"❌ Error loading GRU weights: {e}. Using untrained model.")
            
        # Set eval mode
        self.gru.eval()
        
    def run(self, embedding_tensor: torch.Tensor):
        """
        embedding_tensor: [batch_size, seq_len, input_dim] from Informer
        Returns: [batch_size, output_dim] final prediction with non-negative values
        """
        with torch.no_grad():
            embedding_tensor = embedding_tensor.to(self.device)
            pred = self.gru(embedding_tensor)  # [batch, output_dim]
            
            # Electricity prices are rarely negative, so we can enforce non-negative predictions
            # Apply ReLU-like operation manually to ensure predictions are non-negative
            pred = torch.max(pred, torch.zeros_like(pred))
            
        return pred
