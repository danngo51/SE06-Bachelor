def gru_test():
    return "This is a test from the GRU model."

import os
import torch
import torch.nn as nn
from ml_models.gru.gruModel.gruModel import GRUModel

class GRUWrapper:
    def __init__(self, gru_path, regressor_path, input_dim=512, hidden_dim=128, output_dim=24, device="cpu", bidirectional=False):
        self.device = device

        # Load GRU
        self.gru = GRUModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            bidirectional=bidirectional
        ).to(device)

        if os.path.exists(gru_path):
            self.gru.load_state_dict(torch.load(gru_path, map_location=device))
            print(f"✅ Loaded GRU weights from {gru_path}")
        else:
            raise FileNotFoundError(f"❌ GRU weights not found at {gru_path}")

        # Load Regressor
        self.regressor = nn.Linear(
            hidden_dim * (2 if bidirectional else 1),
            output_dim
        ).to(device)

        if os.path.exists(regressor_path):
            self.regressor.load_state_dict(torch.load(regressor_path, map_location=device))
            print(f"✅ Loaded Regressor weights from {regressor_path}")
        else:
            raise FileNotFoundError(f"❌ Regressor weights not found at {regressor_path}")

        # Set eval mode
        self.gru.eval()
        self.regressor.eval()

    def run(self, embedding_tensor: torch.Tensor):
        """
        embedding_tensor: [batch_size, seq_len, input_dim] from Informer
        Returns: [batch_size, output_dim] final prediction
        """
        with torch.no_grad():
            embedding_tensor = embedding_tensor.to(self.device)
            gru_out = self.gru(embedding_tensor)        # [batch, hidden_dim]
            pred = self.regressor(gru_out)               # [batch, output_dim]
        return pred
