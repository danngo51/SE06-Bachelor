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
            output_dim=output_dim,
            bidirectional=bidirectional
        ).to(device)

        if os.path.exists(gru_path):
            self.gru.load_state_dict(torch.load(gru_path, map_location=device))
            print(f"✅ Loaded GRU weights from {gru_path}")
        else:
            raise FileNotFoundError(f"❌ GRU weights not found at {gru_path}")

        # Set eval mode
        self.gru.eval()

    def run(self, embedding_tensor: torch.Tensor):
        """
        embedding_tensor: [batch_size, seq_len, input_dim] from Informer
        Returns: [batch_size, output_dim] final prediction
        """
        with torch.no_grad():
            embedding_tensor = embedding_tensor.to(self.device)
            pred = self.gru(embedding_tensor)  # [batch, output_dim]
        return pred
