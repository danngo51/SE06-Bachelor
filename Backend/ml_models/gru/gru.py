def gru_test():
    return "This is a test from the GRU model."

import torch
import os
from gruModel.gruModel import GRUModel

class GRUWrapper:
    def __init__(self, input_dim=512, hidden_dim=128, model_path=None, device="cpu", bidirectional=False):
        self.device = device
        self.model = GRUModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            bidirectional=bidirectional
        ).to(device)

        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            print("✅ Loaded GRU weights from:", model_path)

        self.model.eval()

    def run(self, embedding_tensor: torch.Tensor):
        """
        embedding_tensor: [batch, seq_len, input_dim] from Informer
        Returns: [batch, hidden_dim] output from GRU
        """
        with torch.no_grad():
            return self.model(embedding_tensor.to(self.device))
