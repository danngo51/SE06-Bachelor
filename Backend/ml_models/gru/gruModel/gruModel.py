import torch
import torch.nn as nn

class GRUModel(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=128, num_layers=1, output_dim=24):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.regressor = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: [batch, seq_len, input_dim] → e.g. [32, 168, 512]
        _, hidden = self.gru(x)  # hidden: [num_layers, batch, hidden_dim]
        output = self.regressor(hidden[-1])  # use last layer’s hidden state
        return output  # e.g. [batch, 24] → next 24h price prediction
