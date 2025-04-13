import torch
import torch.nn as nn

class GRUModel(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=128, num_layers=1, bidirectional=False):
        """
        input_dim:      Input size from Informer encoder (e.g., 512)
        hidden_dim:     Number of GRU units
        num_layers:     GRU depth
        bidirectional:  Whether to use bidirectional GRU
        """
        super().__init__()
        self.bidirectional = bidirectional
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

    def forward(self, x):
        """
        x: [batch, seq_len, input_dim]
        Returns: [batch, hidden_dim] (or hidden_dim * 2 if bidirectional)
        """
        out, _ = self.gru(x)

        # Pooled output
        if self.bidirectional:
            pooled = out.mean(dim=1)  # [B, 2H]
        else:
            pooled = out[:, -1, :]    # [B, H]

        return pooled
