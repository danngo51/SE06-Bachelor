import torch.nn as nn

class GRUModel(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=128, num_layers=1, bidirectional=False, dropout=0.0):
        """
        input_dim:      Size of input features (Informer encoder output dim)
        hidden_dim:     GRU hidden state size
        num_layers:     Number of GRU layers
        bidirectional:  Whether to use bidirectional GRU
        dropout:        Dropout between GRU layers
        """
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.bidirectional = bidirectional

    def forward(self, x):
        """
        x: Tensor of shape [batch, seq_len, input_dim]
        Returns: Tensor of shape [batch, hidden_dim] (or hidden_dim * 2 if bidirectional)
        """
        output, _ = self.gru(x)  # output: [B, T, H] or [B, T, 2H]

        # Use the final time step's hidden state (or mean pool)
        if self.bidirectional:
            pooled = output.mean(dim=1)  # [B, 2H]
        else:
            pooled = output[:, -1, :]    # [B, H]

        return pooled
