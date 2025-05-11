import torch
import torch.nn as nn

class GRUModel(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=128, num_layers=1, output_dim=24, bidirectional=False):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=bidirectional)
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        
        # If bidirectional, we need to double the size of the hidden dimension for the regressor
        regressor_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.regressor = nn.Linear(regressor_input_dim, output_dim)

    def forward(self, x):
        # x: [batch, seq_len, input_dim] → e.g. [32, 168, 512]
        _, hidden = self.gru(x)  # hidden: [num_layers * (2 if bidir else 1), batch, hidden_dim]
        
        # If bidirectional, concatenate the last forward and backward hidden states
        if self.bidirectional:
            # For bidirectional, hidden shape: [num_layers*2, batch, hidden_dim]
            # Extract the last forward and backward hidden states
            last_forward = hidden[-2]  # Second-to-last layer is the forward of the last layer
            last_backward = hidden[-1]  # Last layer is the backward of the last layer
            hidden_concat = torch.cat((last_forward, last_backward), dim=1)
            output = self.regressor(hidden_concat)
        else:
            # For unidirectional, hidden shape: [num_layers, batch, hidden_dim]
            output = self.regressor(hidden[-1])  # use last layer's hidden state
            
        return output  # e.g. [batch, 24] → next 24h price prediction
