import torch
import torch.nn as nn

class SFIILSTM(nn.Module):
    """
    LSTM architecture tailored for temporal degradation trajectories.
    Expects input shape: [Batch, Sequence_Length, Features]
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.3):
        super(SFIILSTM, self).__init__()
        self.name = "LSTM"
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 1) # Binary classification output (Degraded vs Intact)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [Batch, Seq_Len, Features]
        # We assume the time dimension is appropriately structured.
        # If input is [Batch, Features], we unsqueeze it to [Batch, 1, Features]
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        lstm_out, _ = self.lstm(x)
        
        # Take the output of the last time step
        last_step_out = lstm_out[:, -1, :]
        
        out = self.fc1(last_step_out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return self.sigmoid(out)
