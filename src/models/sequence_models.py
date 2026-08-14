import torch
import torch.nn as nn
import math

class SFIILSTM(nn.Module):
    """
    Long Short-Term Memory network for continuous SFII prediction based on 
    spatiotemporal feature sequences.
    """
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float = 0.2):
        super(SFIILSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Expecting input shape: [Batch, Time, Features]
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected regression head
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 1) # Predicting continuous SFII score (0-1)
        
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        
        # Take the output of the last time step
        out = out[:, -1, :]
        
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        # SFII is strictly bounded between 0 and 1, so we apply Sigmoid
        return torch.sigmoid(out)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class SFIITransformer(nn.Module):
    """
    Transformer Encoder architecture for SFII prediction. 
    Can capture long-range phenological dependencies better than LSTM, but requires more data.
    """
    def __init__(self, input_size: int, d_model: int, nhead: int, num_layers: int, dropout: float = 0.1):
        super(SFIITransformer, self).__init__()
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        self.fc = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        
        # Output shape: (batch_size, seq_length, d_model)
        x = self.transformer_encoder(x)
        
        # Mean pooling over the sequence length
        x = torch.mean(x, dim=1)
        
        return self.fc(x)
