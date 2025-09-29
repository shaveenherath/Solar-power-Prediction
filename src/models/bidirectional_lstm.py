# models/bidirectional_lstm.py
import torch
import torch.nn as nn

class BiLSTMTimeSeries(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True, bidirectional=True
        )
        self.fc = nn.Linear(hidden_size * 2, 1)  # *2 for bidirectional

    def forward(self, x):
        out, _ = self.lstm(x)  # [B, L, 2*H]
        out = out[:, -1, :]     # last timestep
        out = self.fc(out)
        return out.squeeze(-1)
