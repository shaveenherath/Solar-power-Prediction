# models/lstm_forecast.py
from __future__ import annotations
import torch
import torch.nn as nn


class LSTMForecast(nn.Module):
    """
    Simple LSTM regressor that reads a sequence [B, L, D] and predicts a single scalar per sample.
    """
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,  # PyTorch only applies dropout if num_layers > 1
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, seq_len, features]
        returns: [batch]
        """
        out, _ = self.lstm(x)          # [B, L, H]
        out = out[:, -1, :]            # last timestep [B, H]
        out = self.fc(out)             # [B, 1]
        return out.squeeze(-1)         # [B]
