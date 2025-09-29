# models/transformer_forecast.py
from __future__ import annotations
import torch
import torch.nn as nn


class TimeSeriesTransformer(nn.Module):
    """
    Transformer encoder regressor.
    Uses raw feature dimension as d_model (requires input_size % num_heads == 0).
    """
    def __init__(self, input_size: int, num_heads: int = 8, hidden_dim: int = 128, num_layers: int = 6, dropout: float = 0.1):
        super().__init__()
        print("input" , input_size)
        print("num of heads : " , num_heads)
        assert input_size % num_heads == 0, "input_size must be divisible by num_heads"

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_size,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=False,  # expects [L, B, D]
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(input_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, seq_len, features]
        returns: [batch]
        """
        x = x.transpose(0, 1)  # [L, B, D]
        x = self.encoder(x)    # [L, B, D]
        x = x[-1, :, :]        # last token [B, D]
        x = self.fc(x)         # [B, 1]
        return x.squeeze(-1)
