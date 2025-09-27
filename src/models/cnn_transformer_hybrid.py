# models/cnn_transformer_hybrid.py
from __future__ import annotations
import torch
import torch.nn as nn


class CNNTransformerHybrid(nn.Module):
    """
    1D CNN (local patterns) + Transformer encoder (global dependencies) regressor.
    """
    def __init__(
        self,
        input_size: int,
        cnn_out_channels: int = 64,
        kernel_size: int = 3,
        num_heads: int = 8,
        hidden_dim: int = 128,
        num_layers: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        # CNN expects [B, C=in_channels, L]
        self.conv1 = nn.Conv1d(
            in_channels=input_size,
            out_channels=cnn_out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.bn1 = nn.BatchNorm1d(cnn_out_channels)
        self.relu = nn.ReLU()

        # Transformer on features=cnn_out_channels
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cnn_out_channels,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=False,  # expects [L, B, D]
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(cnn_out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, seq_len, features]
        returns: [batch]
        """
        # CNN: [B, L, D] -> [B, D, L]
        x = x.transpose(1, 2)
        x = self.relu(self.bn1(self.conv1(x)))  # [B, C, L]

        # Transformer: [B, C, L] -> [L, B, C]
        x = x.transpose(1, 2).transpose(0, 1)
        x = self.encoder(x)                     # [L, B, C]
        x = x[-1, :, :]                         # last token [B, C]
        x = self.fc(x)                          # [B, 1]
        return x.squeeze(-1)
