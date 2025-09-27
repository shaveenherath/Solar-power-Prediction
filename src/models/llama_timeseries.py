# models/llama_timeseries.py
from __future__ import annotations
import torch
import torch.nn as nn


class LLaMATimeSeries(nn.Module):
    """
    Lightweight decoder-only style Transformer for time series.
    Adds learnable positional embeddings and predicts a scalar from the last token.
    """
    def __init__(
        self,
        n_features: int,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 48,
    ):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.feature_embed = nn.Linear(n_features, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(max_seq_len, d_model))

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=False,  # expects [L, B, D]
        )
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=n_layers)
        self.fc_out = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, seq_len, features]
        returns: [batch]
        """
        bsz, seq_len, _ = x.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}")

        x = self.feature_embed(x)              # [B, L, D]
        x = x.transpose(0, 1)                  # [L, B, D]
        x = x + self.pos_embed[:seq_len].unsqueeze(1)  # add positional [L, 1, D]
        x = self.transformer(x)                # [L, B, D]
        x = x[-1, :, :]                        # last token [B, D]
        x = self.fc_out(x)                     # [B, 1]
        return x.squeeze(-1)
