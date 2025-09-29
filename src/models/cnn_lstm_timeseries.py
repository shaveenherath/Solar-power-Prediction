# models/cnn_lstm_timeseries.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNLSTMTimeSeries(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, cnn_filters=32, kernel_size=3, dropout=0.1):
        super().__init__()
        self.conv = nn.Conv1d(input_size, cnn_filters, kernel_size, padding=kernel_size//2)
        self.lstm = nn.LSTM(cnn_filters, hidden_size, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: [B, L, F] → Conv1d needs [B, F, L]
        x = x.transpose(1, 2)
        x = F.relu(self.conv(x))
        x = x.transpose(1, 2)  # back to [B, L, F]
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])  # last timestep
        out = self.fc(out)
        return out.squeeze(-1)
