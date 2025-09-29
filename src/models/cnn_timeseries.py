# models/cnn_timeseries.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNTimeseries(nn.Module):
    def __init__(self, input_size, out_size=1, num_filters=64, kernel_size=3, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(input_size, num_filters, kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv1d(num_filters, num_filters, kernel_size, padding=kernel_size//2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters, out_size)

    def forward(self, x):
        # x: [B, L, F] → [B, F, L] for Conv1d
        x = x.transpose(1, 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.mean(dim=2)  # global average pooling over sequence
        x = self.dropout(x)
        x = self.fc(x)
        return x.squeeze(-1)
