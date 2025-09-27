# src/datasets/solar_dataset.py
# ======================================
# DATASET: grouped sliding windows (PyTorch)
# ======================================

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Iterable, Tuple, Optional
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


class SolarDatasetGrouped(Dataset):
    """
    Builds sliding windows per (CampusKey, SiteKey) group to avoid cross-group mixing.
    Target is a single-step (or horizon-step) scalar for each window.

    Args:
        csv_file: Path to master CSV (must include Timestamp, CampusKey, SiteKey, target_col, feature_cols)
        feature_cols: list of feature column names
        target_col: target column name (e.g., 'y_norm')
        lookback: number of past timesteps per sample
        horizon: predict the value at t + horizon (1 = next step)
        group_cols: columns defining groups (default: ('CampusKey','SiteKey'))
        scaler: fitted StandardScaler for features; if None, raw features are used (no scaling)
    """
    def __init__(
        self,
        csv_file: str,
        feature_cols: Iterable[str],
        target_col: str,
        lookback: int = 24,
        horizon: int = 1,
        group_cols: Tuple[str, str] = ("CampusKey", "SiteKey"),
        scaler: Optional[StandardScaler] = None,
    ):
        df = pd.read_csv(csv_file, parse_dates=["Timestamp"])
        df = df.sort_values(list(group_cols) + ["Timestamp"]).reset_index(drop=True)

        self.feature_cols = list(feature_cols)
        self.target_col   = target_col
        self.lookback     = int(lookback)
        self.horizon      = int(horizon)
        self.group_cols   = list(group_cols)

        # Keep original df for reference (e.g., mapping timestamps/capacities if needed)
        self.df = df

        # Transform features (using pre-fitted scaler if provided)
        if scaler is not None:
            F_all = scaler.transform(df[self.feature_cols].values)
        else:
            F_all = df[self.feature_cols].values

        if np.isnan(F_all).any():
            print(f"Warning: {np.isnan(F_all).sum()} NaN values found in features after scaling.")

        self.seq_X, self.seq_y, self.seq_t = [], [], []  # X, y, sequence-end timestamp

        # Build sequences per group to avoid cross-group leakage
        start_idx = 0
        for _, g in df.groupby(self.group_cols, sort=False):
            n = len(g)
            if n < self.lookback + self.horizon:
                start_idx += n
                continue

            F = F_all[start_idx:start_idx + n]
            y = g[self.target_col].values
            t = g["Timestamp"].values

            if np.isnan(y).any():
                key = g[self.group_cols].iloc[0].tolist()
                print(f"Warning: {np.isnan(y).sum()} NaNs in target for group {key} before sequence creation.")

            L, H = self.lookback, self.horizon
            for i in range(n - L - H + 1):
                self.seq_X.append(F[i:i + L])
                self.seq_y.append(y[i + L + H - 1])
                self.seq_t.append(t[i + L + H - 1])

            start_idx += n

        self.seq_X = np.asarray(self.seq_X, dtype=np.float32)     # (N, L, D)
        self.seq_y = np.asarray(self.seq_y, dtype=np.float32)     # (N,)
        self.seq_t = np.asarray(self.seq_t)                        # datetime64

    def __len__(self) -> int:
        return len(self.seq_X)

    def __getitem__(self, idx: int):
        # Returns: (X: [L, D], y: [])
        return torch.tensor(self.seq_X[idx]), torch.tensor(self.seq_y[idx])
