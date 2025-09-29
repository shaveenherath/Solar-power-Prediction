# src/datasets/solar_dataset.py
# ======================================
# DATASET: grouped sliding windows (PyTorch)
# ======================================

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Iterable, Tuple, Optional, Dict, Any, List
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

from src.utils.feature_engineering import make_features


def _make_lag_features(
    df: pd.DataFrame,
    base_cols: List[str],
    lags: List[int] | None = None,
    windows: List[int] | None = None,
    group_cols: Tuple[str, str] = ("CampusKey","SiteKey"),
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Optional: create lagged/rolling features for numeric base columns (NOT the target).
    This is OFF by default; only used if lags/windows provided.
    """
    if not lags and not windows:
        return df, []

    df = df.sort_values(list(group_cols) + ["Timestamp"]).copy()
    new_cols: List[str] = []

    # group-wise ops to avoid leakage across sites
    gobj = df.groupby(list(group_cols), sort=False, group_keys=False)

    # lags
    if lags:
        for col in base_cols:
            for L in lags:
                name = f"{col}_lag{L}"
                df[name] = gobj[col].shift(L)
                new_cols.append(name)

    # rolling means (use centered=False to keep causality)
    if windows:
        for col in base_cols:
            for W in windows:
                name = f"{col}_rollmean{W}"
                df[name] = gobj[col].rolling(window=W, min_periods=W).mean().reset_index(level=list(range(len(group_cols))), drop=True)
                new_cols.append(name)

    return df, new_cols


class SolarDatasetGrouped(Dataset):
    """
    Builds sliding windows per (CampusKey, SiteKey) group to avoid cross-group mixing.
    Target is a single-step (or horizon-step) scalar for each window.

    Args:
        csv_file: path to master CSV
        feature_cols: list of "base" feature names (time encodings/lag features may be appended here)
        target_col: target name (e.g., 'y_norm')
        lookback: steps in the past per sample
        horizon: predict value at t + horizon (1 = next step)
        group_cols: group keys
        scaler: fitted StandardScaler for features; if None, raw features are used
        time_encoding: dict with {scheme, params} (see utils/time_encoding.py). None => no time columns
        lag_cfg: optional dict: {"cols": [...], "lags": [1,2], "windows": [3,6]}
                 Default None => no lag features
        verbose: print drop stats
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
        time_encoding: Dict[str, Any] | None = None,      # NEW
        lag_cfg: Dict[str, Any] | None = None,            # NEW
        verbose: bool = True,
    ):
        # ---------- Load + basic sort ----------
        df = pd.read_csv(csv_file, parse_dates=["Timestamp"])
        df = df.sort_values(list(group_cols) + ["Timestamp"]).reset_index(drop=True)

        base_feature_cols = list(feature_cols)  # start with user-selected features
        all_feature_cols  = base_feature_cols.copy()

        df, all_feature_cols = make_features(
            df, base_cols=list(feature_cols),
            time_encoding=time_encoding,
            lag_cfg=lag_cfg,
            group_cols=group_cols,
        )

        # ---------- Drop rows with NaNs in target ----------
        n0 = len(df)
        df = df.dropna(subset=[target_col]).copy()
        if verbose:
            print(f"[NaN drop] Target '{target_col}': {n0} -> {len(df)} rows")

        # ---------- Drop rows with NaNs in selected features ----------
        # Keep only columns we actually plan to feed to the model
        all_feature_cols = list(dict.fromkeys(all_feature_cols))  # de-dup
        missing = [c for c in all_feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Selected feature(s) not found in data: {missing}")

        n1 = len(df)
        df_feat = df[all_feature_cols]
        na_mask = df_feat.isna().any(axis=1)
        n_drop = int(na_mask.sum())
        if n_drop > 0:
            df = df.loc[~na_mask].copy()
        if verbose:
            print(f"[NaN drop] Features {len(all_feature_cols)} cols: {n1} -> {len(df)} rows (dropped {n_drop})")

        # ---------- Keep reference & feature matrix ----------
        self.df = df
        self.feature_cols = all_feature_cols
        self.target_col   = target_col
        self.lookback     = int(lookback)
        self.horizon      = int(horizon)
        self.group_cols   = list(group_cols)

        # ---------- Transform features (external scaler) ----------
        if scaler is not None:
            F_all = scaler.transform(df[self.feature_cols].values)
        else:
            F_all = df[self.feature_cols].values

        if np.isnan(F_all).any() and verbose:
            print(f"Warning: {np.isnan(F_all).sum()} NaNs found in features after scaling.")

        # ---------- Build sequences per group (no leakage) ----------
        self.seq_X, self.seq_y, self.seq_t = [], [], []
        start_idx = 0
        for _, g in df.groupby(self.group_cols, sort=False):
            n = len(g)
            if n < self.lookback + self.horizon:
                start_idx += n
                continue

            F = F_all[start_idx:start_idx + n]
            y = g[self.target_col].values
            t = g["Timestamp"].values

            L, H = self.lookback, self.horizon
            # if any target NaNs slipped through, warn:
            if np.isnan(y).any() and verbose:
                key = g[self.group_cols].iloc[0].tolist()
                print(f"Warning: {np.isnan(y).sum()} NaNs in target for group {key} (post-clean).")

            for i in range(n - L - H + 1):
                self.seq_X.append(F[i:i + L])
                self.seq_y.append(y[i + L + H - 1])
                self.seq_t.append(t[i + L + H - 1])

            start_idx += n

        self.seq_X = np.asarray(self.seq_X, dtype=np.float32)     # (N, L, D)
        self.seq_y = np.asarray(self.seq_y, dtype=np.float32)     # (N,)
        self.seq_t = np.asarray(self.seq_t)                        # datetime64

        if verbose:
            print(f"[Sequences] Built: N={len(self.seq_X)} | L={self.lookback} | D={self.seq_X.shape[-1] if len(self.seq_X)>0 else 'NA'}")

    def __len__(self) -> int:
        return len(self.seq_X)

    def __getitem__(self, idx: int):
        return torch.tensor(self.seq_X[idx]), torch.tensor(self.seq_y[idx])
