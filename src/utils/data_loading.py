# utils/data_loading.py
# ======================================
# SPLITS & DATALOADERS (PyTorch)
# ======================================

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Subset
from datasets.solar_dataset import SolarDatasetGrouped


def compute_time_cutoffs(
    timestamps: np.ndarray,
    train_frac: float = 0.7,
    val_frac: float = 0.2
) -> Tuple[np.datetime64, np.datetime64]:
    """
    Compute time-based cutoffs t1 (end of train) and t2 (end of val).
    """
    ts_sorted = np.sort(timestamps)
    n = len(ts_sorted)
    t1 = ts_sorted[int(n * train_frac)]
    t2 = ts_sorted[int(n * (train_frac + val_frac))]
    return t1, t2


def make_time_indices(
    seq_timestamps: np.ndarray,
    t1: np.datetime64,
    t2: np.datetime64
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split sequence indices by end-timestamp against t1/t2.
    """
    seq_timestamps = np.asarray(seq_timestamps)
    idx_train = np.where(seq_timestamps <= t1)[0]
    idx_val   = np.where((seq_timestamps > t1) & (seq_timestamps <= t2))[0]
    idx_test  = np.where(seq_timestamps > t2)[0]
    return idx_train, idx_val, idx_test


def create_dataloaders_grouped(
    csv_file: str,
    feature_cols: List[str],
    target_col: str,
    lookback: int = 24,
    horizon: int = 1,
    batch_size: int = 128,
    train_frac: float = 0.7,
    val_frac: float = 0.2,
    num_workers: int = 0,
    pin_memory: bool = False,
    shuffle_train: bool = True,
):
    """
    Creates train/val/test DataLoaders using a time-based split.
    IMPORTANT: StandardScaler is fit on TRAIN ROWS (at the row level) BEFORE building sequences.

    Returns:
        train_loader, val_loader, test_loader, scaler, (idx_train, idx_val, idx_test)
    """
    # 1) Read all rows to compute cutoffs and fit scaler on TRAIN ROWS ONLY
    df_all = pd.read_csv(csv_file, parse_dates=["Timestamp"]).sort_values(
        ["CampusKey", "SiteKey", "Timestamp"]
    )
    t1, t2 = compute_time_cutoffs(df_all["Timestamp"].values, train_frac, val_frac)

    # Train mask at ROW level for fitting scaler
    train_mask_rows = df_all["Timestamp"].values <= t1

    scaler = StandardScaler().fit(df_all.loc[train_mask_rows, feature_cols].values)

    # 2) Build the full dataset with the already-fitted scaler (so every split gets the same transform)
    ds_all = SolarDatasetGrouped(
        csv_file=csv_file,
        feature_cols=feature_cols,
        target_col=target_col,
        lookback=lookback,
        horizon=horizon,
        group_cols=("CampusKey", "SiteKey"),
        scaler=scaler,
    )

    # 3) Time-based split at the SEQUENCE level using sequence end-timestamps
    idx_train, idx_val, idx_test = make_time_indices(ds_all.seq_t, t1, t2)

    ds_train = Subset(ds_all, idx_train)
    ds_val   = Subset(ds_all, idx_val)
    ds_test  = Subset(ds_all, idx_test)

    # 4) DataLoaders
    train_loader = DataLoader(
        ds_train, batch_size=batch_size, shuffle=shuffle_train,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=False
    )
    val_loader = DataLoader(
        ds_val, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=False
    )
    test_loader = DataLoader(
        ds_test, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=False
    )

    return train_loader, val_loader, test_loader, scaler, (idx_train, idx_val, idx_test)
