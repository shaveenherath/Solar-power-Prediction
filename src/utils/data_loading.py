# utils/data_loading.py
# ======================================
# SPLITS & DATALOADERS (PyTorch)
# ======================================

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Dict, Any
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Subset

from src.dataset import SolarDatasetGrouped
from src.utils.feature_engineering import make_features


# ---------- time cutoffs ----------
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


# ---------- optional lag/rolling features (to match dataset) ----------
def _make_lag_features_for_scaler(
    df: pd.DataFrame,
    base_cols: List[str],
    lags: List[int] | None = None,
    windows: List[int] | None = None,
    group_cols: Tuple[str, str] = ("CampusKey", "SiteKey"),
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Creates the SAME lag/rolling features as the dataset, so the scaler is
    fitted on the exact feature set the dataset will produce.
    """
    if not lags and not windows:
        return df, []

    df = df.sort_values(list(group_cols) + ["Timestamp"]).copy()
    new_cols: List[str] = []
    gobj = df.groupby(list(group_cols), sort=False, group_keys=False)

    if lags:
        for col in base_cols:
            for L in lags:
                name = f"{col}_lag{L}"
                df[name] = gobj[col].shift(L)
                new_cols.append(name)

    if windows:
        for col in base_cols:
            for W in windows:
                name = f"{col}_rollmean{W}"
                df[name] = (
                    gobj[col]
                    .rolling(window=W, min_periods=W)
                    .mean()
                    .reset_index(level=list(range(len(group_cols))), drop=True)
                )
                new_cols.append(name)

    return df, new_cols


# ---------- main factory ----------
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
    *,
    time_encoding: Dict[str, Any] | None = None,   # NEW
    lag_cfg: Dict[str, Any] | None = None,         # NEW
):
    """
    Creates train/val/test DataLoaders using a time-based split.

    - Reads all rows
    - Applies time_encoding (if any) to df_all for scaler fit
    - Optionally creates lag/rolling features on df_all (to match dataset)
    - Fits StandardScaler on TRAIN ROWS ONLY, using the final engineered feature set
    - Builds a single SolarDatasetGrouped with the same configs, then splits by sequence time
    - Returns loaders + scaler + split indices
    """
    # 1) Read all rows (no engineering in master)
    df_all = pd.read_csv(csv_file, parse_dates=["Timestamp"]).sort_values(
        ["CampusKey", "SiteKey", "Timestamp"]
    )

    # 2) Compute cutoffs
    t1, t2 = compute_time_cutoffs(df_all["Timestamp"].values, train_frac, val_frac)
    train_mask_rows = df_all["Timestamp"].values <= t1

    df_all_enc, feature_cols_final = make_features(
        df_all,
        base_cols=feature_cols,
        time_encoding=time_encoding,
        lag_cfg=lag_cfg,
        group_cols=("CampusKey","SiteKey"),
    )

    # sanity check
    missing = [c for c in feature_cols_final if c not in df_all_enc.columns]
    if missing:
        raise ValueError(f"[create_dataloaders_grouped] Feature(s) not found in data: {missing}")

    # 6) Fit scaler on TRAIN ROWS ONLY for these columns (drop NaNs only for fit)
    X_train_rows = df_all_enc.loc[train_mask_rows, feature_cols_final]
    n_before = len(X_train_rows)
    X_train_rows_clean = X_train_rows.dropna(axis=0, how="any")
    n_after = len(X_train_rows_clean)
    if n_after < n_before:
        print(f"[Scaler fit] Dropped {n_before - n_after} row(s) with NaNs (TRAIN rows) before fitting scaler.")
    scaler = StandardScaler().fit(X_train_rows_clean.values)

    # 7) Build the full dataset with EXACT same configs (so engineered cols match)
    ds_all = SolarDatasetGrouped(
        csv_file=csv_file,
        feature_cols=feature_cols_final,
        target_col=target_col,
        lookback=lookback,
        horizon=horizon,
        group_cols=("CampusKey", "SiteKey"),
        scaler=scaler,
        time_encoding=time_encoding,   # ensure identical engineering inside dataset
        lag_cfg=lag_cfg,
        verbose=True,
    )

    # 8) Time-based split at the SEQUENCE level using sequence end-timestamps
    idx_train, idx_val, idx_test = make_time_indices(ds_all.seq_t, t1, t2)

    ds_train = Subset(ds_all, idx_train)
    ds_val   = Subset(ds_all, idx_val)
    ds_test  = Subset(ds_all, idx_test)

    # 9) DataLoaders
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

    return train_loader, val_loader, test_loader, ds_all.feature_cols , scaler, (idx_train, idx_val, idx_test)
