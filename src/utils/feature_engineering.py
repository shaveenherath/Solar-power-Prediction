# src/utils/feature_engineering.py
from __future__ import annotations
import pandas as pd
from typing import Dict, Any, List, Tuple
from src.utils.time_encoding import apply_time_encoding


def make_features(
    df: pd.DataFrame,
    base_cols: List[str],
    time_encoding: Dict[str, Any] | None = None,
    lag_cfg: Dict[str, Any] | None = None,
    group_cols: Tuple[str, str] = ("CampusKey", "SiteKey"),
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Apply time encoding + lag/rolling features consistently.
    Returns (df_with_features, final_feature_cols).
    """
    # --- time encoding ---
    print("time enc" , time_encoding)
    df, time_cols = apply_time_encoding(df, time_encoding)
    print("time cols are : " , time_cols)

    # --- lag/rolling features ---
    lag_cols: List[str] = []
    if lag_cfg:
        use_cols = lag_cfg.get("cols", [])
        lags     = lag_cfg.get("lags", [])
        windows  = lag_cfg.get("windows", [])

        if not use_cols:
            # auto-pick numeric base features only
            use_cols = [c for c in base_cols if pd.api.types.is_numeric_dtype(df[c])]

        df = df.sort_values(list(group_cols) + ["Timestamp"]).copy()
        gobj = df.groupby(list(group_cols), sort=False, group_keys=False)

        if lags:
            for col in use_cols:
                for L in lags:
                    name = f"{col}_lag{L}"
                    df[name] = gobj[col].shift(L)
                    lag_cols.append(name)

        if windows:
            for col in use_cols:
                for W in windows:
                    name = f"{col}_rollmean{W}"
                    df[name] = (
                        gobj[col]
                        .rolling(window=W, min_periods=W)
                        .mean()
                        .reset_index(level=list(range(len(group_cols))), drop=True)
                    )
                    lag_cols.append(name)
    print("lag cols are :", lag_cols)
    # --- final list ---
    seen, final_cols = set(), []
    for c in list(base_cols) + time_cols + lag_cols:
        if c not in seen:
            final_cols.append(c); seen.add(c)
    print("final_cols are :" , final_cols )

    return df, final_cols
