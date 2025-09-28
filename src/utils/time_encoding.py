# src/utils/time_encoding.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple

def _ensure_time_parts(df: pd.DataFrame) -> pd.DataFrame:
    if "Timestamp" not in df.columns:
        raise ValueError("Timestamp column is required for time encodings.")
    df = df.copy()
    ts = pd.to_datetime(df["Timestamp"])
    df["_hour"] = ts.dt.hour.astype(int)
    df["_dow"]  = ts.dt.dayofweek.astype(int)
    df["_mon"]  = ts.dt.month.astype(int)
    return df

# -------- Scheme: none/basic (no new columns) --------
def add_basic(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    return df, []

# -------- Scheme: cyclic (classic sin/cos) --------
def add_cyclic(df: pd.DataFrame, drop_old=True) -> Tuple[pd.DataFrame, List[str]]:
    df = _ensure_time_parts(df)
    df["hour_sin"]  = np.sin(2*np.pi*df["_hour"]/24)
    df["hour_cos"]  = np.cos(2*np.pi*df["_hour"]/24)
    df["dow_sin"]   = np.sin(2*np.pi*df["_dow"]/7)
    df["dow_cos"]   = np.cos(2*np.pi*df["_dow"]/7)
    df["month_sin"] = np.sin(2*np.pi*df["_mon"]/12)
    df["month_cos"] = np.cos(2*np.pi*df["_mon"]/12)
    if drop_old:
        df.drop(columns=["_hour","_dow","_mon"], inplace=True)
    return df, ["hour_sin","hour_cos","dow_sin","dow_cos","month_sin","month_cos"]

# -------- Scheme: sinusoidal (fixed transformer-style) --------
def _fixed_sinusoid(pos: np.ndarray, d_model: int, base: float = 10000.0) -> np.ndarray:
    positions = pos.astype(float)[:, None]        # [N,1]
    i = np.arange(d_model)[None, :]               # [1,D]
    denom = np.power(base, (2*(i//2))/d_model)    # [1,D]
    angle = positions / denom                     # [N,D]
    enc = np.zeros_like(angle)
    enc[:, 0::2] = np.sin(angle[:, 0::2])
    enc[:, 1::2] = np.cos(angle[:, 1::2])
    return enc

def add_sinusoidal(df: pd.DataFrame, d_hour=8, d_dow=8, d_mon=8, prefix="pe", drop_old=True) -> Tuple[pd.DataFrame, List[str]]:
    df = _ensure_time_parts(df)
    h_enc = _fixed_sinusoid(df["_hour"].to_numpy(), d_hour)
    d_enc = _fixed_sinusoid(df["_dow"].to_numpy(),  d_dow)
    m_enc = _fixed_sinusoid(df["_mon"].to_numpy(),  d_mon)
    h_cols = [f"{prefix}_hour_{i}" for i in range(d_hour)]
    d_cols = [f"{prefix}_dow_{i}"  for i in range(d_dow)]
    m_cols = [f"{prefix}_mon_{i}"  for i in range(d_mon)]
    df[h_cols] = h_enc
    df[d_cols] = d_enc
    df[m_cols] = m_enc
    if drop_old:
        df.drop(columns=["_hour","_dow","_mon"], inplace=True)
    return df, h_cols + d_cols + m_cols

# -------- Scheme: fourier (multi-harmonic sin/cos per cycle) --------
def add_fourier(df: pd.DataFrame, k_hour=3, k_dow=2, k_mon=2, prefix="fourier", drop_old=True) -> Tuple[pd.DataFrame, List[str]]:
    df = _ensure_time_parts(df)
    new_cols: List[str] = []
    def _harmonics(x: np.ndarray, period: int, K: int, name: str):
        cols = []
        for k in range(1, K+1):
            ang = 2*np.pi*k*x/period
            s = np.sin(ang); c = np.cos(ang)
            s_name = f"{prefix}_{name}_sin{k}"
            c_name = f"{prefix}_{name}_cos{k}"
            df[s_name] = s; df[c_name] = c
            cols += [s_name, c_name]
        return cols
    new_cols += _harmonics(df["_hour"].to_numpy(), 24, k_hour, "hour")
    new_cols += _harmonics(df["_dow"].to_numpy(), 7,  k_dow,  "dow")
    new_cols += _harmonics(df["_mon"].to_numpy(), 12, k_mon,  "mon")
    if drop_old:
        df.drop(columns=["_hour","_dow","_mon"], inplace=True)
    return df, new_cols

# -------- API entry --------
def apply_time_encoding(df: pd.DataFrame, cfg: Dict[str, Any] | None) -> Tuple[pd.DataFrame, List[str]]:
    """
    cfg:
      scheme: basic|none|cyclic|sinusoidal|fourier
      params: dict of scheme-specific params
    """
    if not cfg:
        return add_basic(df)
    scheme = (cfg.get("scheme") or "basic").lower()
    params = cfg.get("params", {}) or {}
    if scheme in ("basic","none"):
        return add_basic(df)
    if scheme == "cyclic":
        return add_cyclic(df, drop_old=True)
    if scheme == "sinusoidal":
        return add_sinusoidal(
            df,
            d_hour=int(params.get("d_hour", 8)),
            d_dow=int(params.get("d_dow", 8)),
            d_mon=int(params.get("d_mon", 8)),
            prefix=str(params.get("prefix", "pe")),
            drop_old=True,
        )
    if scheme == "fourier":
        return add_fourier(
            df,
            k_hour=int(params.get("k_hour", 3)),
            k_dow=int(params.get("k_dow", 2)),
            k_mon=int(params.get("k_mon", 2)),
            prefix=str(params.get("prefix", "fourier")),
            drop_old=True,
        )
    raise ValueError(f"Unknown time-encoding scheme: {scheme}")
