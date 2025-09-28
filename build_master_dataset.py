# scripts/build_master_dataset.py
# =========================
# BUILD MASTER DATASET (no time encodings / no label encodings)
# =========================

import os
import argparse
import numpy as np
import pandas as pd


def load_site_details(path: str) -> pd.DataFrame:
    """Load site file and expose DC_Capacity_kWp, Lat, Lon; keep keys."""
    site = pd.read_csv(path)

    site = site.rename(columns={
        'CampusKe': 'CampusKey',      # if truncated in source
        'kWp': 'DC_Capacity_kWp',
        'lat': 'Lat',
        'Lat': 'Lat',
        'Lon': 'Lon',
        'Number of panels': 'NumPanels'
    })

    if 'CampusKey' not in site.columns or 'SiteKey' not in site.columns:
        raise ValueError("Site file must contain 'CampusKey' and 'SiteKey' columns.")

    site['CampusKey'] = site['CampusKey'].astype(str).str.strip()
    site['SiteKey']   = site['SiteKey'].astype(str).str.strip()

    # numeric coercions (safe)
    for c in ['DC_Capacity_kWp', 'Lat', 'Lon']:
        if c in site.columns:
            site[c] = pd.to_numeric(site[c], errors='coerce')

    keep = [c for c in ['CampusKey','SiteKey','DC_Capacity_kWp','Lat','Lon'] if c in site.columns]
    site = site[keep].drop_duplicates(subset=['CampusKey','SiteKey'])
    return site


def coerce_numeric(df: pd.DataFrame, skip_cols: set) -> pd.DataFrame:
    """
    Try to coerce non-numeric columns into numeric for weather fields etc.
    Leaves non-coercible strings as-is.
    """
    for c in df.columns:
        if c in skip_cols:
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            try:
                df[c] = pd.to_numeric(df[c], errors='coerce')
            except Exception:
                pass
    return df


def build_master_dataset(
    data_folder: str,
    output_name: str = "Solar_Power_Weather_Clean_MASTER.csv",
    filter_daylight: bool = False,
    day_start: int = 6,
    day_end: int = 18,
):
    # --- paths ---
    power_path   = os.path.join(data_folder, "Solar_Energy_Generation.csv")
    weather_path = os.path.join(data_folder, "Weather_Data_reordered_all.csv")
    site_path    = os.path.join(data_folder, "Solar_Site_Details.csv")
    master_out   = os.path.join(data_folder, output_name)

    # --- load ---
    df_power   = pd.read_csv(power_path, parse_dates=['Timestamp'])
    df_weather = pd.read_csv(weather_path, parse_dates=['Timestamp'])

    # basic sanity checks
    for col in ['CampusKey','SiteKey','Timestamp','SolarGeneration']:
        if col not in df_power.columns:
            raise ValueError(f"POWER file missing column: {col}")
    for col in ['CampusKey','Timestamp']:
        if col not in df_weather.columns:
            raise ValueError(f"WEATHER file missing column: {col}")

    # de-dup weather on (CampusKey, Timestamp)
    df_weather = df_weather.drop_duplicates(subset=['CampusKey','Timestamp'])

    # --- merge power + weather ---
    df = pd.merge(
        df_power,
        df_weather,
        on=['CampusKey','Timestamp'],
        how='inner'
    )

    # clean target
    df['SolarGeneration'] = pd.to_numeric(df['SolarGeneration'], errors='coerce').fillna(0.0)

    # optional daylight filter (OFF by default)
    if filter_daylight:
        hours = pd.to_datetime(df['Timestamp']).dt.hour
        df = df[(hours >= day_start) & (hours <= day_end)].copy()

    # attach site details (capacity, lat, lon)
    site_df = load_site_details(site_path)
    df['CampusKey'] = df['CampusKey'].astype(str).str.strip()
    df['SiteKey']   = df['SiteKey'].astype(str).str.strip()
    df = df.merge(site_df, on=['CampusKey','SiteKey'], how='left')

    # --------- target normalization (no feature engineering) ---------
    has_cap = df['DC_Capacity_kWp'].fillna(0) > 0
    df['y_norm_cap'] = np.where(
        has_cap,
        df['SolarGeneration'] / df['DC_Capacity_kWp'],
        np.nan
    )

    # fallback: per-site historical maximum (if capacity missing)
    df['site_max'] = df.groupby(['CampusKey','SiteKey'])['SolarGeneration'].transform('max')
    needs_fallback = df['y_norm_cap'].isna()
    df['y_norm'] = df['y_norm_cap']
    df.loc[needs_fallback, 'y_norm'] = (
        df.loc[needs_fallback, 'SolarGeneration'] /
        df.loc[needs_fallback, 'site_max'].replace(0, np.nan)
    )

    # drop rows where normalization failed
    df = df.dropna(subset=['y_norm']).copy()

    # Try to coerce any stringy numeric weather columns (safe)
    skip_for_coercion = {'Timestamp','CampusKey','SiteKey'}
    df = coerce_numeric(df, skip_for_coercion)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # --- FINAL SAFETY: drop rows with NaNs in target only (features handled at run-time)
    n_before = len(df)
    df = df.dropna(subset=['y_norm']).copy()
    n_after = len(df)
    if n_after < n_before:
        print(f"⚠️ Dropped {n_before - n_after} rows due to NaNs in target normalization.")

    # --- order and save ---
    # keep all original + site fields + normalization columns (no engineered time/encodings)
    base_cols = ['CampusKey','SiteKey','Timestamp','SolarGeneration',
                 'DC_Capacity_kWp','Lat','Lon','y_norm_cap','site_max','y_norm']
    # include any other columns present (e.g., weather variables)
    other_cols = [c for c in df.columns if c not in base_cols]
    out_cols = [c for c in base_cols if c in df.columns] + other_cols

    df.sort_values(['CampusKey','SiteKey','Timestamp']).to_csv(master_out, index=False)

    # --- report ---
    print("✅ Master dataset saved:", master_out)
    print(f"Rows: {len(df)}  | Sites: {df[['CampusKey','SiteKey']].drop_duplicates().shape[0]}")
    print("Target column: y_norm  (primary: kWh/kWp; fallback: kWh/site_max)")
    print(f"Rows using true kWp: {int(has_cap.sum())} | Rows using site_max fallback: {int(needs_fallback.sum())}")

    # Helpful hints for downstream
    numeric_cols = [c for c in other_cols if pd.api.types.is_numeric_dtype(df[c])]
    print(f"Detected numeric candidates (weather/features to pick later): {numeric_cols[:20]}{'...' if len(numeric_cols)>20 else ''}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Master Solar Power + Weather dataset (no encodings)")
    parser.add_argument("--data_folder", type=str, required=True,
                        help="Path to folder containing input CSV files (Power, Weather, Site)")
    parser.add_argument("--output_name", type=str, default="Solar_Power_Weather_Clean_MASTER.csv",
                        help="Output CSV filename")
    parser.add_argument("--filter_daylight", action="store_true",
                        help="If set, keep only rows within [--day_start, --day_end]")
    parser.add_argument("--day_start", type=int, default=6, help="Daylight start hour (inclusive)")
    parser.add_argument("--day_end", type=int, default=18, help="Daylight end hour (inclusive)")
    args = parser.parse_args()

    build_master_dataset(
        data_folder=args.data_folder,
        output_name=args.output_name,
        filter_daylight=args.filter_daylight,
        day_start=args.day_start,
        day_end=args.day_end,
    )
