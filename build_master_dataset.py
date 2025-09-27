# build_master_dataset.py
# =========================
# BUILD MASTER DATASET
# =========================

import os
import argparse
import numpy as np
import pandas as pd


# ---------- helpers ----------
def preprocess_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time cyclics + label encodings."""
    from sklearn.preprocessing import LabelEncoder
    df = df.copy()
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    # encoders (IDs as integers for model input)
    for col, new in [('CampusKey', 'CampusKey_encoded'),
                     ('SiteKey', 'SiteKey_encoded')]:
        le = LabelEncoder()
        df[new] = le.fit_transform(df[col].astype(str))

    # time features
    df['hour'] = df['Timestamp'].dt.hour
    df['dayofweek'] = df['Timestamp'].dt.dayofweek
    df['month'] = df['Timestamp'].dt.month
    df['hour_sin'] = np.sin(2*np.pi*df['hour']/24)
    df['hour_cos'] = np.cos(2*np.pi*df['hour']/24)
    df['dow_sin'] = np.sin(2*np.pi*df['dayofweek']/7)
    df['dow_cos'] = np.cos(2*np.pi*df['dayofweek']/7)
    df['month_sin'] = np.sin(2*np.pi*df['month']/12)
    df['month_cos'] = np.cos(2*np.pi*df['month']/12)
    return df


def load_site_details(path: str) -> pd.DataFrame:
    """Load site file and expose DC_Capacity_kWp, Lat, Lon."""
    site = pd.read_csv(path)

    site = site.rename(columns={
        'CampusKe': 'CampusKey',      # if truncated
        'kWp': 'DC_Capacity_kWp',
        'lat': 'Lat', 'Lat': 'Lat',
        'Lon': 'Lon',
        'Number of panels': 'NumPanels'
    })

    if 'CampusKey' not in site.columns or 'SiteKey' not in site.columns:
        raise ValueError("Site file must contain 'CampusKey' and 'SiteKey' columns.")

    site['CampusKey'] = site['CampusKey'].astype(str).str.strip()
    site['SiteKey'] = site['SiteKey'].astype(str).str.strip()

    for c in ['DC_Capacity_kWp', 'Lat', 'Lon']:
        if c in site.columns:
            site[c] = pd.to_numeric(site[c], errors='coerce')

    keep = [c for c in ['CampusKey', 'SiteKey',
                        'DC_Capacity_kWp', 'Lat', 'Lon'] if c in site.columns]
    site = site[keep].drop_duplicates(subset=['CampusKey', 'SiteKey'])
    return site


def coerce_numeric(df: pd.DataFrame, skip_cols: set) -> pd.DataFrame:
    """Try to coerce non-numeric columns into numeric."""
    for c in df.columns:
        if c in skip_cols:
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


# ---------- main ----------
def build_master_dataset(data_folder: str):
    power_path = os.path.join(data_folder, "Solar_Energy_Generation.csv")
    weather_path = os.path.join(data_folder, "Weather_Data_reordered_all.csv")
    site_path = os.path.join(data_folder, "Solar_Site_Details.csv")
    master_out = os.path.join(data_folder, "Solar_Power_Weather_Clean_MASTER.csv")

    df_power = pd.read_csv(power_path, parse_dates=['Timestamp'])
    df_weather = pd.read_csv(weather_path, parse_dates=['Timestamp'])

    for col in ['CampusKey', 'SiteKey', 'Timestamp', 'SolarGeneration']:
        if col not in df_power.columns:
            raise ValueError(f"POWER file missing column: {col}")
    for col in ['CampusKey', 'Timestamp']:
        if col not in df_weather.columns:
            raise ValueError(f"WEATHER file missing column: {col}")

    df_weather = df_weather.drop_duplicates(subset=['CampusKey', 'Timestamp'])

    df = pd.merge(df_power, df_weather, on=['CampusKey', 'Timestamp'], how='inner')
    df['SolarGeneration'] = pd.to_numeric(df['SolarGeneration'], errors='coerce').fillna(0.0)

    df['hour'] = df['Timestamp'].dt.hour
    df = df[(df['hour'] >= 6) & (df['hour'] <= 18)].copy()

    site_df = load_site_details(site_path)
    df['CampusKey'] = df['CampusKey'].astype(str).str.strip()
    df['SiteKey'] = df['SiteKey'].astype(str).str.strip()
    df = df.merge(site_df, on=['CampusKey', 'SiteKey'], how='left')

    has_cap = df['DC_Capacity_kWp'].fillna(0) > 0
    df['y_norm_cap'] = np.where(
        has_cap,
        df['SolarGeneration'] / df['DC_Capacity_kWp'],
        np.nan
    )
    df['site_max'] = df.groupby(['CampusKey', 'SiteKey'])['SolarGeneration'].transform('max')
    needs_fallback = df['y_norm_cap'].isna()
    df['y_norm'] = df['y_norm_cap']
    df.loc[needs_fallback, 'y_norm'] = (
        df.loc[needs_fallback, 'SolarGeneration'] /
        df.loc[needs_fallback, 'site_max'].replace(0, np.nan)
    )
    df = df.dropna(subset=['y_norm']).copy()

    df = preprocess_features(df)
    skip_for_coercion = set(['Timestamp', 'CampusKey', 'SiteKey'])
    df = coerce_numeric(df, skip_for_coercion)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    exclude = set([
        'Timestamp', 'CampusKey', 'SiteKey',
        'SolarGeneration', 'y_norm', 'y_norm_cap', 'site_max',
        'hour', 'dayofweek', 'month'
    ])
    feature_cols = [
        'CampusKey_encoded', 'SiteKey_encoded',
        'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos'
    ]
    for c in df.columns:
        if c in exclude or c in feature_cols:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            feature_cols.append(c)

    cols_needed = feature_cols + ['y_norm']
    na_mask = df[cols_needed].isna().any(axis=1)
    n_bad = int(na_mask.sum())
    if n_bad > 0:
        print(f"⚠️ Dropping {n_bad} rows due to NaNs in features/target.")
    df = df.loc[~na_mask].copy()

    out_cols = ['CampusKey', 'SiteKey', 'Timestamp', 'SolarGeneration', 'y_norm',
                'DC_Capacity_kWp', 'Lat', 'Lon'] + feature_cols
    out_cols = [c for c in out_cols if c in df.columns]

    df.sort_values(['CampusKey', 'SiteKey', 'Timestamp']).to_csv(master_out, index=False)

    print("✅ Master dataset saved:", master_out)
    print(f"Rows: {len(df)}  | Sites: {df[['CampusKey','SiteKey']].drop_duplicates().shape[0]}")
    print(f"Features ({len(feature_cols)}): {feature_cols}")
    print("Target column: y_norm  (primary: kWh/kWp per interval; fallback: kWh/site_max)")
    print(f"Rows using true kWp: {int(has_cap.sum())} | Rows using site_max fallback: {int(needs_fallback.sum())}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Master Solar Power + Weather dataset")
    parser.add_argument("--data_folder", type=str, required=True,
                        help="Path to folder containing input CSV files (Power, Weather, Site)")
    args = parser.parse_args()

    build_master_dataset(args.data_folder)
