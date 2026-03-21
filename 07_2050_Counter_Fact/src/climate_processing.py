"""
Stage 2: Climate Processing — Delta method + NDVI prediction.

Functions:
  - compute_ensemble_deltas: Load CMIP6 CSVs, compute per-point delta factors
  - train_ndvi_models: Fit NDVI = f(rain, temp, country) from current data
  - apply_deltas_to_observations: Replace 7 climate features with 2050 values
  - print_signal_diagnostics: Verify climate change signal magnitudes
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import joblib

from config import (
    BIRL_SAMPLE_PATH, CMIP6_RAW_DIR, CMIP6_PROC_DIR, DATA_DIR,
    GCMS, SSPS, COUNTRIES,
    MULTIPLICATIVE_FEATURES, ADDITIVE_FEATURES, NDVI_FEATURES,
)

log = logging.getLogger("climate")


# ═══════════════════════════════════════════════
# 2a. Ensemble Delta Factors
# ═══════════════════════════════════════════════

def compute_ensemble_deltas(cmip6_dir: Path = CMIP6_RAW_DIR) -> pd.DataFrame:
    """Load 20 CMIP6 CSVs, compute per-point ensemble-mean delta factors.

    Returns DataFrame with columns:
      point_idx, country, latitude, longitude,
      delta_rain_{ssp}, delta_rain_cv_{ssp}, delta_tmean_{ssp}, delta_tmax_{ssp}
      for each ssp in [ssp245, ssp585].
    """
    all_data = {}
    for gcm in GCMS:
        for ssp in SSPS:
            for period in ["baseline", "future"]:
                fname = f"cmip6_{gcm}_{ssp}_{period}.csv"
                fpath = cmip6_dir / fname
                if not fpath.exists():
                    log.warning(f"Missing: {fname}")
                    continue
                df = pd.read_csv(fpath)
                all_data[(gcm, ssp, period)] = df
                log.info(f"  Loaded {fname}: {len(df)} rows")

    # Check we have at least some data
    if not all_data:
        raise FileNotFoundError(f"No CMIP6 CSVs found in {cmip6_dir}")

    # For each SSP, compute ensemble mean of baseline and future across GCMs
    results = None
    for ssp in SSPS:
        baseline_dfs = []
        future_dfs = []
        for gcm in GCMS:
            key_b = (gcm, ssp, "baseline")
            key_f = (gcm, ssp, "future")
            if key_b in all_data and key_f in all_data:
                baseline_dfs.append(all_data[key_b])
                future_dfs.append(all_data[key_f])

        if not baseline_dfs:
            log.warning(f"No complete GCM data for {ssp}")
            continue

        n_gcms = len(baseline_dfs)
        log.info(f"  {ssp}: {n_gcms} GCMs available for ensemble")

        # Stack and average across GCMs (per point)
        climate_vars = ["rainfall_gs_sum", "rainfall_gs_cv", "tmean_gs", "tmax_gs"]
        id_cols = ["point_idx", "country", "latitude", "longitude"]

        baseline_cat = pd.concat(baseline_dfs, ignore_index=True)
        future_cat = pd.concat(future_dfs, ignore_index=True)

        baseline_ens = baseline_cat.groupby(id_cols)[climate_vars].mean().reset_index()
        future_ens = future_cat.groupby(id_cols)[climate_vars].mean().reset_index()

        # Compute deltas
        merged = baseline_ens.merge(future_ens, on=id_cols, suffixes=("_bl", "_ft"))

        # Rainfall: multiplicative delta
        merged[f"delta_rain_{ssp}"] = (
            merged["rainfall_gs_sum_ft"] / merged["rainfall_gs_sum_bl"].clip(lower=1.0)
        )
        merged[f"delta_rain_cv_{ssp}"] = (
            merged["rainfall_gs_cv_ft"] / merged["rainfall_gs_cv_bl"].clip(lower=0.01)
        )
        # Temperature: additive delta
        merged[f"delta_tmean_{ssp}"] = merged["tmean_gs_ft"] - merged["tmean_gs_bl"]
        merged[f"delta_tmax_{ssp}"] = merged["tmax_gs_ft"] - merged["tmax_gs_bl"]

        delta_cols = [f"delta_rain_{ssp}", f"delta_rain_cv_{ssp}",
                      f"delta_tmean_{ssp}", f"delta_tmax_{ssp}"]

        if results is None:
            results = merged[id_cols + delta_cols].copy()
        else:
            results = results.merge(merged[id_cols + delta_cols], on=id_cols)

    if results is None:
        raise ValueError("Could not compute deltas — no CMIP6 data loaded")

    log.info(f"  Deltas computed for {len(results)} points")
    return results


# ═══════════════════════════════════════════════
# 2b. NDVI Prediction Models
# ═══════════════════════════════════════════════

def train_ndvi_models(df: pd.DataFrame) -> tuple:
    """Train NDVI = f(rainfall, temperature, country) from current data.

    Returns: (ndvi_growing_model, ndvi_preseason_model, r2_growing, r2_preseason)
    """
    # Encode country as integer
    country_map = {c: i for i, c in enumerate(COUNTRIES)}

    features = ["rainfall_growing_sum_final", "era5_tmean_growing_final"]

    trained = {}
    for target_name in ["ndvi_growing_mean_final", "ndvi_preseason_mean_final"]:
        mask = df[features + [target_name]].notna().all(axis=1)
        X = df.loc[mask, features].copy()
        X["country_enc"] = df.loc[mask, "country"].map(country_map)
        y = df.loc[mask, target_name]

        model = GradientBoostingRegressor(
            n_estimators=200, max_depth=4, random_state=42,
            subsample=0.8, learning_rate=0.05,
        )
        model.fit(X, y)
        r2 = model.score(X, y)
        log.info(f"  NDVI model '{target_name}': R²={r2:.3f} (n={len(y):,})")
        trained[target_name] = (model, r2)

    ndvi_grow_model, r2_grow = trained["ndvi_growing_mean_final"]
    ndvi_pre_model, r2_pre = trained["ndvi_preseason_mean_final"]

    # Save models
    joblib.dump(ndvi_grow_model, DATA_DIR / "ndvi_model_growing.joblib")
    joblib.dump(ndvi_pre_model, DATA_DIR / "ndvi_model_preseason.joblib")
    log.info("  NDVI models saved to data/")

    return ndvi_grow_model, ndvi_pre_model, r2_grow, r2_pre


def predict_ndvi_2050(
    df_2050: pd.DataFrame,
    ndvi_grow_model,
    ndvi_pre_model,
) -> pd.DataFrame:
    """Predict 2050 NDVI from adjusted rainfall + temperature."""
    country_map = {c: i for i, c in enumerate(COUNTRIES)}

    features = ["rainfall_growing_sum_final", "era5_tmean_growing_final"]
    mask = df_2050[features].notna().all(axis=1)

    X = df_2050.loc[mask, features].copy()
    X["country_enc"] = df_2050.loc[mask, "country"].map(country_map)

    df_2050 = df_2050.copy()
    df_2050.loc[mask, "ndvi_growing_mean_final"] = ndvi_grow_model.predict(X)
    df_2050.loc[mask, "ndvi_preseason_mean_final"] = ndvi_pre_model.predict(X)

    return df_2050


# ═══════════════════════════════════════════════
# 2c. Apply Deltas to Observations
# ═══════════════════════════════════════════════

def apply_deltas_to_observations(
    df: pd.DataFrame,
    deltas: pd.DataFrame,
    ndvi_grow_model,
    ndvi_pre_model,
    scenario: str,
) -> pd.DataFrame:
    """Apply delta factors to all 222,023 observations for one SSP scenario.

    Returns: DataFrame copy with 7 climate columns replaced by 2050 values.
    """
    ssp = scenario  # "ssp245" or "ssp585"

    # Build GPS→delta lookup (exact join on coordinates)
    delta_lookup = deltas.set_index(["latitude", "longitude"])

    df_2050 = df.copy()

    # Match each observation to its delta point
    # Deduplicate deltas by coordinates (border GPS points may appear in 2 countries)
    delta_cols = [f"delta_rain_{ssp}", f"delta_rain_cv_{ssp}",
                  f"delta_tmean_{ssp}", f"delta_tmax_{ssp}"]
    delta_dedup = (deltas[["latitude", "longitude"] + delta_cols]
                   .groupby(["latitude", "longitude"])[delta_cols]
                   .mean().reset_index())

    gps_key = df[["gps_lat_final", "gps_lon_final"]].copy()
    gps_key.columns = ["latitude", "longitude"]

    # Merge deltas by coordinates (now 1:1)
    merged = gps_key.merge(delta_dedup, on=["latitude", "longitude"], how="left")

    # Apply multiplicative deltas (rainfall)
    delta_rain = merged[f"delta_rain_{ssp}"].values
    delta_rain_cv = merged[f"delta_rain_cv_{ssp}"].values
    delta_tmean = merged[f"delta_tmean_{ssp}"].values
    delta_tmax = merged[f"delta_tmax_{ssp}"].values

    # Rainfall: multiplicative
    df_2050["rainfall_growing_sum_final"] = df["rainfall_growing_sum_final"] * delta_rain
    df_2050["rainfall_10yr_mean_final"] = df["rainfall_10yr_mean_final"] * delta_rain
    df_2050["rainfall_10yr_cv_final"] = df["rainfall_10yr_cv_final"] * delta_rain_cv

    # Temperature: additive
    df_2050["era5_tmean_growing_final"] = df["era5_tmean_growing_final"] + delta_tmean
    df_2050["era5_tmax_growing_final"] = df["era5_tmax_growing_final"] + delta_tmax

    # NDVI: predict from adjusted rain + temp
    df_2050 = predict_ndvi_2050(df_2050, ndvi_grow_model, ndvi_pre_model)

    # Count how many observations got deltas
    n_matched = merged[f"delta_rain_{ssp}"].notna().sum()
    n_total = len(df)
    log.info(f"  Applied {ssp} deltas: {n_matched:,}/{n_total:,} obs matched")

    return df_2050


# ═══════════════════════════════════════════════
# 2d. Signal Magnitude Diagnostics
# ═══════════════════════════════════════════════

def print_signal_diagnostics(df_current: pd.DataFrame, df_2050: pd.DataFrame, ssp: str):
    """Print per-country median climate change signals."""
    print(f"\n{'='*70}")
    print(f"Climate Change Signal Diagnostics ({ssp})")
    print(f"{'='*70}")
    print(f"{'Country':<12} {'Rain Δ%':>8} {'Rain CV Δ%':>10} {'Tmean Δ°C':>10} {'Tmax Δ°C':>9} {'NDVI Δ%':>8}")
    print("-" * 70)

    for country in COUNTRIES:
        mask = df_current["country"] == country
        rain_c = df_current.loc[mask, "rainfall_growing_sum_final"].median()
        rain_f = df_2050.loc[mask, "rainfall_growing_sum_final"].median()
        cv_c = df_current.loc[mask, "rainfall_10yr_cv_final"].median()
        cv_f = df_2050.loc[mask, "rainfall_10yr_cv_final"].median()
        tm_c = df_current.loc[mask, "era5_tmean_growing_final"].median()
        tm_f = df_2050.loc[mask, "era5_tmean_growing_final"].median()
        tx_c = df_current.loc[mask, "era5_tmax_growing_final"].median()
        tx_f = df_2050.loc[mask, "era5_tmax_growing_final"].median()
        nd_c = df_current.loc[mask, "ndvi_growing_mean_final"].median()
        nd_f = df_2050.loc[mask, "ndvi_growing_mean_final"].median()

        rain_pct = (rain_f / rain_c - 1) * 100 if rain_c > 0 else float("nan")
        cv_pct = (cv_f / cv_c - 1) * 100 if cv_c > 0 else float("nan")
        tm_delta = tm_f - tm_c if not np.isnan(tm_c) else float("nan")
        tx_delta = tx_f - tx_c if not np.isnan(tx_c) else float("nan")
        nd_pct = (nd_f / nd_c - 1) * 100 if nd_c > 0 else float("nan")

        print(f"{country:<12} {rain_pct:>+7.1f}% {cv_pct:>+9.1f}% "
              f"{tm_delta:>+9.1f}°C {tx_delta:>+8.1f}°C {nd_pct:>+7.1f}%")

    print(f"{'='*70}")
    print("Expected: temp +1.5–2.5°C, rain ±15%, CV +10–30%\n")
