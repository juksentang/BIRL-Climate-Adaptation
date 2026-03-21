"""
Stage 3: Counterfactual Matrix Generation.

Re-run trained LightGBM environment models with 2050 climate features
to produce new (q10, q50, q90, sigma) income distribution matrices.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

from config import (
    MODEL_MU_PATH, MODEL_SIGMA_PATH, SIGMA_SCALE, Z10, Z90,
    FEATURE_COLS, CAT_FEATURES, CROP_ORDER, INTENSITY_ORDER, N_ACTIONS,
    BIRL_SAMPLE_PATH, COUNTRIES,
)

# Exact category lists matching model training (from model.pandas_categorical)
ACTION_CROP_CATS = ['legumes', 'maize', 'other', 'rice', 'sorghum_millet',
                    'teff', 'tree_crops', 'tubers', 'wheat_barley']
COUNTRY_CATS = ['Ethiopia', 'Malawi', 'Mali', 'Nigeria', 'Tanzania', 'Uganda']
SEASON_CATS = [1.0, 2.0]
INTENSITY_MAP = {'low': 0, 'medium': 1, 'high': 2}

log = logging.getLogger("cf_engine")


def load_env_models() -> tuple:
    """Load trained LightGBM boosters.

    Returns: (model_mu, model_sigma)
    """
    model_mu = lgb.Booster(model_file=str(MODEL_MU_PATH))
    model_sigma = lgb.Booster(model_file=str(MODEL_SIGMA_PATH))
    log.info(f"  Loaded model_mu ({model_mu.num_trees()} trees) "
             f"and model_sigma ({model_sigma.num_trees()} trees)")
    return model_mu, model_sigma


def generate_counterfactual_matrix(
    model_mu: lgb.Booster,
    model_sigma: lgb.Booster,
    df: pd.DataFrame,
) -> tuple:
    """Generate (q10, q50, q90, sigma) matrices for given climate state.

    Args:
        model_mu: LightGBM booster for E[log(y)|s,a]
        model_sigma: LightGBM booster for Std[log(y)|s,a]
        df: DataFrame with 34 state features (climate already replaced if 2050)

    Returns:
        (q10, q50, q90, sigma_usd) — each shape (N_obs, 27), float32
    """
    N = len(df)
    q10_mat = np.zeros((N, N_ACTIONS), dtype=np.float32)
    q50_mat = np.zeros((N, N_ACTIONS), dtype=np.float32)
    q90_mat = np.zeros((N, N_ACTIONS), dtype=np.float32)
    sigma_mat = np.zeros((N, N_ACTIONS), dtype=np.float32)

    # State features (everything except action_crop and input_intensity)
    state_cols = [c for c in FEATURE_COLS if c not in ("action_crop", "input_intensity")]

    for aid in range(N_ACTIONS):
        crop_idx = aid // 3
        intensity_idx = aid % 3
        crop_name = CROP_ORDER[crop_idx]
        intensity_name = INTENSITY_ORDER[intensity_idx]

        # Build feature matrix for this action
        X = df[state_cols].copy()
        X.insert(0, "input_intensity", intensity_name)
        X.insert(0, "action_crop", crop_name)

        # Ensure column order matches FEATURE_COLS exactly
        X = X[FEATURE_COLS]

        # Encode features to match model training expectations
        X["input_intensity"] = X["input_intensity"].map(INTENSITY_MAP)
        X["action_crop"] = pd.Categorical(X["action_crop"], categories=ACTION_CROP_CATS)
        X["country"] = pd.Categorical(X["country"], categories=COUNTRY_CATS)
        X["season"] = pd.Categorical(X["season"], categories=SEASON_CATS)

        # Predict in log-space
        mu = model_mu.predict(X)
        sig_raw = model_sigma.predict(X)
        sig = np.clip(sig_raw, 0.1, None) * SIGMA_SCALE

        # Convert to USD: q = exp(mu + z*sigma) - 1
        q10_mat[:, aid] = np.exp(mu + Z10 * sig) - 1
        q50_mat[:, aid] = np.exp(mu) - 1
        q90_mat[:, aid] = np.exp(mu + Z90 * sig) - 1
        sigma_mat[:, aid] = (q90_mat[:, aid] - q10_mat[:, aid]) / 2.56

        if aid % 9 == 0:
            log.info(f"    Action {aid}/{N_ACTIONS}: {crop_name}_{intensity_name} done")

    log.info(f"  CF matrix: ({N:,}, {N_ACTIONS}) generated")
    return q10_mat, q50_mat, q90_mat, sigma_mat


def validate_cf_matrices(
    baseline_npz_path: Path,
    cf_q50: np.ndarray,
    df: pd.DataFrame,
    label: str,
):
    """Compare 2050 matrices to baseline and print diagnostics."""
    baseline = np.load(str(baseline_npz_path))
    bl_q50 = baseline["q50"]

    print(f"\n{'='*60}")
    print(f"Counterfactual Validation: {label}")
    print(f"{'='*60}")
    print(f"{'Country':<12} {'BL q50 med':>11} {'CF q50 med':>11} {'Δ%':>8}")
    print("-" * 50)

    for country in COUNTRIES:
        mask = (df["country"] == country).values
        bl_med = np.nanmedian(bl_q50[mask])
        cf_med = np.nanmedian(cf_q50[mask])
        pct = (cf_med / bl_med - 1) * 100 if bl_med > 0 else float("nan")
        print(f"{country:<12} ${bl_med:>10.1f} ${cf_med:>10.1f} {pct:>+7.1f}%")

    bl_all = np.nanmedian(bl_q50)
    cf_all = np.nanmedian(cf_q50)
    pct_all = (cf_all / bl_all - 1) * 100
    print("-" * 50)
    print(f"{'ALL':<12} ${bl_all:>10.1f} ${cf_all:>10.1f} {pct_all:>+7.1f}%")
    print(f"{'='*60}\n")
