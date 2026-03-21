"""
Stage 3 Runner: Generate 2050 counterfactual income matrices.

Prerequisites:
  - data/cmip6_processed/obs_climate_{ssp}.parquet (from Stage 2)
  - LightGBM models from Step 04

Outputs:
  - data/ssp245_cf.npz (q10, q50, q90, sigma — 222,023 × 27)
  - data/ssp585_cf.npz
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import (
    BIRL_SAMPLE_PATH, CMIP6_PROC_DIR, DATA_DIR, ENV_OUTPUT_PATH,
    FEATURE_COLS, CLIMATE_FEATURES, SSPS, COUNTRIES,
)
from counterfactual_engine import (
    load_env_models,
    generate_counterfactual_matrix,
    validate_cf_matrices,
)

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
log = logging.getLogger("02_cf")


def main():
    log.info("=" * 60)
    log.info("Stage 3: Counterfactual Matrix Generation")
    log.info("=" * 60)

    # Load env models
    log.info("Loading LightGBM models...")
    model_mu, model_sigma = load_env_models()

    # Load baseline observations
    log.info("Loading birl_sample...")
    df = pd.read_parquet(BIRL_SAMPLE_PATH)
    log.info(f"  {df.shape[0]:,} obs")

    for ssp in SSPS:
        log.info(f"\n{'='*40}")
        log.info(f"Generating counterfactual matrix: {ssp}")
        log.info(f"{'='*40}")

        # Load 2050 climate columns
        climate_path = CMIP6_PROC_DIR / f"obs_climate_{ssp}.parquet"
        if not climate_path.exists():
            log.warning(f"  Missing {climate_path}, skipping {ssp}")
            continue

        climate_2050 = pd.read_parquet(climate_path)

        # Replace climate features in df
        df_2050 = df.copy()
        for col in CLIMATE_FEATURES:
            if col in climate_2050.columns:
                df_2050[col] = climate_2050[col].values

        # Generate counterfactual matrices
        q10, q50, q90, sigma = generate_counterfactual_matrix(model_mu, model_sigma, df_2050)

        # Save
        out_path = DATA_DIR / f"{ssp}_cf.npz"
        np.savez_compressed(
            str(out_path),
            q10=q10, q50=q50, q90=q90, sigma=sigma,
        )
        log.info(f"  Saved: {out_path} ({out_path.stat().st_size/1e6:.1f} MB)")

        # Validate
        validate_cf_matrices(ENV_OUTPUT_PATH, q50, df, label=f"{ssp} 2050")

    log.info("\nStage 3 complete.")


if __name__ == "__main__":
    main()
