"""
Stage 2 Runner: Process CMIP6 data → delta factors → apply to observations.

Prerequisites:
  - CMIP6 CSVs in data/cmip6_raw/ (from GEE extraction)
  - birl_sample.parquet (from Step 06)

Outputs:
  - data/cmip6_processed/obs_climate_ssp245.parquet
  - data/cmip6_processed/obs_climate_ssp585.parquet
  - data/ndvi_model_growing.joblib, ndvi_model_preseason.joblib
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import (
    BIRL_SAMPLE_PATH, CMIP6_RAW_DIR, CMIP6_PROC_DIR,
    CLIMATE_FEATURES, SSPS,
)
from climate_processing import (
    compute_ensemble_deltas,
    train_ndvi_models,
    apply_deltas_to_observations,
    print_signal_diagnostics,
)

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
log = logging.getLogger("01_climate")


def main():
    log.info("=" * 60)
    log.info("Stage 2: Climate Processing")
    log.info("=" * 60)

    # Load baseline observations
    log.info("Loading birl_sample...")
    df = pd.read_parquet(BIRL_SAMPLE_PATH)
    log.info(f"  {df.shape[0]:,} obs, {df.shape[1]} cols")

    # Step 2a: Compute ensemble deltas
    log.info("\nStep 2a: Computing ensemble deltas from CMIP6...")
    deltas = compute_ensemble_deltas(CMIP6_RAW_DIR)
    deltas.to_parquet(CMIP6_PROC_DIR / "ensemble_deltas.parquet", index=False)
    log.info(f"  Saved deltas: {len(deltas)} points")

    # Step 2b: Train NDVI prediction models
    log.info("\nStep 2b: Training NDVI prediction models...")
    ndvi_grow, ndvi_pre, r2_g, r2_p = train_ndvi_models(df)
    log.info(f"  NDVI growing R²={r2_g:.3f}, preseason R²={r2_p:.3f}")

    # Step 2c-d: Apply deltas for each SSP
    for ssp in SSPS:
        log.info(f"\nStep 2c: Applying {ssp} deltas...")
        df_2050 = apply_deltas_to_observations(df, deltas, ndvi_grow, ndvi_pre, ssp)

        # Save adjusted climate columns
        out_cols = ["gps_lat_final", "gps_lon_final", "country"] + CLIMATE_FEATURES
        df_2050[out_cols].to_parquet(
            CMIP6_PROC_DIR / f"obs_climate_{ssp}.parquet", index=False
        )

        # Diagnostics
        print_signal_diagnostics(df, df_2050, ssp)

    log.info("\nStage 2 complete.")


if __name__ == "__main__":
    main()
