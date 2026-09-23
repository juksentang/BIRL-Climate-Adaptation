"""
Stage 4 Runner: CE computation for all scenarios.

Prerequisites:
  - data/ssp245_cf.npz, ssp585_cf.npz (from Stage 3)
  - posterior.pkl (from Step 06 MCMC)

Outputs:
  - results/ce_by_scenario_country.csv
  - results/ce_posterior_samples.npz
  - results/climate_loss.csv
  - results/policy_value.csv
  - results/synergy.csv
"""

import sys
import gc
import logging
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import RESULTS_DIR
from welfare_engine import (
    load_country_posteriors,
    build_feasibility_data,
    run_all_scenarios,
    compute_derived_metrics,
)
from reporting import print_summary_table

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
log = logging.getLogger("03_welfare")


def main():
    log.info("=" * 60)
    log.info("Stage 4: CE Computation & Welfare Analysis")
    log.info("=" * 60)

    t0 = time.time()

    # Load posterior (uses lightweight .npz if available, ~512KB vs 2.8GB pkl)
    log.info("Loading MCMC posterior...")
    posterior = load_country_posteriors()

    # Build feasibility data
    log.info("Building feasibility masks...")
    feas_data = build_feasibility_data()
    gc.collect()  # reclaim parquet heap fragmentation

    # Run all scenarios
    log.info(f"\nRunning 6 scenarios × 6 countries × {posterior['n_samples']} samples...")
    results_df, posterior_ce = run_all_scenarios(posterior, feas_data)

    # Save main results
    results_df.to_csv(RESULTS_DIR / "ce_by_scenario_country.csv", index=False)
    log.info(f"  Saved: results/ce_by_scenario_country.csv")

    # Save posterior CE arrays for visualization
    save_dict = {}
    for (scenario, country), arr in posterior_ce.items():
        key = f"{scenario}__{country}"
        save_dict[key] = arr
    np.savez_compressed(str(RESULTS_DIR / "ce_posterior_samples.npz"), **save_dict)
    log.info(f"  Saved: results/ce_posterior_samples.npz")

    # Compute derived metrics
    log.info("\nComputing derived metrics...")
    metrics = compute_derived_metrics(results_df, posterior_ce)

    metrics["climate_loss"].to_csv(RESULTS_DIR / "climate_loss.csv", index=False)
    metrics["policy_value"].to_csv(RESULTS_DIR / "policy_value.csv", index=False)
    metrics["synergy"].to_csv(RESULTS_DIR / "synergy.csv", index=False)
    log.info("  Saved: climate_loss.csv, policy_value.csv, synergy.csv")

    # Print summary
    print_summary_table(
        results_df,
        metrics["climate_loss"],
        metrics["policy_value"],
        metrics["synergy"],
    )

    elapsed = time.time() - t0
    log.info(f"\nStage 4 complete ({elapsed:.0f}s)")


if __name__ == "__main__":
    main()
