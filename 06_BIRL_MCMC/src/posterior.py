"""
Posterior extraction: convert raw MCMC samples to household/country/global parameter tables.

Updated for hier_noalpha / R3 models:
  - No alpha parameters
  - Country-level unbounded ρ/γ (rho_c_unbounded, gamma_c_unbounded)
  - Global beta (log_beta)
  - Within-country sigma (sigma_rho_W, sigma_gamma_W)
"""

import logging

import numpy as np
import pandas as pd

log = logging.getLogger("birl")


def extract_and_save_posterior(np_samples, hh_ids, hh_ctry, ctry_list, out_dir):
    """Extract HH/country/global params from posterior samples and save."""
    # --- Household-level ---
    rows = {"hh_id_merge": hh_ids, "country": hh_ctry}
    for name in ["rho", "gamma"]:
        key = f"{name}_i"
        if key not in np_samples:
            continue
        s = np_samples[key]  # (n_total_samples, N_hh)
        rows[f"{name}_mean"] = s.mean(0)
        rows[f"{name}_std"] = s.std(0)
        rows[f"{name}_q025"] = np.quantile(s, 0.025, axis=0)
        rows[f"{name}_q975"] = np.quantile(s, 0.975, axis=0)

    hh_df = pd.DataFrame(rows)
    hh_df.to_parquet(out_dir / "main_hh_params.parquet", index=False)
    log.info(f"  HH params: {hh_df.shape}")

    # --- Country-level ---
    c_rows = {"country": ctry_list}

    # ρ country unbounded → transform to bounded for reporting
    if "rho_c_unbounded" in np_samples:
        rho_c_ub = np_samples["rho_c_unbounded"]  # (n_samples, N_country)
        from jax.nn import sigmoid
        rho_c_bounded = 0.1 + 4.9 * sigmoid(rho_c_ub)
        c_rows["rho_c_mean"] = rho_c_bounded.mean(0)
        c_rows["rho_c_q025"] = np.quantile(rho_c_bounded, 0.025, axis=0)
        c_rows["rho_c_q975"] = np.quantile(rho_c_bounded, 0.975, axis=0)

    # γ country unbounded → transform
    if "gamma_c_unbounded" in np_samples:
        gamma_c_ub = np_samples["gamma_c_unbounded"]
        from jax.nn import sigmoid
        gamma_c_bounded = 0.1 + 29.9 * sigmoid(gamma_c_ub)
        c_rows["gamma_c_mean"] = gamma_c_bounded.mean(0)
        c_rows["gamma_c_q025"] = np.quantile(gamma_c_bounded, 0.025, axis=0)
        c_rows["gamma_c_q975"] = np.quantile(gamma_c_bounded, 0.975, axis=0)

    c_df = pd.DataFrame(c_rows)
    c_df.to_csv(out_dir / "main_country_params.csv", index=False)
    log.info(f"  Country params: {c_df.shape}")

    # --- Global-level ---
    g_rows = {}
    for k in ["mu_rho_G", "sigma_rho_G", "sigma_rho_W",
              "mu_gamma_G", "sigma_gamma_G", "sigma_gamma_W",
              "log_beta"]:
        if k in np_samples:
            s = np_samples[k]
            g_rows[k] = [float(s.mean())]
            g_rows[f"{k}_q025"] = [float(np.quantile(s, 0.025))]
            g_rows[f"{k}_q975"] = [float(np.quantile(s, 0.975))]

    # Add beta (exp of log_beta) for convenience
    if "beta" in np_samples:
        s = np_samples["beta"]
        g_rows["beta"] = [float(s.mean())]
        g_rows["beta_q025"] = [float(np.quantile(s, 0.025))]
        g_rows["beta_q975"] = [float(np.quantile(s, 0.975))]

    pd.DataFrame(g_rows).to_csv(out_dir / "main_global_params.csv", index=False)
    log.info(f"  Global params saved")

    return hh_df
