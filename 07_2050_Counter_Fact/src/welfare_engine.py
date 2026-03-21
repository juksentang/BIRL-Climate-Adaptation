"""
Stage 4: CE Computation & Welfare Analysis.

Port of BIRL Stone-Geary utility from JAX to NumPy.
Computes certainty equivalents for 6 scenarios × 6 countries × 3,000 posterior samples.

Key functions:
  - load_country_posteriors: Load and transform MCMC posterior samples
  - build_feasibility_data: Reconstruct empirical feasibility masks
  - stone_geary_numpy: Vectorized CRRA utility (matches models.py:40-52)
  - compute_ce_matrix: CE for all obs × all actions
  - run_all_scenarios: Full pipeline
"""

import logging
import pickle
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    POSTERIOR_PATH, BIRL_SAMPLE_PATH, ACTION_CONFIG_PATH, ENV_OUTPUT_PATH,
    RHO_LO, RHO_HI, GAMMA_LO, GAMMA_HI,
    COUNTRIES, SCENARIOS, THIN_FACTOR, N_ACTIONS,
    DATA_DIR, RESULTS_DIR,
)

log = logging.getLogger("welfare")


# ═══════════════════════════════════════════════
# 4a. Posterior Loading
# ═══════════════════════════════════════════════

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def load_country_posteriors(
    posterior_path: Path = POSTERIOR_PATH,
    thin: int = THIN_FACTOR,
) -> dict:
    """Load MCMC posterior, extract country-level rho and gamma.

    Applies sigmoid transform matching models.py:86-102:
      rho_c = 0.1 + 4.9 * sigmoid(rho_c_unbounded)
      gamma_c = 0.1 + 29.9 * sigmoid(gamma_c_unbounded)

    Returns: {
        'rho_c':     ndarray (K, 6),  # bounded [0.1, 5.0]
        'gamma_c':   ndarray (K, 6),  # bounded [0.1, 30.0]
        'countries':  list of 6 country names (alphabetical)
        'n_samples': K (after thinning)
    }
    """
    # Try lightweight .npz first (512 KB vs 2.8 GB pickle)
    lightweight_path = DATA_DIR / "posterior_country_params.npz"
    if lightweight_path.exists():
        log.info(f"Loading lightweight posterior from {lightweight_path} ...")
        data = np.load(str(lightweight_path))
        rho_c_ub = data["rho_c_unbounded"]
        gamma_c_ub = data["gamma_c_unbounded"]
    else:
        log.info(f"Loading full posterior from {posterior_path} ...")
        with open(posterior_path, "rb") as f:
            checkpoint = pickle.load(f)
        if "posterior" in checkpoint:
            samples = checkpoint["posterior"]
        else:
            samples = checkpoint
        rho_c_ub = np.array(samples["rho_c_unbounded"])
        gamma_c_ub = np.array(samples["gamma_c_unbounded"])
        del checkpoint, samples

    log.info(f"  Raw posterior: {rho_c_ub.shape[0]} samples × {rho_c_ub.shape[1]} countries")

    # Thin
    rho_c_ub = rho_c_ub[::thin]
    gamma_c_ub = gamma_c_ub[::thin]
    K = rho_c_ub.shape[0]
    log.info(f"  After thinning (1/{thin}): {K} samples")

    # Sigmoid transform to bounded space
    rho_c = RHO_LO + (RHO_HI - RHO_LO) * _sigmoid(rho_c_ub)
    gamma_c = GAMMA_LO + (GAMMA_HI - GAMMA_LO) * _sigmoid(gamma_c_ub)

    # Verify against known point estimates
    log.info(f"  rho_c medians: {np.median(rho_c, axis=0).round(2).tolist()}")
    log.info(f"  gamma_c medians: {np.median(gamma_c, axis=0).round(2).tolist()}")

    return {
        "rho_c": rho_c.astype(np.float64),
        "gamma_c": gamma_c.astype(np.float64),
        "countries": COUNTRIES,
        "n_samples": K,
    }


# ═══════════════════════════════════════════════
# 4b. Feasibility Mask
# ═══════════════════════════════════════════════

def build_feasibility_data(
    birl_path: Path = BIRL_SAMPLE_PATH,
    action_cfg_path: Path = ACTION_CONFIG_PATH,
) -> dict:
    """Reconstruct empirical feasibility mask and obs-to-cz mapping.

    Replicates logic from data_loader.py:88-97 and 128-162.

    Returns: {
        'feasibility_mask': ndarray (N_CZ, 27) boolean,
        'obs_cz_idx': ndarray (N_obs,) int,
        'obs_country_idx': ndarray (N_obs,) int,
        'country_to_idx': dict,
        'N_CZ': int,
    }
    """
    df = pd.read_parquet(birl_path)

    with open(action_cfg_path) as f:
        action_cfg = json.load(f)

    # Country index (alphabetical)
    country_to_idx = {c: i for i, c in enumerate(COUNTRIES)}
    df["country_idx"] = df["country"].map(country_to_idx).astype(np.int32)

    # Country × Zone composite index
    df["cz_key"] = df["country"] + "_" + df["action_zone"].astype(str)
    cz_keys_sorted = sorted(df["cz_key"].unique())
    cz_to_idx = {k: i for i, k in enumerate(cz_keys_sorted)}
    df["cz_idx"] = df["cz_key"].map(cz_to_idx).astype(np.int32)

    N_CZ = len(cz_keys_sorted)
    log.info(f"  N_CZ = {N_CZ}, CZ combos: {cz_keys_sorted}")

    # Build empirical feasibility mask
    feas = np.zeros((N_CZ, N_ACTIONS), dtype=bool)
    for (cz_key, action_id), _ in df.groupby(["cz_key", "action_id"]).size().items():
        feas[cz_to_idx[cz_key], action_id] = True

    log.info(f"  Feasibility: {feas.sum(1).tolist()} actions per CZ")

    return {
        "feasibility_mask": feas,
        "obs_cz_idx": df["cz_idx"].values,
        "obs_country_idx": df["country_idx"].values,
        "country_to_idx": country_to_idx,
        "N_CZ": N_CZ,
    }


# ═══════════════════════════════════════════════
# 4c. Stone-Geary Utility (NumPy port)
# ═══════════════════════════════════════════════

def stone_geary_numpy(Y, gamma, rho, eps=1.0):
    """Stone-Geary CRRA: (surplus^(1-rho) - 1) / (1-rho).

    Exact port of models.py:40-52 from JAX to NumPy.
    Handles rho≈1 via Taylor expansion.

    Y: ndarray, any shape
    gamma, rho: scalar float
    Returns: same shape as Y
    """
    surplus = np.maximum(Y - gamma, eps)
    a = 1.0 - rho
    log_s = np.log(surplus)
    taylor = log_s * (1.0 + 0.5 * a * log_s)
    safe_a = np.where(np.abs(a) < 0.02, 1.0, a)
    power = (np.exp(safe_a * log_s) - 1.0) / safe_a
    return np.where(np.abs(a) < 0.02, taylor, power)


# ═══════════════════════════════════════════════
# 4d. CE Computation
# ═══════════════════════════════════════════════

def compute_ce_matrix(q10, q50, q90, rho, gamma, eps=1.0):
    """Compute CE for all obs × all actions.

    5-point quadrature matching models.py:112-119.

    q10, q50, q90: (N_obs, N_actions)
    rho, gamma: scalar
    Returns: CE matrix (N_obs, N_actions)
    """
    y1 = q10
    y2 = 0.5 * (q10 + q50)
    y3 = q50
    y4 = 0.5 * (q50 + q90)
    y5 = q90

    eu = (0.10 * stone_geary_numpy(y1, gamma, rho, eps)
        + 0.20 * stone_geary_numpy(y2, gamma, rho, eps)
        + 0.40 * stone_geary_numpy(y3, gamma, rho, eps)
        + 0.20 * stone_geary_numpy(y4, gamma, rho, eps)
        + 0.10 * stone_geary_numpy(y5, gamma, rho, eps))

    # Invert: CE = gamma + ((1-rho)*EU + 1)^(1/(1-rho))
    a = 1.0 - rho
    if abs(a) < 0.02:
        # rho ≈ 1: U = log(surplus), CE = gamma + exp(EU)
        ce = gamma + np.exp(eu)
    else:
        inner = a * eu + 1.0
        inner = np.maximum(inner, 1e-10)
        ce = gamma + np.power(inner, 1.0 / a)

    return ce


def compute_country_ce_for_sample(
    q10, q50, q90, rho, gamma,
    country_mask, obs_cz_idx, feasibility_mask,
):
    """For one country and one (rho, gamma) pair, compute median optimal CE.

    Args:
        q10, q50, q90: (N_obs, 27) full arrays
        rho, gamma: scalar
        country_mask: boolean (N_obs,)
        obs_cz_idx: int (N_obs,)
        feasibility_mask: (N_CZ, 27) boolean

    Returns: float (median CE across obs in this country)
    """
    # Subset to country
    c_q10 = q10[country_mask]
    c_q50 = q50[country_mask]
    c_q90 = q90[country_mask]
    c_cz = obs_cz_idx[country_mask]

    # CE for all actions
    ce_mat = compute_ce_matrix(c_q10, c_q50, c_q90, rho, gamma)

    # Mask infeasible actions
    c_feas = feasibility_mask[c_cz]
    ce_mat = np.where(c_feas, ce_mat, -np.inf)

    # Optimal CE per obs (max across actions)
    optimal_ce = ce_mat.max(axis=1)

    # Filter out any -inf (all actions infeasible — shouldn't happen)
    valid = optimal_ce > -np.inf
    if valid.sum() == 0:
        return np.nan

    return float(np.median(optimal_ce[valid]))


def compute_country_ce_batched(
    c_q10, c_q50, c_q90, c_feas,
    rho_samples, gamma_samples,
    floor_coverage=0.0,
    max_obs=10000,
):
    """Batch CE computation across all posterior samples for one country.

    Safety net model (floor_coverage > 0):
      The government guarantees a minimum income above the subsistence level:
        floor = gamma + (country_median_q50 - gamma) * floor_coverage
      This ensures positive surplus under the safety net.
      Y_effective = max(Y, floor)
      gamma is UNCHANGED — preference parameter, not a policy lever.

    Insurance model (via rho_samples already scaled by rho_factor):
      rho_effective = rho * rho_factor. Reduces effective risk aversion.

    Args:
        c_q10, c_q50, c_q90: (N_c, 27) — country subset
        c_feas: (N_c, 27) boolean — feasibility mask for country obs
        rho_samples: (K,) posterior samples for rho (already scaled by rho_factor)
        gamma_samples: (K,) posterior samples for gamma (never modified)
        floor_coverage: if > 0, apply safety net income floor
        max_obs: subsample to this many obs if country is larger

    Returns: (K,) array of median CE values
    """
    N_c, N_a = c_q50.shape
    K = len(rho_samples)
    results = np.empty(K, dtype=np.float64)

    # Subsample observations if country is large
    if N_c > max_obs:
        rng = np.random.RandomState(42)
        idx = rng.choice(N_c, max_obs, replace=False)
        idx.sort()
        c_q10 = c_q10[idx]
        c_q50 = c_q50[idx]
        c_q90 = c_q90[idx]
        c_feas = c_feas[idx]

    # Pre-compute country-level median income for floor calculation
    # Use median of the best q50 per obs (proxy for "typical income")
    best_q50_per_obs = np.where(c_feas, c_q50, np.nan)
    country_median_income = np.nanmedian(np.nanmax(best_q50_per_obs, axis=1))

    for k in range(K):
        rho = rho_samples[k]
        gamma = gamma_samples[k]
        eps = 1.0

        # Base quadrature points
        y1 = c_q10
        y2 = 0.5 * (c_q10 + c_q50)
        y3 = c_q50
        y4 = 0.5 * (c_q50 + c_q90)
        y5 = c_q90

        # Safety net: income floor above gamma
        # floor = gamma + (median_income - gamma) * coverage
        # At coverage=0.5: floor is halfway between gamma and median income
        # This guarantees floor > gamma → positive surplus
        if floor_coverage > 0:
            buffer = max(country_median_income - gamma, 0) * floor_coverage
            floor = gamma + buffer
            y1 = np.maximum(y1, floor)
            y2 = np.maximum(y2, floor)
            y3 = np.maximum(y3, floor)
            y4 = np.maximum(y4, floor)
            y5 = np.maximum(y5, floor)

        # Vectorized utility across all obs × actions
        # gamma is UNCHANGED — preference parameter stays the same
        eu = (0.10 * stone_geary_numpy(y1, gamma, rho, eps)
            + 0.20 * stone_geary_numpy(y2, gamma, rho, eps)
            + 0.40 * stone_geary_numpy(y3, gamma, rho, eps)
            + 0.20 * stone_geary_numpy(y4, gamma, rho, eps)
            + 0.10 * stone_geary_numpy(y5, gamma, rho, eps))

        # CE inversion (gamma unchanged)
        a = 1.0 - rho
        if abs(a) < 0.02:
            ce_mat = gamma + np.exp(eu)
        else:
            inner = np.maximum(a * eu + 1.0, 1e-10)
            ce_mat = gamma + np.power(inner, 1.0 / a)

        # Mask infeasible
        ce_mat = np.where(c_feas, ce_mat, -np.inf)
        optimal_ce = ce_mat.max(axis=1)
        valid = optimal_ce > -np.inf
        results[k] = np.median(optimal_ce[valid]) if valid.any() else np.nan

    return results


# ═══════════════════════════════════════════════
# 4e-f. Full Scenario Runner
# ═══════════════════════════════════════════════

def load_cf_matrices(scenario_climate: str) -> dict:
    """Load counterfactual matrices for a given climate scenario.

    scenario_climate: "baseline", "ssp245", or "ssp585"
    Returns: dict with 'q10', 'q50', 'q90' arrays (N_obs, 27)
    """
    if scenario_climate == "baseline":
        path = ENV_OUTPUT_PATH
    elif scenario_climate == "ssp245":
        path = DATA_DIR / "ssp245_cf.npz"
    elif scenario_climate == "ssp585":
        path = DATA_DIR / "ssp585_cf.npz"
    else:
        raise ValueError(f"Unknown climate scenario: {scenario_climate}")

    data = np.load(str(path))
    # NaN → 0 for quantiles (same as data_loader.py:66-68)
    # Keep float32 to save ~206 MB vs float64
    q10 = np.nan_to_num(data["q10"], nan=0.0).astype(np.float32)
    q50 = np.nan_to_num(data["q50"], nan=0.0).astype(np.float32)
    q90 = np.nan_to_num(data["q90"], nan=0.0).astype(np.float32)

    return {"q10": q10, "q50": q50, "q90": q90}


def run_all_scenarios(
    posterior: dict,
    feas_data: dict,
) -> pd.DataFrame:
    """Compute CE for all 6 scenarios × 6 countries × K posterior samples.

    Returns: DataFrame with columns:
        scenario, country, ce_median, ce_q025, ce_q975
    """
    rho_c = posterior["rho_c"]       # (K, 6)
    gamma_c = posterior["gamma_c"]   # (K, 6)
    K = posterior["n_samples"]

    obs_cz_idx = feas_data["obs_cz_idx"]
    obs_country_idx = feas_data["obs_country_idx"]
    feas_mask = feas_data["feasibility_mask"]
    country_to_idx = feas_data["country_to_idx"]

    # Pre-compute country masks
    country_masks = {}
    for country in COUNTRIES:
        cidx = country_to_idx[country]
        country_masks[country] = obs_country_idx == cidx

    # Load climate matrices one at a time to save ~274 MB
    # Group scenarios by climate key so each is loaded/freed once
    from collections import OrderedDict
    import gc

    rows = []
    posterior_ce = {}

    # Pre-compute country subsets
    country_subsets = {}
    for country in COUNTRIES:
        c_mask = country_masks[country]
        c_cz = obs_cz_idx[c_mask]
        c_feas = feas_mask[c_cz]
        country_subsets[country] = {"mask": c_mask, "feas": c_feas}

    # Group scenarios by climate key
    climate_groups = OrderedDict()
    for scenario_name, scenario_cfg in SCENARIOS.items():
        ckey = scenario_cfg["climate"]
        climate_groups.setdefault(ckey, []).append((scenario_name, scenario_cfg))

    for climate_key, scenario_list in climate_groups.items():
        cf = load_cf_matrices(climate_key)

        for scenario_name, scenario_cfg in scenario_list:
            rho_factor = scenario_cfg["rho_factor"]
            floor_coverage = scenario_cfg["floor_coverage"]

            log.info(f"Scenario {scenario_name}: climate={climate_key}, "
                     f"rho×{rho_factor}, floor_coverage={floor_coverage}")

            for ci, country in enumerate(COUNTRIES):
                c_mask = country_subsets[country]["mask"]
                c_feas = country_subsets[country]["feas"]

                # Country subset of matrices
                c_q10 = cf["q10"][c_mask]
                c_q50 = cf["q50"][c_mask]
                c_q90 = cf["q90"][c_mask]

                # Country-specific posterior samples
                # Insurance: scale rho. Safety net: income floor. gamma NEVER changes.
                rho_samples = rho_c[:, ci] * rho_factor
                gamma_samples = gamma_c[:, ci]  # always unmodified

                # Batched CE computation
                ce_values = compute_country_ce_batched(
                    c_q10, c_q50, c_q90, c_feas,
                    rho_samples, gamma_samples,
                    floor_coverage=floor_coverage,
                )

                ce_med = np.nanmedian(ce_values)
                ce_q025 = np.nanpercentile(ce_values, 2.5)
                ce_q975 = np.nanpercentile(ce_values, 97.5)

                rows.append({
                    "scenario": scenario_name,
                    "country": country,
                    "ce_median": ce_med,
                    "ce_q025": ce_q025,
                    "ce_q975": ce_q975,
                    "rho_factor": rho_factor,
                    "floor_coverage": floor_coverage,
                })

                posterior_ce[(scenario_name, country)] = ce_values

                log.info(f"  {country}: CE = ${ce_med:.2f} "
                         f"[${ce_q025:.2f}, ${ce_q975:.2f}]")

        # Free climate matrices after processing all scenarios for this climate
        del cf
        gc.collect()

    results_df = pd.DataFrame(rows)
    return results_df, posterior_ce


# ═══════════════════════════════════════════════
# 4g. Derived Metrics
# ═══════════════════════════════════════════════

def compute_derived_metrics(
    results_df: pd.DataFrame,
    posterior_ce: dict,
) -> dict:
    """Compute climate loss, policy value, synergy from CE results.

    Returns dict of DataFrames: climate_loss, policy_value, synergy.
    """
    rows_loss = []
    rows_policy = []
    rows_synergy = []

    for country in COUNTRIES:
        ce_s0 = posterior_ce.get(("S0_current", country))
        ce_s1 = posterior_ce.get(("S1_ssp245", country))
        ce_s2 = posterior_ce.get(("S2_ssp585", country))
        ce_s3 = posterior_ce.get(("S3_insurance", country))
        ce_s4 = posterior_ce.get(("S4_safety_net", country))
        ce_s5 = posterior_ce.get(("S5_combined", country))

        if ce_s0 is None or ce_s2 is None:
            continue

        # Climate loss = CE(S0) - CE(S2)
        loss_585 = ce_s0 - ce_s2
        loss_245 = ce_s0 - ce_s1 if ce_s1 is not None else None

        rows_loss.append({
            "country": country,
            "loss_ssp585_median": np.nanmedian(loss_585),
            "loss_ssp585_q025": np.nanpercentile(loss_585, 2.5),
            "loss_ssp585_q975": np.nanpercentile(loss_585, 97.5),
            "loss_ssp585_pct_median": np.nanmedian(loss_585 / ce_s0 * 100),
            "loss_ssp245_median": np.nanmedian(loss_245) if loss_245 is not None else np.nan,
        })

        if ce_s3 is not None and ce_s4 is not None and ce_s5 is not None:
            # Policy value = CE(policy) - CE(S2)
            ins_val = ce_s3 - ce_s2
            sn_val = ce_s4 - ce_s2
            comb_val = ce_s5 - ce_s2
            synergy = comb_val - (ins_val + sn_val)

            rows_policy.append({
                "country": country,
                "insurance_median": np.nanmedian(ins_val),
                "insurance_q025": np.nanpercentile(ins_val, 2.5),
                "insurance_q975": np.nanpercentile(ins_val, 97.5),
                "safety_net_median": np.nanmedian(sn_val),
                "safety_net_q025": np.nanpercentile(sn_val, 2.5),
                "safety_net_q975": np.nanpercentile(sn_val, 97.5),
                "combined_median": np.nanmedian(comb_val),
                "combined_q025": np.nanpercentile(comb_val, 2.5),
                "combined_q975": np.nanpercentile(comb_val, 97.5),
                "policy_ratio": np.nanmedian(ins_val) / max(np.nanmedian(sn_val), 0.01),
            })

            rows_synergy.append({
                "country": country,
                "synergy_median": np.nanmedian(synergy),
                "synergy_q025": np.nanpercentile(synergy, 2.5),
                "synergy_q975": np.nanpercentile(synergy, 97.5),
                "synergy_pct": np.nanmedian(synergy / np.maximum(comb_val, 0.01) * 100),
            })

    return {
        "climate_loss": pd.DataFrame(rows_loss),
        "policy_value": pd.DataFrame(rows_policy),
        "synergy": pd.DataFrame(rows_synergy),
    }
