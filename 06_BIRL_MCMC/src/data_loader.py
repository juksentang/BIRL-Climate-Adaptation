"""
Data loading, index encoding, and JAX array construction.
Returns a frozen BIRLData dataclass with everything needed by the pipeline.

Round 1 changes vs 05_Archived:
  - P0+P7: Country×zone empirical feasibility mask (replaces zone-only mask)
  - P2+P8: Pre-computed log-space arrays for belief bias
"""

import json
import logging
import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import jax.numpy as jnp

log = logging.getLogger("birl")


@dataclasses.dataclass(frozen=True)
class BIRLData:
    """All data arrays and metadata needed by the BIRL pipeline."""
    # Dimensions
    N_OBS: int
    N_HH: int
    N_COUNTRY: int
    N_ACTIONS: int
    N_ZONES: int
    N_CZ: int                     # number of non-empty country×zone combos

    # Metadata
    countries: list                # sorted country name strings
    hh_list: np.ndarray            # sorted unique hh_id_merge
    hh_to_idx: dict                # {hh_id_merge: int}
    hh_countries: np.ndarray       # string country per HH (for output)
    action_cfg: dict               # raw action_space_config.json

    # Main model data dict (passed as **kwargs to model functions)
    main_data: dict

    # Robustness-specific arrays
    survival_thresh_jax: object    # jnp array (N_obs,)
    totcons_jax: object            # jnp array (N_obs,)
    hh_gamma_fixed: object         # jnp array (N_hh,) for R4

    # R9 subset
    r9_data: dict
    r9_hh_list: np.ndarray         # R9 unique hh_id_merge (for Spearman matching)

    # Extended model data (main_data + z-scored HH covariates)
    main_data_ext: dict


def _build_log_space_arrays(q10_raw, q50_raw, q90_raw, sigma_raw):
    """Pre-compute log-space arrays for belief bias.

    Returns (cf_q10, cf_q50, cf_q90, cf_sigma,
             cf_log_q10, cf_log_q50, cf_log_q90, cf_sigma_log)
    all as jnp float32 arrays.
    """
    _eps = 0.01

    # Standard USD-space arrays (NaN→0 for q, NaN→1 for sigma)
    cf_q10 = jnp.array(np.nan_to_num(q10_raw, nan=0.0), dtype=jnp.float32)
    cf_q50 = jnp.array(np.nan_to_num(q50_raw, nan=0.0), dtype=jnp.float32)
    cf_q90 = jnp.array(np.nan_to_num(q90_raw, nan=0.0), dtype=jnp.float32)
    cf_sigma = jnp.array(np.nan_to_num(sigma_raw, nan=1.0), dtype=jnp.float32)

    # Log-space: floor at eps to handle negative / zero values
    q10_safe = np.maximum(np.nan_to_num(q10_raw, nan=_eps), _eps).astype(np.float32)
    q50_safe = np.maximum(np.nan_to_num(q50_raw, nan=_eps), _eps).astype(np.float32)
    q90_safe = np.maximum(np.nan_to_num(q90_raw, nan=_eps), _eps).astype(np.float32)

    cf_log_q10 = jnp.array(np.log(q10_safe))
    cf_log_q50 = jnp.array(np.log(q50_safe))
    cf_log_q90 = jnp.array(np.log(q90_safe))

    # sigma_log from q90/q50 ratio (lognormal z_{0.9} = 1.282)
    ratio = np.maximum(q90_safe / q50_safe, 1.0 + 1e-4)
    sigma_log_np = np.clip(np.log(ratio) / 1.282, 0.01, 5.0).astype(np.float32)
    cf_sigma_log = jnp.array(sigma_log_np)

    return cf_q10, cf_q50, cf_q90, cf_sigma, cf_log_q10, cf_log_q50, cf_log_q90, cf_sigma_log


def _build_empirical_mask(df, n_actions, cz_to_idx):
    """Build empirical feasibility mask from observed (cz_key, action_id) pairs.

    Returns (N_CZ, N_ACTIONS) boolean array.
    """
    n_cz = len(cz_to_idx)
    feas = np.zeros((n_cz, n_actions), dtype=bool)
    for (cz_key, action_id), _ in df.groupby(["cz_key", "action_id"]).size().items():
        feas[cz_to_idx[cz_key], action_id] = True
    return feas


def load_all(data_dir: Path) -> BIRLData:
    """Load parquet + npz + json, encode indices, build JAX arrays. Returns BIRLData."""
    import time
    t0 = time.time()

    # ── Load raw files ──
    df = pd.read_parquet(data_dir / "birl_sample.parquet")
    env = np.load(str(data_dir / "env_model_output.npz"), allow_pickle=True)
    with open(data_dir / "action_space_config.json") as f:
        action_cfg = json.load(f)

    log.info(f"  Parquet: {df.shape[0]:,} x {df.shape[1]}  ({time.time()-t0:.1f}s)")

    # ── Household index ──
    hh_list = np.sort(df["hh_id_merge"].unique())
    hh_to_idx = {h: i for i, h in enumerate(hh_list)}
    df["hh_idx"] = df["hh_id_merge"].map(hh_to_idx).astype(np.int32)

    # ── Country index (alphabetical) ──
    countries = sorted(df["country"].unique())
    country_to_idx = {c: i for i, c in enumerate(countries)}
    df["country_idx"] = df["country"].map(country_to_idx).astype(np.int32)

    # ── HH -> country mapping ──
    hh_info = df.drop_duplicates("hh_id_merge").set_index("hh_idx").sort_index()
    hh_country_np = hh_info["country_idx"].values.astype(np.int32)
    hh_countries = hh_info["country"].values

    # ── Zone index (keep for reference) ──
    zone_order = sorted(action_cfg["zones"])
    zone_to_idx = {z: i for i, z in enumerate(zone_order)}
    df["zone_idx"] = df["action_zone"].map(zone_to_idx).astype(np.int32)

    # ── Country × Zone composite index (P0+P7 fix) ──
    df["cz_key"] = df["country"] + "_" + df["action_zone"].astype(str)
    cz_keys_sorted = sorted(df["cz_key"].unique())
    cz_to_idx = {k: i for i, k in enumerate(cz_keys_sorted)}
    df["cz_idx"] = df["cz_key"].map(cz_to_idx).astype(np.int32)

    N_OBS = len(df)
    N_HH = len(hh_list)
    N_COUNTRY = len(countries)
    N_ACTIONS = action_cfg["n_actions"]
    N_ZONES = len(zone_order)
    N_CZ = len(cz_keys_sorted)

    log.info(f"  N_obs={N_OBS:,}  N_hh={N_HH:,}  N_country={N_COUNTRY}  "
             f"N_actions={N_ACTIONS}  N_zones={N_ZONES}  N_cz={N_CZ}")
    log.info(f"  Countries: {countries}")
    log.info(f"  CZ combos: {cz_keys_sorted}")

    # ── Empirical feasibility mask: shape (N_CZ, N_ACTIONS) ──
    feas_np = _build_empirical_mask(df, N_ACTIONS, cz_to_idx)
    feas_jax = jnp.array(feas_np)
    log.info(f"  Feasibility (empirical): {feas_np.sum(1).tolist()} actions/cz")

    # Verify: every obs's chosen action is feasible
    obs_cz = df["cz_idx"].values
    obs_act = df["action_id"].values
    n_infeasible = int(np.sum(~feas_np[obs_cz, obs_act]))
    assert n_infeasible == 0, (
        f"BUG: {n_infeasible} obs chose infeasible actions after empirical mask"
    )
    log.info(f"  Feasibility check: 0 infeasible observations (PASS)")

    # ── JAX arrays: indices ──
    obs_action_jax = jnp.array(df["action_id"].values, dtype=jnp.int32)
    obs_hh_idx_jax = jnp.array(df["hh_idx"].values, dtype=jnp.int32)
    obs_cz_idx_jax = jnp.array(df["cz_idx"].values, dtype=jnp.int32)
    hh_country_jax = jnp.array(hh_country_np, dtype=jnp.int32)

    # ── JAX arrays: environment model (USD + log-space) ──
    (cf_q10, cf_q50, cf_q90, cf_sigma,
     cf_log_q10, cf_log_q50, cf_log_q90, cf_sigma_log) = _build_log_space_arrays(
        env["q10"], env["q50"], env["q90"], env["sigma"]
    )
    log.info(f"  Log-space: sigma_log median={float(jnp.median(cf_sigma_log)):.3f}, "
             f"max={float(jnp.max(cf_sigma_log)):.3f}")

    # ── Main data dict ──
    main_data = dict(
        obs_action=obs_action_jax,
        obs_hh_idx=obs_hh_idx_jax,
        hh_country_idx=hh_country_jax,
        obs_cz_idx=obs_cz_idx_jax,
        feasibility_mask=feas_jax,
        cf_q10=cf_q10, cf_q50=cf_q50, cf_q90=cf_q90, cf_sigma=cf_sigma,
        cf_log_q10=cf_log_q10, cf_log_q50=cf_log_q50,
        cf_log_q90=cf_log_q90, cf_sigma_log=cf_sigma_log,
        N_hh=N_HH,
        N_country=N_COUNTRY,
    )

    # ── R1 extra data ──
    survival_thresh_jax = jnp.array(
        df["survival_threshold_P25"].fillna(0).values, dtype=jnp.float32)
    totcons_jax = jnp.array(
        df["totcons_USD"].fillna(0).values, dtype=jnp.float32)

    # ── R4: per-household fixed gamma ──
    hh_gamma_fixed = jnp.array(
        df.groupby("hh_idx")["survival_threshold_P25"]
        .median().sort_index().fillna(20.0).values,
        dtype=jnp.float32,
    )

    # ── R9: main-crop subset (one obs per decision_id, highest harvest value) ──
    r9_sort_val = df["harvest_value_USD"].fillna(0)
    r9_order = r9_sort_val.values.argsort()[::-1]
    r9_seen = set()
    r9_keep = []
    for i in r9_order:
        did = df.iloc[i]["decision_id"]
        if did not in r9_seen:
            r9_seen.add(did)
            r9_keep.append(i)
    r9_idx = np.sort(np.array(r9_keep))
    r9_df = df.iloc[r9_idx].reset_index(drop=True)

    r9_hh_list = np.sort(r9_df["hh_id_merge"].unique())
    r9_hh_to_idx = {h: i for i, h in enumerate(r9_hh_list)}
    r9_df["hh_idx"] = r9_df["hh_id_merge"].map(r9_hh_to_idx).astype(np.int32)
    r9_hh_info = r9_df.drop_duplicates("hh_id_merge").set_index("hh_idx").sort_index()
    r9_hh_country = r9_hh_info["country_idx"].values.astype(np.int32)

    # R9 country×zone mask
    r9_df["cz_key"] = r9_df["country"] + "_" + r9_df["action_zone"].astype(str)
    r9_cz_keys = sorted(r9_df["cz_key"].unique())
    r9_cz_to_idx = {k: i for i, k in enumerate(r9_cz_keys)}
    r9_df["cz_idx"] = r9_df["cz_key"].map(r9_cz_to_idx).astype(np.int32)
    r9_feas = _build_empirical_mask(r9_df, N_ACTIONS, r9_cz_to_idx)

    # R9 log-space arrays
    (r9_cf_q10, r9_cf_q50, r9_cf_q90, r9_cf_sigma,
     r9_cf_log_q10, r9_cf_log_q50, r9_cf_log_q90, r9_cf_sigma_log) = _build_log_space_arrays(
        env["q10"][r9_idx], env["q50"][r9_idx],
        env["q90"][r9_idx], env["sigma"][r9_idx]
    )

    r9_data = dict(
        obs_action=jnp.array(r9_df["action_id"].values, dtype=jnp.int32),
        obs_hh_idx=jnp.array(r9_df["hh_idx"].values, dtype=jnp.int32),
        hh_country_idx=jnp.array(r9_hh_country, dtype=jnp.int32),
        obs_cz_idx=jnp.array(r9_df["cz_idx"].values, dtype=jnp.int32),
        feasibility_mask=jnp.array(r9_feas),
        cf_q10=r9_cf_q10, cf_q50=r9_cf_q50, cf_q90=r9_cf_q90, cf_sigma=r9_cf_sigma,
        cf_log_q10=r9_cf_log_q10, cf_log_q50=r9_cf_log_q50,
        cf_log_q90=r9_cf_log_q90, cf_sigma_log=r9_cf_sigma_log,
        N_hh=len(r9_hh_list),
        N_country=N_COUNTRY,
    )
    log.info(f"  R9 subset: {len(r9_idx):,} obs, {len(r9_hh_list):,} HH")

    del env

    # ── Extended model: z-scored household covariates ──
    def _hh_zscore(col):
        raw = df.groupby("hh_idx")[col].median().sort_index()
        raw = raw.fillna(raw.median())
        mu, sd = float(raw.mean()), float(raw.std())
        if sd < 1e-10:
            sd = 1.0
        return jnp.array(((raw.values - mu) / sd).astype(np.float32))

    hh_education = _hh_zscore("formal_education_manager")
    hh_conflict = _hh_zscore("conflict_events_25km_12m")
    hh_asset = _hh_zscore("hh_asset_index")
    hh_female = _hh_zscore("female_manager")
    hh_age = _hh_zscore("age_manager")

    main_data_ext = dict(
        **main_data,
        hh_education=hh_education,
        hh_conflict=hh_conflict,
        hh_asset=hh_asset,
        hh_female=hh_female,
        hh_age=hh_age,
    )
    log.info(f"  Extended covariates: 5 HH-level z-scored arrays")

    log.info(f"  Data ready ({time.time()-t0:.1f}s total)")

    return BIRLData(
        N_OBS=N_OBS, N_HH=N_HH, N_COUNTRY=N_COUNTRY,
        N_ACTIONS=N_ACTIONS, N_ZONES=N_ZONES, N_CZ=N_CZ,
        countries=countries, hh_list=hh_list, hh_to_idx=hh_to_idx,
        hh_countries=hh_countries, action_cfg=action_cfg,
        main_data=main_data,
        survival_thresh_jax=survival_thresh_jax,
        totcons_jax=totcons_jax,
        hh_gamma_fixed=hh_gamma_fixed,
        r9_data=r9_data, r9_hh_list=r9_hh_list,
        main_data_ext=main_data_ext,
    )
