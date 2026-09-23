"""
Data loading for BIRL v2 (adapted from 06_BIRL_MCMC/src/data_loader.py).

Reads (read-only) from 06's data directory:
  birl_sample.parquet      obs-level table (country, hh_id_merge, action_id, action_zone, ...)
  env_model_output.npz     q10/q50/q90/sigma  (N_obs, 27) predicted per-plot income (USD)
  action_space_config.json feasibility zones, action labels

Returns a plain dict (see load_data) with the arrays the v2 models need plus
the per-country income scale m_c (median over observations of the CHOSEN
action's q50), which is a data constant used for gamma_c = s_c * m_c,
eps_c = EPS_FRAC * m_c and reward = CE / m_c.
"""

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import jax.numpy as jnp

from cropchoice.config import LOG_CLIP
from cropchoice.models_v2 import add_precomputed

log = logging.getLogger("birl_v2")

_LOG_FLOOR = 0.01   # as in 06: floor q at 0.01 USD before taking logs
Q_MAX_WARN = 1e5    # USD: warn loudly if any q exceeds this (float32 logit resolution, see below)


def compute_m_c(q50, action_id, country_idx, n_country):
    """Per-country median of the chosen action's q50 (numpy, float64)."""
    chosen = q50[np.arange(len(action_id)), action_id].astype(np.float64)
    m_c = np.zeros(n_country, dtype=np.float64)
    for c in range(n_country):
        m_c[c] = np.nanmedian(chosen[country_idx == c])   # NaN-skipping, like pandas .median()
    return m_c


def _build_empirical_mask(df, n_actions, cz_to_idx):
    """Empirical (country x zone, action) feasibility mask from observed pairs."""
    feas = np.zeros((len(cz_to_idx), n_actions), dtype=bool)
    for (cz_key, action_id), _ in df.groupby(["cz_key", "action_id"]).size().items():
        feas[cz_to_idx[cz_key], action_id] = True
    return feas


def _log_arrays(q10_raw, q50_raw, q90_raw):
    """Log-space arrays floored at _LOG_FLOOR (NaN -> floor), float32 jnp."""
    out = []
    for q in (q10_raw, q50_raw, q90_raw):
        q_safe = np.maximum(np.nan_to_num(q, nan=_LOG_FLOOR), _LOG_FLOOR).astype(np.float32)
        out.append(jnp.array(np.log(q_safe)))
    return out


def load_data(data_dir: Path) -> dict:
    """Load parquet + npz + json and build the v2 data dict.

    Keys used by the models:
      obs_action, obs_country_idx, obs_cz_idx, feasibility_mask,
      cf_log_q10, cf_log_q50, cf_log_q90, m_c, N_country
    Metadata / robustness keys:
      countries, hh_id, obs_hh_idx, hh_list, hh_country_idx, cz_keys,
      action_labels, action_cfg, N_obs, N_actions, N_cz, N_hh, m_c_np
    """
    t0 = time.time()
    data_dir = Path(data_dir)

    df = pd.read_parquet(
        data_dir / "birl_sample.parquet",
        columns=["country", "hh_id_merge", "action_id", "action_zone", "harvest_value_USD_w"],
    )
    env = np.load(str(data_dir / "env_model_output.npz"), allow_pickle=True)
    with open(data_dir / "action_space_config.json") as f:
        action_cfg = json.load(f)
    log.info(f"  Parquet: {df.shape[0]:,} rows  ({time.time()-t0:.1f}s)")

    # env rows must align with parquet rows
    hh_env = env["hh_id"]
    assert len(hh_env) == len(df), "env_model_output.npz rows != parquet rows"
    assert (hh_env == df["hh_id_merge"].values).all(), "hh_id misaligned between npz and parquet"

    # ── Country index (sorted names) ──
    countries = sorted(df["country"].unique())
    country_to_idx = {c: i for i, c in enumerate(countries)}
    country_idx = df["country"].map(country_to_idx).values.astype(np.int32)

    # ── Household index ──
    hh_list = np.sort(df["hh_id_merge"].unique())
    hh_to_idx = {h: i for i, h in enumerate(hh_list)}
    hh_idx = df["hh_id_merge"].map(hh_to_idx).values.astype(np.int32)
    hh_country = np.zeros(len(hh_list), dtype=np.int32)
    hh_country[hh_idx] = country_idx

    # ── Country x zone composite index + empirical feasibility mask ──
    df["cz_key"] = df["country"] + "_" + df["action_zone"].astype(str)
    cz_keys = sorted(df["cz_key"].unique())
    cz_to_idx = {k: i for i, k in enumerate(cz_keys)}
    cz_idx = df["cz_key"].map(cz_to_idx).values.astype(np.int32)

    N_obs = len(df)
    N_country = len(countries)
    N_actions = int(action_cfg["n_actions"])
    N_cz = len(cz_keys)
    N_hh = len(hh_list)

    feas_np = _build_empirical_mask(df, N_actions, cz_to_idx)
    obs_act = df["action_id"].values.astype(np.int32)
    n_infeasible = int(np.sum(~feas_np[cz_idx, obs_act]))
    assert n_infeasible == 0, f"{n_infeasible} obs chose infeasible actions"

    # ── Income scale m_c ──
    m_c_np = compute_m_c(env["q50"], obs_act, country_idx, N_country)

    log.info(f"  N_obs={N_obs:,}  N_country={N_country}  N_actions={N_actions}  "
             f"N_cz={N_cz}  N_hh={N_hh:,}")
    log.info(f"  Countries: {countries}")
    log.info(f"  m_c (median chosen q50, USD): "
             + ", ".join(f"{c}={m:.2f}" for c, m in zip(countries, m_c_np)))
    log.info(f"  Feasible actions per cz: {feas_np.sum(1).tolist()}")

    cf_log_q10, cf_log_q50, cf_log_q90 = _log_arrays(env["q10"], env["q50"], env["q90"])

    # ── Income range check: reward = CE / m_c and logits = beta * reward are float32;
    #    a q90 of 1e8 USD would give logits ~1e8 where float32 resolves only ~10 nat,
    #    so outliers must be visible.  LOG_CLIP (= exp(20) = 4.9e8 USD) only guards exp().
    q90_max_c = {}
    for c, name in enumerate(countries):
        sel = country_idx == c
        q90_max_c[name] = float(np.nanmax(env["q90"][sel])) if sel.any() else float("nan")
    q_max = max(float(np.nanmax(env["q90"])), float(np.nanmax(env["q50"])),
                float(np.nanmax(env["q10"])))
    log.info("  max q90 per country (USD): "
             + ", ".join(f"{n}={v:.4g}" for n, v in q90_max_c.items()))
    if q_max > Q_MAX_WARN:
        log.warning(f"  max predicted income {q_max:.4g} USD exceeds {Q_MAX_WARN:.0e}: "
                    f"logits = beta * CE / m_c may lose float32 resolution "
                    f"(check max_abs_logit in model_diagnostics.json)")
    if q_max > np.exp(LOG_CLIP):
        log.warning(f"  max predicted income {q_max:.4g} USD exceeds exp(LOG_CLIP) = "
                    f"{np.exp(LOG_CLIP):.3g}: values are clipped in the model")

    data = dict(
        # model inputs
        obs_action=jnp.array(obs_act, dtype=jnp.int32),
        obs_country_idx=jnp.array(country_idx, dtype=jnp.int32),
        obs_cz_idx=jnp.array(cz_idx, dtype=jnp.int32),
        feasibility_mask=jnp.array(feas_np),
        cf_log_q10=cf_log_q10, cf_log_q50=cf_log_q50, cf_log_q90=cf_log_q90,
        m_c=jnp.array(m_c_np, dtype=jnp.float32),
        N_country=N_country,
        # metadata (not passed to the model)
        countries=countries,
        m_c_np=m_c_np,
        hh_id=df["hh_id_merge"].values,
        obs_hh_idx=hh_idx,
        hh_list=hh_list,
        hh_country_idx=hh_country,
        cz_keys=cz_keys,
        action_labels=action_cfg["action_labels"],
        action_cfg=action_cfg,
        N_obs=N_obs, N_actions=N_actions, N_cz=N_cz, N_hh=N_hh,
        q90_max_usd=q90_max_c, q_max_usd=q_max,
    )
    del env
    # parameter-independent arrays used by compute_logits (q10/q50/q90 in USD,
    # mask_obs = feasibility_mask[obs_cz_idx]) computed once here, not per gradient
    data = add_precomputed(data)
    log.info(f"  Data ready ({time.time()-t0:.1f}s)")
    return data


MODEL_KEYS = ("obs_action", "obs_country_idx", "obs_cz_idx", "feasibility_mask",
              "cf_log_q10", "cf_log_q50", "cf_log_q90", "m_c", "N_country",
              "q10", "q50", "q90", "mask_obs")
_OBS_KEYS = ("obs_action", "obs_country_idx", "obs_cz_idx", "cf_log_q10", "cf_log_q50", "cf_log_q90",
             "q10", "q50", "q90", "mask_obs")


def model_kwargs(data: dict) -> dict:
    """The `data` dict passed to the numpyro models (model(data=model_kwargs(d), ...)).
    Adds the precomputed q10/q50/q90/mask_obs arrays if the dict lacks them."""
    data = add_precomputed(data)
    return {k: data[k] for k in MODEL_KEYS}


def select_obs(data: dict, idx) -> dict:
    """Row subset of the data dict (observation-level arrays only; masks, m_c,
    metadata unchanged).  Used by the tests on 'random real obs'."""
    idx = np.asarray(idx)
    out = {k: v for k, v in data.items() if k not in _OBS_KEYS}
    for k in _OBS_KEYS:
        if k in data:
            out[k] = jnp.asarray(np.asarray(data[k])[idx])
    if "obs_hh_idx" in data:
        out["obs_hh_idx"] = np.asarray(data["obs_hh_idx"])[idx]
    if "hh_id" in data:
        out["hh_id"] = np.asarray(data["hh_id"])[idx]
    out["N_obs"] = int(len(idx))
    return out
