"""
NumPyro models for BIRL MCMC.

Two models:
  - birl_hier_noalpha: Main model. Hierarchical ρ + γ, no α, learnable β.
  - birl_r3:           Robustness. Fixed ρ=1.5, hierarchical γ, no α, learnable β.

Design based on SVI experiments (05_BIRL_SVI):
  - α dropped: not identifiable from crop choice data (α-γ compensation)
  - β learnable: SVI found β≈0.14 (fixed β=5 was 36x too high)
  - ρ bounded [0.1, 5.0]: prevents divergence to infinity
  - γ bounded [0.1, 30.0]: upper bound from consumption median $33.8
  - Reward centered (mean-only): preserves natural scale for parameter identification
"""

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro import plate, sample, deterministic


# ── Bounds ──
RHO_LO, RHO_HI = 0.1, 5.0
GAMMA_LO, GAMMA_HI = 0.1, 30.0


def _center_reward(reward, mask):
    """Mean-subtract reward over feasible actions per obs.

    Center-only (no std division) to preserve natural utility scale.
    This lets MCMC distinguish high-ρ obs (narrow spread) from low-ρ obs
    (wide spread), which full z-score would erase.
    """
    n = mask.sum(axis=-1, keepdims=True).clip(min=1)
    mean = jnp.where(mask, reward, 0.0).sum(axis=-1, keepdims=True) / n
    return reward - mean


def _stone_geary(Y, gamma, rho, eps=1.0):
    """Stone-Geary CRRA: (surplus^(1-ρ) - 1) / (1-ρ).

    Normalized form with log(surplus) limit at ρ=1.
    JAX-safe: both branches always finite.
    """
    surplus = jnp.maximum(Y - gamma, eps)
    a = 1.0 - rho
    log_s = jnp.log(surplus)
    taylor = log_s * (1.0 + 0.5 * a * log_s)
    safe_a = jnp.where(jnp.abs(a) < 0.02, 1.0, a)
    power = (jnp.exp(safe_a * log_s) - 1.0) / safe_a
    return jnp.where(jnp.abs(a) < 0.02, taylor, power)


# =====================================================================
# Main model: Hierarchical ρ + γ, no α
# =====================================================================

def birl_hier_noalpha(
    obs_action, obs_hh_idx, hh_country_idx, obs_cz_idx,
    feasibility_mask,
    cf_q10, cf_q50, cf_q90, cf_sigma,
    cf_log_q10, cf_log_q50, cf_log_q90, cf_sigma_log,
    N_hh, N_country,
):
    N_obs, N_actions = cf_q50.shape

    # ── Learnable β ──
    log_beta = sample("log_beta", dist.Normal(1.0, 0.5))
    beta = deterministic("beta", jnp.exp(jnp.clip(log_beta, -2.0, 4.0)))

    # ── Global hyperpriors ──
    mu_rho_G = sample("mu_rho_G", dist.Normal(0.0, 1.0))
    sigma_rho_G = sample("sigma_rho_G", dist.HalfNormal(0.5))
    sigma_rho_W = sample("sigma_rho_W", dist.HalfNormal(0.3))

    mu_gamma_G = sample("mu_gamma_G", dist.Normal(0.0, 1.0))
    sigma_gamma_G = sample("sigma_gamma_G", dist.HalfNormal(0.5))
    sigma_gamma_W = sample("sigma_gamma_W", dist.HalfNormal(0.3))

    # ── Country level (non-centered) ──
    with plate("countries", N_country):
        rho_c_raw = sample("rho_c_raw", dist.Normal(0, 1))
        gamma_c_raw = sample("gamma_c_raw", dist.Normal(0, 1))

    rho_c = deterministic("rho_c_unbounded",
                          mu_rho_G + sigma_rho_G * rho_c_raw)
    gamma_c = deterministic("gamma_c_unbounded",
                            mu_gamma_G + sigma_gamma_G * gamma_c_raw)

    # ── Household level (offset around country mean) ──
    with plate("households", N_hh):
        rho_offset = sample("rho_offset", dist.Normal(0, 1))
        gamma_offset = sample("gamma_offset", dist.Normal(0, 1))

    rho_i_ub = rho_c[hh_country_idx] + sigma_rho_W * rho_offset
    rho_i = deterministic("rho_i",
                          RHO_LO + (RHO_HI - RHO_LO) * jax.nn.sigmoid(rho_i_ub))

    gamma_i_ub = gamma_c[hh_country_idx] + sigma_gamma_W * gamma_offset
    gamma_i = deterministic("gamma_i",
                            GAMMA_LO + (GAMMA_HI - GAMMA_LO) * jax.nn.sigmoid(gamma_i_ub))

    # ── Observation level ──
    ro = rho_i[obs_hh_idx][:, None]
    go = gamma_i[obs_hh_idx][:, None]

    q10 = jnp.exp(jnp.clip(cf_log_q10, -20.0, 20.0))
    q50 = jnp.exp(jnp.clip(cf_log_q50, -20.0, 20.0))
    q90 = jnp.exp(jnp.clip(cf_log_q90, -20.0, 20.0))

    y1, y2, y3 = q10, 0.5 * (q10 + q50), q50
    y4, y5 = 0.5 * (q50 + q90), q90

    reward = (0.10 * _stone_geary(y1, go, ro)
            + 0.20 * _stone_geary(y2, go, ro)
            + 0.40 * _stone_geary(y3, go, ro)
            + 0.20 * _stone_geary(y4, go, ro)
            + 0.10 * _stone_geary(y5, go, ro))

    mask = feasibility_mask[obs_cz_idx]
    reward_c = _center_reward(reward, mask)
    logits = jnp.where(mask, beta * reward_c, -1e10)

    with plate("observations", N_obs):
        sample("obs_action", dist.Categorical(logits=logits), obs=obs_action)


# =====================================================================
# R3: Fixed ρ=1.5, hierarchical γ
# =====================================================================

def birl_r3(
    obs_action, obs_hh_idx, hh_country_idx, obs_cz_idx,
    feasibility_mask,
    cf_q10, cf_q50, cf_q90, cf_sigma,
    cf_log_q10, cf_log_q50, cf_log_q90, cf_sigma_log,
    N_hh, N_country,
):
    N_obs, N_actions = cf_q50.shape

    # ── Learnable β ──
    log_beta = sample("log_beta", dist.Normal(1.0, 0.5))
    beta = deterministic("beta", jnp.exp(jnp.clip(log_beta, -2.0, 4.0)))

    # ── Fixed ρ = 1.5 for all households ──
    rho_i = deterministic("rho_i", jnp.full(N_hh, 1.5))

    # ── Global hyperpriors (γ only) ──
    mu_gamma_G = sample("mu_gamma_G", dist.Normal(0.0, 1.0))
    sigma_gamma_G = sample("sigma_gamma_G", dist.HalfNormal(0.5))
    sigma_gamma_W = sample("sigma_gamma_W", dist.HalfNormal(0.3))

    # ── Country level (non-centered) ──
    with plate("countries", N_country):
        gamma_c_raw = sample("gamma_c_raw", dist.Normal(0, 1))

    gamma_c = deterministic("gamma_c_unbounded",
                            mu_gamma_G + sigma_gamma_G * gamma_c_raw)

    # ── Household level ──
    with plate("households", N_hh):
        gamma_offset = sample("gamma_offset", dist.Normal(0, 1))

    gamma_i_ub = gamma_c[hh_country_idx] + sigma_gamma_W * gamma_offset
    gamma_i = deterministic("gamma_i",
                            GAMMA_LO + (GAMMA_HI - GAMMA_LO) * jax.nn.sigmoid(gamma_i_ub))

    # ── Observation level ──
    ro = rho_i[obs_hh_idx][:, None]
    go = gamma_i[obs_hh_idx][:, None]

    q10 = jnp.exp(jnp.clip(cf_log_q10, -20.0, 20.0))
    q50 = jnp.exp(jnp.clip(cf_log_q50, -20.0, 20.0))
    q90 = jnp.exp(jnp.clip(cf_log_q90, -20.0, 20.0))

    y1, y2, y3 = q10, 0.5 * (q10 + q50), q50
    y4, y5 = 0.5 * (q50 + q90), q90

    reward = (0.10 * _stone_geary(y1, go, ro)
            + 0.20 * _stone_geary(y2, go, ro)
            + 0.40 * _stone_geary(y3, go, ro)
            + 0.20 * _stone_geary(y4, go, ro)
            + 0.10 * _stone_geary(y5, go, ro))

    mask = feasibility_mask[obs_cz_idx]
    reward_c = _center_reward(reward, mask)
    logits = jnp.where(mask, beta * reward_c, -1e10)

    with plate("observations", N_obs):
        sample("obs_action", dist.Categorical(logits=logits), obs=obs_action)
