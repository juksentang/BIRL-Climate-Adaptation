"""
NumPyro models for BIRL v2 (Phase 1: country-level parameters only).

Utility: Stone-Geary CRRA on per-plot predicted income Y (USD) with a SMOOTH
subsistence floor (spec-critic P4).  With d = Y - gamma - eps and k = eps,

    surplus = eps + 0.5 * (d + sqrt(d^2 + k^2))

which is >= eps, C1, and equals max(Y - gamma, eps) away from the kink.
    gamma_c = s_c * m_c,  s_c in (0, s_max);   eps_c = EPS_FRAC * m_c
    m_c     = per-country median over observations of the CHOSEN action's q50.

Certainty equivalent (spec-critic P1): computed DIRECTLY as a weighted power
mean of the five surplus nodes in log space, never by forming EU and
inverting it.  With p = 1 - rho and weights w_k on the nodes
(q10, (q10+q50)/2, q50, (q50+q90)/2, q90):

    m = sum_k w_k log x_k,  c_k = log x_k - m
    log CE_surplus = m + log1p( sum_k w_k expm1(p c_k) ) / p             |p| >= P_TAYLOR
    log CE_surplus = m + (p/2) Var_w(log x) + (p^2/6) kappa3_w(log x)  |p| <  P_TAYLOR
    CE = gamma + exp(log CE_surplus)

The direct branch is algebraically (1/p) logsumexp_k(p log x_k + log w_k)
(spec), written in the centred, cancellation-free form so that it stays
float32-accurate down to |p| = P_TAYLOR = 1e-3 (the sum inside log1p is
O(p^2 Var), so no digits are lost when dividing by p).  The Taylor branch is
the cumulant expansion of the power mean around p = 0 (the third-cumulant term
is an addition to the spec's 2nd-order form; it costs nothing) and is now only
the p -> 0 limit: at |p| = 1e-3 the two branches differ by ~1e-8 relative, so
the log-likelihood is continuous in rho_c to ~1e-3 nat (tests T2).

Reward and choice:
    reward_i(a) = CE_i(a) / m_c[country(i)],  mean-centred over feasible actions
    logits_i(a) = beta_c[country(i)] * reward_c_i(a);  -1e10 where infeasible
    obs_action ~ Categorical(logits)

ONE function, compute_logits(rho_c, s_c, beta_c, data), is used by the
numpyro models, by simulate_actions and by log_likelihood.

Parameters (6 countries), CENTRED by default (spec-critic P8), non-centred
with noncentered=True, independent wide priors with flat_priors=True (P7):
    mu_rho ~ N(0,1);  sigma_rho ~ HalfN(1);  rho_lat_c ~ N(mu_rho, sigma_rho)
        rho_c  = RHO_LO + (RHO_HI - RHO_LO) * sigmoid(rho_lat_c)
    mu_s   ~ N(-1,1); sigma_s   ~ HalfN(1);  s_lat_c   ~ N(mu_s, sigma_s)
        s_c    = s_max * sigmoid(s_lat_c)
    mu_lb  ~ N(1,1);  sigma_lb  ~ HalfN(1);  lb_c      ~ N(mu_lb, sigma_lb)
        beta_c = exp(clip(lb_c, -4, 6))
    flat_priors: rho_lat_c ~ N(0,2), s_lat_c ~ N(-1,2), lb_c ~ N(1,2), no hyperparameters.
Deterministic sites: rho_c, s_c, gamma_c (USD), beta_c and (non-centred)
rho_lat_c, s_lat_c, lb_c, so every run exposes the same site names.
v2_country_gfix: s_c fixed at s_fixed (default 0.3); only rho and beta sampled.
"""

import numpy as np
import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from numpyro import plate, sample, deterministic

from cropchoice.config import (S_MAX, EPS_FRAC, RHO_LO, RHO_HI, LOG_BETA_LO, LOG_BETA_HI,
                        P_TAYLOR, QUAD_W, LOG_CLIP)

INFEASIBLE_LOGIT = -1e10
T_CLIP = 60.0     # clip on p * c_k inside the direct power-mean branch (overflow guard only)

# keys of the data dict that compute_logits reads (loader dict or model kwargs).
# The loader / model_kwargs also provide the parameter-independent arrays
# q10, q50, q90 (= exp(clip(cf_log_q*)), float32) and mask_obs
# (= feasibility_mask[obs_cz_idx]); when present they are used directly so
# that no exp / clip / gather is redone inside every gradient evaluation.
# Any dict with only the log arrays + mask (e.g. the toy fixture) still works.
LOGIT_KEYS = ("obs_country_idx", "obs_cz_idx", "feasibility_mask",
              "cf_log_q10", "cf_log_q50", "cf_log_q90", "m_c")
PRECOMPUTED_KEYS = ("q10", "q50", "q90", "mask_obs")


# =====================================================================
# Smooth floor, nodes, power mean, certainty equivalent
# =====================================================================

def smooth_surplus(Y, gamma, eps):
    """Smooth version of max(Y - gamma, eps):  eps + 0.5 * (d + sqrt(d^2 + k^2)),
    d = Y - gamma - eps, k = eps.  Always >= eps (up to float rounding), C1.

    For d < 0 the algebraically identical form 0.5 k^2 / (sqrt(d^2 + k^2) - d)
    is used so that the tiny positive remainder is not lost to cancellation
    in float32.  Both branches of the jnp.where are finite everywhere, so
    gradients are finite.
    """
    d = Y - gamma - eps
    k = eps
    r = jnp.sqrt(d * d + k * k)
    neg = d < 0
    denom = jnp.where(neg, r - d, 1.0)
    return eps + jnp.where(neg, 0.5 * k * k / denom, 0.5 * (d + r))


def five_points(q10, q50, q90):
    """Quadrature nodes: q10, (q10+q50)/2, q50, (q50+q90)/2, q90."""
    return (q10, 0.5 * (q10 + q50), q50, 0.5 * (q50 + q90), q90)


def log_power_mean(log_x, p, w=QUAD_W):
    """log of the weighted power mean  ( sum_k w_k x_k^p )^(1/p)  from log x_k.

    log_x: list/tuple of K arrays (same shape, e.g. (N, A)); p broadcastable
    to that shape (e.g. (N, 1)); w: K positive weights summing to 1.

    Both branches share m = sum_k w_k log x_k and c_k = log x_k - m.
    |p| >= P_TAYLOR:  m + log1p(sum_k w_k expm1(p c_k)) / p
                      exact for every p != 0 (= (1/p) logsumexp(p log x_k + log w_k))
                      and cancellation-free: the sum is O(p^2 Var_w), so it is
                      accurate to float32 eps in absolute terms and the division
                      by p loses nothing.  sum_k w_k c_k = 0 guarantees at least
                      one p c_k >= 0, so the sum is >= min_k w_k - 1 > -1 and
                      log1p is finite; p c_k is clipped at +-T_CLIP (exp(60) is
                      far below float32 overflow) as a guard that is unreachable
                      for real income spreads (|p c_k| <= 5 * log(1e4/eps) ~ 50).
    |p| <  P_TAYLOR:  cumulant expansion m + p/2 var + p^2/6 kappa3 (exact at
                      p = 0: weighted geometric mean).  p_safe = 1 in this region
                      keeps the unused direct branch finite (JAX evaluates both).
    """
    near = jnp.abs(p) < P_TAYLOR
    p_safe = jnp.where(near, 1.0, p)

    m = w[0] * log_x[0]
    for wk, lx in zip(w[1:], log_x[1:]):
        m = m + wk * lx
    cs = [lx - m for lx in log_x]

    # direct branch (centred, exact)
    s = w[0] * jnp.expm1(jnp.clip(p_safe * cs[0], -T_CLIP, T_CLIP))
    for wk, c in zip(w[1:], cs[1:]):
        s = s + wk * jnp.expm1(jnp.clip(p_safe * c, -T_CLIP, T_CLIP))
    direct = m + jnp.log1p(s) / p_safe

    # Taylor branch (p -> 0 limit)
    var = w[0] * cs[0] * cs[0]
    k3 = w[0] * cs[0] * cs[0] * cs[0]
    for wk, c in zip(w[1:], cs[1:]):
        var = var + wk * c * c
        k3 = k3 + wk * c * c * c
    taylor = m + 0.5 * p * var + (p * p / 6.0) * k3
    return jnp.where(near, taylor, direct)


def ce_from_nodes(nodes, gamma, rho, eps):
    """CE = gamma + power-mean of the smooth surpluses of the K income nodes."""
    log_x = [jnp.log(smooth_surplus(y, gamma, eps)) for y in nodes]
    p = 1.0 - rho
    return gamma + jnp.exp(log_power_mean(log_x, p))


def certainty_equivalent(q10, q50, q90, gamma, rho, eps):
    """Certainty equivalent (USD) of the 5-node income distribution.

    q10, q50, q90: income quantiles (USD, same shape, e.g. (N, A));
    gamma, rho, eps broadcastable to them (e.g. (N, 1) or scalars).
    Identity: q10 = q50 = q90 = Y0  =>  CE = gamma + smooth_surplus(Y0), which
    equals Y0 away from the floor.
    """
    return ce_from_nodes(five_points(q10, q50, q90), gamma, rho, eps)


# =====================================================================
# Reward, logits, likelihood
# =====================================================================

def center_reward(reward, mask):
    """Subtract the per-observation mean over feasible actions (06's _center_reward)."""
    n = mask.sum(axis=-1, keepdims=True).clip(min=1)
    mean = jnp.where(mask, reward, 0.0).sum(axis=-1, keepdims=True) / n
    return reward - mean


def q_from_log(cf_log_q):
    """exp of the loader's log-q arrays with the +-LOG_CLIP clip used in 06."""
    return jnp.exp(jnp.clip(cf_log_q, -LOG_CLIP, LOG_CLIP))


def q_arrays(data):
    """(q10, q50, q90) in USD: the precomputed arrays when the dict has them,
    else exp(clip(log q)) (identical numbers, recomputed per call)."""
    if all(k in data for k in ("q10", "q50", "q90")):
        return data["q10"], data["q50"], data["q90"]
    return (q_from_log(data["cf_log_q10"]), q_from_log(data["cf_log_q50"]),
            q_from_log(data["cf_log_q90"]))


def obs_mask(data):
    """(N_obs, N_actions) feasibility of every action for every observation."""
    if "mask_obs" in data:
        return data["mask_obs"]
    return data["feasibility_mask"][data["obs_cz_idx"]]


def add_precomputed(data):
    """Return a copy of `data` with q10/q50/q90 (float32 USD) and mask_obs added
    (parameter-independent work hoisted out of the likelihood)."""
    out = dict(data)
    if not all(k in out for k in ("q10", "q50", "q90")):
        out["q10"], out["q50"], out["q90"] = q_arrays(data)
    if "mask_obs" not in out:
        out["mask_obs"] = obs_mask(data)
    return out


def per_obs_params(rho_c, s_c, beta_c, m_c, obs_country_idx):
    """Broadcast country parameters to (N_obs, 1): rho, gamma, eps, m, beta."""
    gamma_c = s_c * m_c
    eps_c = EPS_FRAC * m_c
    # One-hot matmul instead of x[obs_country_idx]: the transpose of a gather is
    # a scatter-add of N_obs values into N_country slots, which XLA runs as
    # contended atomics on GPU (measured 2.5 s per gradient vs 1.6 ms forward).
    # The transpose of a matmul is a GEMM.  HIGHEST precision so the selection
    # is exact (GPU float32 matmuls default to TF32, ~1e-3 relative error).
    n_c = rho_c.shape[0]
    P = jnp.stack([rho_c, gamma_c, eps_c, m_c, beta_c], axis=1)            # (C, 5), keeps
    oh = jax.nn.one_hot(obs_country_idx, n_c, dtype=P.dtype)               # the input dtype
    O = jnp.dot(oh, P, precision=jax.lax.Precision.HIGHEST)                # (N, 5)
    return O[:, 0:1], O[:, 1:2], O[:, 2:3], O[:, 3:4], O[:, 4:5]


def compute_ce(rho_c, s_c, data):
    """CE (USD), shape (N_obs, N_actions), for country parameters rho_c, s_c."""
    m_c = data["m_c"]
    ro, go, eo, _, _ = per_obs_params(rho_c, s_c, jnp.zeros_like(rho_c), m_c,
                                      data["obs_country_idx"])
    return certainty_equivalent(*q_arrays(data), go, ro, eo)


def compute_logits(rho_c, s_c, beta_c, data):
    """Choice logits (N_obs, N_actions) for country-level parameters.

    data: the loader dict, the model kwargs dict, or any dict with LOGIT_KEYS.
    Used unchanged by the numpyro models, simulate_actions and log_likelihood.
    """
    m_c = data["m_c"]
    ro, go, eo, mo, bo = per_obs_params(rho_c, s_c, beta_c, m_c, data["obs_country_idx"])
    ce = certainty_equivalent(*q_arrays(data), go, ro, eo)
    reward = ce / mo
    mask = obs_mask(data)
    reward_c = center_reward(reward, mask)
    return jnp.where(mask, bo * reward_c, INFEASIBLE_LOGIT)


def _logit_data(data):
    return {k: data[k] for k in LOGIT_KEYS + PRECOMPUTED_KEYS if k in data}


@jax.jit
def _loglik_jit(rho_c, s_c, beta_c, obs_action, ldata):
    logits = compute_logits(rho_c, s_c, beta_c, ldata)
    logp = jax.nn.log_softmax(logits, axis=-1)
    return jnp.sum(jnp.take_along_axis(logp, obs_action[:, None], axis=-1))


def log_likelihood(rho_c, s_c, beta_c, data, obs_action=None):
    """Summed categorical log-likelihood of obs_action (default data['obs_action'])."""
    obs = data["obs_action"] if obs_action is None else obs_action
    return _loglik_jit(jnp.asarray(rho_c, jnp.float32), jnp.asarray(s_c, jnp.float32),
                       jnp.asarray(beta_c, jnp.float32), jnp.asarray(obs, jnp.int32),
                       _logit_data(data))


# =====================================================================
# Parameter transforms (shared by the models and by numpy post-processing)
# =====================================================================

def country_params_from_latents(rho_lat, s_lat, lb, m_c, s_max=S_MAX):
    """(rho_c, s_c, gamma_c, beta_c) from the unconstrained country latents (JAX)."""
    rho_c = RHO_LO + (RHO_HI - RHO_LO) * jax.nn.sigmoid(rho_lat)
    s_c = s_max * jax.nn.sigmoid(s_lat)
    beta_c = jnp.exp(jnp.clip(lb, LOG_BETA_LO, LOG_BETA_HI))
    return rho_c, s_c, s_c * m_c, beta_c


def loglik_from_means(mu_rho, mu_s, mu_lb, data, s_max=S_MAX):
    """Summed log-likelihood when every country sits at the latent means
    (rho_lat_c = mu_rho, s_lat_c = mu_s, lb_c = mu_lb).  Used by test T4."""
    n = data["m_c"].shape[0]
    rho_c, s_c, _, beta_c = country_params_from_latents(
        jnp.full((n,), mu_rho), jnp.full((n,), mu_s), jnp.full((n,), mu_lb), data["m_c"], s_max)
    return log_likelihood(rho_c, s_c, beta_c, data)


def _sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-x))


def latent_for_rho(rho):
    """Inverse of rho_c = RHO_LO + (RHO_HI - RHO_LO) sigmoid(lat) (numpy, float64)."""
    u = (np.asarray(rho, np.float64) - RHO_LO) / (RHO_HI - RHO_LO)
    return np.log(u) - np.log1p(-u)


def latent_for_s(s, s_max=S_MAX):
    u = np.asarray(s, np.float64) / s_max
    return np.log(u) - np.log1p(-u)


def derive_country_params(latents, m_c, s_max=S_MAX, s_fixed=None):
    """Map latent site values (numpy dict; arrays may carry a leading sample dim)
    to rho_c, s_c, gamma_c, beta_c.  Accepts centred sites (rho_lat_c, s_lat_c,
    lb_c), non-centred sites (mu_* + sigma_* * *_raw) or a mix; used for SVI
    guide medians / draws, which do not carry the deterministic sites."""
    def lat(name, mu, sigma, raw):
        if name in latents:
            return np.asarray(latents[name], np.float64)
        return (np.asarray(latents[mu], np.float64)[..., None]
                + np.asarray(latents[sigma], np.float64)[..., None]
                * np.asarray(latents[raw], np.float64))
    m_c = np.asarray(m_c, np.float64)
    rho_lat = lat("rho_lat_c", "mu_rho", "sigma_rho", "rho_raw")
    lb = lat("lb_c", "mu_lb", "sigma_lb", "lb_raw")
    rho_c = RHO_LO + (RHO_HI - RHO_LO) * _sigmoid_np(rho_lat)
    if s_fixed is None:
        s_c = s_max * _sigmoid_np(lat("s_lat_c", "mu_s", "sigma_s", "s_raw"))
    else:
        s_c = np.full_like(rho_c, float(s_fixed))
    beta_c = np.exp(np.clip(lb, LOG_BETA_LO, LOG_BETA_HI))
    return {"rho_c": rho_c, "s_c": s_c, "gamma_c": s_c * m_c, "beta_c": beta_c}


# =====================================================================
# Models
# =====================================================================

def _country_params(N_country, m_c, s_max, s_fixed, flat_priors, noncentered):
    """Sample the country-level parameters. Returns rho_c, s_c, beta_c."""
    sample_s = s_fixed is None
    if flat_priors:
        with plate("countries", N_country):
            rho_lat = sample("rho_lat_c", dist.Normal(0.0, 2.0))
            lb = sample("lb_c", dist.Normal(1.0, 2.0))
            s_lat = sample("s_lat_c", dist.Normal(-1.0, 2.0)) if sample_s else None
    else:
        mu_rho = sample("mu_rho", dist.Normal(0.0, 1.0))
        sigma_rho = sample("sigma_rho", dist.HalfNormal(1.0))
        mu_lb = sample("mu_lb", dist.Normal(1.0, 1.0))
        sigma_lb = sample("sigma_lb", dist.HalfNormal(1.0))
        if sample_s:
            mu_s = sample("mu_s", dist.Normal(-1.0, 1.0))
            sigma_s = sample("sigma_s", dist.HalfNormal(1.0))
        with plate("countries", N_country):
            if noncentered:
                rho_raw = sample("rho_raw", dist.Normal(0.0, 1.0))
                lb_raw = sample("lb_raw", dist.Normal(0.0, 1.0))
                s_raw = sample("s_raw", dist.Normal(0.0, 1.0)) if sample_s else None
            else:
                rho_lat = sample("rho_lat_c", dist.Normal(mu_rho, sigma_rho))
                lb = sample("lb_c", dist.Normal(mu_lb, sigma_lb))
                s_lat = sample("s_lat_c", dist.Normal(mu_s, sigma_s)) if sample_s else None
        if noncentered:
            rho_lat = deterministic("rho_lat_c", mu_rho + sigma_rho * rho_raw)
            lb = deterministic("lb_c", mu_lb + sigma_lb * lb_raw)
            s_lat = deterministic("s_lat_c", mu_s + sigma_s * s_raw) if sample_s else None

    if not sample_s:
        s_fixed = float(s_fixed)
        if 0.0 < s_fixed < s_max:
            deterministic("s_lat_c", jnp.full((N_country,), float(latent_for_s(s_fixed, s_max))))
        s_lat = jnp.zeros((N_country,))   # placeholder, overwritten below

    rho_c, s_c, gamma_c, beta_c = country_params_from_latents(rho_lat, s_lat, lb, m_c, s_max)
    if not sample_s:
        s_c = jnp.full((N_country,), s_fixed)
        gamma_c = s_c * m_c
    rho_c = deterministic("rho_c", rho_c)
    s_c = deterministic("s_c", s_c)
    deterministic("gamma_c", gamma_c)
    beta_c = deterministic("beta_c", beta_c)
    return rho_c, s_c, beta_c


def v2_country(data, s_max=S_MAX, flat_priors=False, noncentered=False, s_fixed=None,
               asc=None, beta_fixed=None):
    """Main variant: rho_c, s_c (gamma_c = s_c * m_c), beta_c sampled per country.

    data: dict from data_loader.model_kwargs (obs_action, obs_country_idx,
    obs_cz_idx, feasibility_mask, cf_log_q10/50/90, m_c, N_country).
    asc: None | "global" | "country" -- alternative-specific constants added to
         the logits (action 0 is the reference, alpha_0 = 0); "country" gives
         one set per country.  Experimental (2026-09-20 reward-spec tests).
    beta_fixed: if given, beta_c is overridden by this constant (beta_fixed=0
         with asc gives the pure choice-frequency null model).
    """
    rho_c, s_c, beta_c = _country_params(data["N_country"], data["m_c"], s_max, s_fixed,
                                         flat_priors, noncentered)
    if beta_fixed is not None:
        beta_c = jnp.full_like(beta_c, float(beta_fixed))
    logits = compute_logits(rho_c, s_c, beta_c, data)
    if asc is not None:
        n_a = logits.shape[1]
        if asc == "global":
            a = sample("asc", dist.Normal(0.0, 3.0).expand([n_a - 1]).to_event(1))
            alpha = jnp.concatenate([jnp.zeros((1,), a.dtype), a])[None, :]          # (1, A)
        elif asc == "country":
            n_c = data["N_country"]
            a = sample("asc_c", dist.Normal(0.0, 3.0).expand([n_c, n_a - 1]).to_event(2))
            alpha_c = jnp.concatenate([jnp.zeros((n_c, 1), a.dtype), a], axis=1)     # (C, A)
            oh = jax.nn.one_hot(data["obs_country_idx"], n_c, dtype=alpha_c.dtype)
            alpha = jnp.dot(oh, alpha_c, precision=jax.lax.Precision.HIGHEST)       # (N, A)
        else:
            raise ValueError(f"asc must be None, 'global' or 'country', got {asc!r}")
        logits = logits + alpha        # infeasible stays ~ -1e10
    with plate("observations", logits.shape[0]):
        sample("obs_action", dist.Categorical(logits=logits), obs=data["obs_action"])


def v2_country_gfix(data, s_fixed=0.3, s_max=S_MAX, flat_priors=False, noncentered=False):
    """Variant with s_c fixed at s_fixed; only rho_c and beta_c sampled."""
    v2_country(data, s_max=s_max, flat_priors=flat_priors, noncentered=noncentered,
               s_fixed=s_fixed)


MODELS = {"v2_country": v2_country, "v2_country_gfix": v2_country_gfix}

# Site groups used by the diagnostics (presence in the posterior decides)
HYPER_SITES = ["mu_rho", "sigma_rho", "mu_s", "sigma_s", "mu_lb", "sigma_lb"]
LATENT_SITES = ["rho_lat_c", "s_lat_c", "lb_c", "rho_raw", "s_raw", "lb_raw"]
COUNTRY_SITES = ["rho_c", "s_c", "s_lat_c", "gamma_c", "beta_c", "rho_lat_c", "lb_c"]


# =====================================================================
# Simulation
# =====================================================================

@jax.jit
def _simulate_jit(rng_key, rho_c, s_c, beta_c, ldata):
    logits = compute_logits(rho_c, s_c, beta_c, ldata)
    return jax.random.categorical(rng_key, logits, axis=-1).astype(jnp.int32)


def simulate_actions(rng_key, rho_c, s_c, beta_c, data):
    """One action per observation from Categorical(compute_logits(rho_c, s_c, beta_c, data)).

    Infeasible actions have logit -1e10 and are never drawn. Returns jnp int32 (N_obs,).
    """
    return _simulate_jit(rng_key, jnp.asarray(rho_c, jnp.float32), jnp.asarray(s_c, jnp.float32),
                         jnp.asarray(beta_c, jnp.float32), _logit_data(data))
