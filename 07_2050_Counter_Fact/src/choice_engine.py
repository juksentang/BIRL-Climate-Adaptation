"""Choice-model counterfactual engine (Step 07 stage 3, rewritten on the Step 08
semi-parametric choice model).  See ../CHOICE_CF_SPEC.md.

Pure functions, JAX for the per-draw batch, numpy for bookkeeping.

  mu, sig      = mu_sigma(q10, q50, q90)                  log-income moments
  q'           = apply_policy(q10, q50, q90, policy, params, m_obs, ctx)
  out          = scenario_metrics(q10', q50', q90', draws, mask, ci, C, m_c, ...)

Choice model (exactly as estimated in 08_BIRL_v2/slurm/run_semipar.py):
  V_ia = a_c mu_ia + b_c sig_ia + c_c sig_ia^2 ;  U_ia = V_ia + ASC_{c,a}
  P(a|i) = softmax over feasible actions of U_ia   (V is NOT centred: the
  logsum needs the absolute level; centring would cancel in P anyway).
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.stats import norm as jnorm

Z = 1.2816                       # norm.ppf(0.9)
W5 = np.array([0.1, 0.2, 0.4, 0.2, 0.1], np.float32)
INFEASIBLE = -1e10
HI = jax.lax.Precision.HIGHEST

POLICIES = ("none", "transfer", "contraction", "safety_net", "insurance", "both")
DEFAULT_PARAMS = dict(t=0.10, lam=0.5, f=0.5, f_both=0.3, trigger=0.5, basis=0.3, loading=0.2, theta=0.3)
# "both" = a lower floor (f_both) plus insurance triggered at `trigger`: with f == trigger the
# insurance would never pay after the floor, so the combined policy uses its own floor share.


# ─────────────────────────────────────────────────────────────── posterior ──
def load_posterior_thinned(path, K=200):
    """Flatten chains, keep every (n_total // K)-th draw.  Returns dict of numpy:
    a, b, c: (K, C);  asc: (K, C, A) with the reference action's 0 prepended."""
    d = np.load(path)
    a = d["a_c"].reshape(-1, d["a_c"].shape[-1])
    b = d["b_c"].reshape(-1, d["b_c"].shape[-1])
    c = d["c_c"].reshape(-1, d["c_c"].shape[-1])
    asc = d["asc_c"].reshape(-1, *d["asc_c"].shape[-2:])
    n = a.shape[0]
    step = max(n // K, 1)
    idx = np.arange(0, n, step)[:K]
    asc_full = np.concatenate([np.zeros((len(idx), asc.shape[1], 1), asc.dtype), asc[idx]], axis=2)
    return {"a": a[idx].astype(np.float32), "b": b[idx].astype(np.float32),
            "c": c[idx].astype(np.float32), "asc": asc_full.astype(np.float32),
            "n_total": int(n), "idx": idx}


# ─────────────────────────────────────────────────────────────── moments ──
def mu_sigma(q10, q50, q90):
    """(mu, sigma) of log income from the three quantiles, as in the estimation."""
    mu = jnp.log1p(jnp.maximum(q50, 0.0))
    sig = (jnp.log1p(jnp.maximum(q90, 0.0)) - jnp.log1p(jnp.maximum(q10, 0.0))) / (2 * Z)
    return mu, jnp.clip(sig, 0.01, 5.0)


def five_nodes(q10, q50, q90):
    """(5, ...) income nodes of the quadrature rule used throughout the pipeline."""
    return jnp.stack([q10, 0.5 * (q10 + q50), q50, 0.5 * (q50 + q90), q90], axis=0)


def e5(nodes):
    return jnp.tensordot(jnp.asarray(W5), nodes, axes=(0, 0))


# ─────────────────────────────────────────────────────────────── policies ──
def _country_mean(x_obs, ci, C):
    """Mean of a per-observation vector within each country -> (C,)."""
    oh = jax.nn.one_hot(ci, C, dtype=jnp.float32)
    return jnp.dot(oh.T, x_obs, precision=HI) / jnp.maximum(oh.sum(0), 1.0)


def cost_context(q10, q50, q90, obs_action, ci, C):
    """Baseline chosen-action nodes (5, N) and country sizes, used for the
    transfer-type costs (floor cost, expected payout, premium)."""
    idx = jnp.arange(q50.shape[0])
    nodes = five_nodes(q10[idx, obs_action], q50[idx, obs_action], q90[idx, obs_action])   # (5, N)
    return {"chosen_nodes": nodes, "ci": ci, "C": C}


def apply_policy(q10, q50, q90, policy, params, m_obs, ctx):
    """Monotone income transforms applied to the quantiles pointwise.
    Returns (q10', q50', q90', info) with info holding per-country scalars:
      cost (public transfer cost per plot-season, USD), payout, premium, F, T."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    ci, C = ctx["ci"], ctx["C"]
    m_c = _country_mean(m_obs, ci, C)                       # (C,) = m_c itself
    zeros = jnp.zeros((C,), jnp.float32)
    info = {"cost": zeros, "payout": zeros, "premium": zeros, "F": zeros, "T": zeros}
    if policy == "none":
        return q10, q50, q90, info
    if policy == "transfer":
        d = p["t"] * m_obs[:, None]
        info["cost"] = p["t"] * m_c
        return q10 + d, q50 + d, q90 + d, info
    if policy == "contraction":
        lam = p["lam"]
        return q50 + lam * (q10 - q50), q50, q50 + lam * (q90 - q50), info

    chosen = ctx["chosen_nodes"]                            # (5, N) baseline chosen action
    if policy in ("safety_net", "both"):
        f_share = p["f"] if policy == "safety_net" else p["f_both"]
        F_obs = f_share * m_obs
        floor_cost_obs = e5(jnp.maximum(F_obs[None, :] - chosen, 0.0))
        info["F"] = f_share * m_c
        info["cost"] = _country_mean(floor_cost_obs, ci, C)
        q10, q50, q90 = (jnp.maximum(q, F_obs[:, None]) for q in (q10, q50, q90))
        if policy == "safety_net":
            return q10, q50, q90, info
        chosen = jnp.maximum(chosen, F_obs[None, :])        # insurance payouts are computed on the floored distribution
    # insurance (alone, or after the floor)
    T_obs = p["trigger"] * m_obs
    payout_obs = e5((1.0 - p["basis"]) * jnp.maximum(T_obs[None, :] - chosen, 0.0))   # E[payout] on baseline chosen action
    payout_c = _country_mean(payout_obs, ci, C)
    premium_c = (1.0 + p["loading"]) * payout_c
    prem_obs = jnp.dot(jax.nn.one_hot(ci, C, dtype=jnp.float32), premium_c, precision=HI)
    info["T"] = p["trigger"] * m_c
    info["payout"] = payout_c
    info["premium"] = premium_c
    info["cost"] = info["cost"] + p["loading"] * payout_c    # public cost if the loading is subsidised

    def g(q):
        return q + (1.0 - p["basis"]) * jnp.maximum(T_obs[:, None] - q, 0.0) - prem_obs[:, None]
    return g(q10), g(q50), g(q90), info


# ─────────────────────────────────────────────────────────────── metrics ──
def _softmax_feasible(U, mask):
    U = jnp.where(mask, U, INFEASIBLE)
    m = jnp.max(U, axis=-1, keepdims=True)
    e = jnp.where(mask, jnp.exp(U - m), 0.0)
    s = jnp.sum(e, axis=-1, keepdims=True)
    return e / s, (jnp.log(s) + m)[..., 0]                 # P (..., N, A), logsum (..., N)


def _batch_metrics(q10, q50, q90, q10c, q50c, q90c, a, b, c, asc, mask, oh, m_obs, theta_obs,
                   F_obs, T_obs, basis, shift_obs, prem_obs, lam, rhos, policy_kind):
    """One batch of draws.  q*  : post-policy quantiles (N, A)
                            q*c : climate (pre-policy) quantiles, for the behavioural cost
                            a,b,c: (Kb, C); asc: (Kb, C, A); oh: (N, C) one-hot.
    Returns per-country arrays (Kb, C, ...)."""
    mu, sig = mu_sigma(q10, q50, q90)
    nodes = five_nodes(q10, q50, q90)                                       # (5, N, A)
    ey = e5(nodes)                                                          # (N, A)
    # P(y' < theta) exactly: y' = g(y) is monotone, so P(g(y) < theta) = P(y < g^{-1}(theta)) evaluated on the
    # PRE-policy lognormal (mu_c, sig_c).  Refitting a lognormal to the transformed quantiles (mu, sig) would
    # put mass below a hard floor and understate what floors and indemnities do to the downside.
    mu_c, sig_c = mu_sigma(q10c, q50c, q90c)
    th = theta_obs[:, None] * jnp.ones_like(q50c)                                      # (N, A)
    if policy_kind == "transfer":
        ystar = th - shift_obs[:, None]
    elif policy_kind == "contraction":
        ystar = q50c + (th - q50c) / lam
    elif policy_kind == "floor":
        ystar = jnp.where(th > F_obs[:, None], th, -1.0)
    elif policy_kind in ("insurance", "both"):
        T = T_obs[:, None]; pi = prem_obs[:, None]
        above = th >= T - pi                                                            # g(T) = T - pi
        y_lo = jnp.where(basis > 0, (th + pi - (1.0 - basis) * T) / jnp.maximum(basis, 1e-6), -1.0)
        ystar = jnp.where(above, th + pi, y_lo)
        if policy_kind == "both":                                                       # floor applied before the indemnity
            ystar = jnp.where(ystar > F_obs[:, None], ystar, -1.0)
    else:
        ystar = th
    pb = jnp.where(ystar > 0, jnorm.cdf((jnp.log1p(jnp.maximum(ystar, 0.0)) - mu_c) / sig_c), 0.0)   # (N, A)
    es = e5(jnp.maximum(theta_obs[None, :, None] - nodes, 0.0))                          # expected shortfall E[(theta-y')+], exact on transformed nodes (N, A)
    ce = {r: jnp.exp(mu + 0.5 * (1.0 - r) * sig * sig) - 1.0 for r in rhos}
    nodes_c = five_nodes(q10c, q50c, q90c)
    if policy_kind == "floor":
        cb = e5(jnp.maximum(F_obs[None, :, None] - nodes_c, 0.0))            # (N, A)
    elif policy_kind == "insurance":
        cb = e5((1.0 - basis) * jnp.maximum(T_obs[None, :, None] - nodes_c, 0.0))
    elif policy_kind == "both":
        cb = (e5(jnp.maximum(F_obs[None, :, None] - nodes_c, 0.0))
              + e5((1.0 - basis) * jnp.maximum(T_obs[None, :, None] - jnp.maximum(nodes_c, F_obs[None, :, None]), 0.0)))
    else:
        cb = jnp.zeros_like(ey)
    n_c = jnp.maximum(oh.sum(0), 1.0)                                       # (C,)

    def per_draw(a_k, b_k, c_k, asc_k):
        ao = jnp.dot(oh, a_k, precision=HI)[:, None]
        bo = jnp.dot(oh, b_k, precision=HI)[:, None]
        co = jnp.dot(oh, c_k, precision=HI)[:, None]
        U = ao * mu + bo * sig + co * sig * sig + jnp.dot(oh, asc_k, precision=HI)   # (N, A)
        P, ls = _softmax_feasible(U, mask)
        cm = lambda v: jnp.dot(oh.T, v, precision=HI) / n_c                 # (N,) -> (C,)
        out = {
            "shares": jnp.dot(oh.T, P, precision=HI) / n_c[:, None],        # (C, A)
            "exp_income": cm(jnp.sum(P * ey, -1)),
            "exp_median": cm(jnp.sum(P * q50, -1)),
            "p_below_theta": cm(jnp.sum(P * pb, -1)),
            "exp_shortfall": cm(jnp.sum(P * es, -1)),
            "logsum": cm(ls),
            "cost_behav": cm(jnp.sum(P * cb, -1)),
        }
        for r in rhos:
            out[f"ce_rho{r}"] = cm(jnp.sum(P * ce[r], -1))
        return out

    return jax.vmap(per_draw)(a, b, c, asc)


_batch_metrics_jit = jax.jit(_batch_metrics, static_argnames=("rhos", "policy_kind"))


def scenario_metrics(q10, q50, q90, q10c, q50c, q90c, draws, mask, ci, C, m_obs, params,
                     policy, batch=10, rhos=(1.5, 2.5, 3.5), info=None):
    """Loop the draws in batches on the device.  Returns dict of numpy arrays:
    shares (K, C, A); exp_income, exp_median, p_below_theta, logsum, cost_behav,
    ce_rho* : (K, C)."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    kind = {"safety_net": "floor", "insurance": "insurance", "both": "both",
            "transfer": "transfer", "contraction": "contraction"}.get(policy, "none")
    oh = jax.nn.one_hot(jnp.asarray(ci), C, dtype=jnp.float32)
    theta_obs = p["theta"] * m_obs
    F_obs = (p["f_both"] if policy == "both" else p["f"]) * m_obs
    T_obs = p["trigger"] * m_obs
    shift_obs = (p["t"] if policy == "transfer" else 0.0) * m_obs
    prem_c = jnp.asarray(info["premium"], jnp.float32) if info is not None else jnp.zeros((C,), jnp.float32)
    prem_obs = jnp.dot(oh, prem_c, precision=HI)
    lam = float(p["lam"]) if policy == "contraction" else 1.0
    K = draws["a"].shape[0]
    outs = []
    for s in range(0, K, batch):
        sl = slice(s, s + batch)
        o = _batch_metrics_jit(q10, q50, q90, q10c, q50c, q90c,
                               jnp.asarray(draws["a"][sl]), jnp.asarray(draws["b"][sl]),
                               jnp.asarray(draws["c"][sl]), jnp.asarray(draws["asc"][sl]),
                               mask, oh, m_obs, theta_obs, F_obs, T_obs, float(p["basis"]),
                               shift_obs, prem_obs, lam, tuple(rhos), kind)
        outs.append({k: np.asarray(v) for k, v in o.items()})
    return {k: np.concatenate([o[k] for o in outs], axis=0) for k in outs[0]}


def observed_shares(obs_action, ci, C, A):
    """Observed action frequencies per country (C, A) for the PPC."""
    out = np.zeros((C, A), np.float64)
    for c in range(C):
        sel = np.asarray(ci) == c
        out[c] = np.bincount(np.asarray(obs_action)[sel], minlength=A) / max(sel.sum(), 1)
    return out
