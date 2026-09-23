"""Policies as monotone income transforms applied to the quantiles pointwise
(Step 07 stage 3; spec in 07_2050_Counter_Fact/CHOICE_CF_SPEC.md).

  none          y' = y
  transfer      y' = y + t m_c                       cost = t m_c
  contraction   q_k' = q50 + lam (q_k - q50)         (sigma-only probe, no cost)
  safety_net    y' = max(y, F),  F = f m_c           cost = E[(F - y)+] on the chosen action's baseline nodes
  insurance     y' = y + (1-basis)(T - y)+ - pi,  T = trigger m_c, pi = (1+loading) E[payout]
                                                     cost = loading * E[payout]  (subsidised loading)
  both          floor at f_both m_c, then insurance (payouts on the floored distribution)

All transforms are non-decreasing in y, so quantiles map through them
pointwise; the counterfactual recomputes (mu, sigma) from the transformed
quantiles for the choice model and uses the inverse transform of the threshold
for exact downside probabilities (see `cropchoice.counterfactual`).
"""
import jax
import jax.numpy as jnp

from cropchoice.broadcast import HI, one_hot, per_obs
from cropchoice.quantiles import five_nodes, e5

POLICIES = ("none", "transfer", "contraction", "safety_net", "insurance", "both")
DEFAULT_PARAMS = dict(t=0.10, lam=0.5, f=0.5, f_both=0.3, trigger=0.5, basis=0.3, loading=0.2, theta=0.3)
# "both" = a lower floor (f_both) plus insurance triggered at `trigger`: with f == trigger the
# insurance would never pay after the floor, so the combined policy uses its own floor share.
POLICY_KIND = {"safety_net": "floor", "insurance": "insurance", "both": "both",
               "transfer": "transfer", "contraction": "contraction"}


def _country_mean(x_obs, ci, C):
    """Mean of a per-observation vector within each country -> (C,)."""
    oh = one_hot(ci, C)
    return jnp.dot(oh.T, x_obs, precision=HI) / jnp.maximum(oh.sum(0), 1.0)


def cost_context(q10, q50, q90, obs_action, ci, C):
    """Baseline chosen-action nodes (5, N) and country index, used for the
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
    prem_obs = per_obs(one_hot(ci, C), premium_c)
    info["T"] = p["trigger"] * m_c
    info["payout"] = payout_c
    info["premium"] = premium_c
    info["cost"] = info["cost"] + p["loading"] * payout_c    # public cost if the loading is subsidised

    def g(q):
        return q + (1.0 - p["basis"]) * jnp.maximum(T_obs[:, None] - q, 0.0) - prem_obs[:, None]
    return g(q10), g(q50), g(q90), info


def inverse_threshold(theta, policy_kind, q50c, F_obs, T_obs, prem_obs, basis, shift_obs, lam):
    """y* such that P(g(y) < theta) = P(y < y*) for the transform g of `policy_kind`,
    evaluated on the PRE-policy distribution.  Arrays broadcast to (N, A); a
    negative y* means the event is impossible after the policy (probability 0).
    Refitting a lognormal to the transformed quantiles would put mass below a
    hard floor and understate what floors and indemnities do to the downside."""
    th = theta[:, None] * jnp.ones_like(q50c)
    if policy_kind == "transfer":
        return th - shift_obs[:, None]
    if policy_kind == "contraction":
        return q50c + (th - q50c) / lam
    if policy_kind == "floor":
        return jnp.where(th > F_obs[:, None], th, -1.0)
    if policy_kind in ("insurance", "both"):
        T = T_obs[:, None]; pi = prem_obs[:, None]
        above = th >= T - pi                                                            # g(T) = T - pi
        y_lo = jnp.where(basis > 0, (th + pi - (1.0 - basis) * T) / jnp.maximum(basis, 1e-6), -1.0)
        ystar = jnp.where(above, th + pi, y_lo)
        if policy_kind == "both":                                                       # floor applied before the indemnity
            ystar = jnp.where(ystar > F_obs[:, None], ystar, -1.0)
        return ystar
    return th
