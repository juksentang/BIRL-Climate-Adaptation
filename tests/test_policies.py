"""Policy layer of the counterfactual (cropchoice.policies / counterfactual): the
monotone transforms, the inverse-threshold downside probability, cost accounting."""
import numpy as np
import pytest
import jax.numpy as jnp
from jax.scipy.stats import norm as jnorm

from cropchoice.config import DATA_DIR  # noqa: F401  (device setup first)
from cropchoice.policies import POLICIES, DEFAULT_PARAMS, POLICY_KIND, cost_context, apply_policy, inverse_threshold
from cropchoice.counterfactual import scenario_metrics, toy_inputs
from cropchoice.quantiles import mu_sigma, five_nodes, e5, W5
from cropchoice.broadcast import one_hot, per_obs

pytestmark = pytest.mark.toy


@pytest.fixture(scope="module")
def inp():
    return toy_inputs(K=4, N=300, A=9, C=6, seed=3)


def _setup(inp, params, policy):
    countries, m_c, ci, mask, obs_action, draws = (inp[k] for k in ("countries", "m_c", "ci", "mask", "obs_action", "draws"))
    C = len(countries)
    m_obs = jnp.asarray(m_c[ci])
    q10, q50, q90 = (jnp.asarray(x) for x in inp["quant"]["baseline"])
    ctx = cost_context(q10, q50, q90, jnp.asarray(obs_action), jnp.asarray(ci), C)
    p10, p50, p90, info = apply_policy(q10, q50, q90, policy, params, m_obs, ctx)
    return (q10, q50, q90), (p10, p50, p90), info, m_obs, ctx


@pytest.mark.parametrize("policy", POLICIES)
def test_transforms_are_monotone_and_ordered(inp, policy):
    """q10' <= q50' <= q90' everywhere, and the transform is non-decreasing in y
    (applying it to the ordered quantiles keeps the order for every policy)."""
    (q10, q50, q90), (p10, p50, p90), info, _, _ = _setup(inp, DEFAULT_PARAMS, policy)
    assert bool(jnp.all(p10 <= p50 + 1e-6)) and bool(jnp.all(p50 <= p90 + 1e-6))
    assert np.all(np.isfinite(np.asarray(p50)))
    if policy in ("safety_net", "both"):
        F = (DEFAULT_PARAMS["f"] if policy == "safety_net" else DEFAULT_PARAMS["f_both"]) * np.asarray(_setup(inp, DEFAULT_PARAMS, policy)[3])
        assert bool(jnp.all(p10 >= jnp.asarray(F)[:, None] - 1e-6))            # floor binds
    if policy == "transfer":
        assert np.allclose(np.asarray(p50 - q50), DEFAULT_PARAMS["t"] * np.asarray(_setup(inp, DEFAULT_PARAMS, policy)[3])[:, None], atol=1e-4)
    if policy == "contraction":
        assert np.allclose(np.asarray(p50), np.asarray(q50))                     # median fixed
        assert bool(jnp.all(p90 - p10 <= q90 - q10 + 1e-6))                      # spread shrinks


def _exact_pb_by_integration(q10, q50, q90, g, theta, n=4001):
    """Numerical P(g(y) < theta) for lognormal y with (mu, sigma) from the quantiles:
    integrate the density on a fine grid of y (the 'truth' the inverse formula must match)."""
    mu, sig = (np.asarray(v, float) for v in mu_sigma(q10, q50, q90))
    z = np.linspace(-6, 6, n)[:, None, None]
    y = np.expm1(mu[None] + sig[None] * z)
    below = (g(y) < theta[None, :, None]).astype(float)
    w = np.exp(-0.5 * z ** 2); w /= w.sum(0, keepdims=True)
    return (w * below).sum(0)


@pytest.mark.parametrize("policy", ["transfer", "contraction", "safety_net", "insurance", "both"])
def test_inverse_threshold_matches_numerical_integration(inp, policy):
    """P(y' < theta) computed with the inverse-transformed threshold on the pre-policy
    lognormal must equal P(g(y) < theta) obtained by integrating the transform numerically."""
    p = dict(DEFAULT_PARAMS)
    (q10, q50, q90), _, info, m_obs, ctx = _setup(inp, p, policy)
    ci = np.asarray(ctx["ci"]); C = ctx["C"]
    m = np.asarray(m_obs); theta = p["theta"] * m
    F = (p["f_both"] if policy == "both" else p["f"]) * m; T = p["trigger"] * m
    prem = np.asarray(per_obs(one_hot(ci, C), jnp.asarray(info["premium"])))
    shift = (p["t"] if policy == "transfer" else 0.0) * m
    lam = p["lam"] if policy == "contraction" else 1.0
    kind = POLICY_KIND[policy]
    ystar = np.asarray(inverse_threshold(jnp.asarray(theta), kind, q50, jnp.asarray(F), jnp.asarray(T), jnp.asarray(prem), p["basis"], jnp.asarray(shift), lam))
    mu, sig = (np.asarray(v, float) for v in mu_sigma(q10, q50, q90))
    pb = np.where(ystar > 0, jnorm.cdf((np.log1p(np.maximum(ystar, 0)) - mu) / sig), 0.0)
    q50n = np.asarray(q50)

    def g(y):
        if policy == "transfer":
            return y + shift[:, None]
        if policy == "contraction":
            return q50n[None] + lam * (y - q50n[None])
        if policy == "safety_net":
            return np.maximum(y, F[None, :, None])
        yy = np.maximum(y, F[None, :, None]) if policy == "both" else y
        return yy + (1 - p["basis"]) * np.maximum(T[None, :, None] - yy, 0) - prem[None, :, None]
    ref = _exact_pb_by_integration(q10, q50, q90, g, theta)
    pb = np.asarray(pb)
    # Convention of the pipeline: an inverse threshold y* <= 0 counts as "impossible" (P = 0).  The
    # fitted distribution is lognormal in 1 + y and so carries a little mass on y in (-1, 0]
    # (the env model's exp(.) - 1 convention); that mass is ignored there.  Check the inverse
    # formula exactly where y* > 0, and bound the neglected mass elsewhere by P(y <= 0).
    pos = ystar > 0
    if pos.any():                                                       # floors above theta make every cell impossible
        assert np.abs(pb[pos] - ref[pos]).max() < 2e-3
    p_neg = jnorm.cdf((np.log1p(0.0) - mu) / sig)                     # P(y <= 0) under the fitted law
    assert np.all(ref[~pos] <= np.asarray(p_neg)[~pos] + 2e-3)


def test_insurance_zero_basis_boundary(inp):
    """basis = 0: full indemnity, so nothing ends below T - premium; the inverse threshold
    below that level is 'impossible' (negative) and P(y' < theta) is 0 for theta < T - pi."""
    p = dict(DEFAULT_PARAMS, basis=0.0, theta=0.3, trigger=0.6)
    (q10, q50, q90), (p10, p50, p90), info, m_obs, ctx = _setup(inp, p, "insurance")
    C = ctx["C"]; ci = np.asarray(ctx["ci"]); m = np.asarray(m_obs)
    prem = np.asarray(per_obs(one_hot(ci, C), jnp.asarray(info["premium"])))
    worst = p["trigger"] * m - prem
    assert np.all(np.asarray(p10) >= worst[:, None] - 1e-5)
    theta = p["theta"] * m
    ystar = np.asarray(inverse_threshold(jnp.asarray(theta), "insurance", q50, jnp.zeros_like(m_obs), jnp.asarray(p["trigger"] * m),
                                         jnp.asarray(prem), 0.0, jnp.zeros_like(m_obs), 1.0))
    low = theta < worst
    assert np.all(ystar[low] < 0) and np.all(ystar[~low] > 0)


def test_both_floor_precedes_indemnity(inp):
    """In 'both' the payout is computed on the floored distribution: with the floor at or
    above the trigger the insurance never pays and the premium is zero."""
    p = dict(DEFAULT_PARAMS, f_both=0.5, trigger=0.5)
    _, _, info, _, _ = _setup(inp, p, "both")
    assert float(jnp.max(info["payout"])) < 1e-6 and float(jnp.max(info["premium"])) < 1e-6
    p2 = dict(DEFAULT_PARAMS, f_both=0.3, trigger=0.5)
    _, _, info2, _, _ = _setup(inp, p2, "both")
    _, _, info_ins, _, _ = _setup(inp, p2, "insurance")
    assert bool(jnp.all(info2["payout"] <= info_ins["payout"] + 1e-6))          # floor reduces the payout


def test_cost_accounting(inp):
    """transfer: cost = t m_c; safety net: cost = country mean of E[(F - y)+] on the chosen
    action (transfer-type cost); insurance: cost = loading * E[payout], premium = (1+loading) E[payout];
    behaviour-adjusted cost of 'none'/'contraction' is 0."""
    p = dict(DEFAULT_PARAMS)
    countries, m_c, ci = inp["countries"], inp["m_c"], inp["ci"]
    _, _, info_t, _, _ = _setup(inp, p, "transfer")
    assert np.allclose(np.asarray(info_t["cost"]), p["t"] * m_c, rtol=1e-5)
    _, _, info_f, m_obs, ctx = _setup(inp, p, "safety_net")
    chosen = np.asarray(ctx["chosen_nodes"]); F = p["f"] * np.asarray(m_obs)
    floor_cost = (W5[:, None] * np.maximum(F[None] - chosen, 0)).sum(0)
    ref = np.array([floor_cost[ci == c].mean() for c in range(len(countries))])
    assert np.allclose(np.asarray(info_f["cost"]), ref, rtol=1e-4)
    _, _, info_i, _, _ = _setup(inp, p, "insurance")
    assert np.allclose(np.asarray(info_i["premium"]), (1 + p["loading"]) * np.asarray(info_i["payout"]), rtol=1e-5)
    assert np.allclose(np.asarray(info_i["cost"]), p["loading"] * np.asarray(info_i["payout"]), rtol=1e-5)


def test_scenario_metrics_shapes_and_shares(inp):
    countries, m_c, ci, mask, draws = (inp[k] for k in ("countries", "m_c", "ci", "mask", "draws"))
    C = len(countries); K = draws["a"].shape[0]
    q10, q50, q90 = (jnp.asarray(x) for x in inp["quant"]["baseline"])
    m_obs = jnp.asarray(m_c[ci]); ctx = cost_context(q10, q50, q90, jnp.asarray(inp["obs_action"]), jnp.asarray(ci), C)
    p10, p50, p90, info = apply_policy(q10, q50, q90, "insurance", DEFAULT_PARAMS, m_obs, ctx)
    out = scenario_metrics(p10, p50, p90, q10, q50, q90, draws, jnp.asarray(mask), ci, C, m_obs, DEFAULT_PARAMS, "insurance", batch=2, info=info)
    assert out["shares"].shape == (K, C, mask.shape[1]) and np.allclose(out["shares"].sum(-1), 1, atol=1e-5)
    for k in ("exp_income", "p_below_theta", "exp_shortfall", "logsum", "ce_rho2.5"):
        assert out[k].shape == (K, C) and np.isfinite(out[k]).all()
    assert np.all((out["p_below_theta"] >= 0) & (out["p_below_theta"] <= 1))
