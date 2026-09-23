"""
Unit tests for BIRL v2 (tests T1-T7 of the spec + a few guards).

    python3 -m pytest -m toy tests/          # T1, T2, T3, T7 (+ toy variants of T4, T5): no data
    python3 -m pytest -m realdata tests/     # T2-T6 on ALL 222K x 27 cells of the 06 data (cluster)
    python3 -m pytest tests/                 # everything

T2 / T3 on real data run over every observation and action (the cells that
stress the Taylor switch and the float32 path -- two nodes inside the floor and
q90/q50 >> 100 -- are rare, so a random subset would miss them); assertion
messages name the worst cell (obs, action, country, q10/q50/q90).

The float64 numpy reference `ce_reference_np` is written independently of
src/models.py (spec-critic P11).
"""

import numpy as np
import pandas as pd
import pytest

from src.config import DATA_DIR, EPS_FRAC, S_MAX, QUAD_W, P_TAYLOR, RHO_LO, RHO_HI  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from src.models import (smooth_surplus, certainty_equivalent, ce_from_nodes, five_points,  # noqa: E402
                        compute_logits, simulate_actions, log_likelihood, loglik_from_means,
                        country_params_from_latents, derive_country_params, latent_for_rho,
                        latent_for_s, INFEASIBLE_LOGIT, log_power_mean, add_precomputed)
from tests.conftest import M_C_REF, make_toy_data  # noqa: E402

T1_RHOS = [0.3, 0.985, 0.995, 1.0, 1.005, 1.015, 1.03, 2.5, 4.5]
T3_RHOS = [0.3, 0.99, 1.0, 2.5, 4.5, 1.0 - P_TAYLOR * 1.5, 1.0 + P_TAYLOR * 1.5]   # last two: direct branch near the switch
T4_RHOS = [0.1, 0.99, 1.0, 1.01, 4.99]
T2_SWITCHES = (1.0 - P_TAYLOR, 1.0 + P_TAYLOR)   # rho where |1 - rho| = P_TAYLOR
T2_H = 1e-4                                      # half-spacing of the rho grid around a switch
T2_LOGLIK_TOL_NAT = 1e-2                         # per-country summed log-lik jump across the switch


# =====================================================================
# Independent float64 numpy reference (spec-critic P11)
# =====================================================================

def smooth_surplus_np(Y, gamma, eps):
    Y = np.asarray(Y, np.float64)
    d = Y - gamma - eps
    return eps + 0.5 * (d + np.sqrt(d * d + eps * eps))


def ce_reference_np(q10, q50, q90, gamma, rho, eps):
    """CE = gamma + ( sum_k w_k surplus_k^(1-rho) )^(1/(1-rho)), geometric mean at rho = 1.
    gamma, eps: (N, 1) or scalar; rho scalar."""
    q10, q50, q90 = (np.asarray(q, np.float64) for q in (q10, q50, q90))
    nodes = np.stack([q10, 0.5 * (q10 + q50), q50, 0.5 * (q50 + q90), q90], axis=-1)
    x = smooth_surplus_np(nodes, np.asarray(gamma, np.float64)[..., None],
                          np.asarray(eps, np.float64)[..., None])
    w = np.asarray(QUAD_W, np.float64)
    p = 1.0 - rho
    if abs(p) < 1e-12:
        log_ce = np.sum(w * np.log(x), axis=-1)
    else:
        log_ce = np.log(np.sum(w * x ** p, axis=-1)) / p
    return np.asarray(gamma, np.float64) + np.exp(log_ce)


def _q(data):
    return [np.exp(np.clip(np.asarray(data[k], np.float64), -20.0, 20.0))
            for k in ("cf_log_q10", "cf_log_q50", "cf_log_q90")]


def _obs_gamma_eps(data, s=0.3):
    m = np.asarray(data["m_c_np"], np.float64)[np.asarray(data["obs_country_idx"])]
    return (s * m)[:, None], (EPS_FRAC * m)[:, None]


def _ce_jax(data, rho, s=0.3):
    q = [jnp.asarray(v, jnp.float32) for v in _q(data)]
    g, e = _obs_gamma_eps(data, s)
    return np.asarray(certainty_equivalent(*q, jnp.asarray(g, jnp.float32),
                                           jnp.float32(rho), jnp.asarray(e, jnp.float32)),
                      np.float64)


def _worst_cell(rel, data, label):
    """Diagnostic string for the arg-max |rel| cell of an (N_obs, N_actions) array."""
    rel = np.asarray(rel, np.float64)
    i, a = np.unravel_index(int(np.nanargmax(np.abs(rel))), rel.shape)
    q10, q50, q90 = (float(q[i, a]) for q in _q(data))
    ctry = data["countries"][int(np.asarray(data["obs_country_idx"])[i])]
    return (f"{label}: max = {rel[i, a]:.3e} at obs {i} action {a} ({ctry}): "
            f"q10={q10:.4g} q50={q50:.4g} q90={q90:.4g} USD, m_c={float(data['m_c_np'][int(np.asarray(data['obs_country_idx'])[i])]):.4g}"
            f"; {int(np.sum(np.abs(rel) > 1e-4))} cells > 1e-4, {int(np.sum(np.abs(rel) > 1e-5))} > 1e-5")


# =====================================================================
# T1  identity: q10 = q50 = q90 = Y0  =>  CE == Y0
# =====================================================================

@pytest.mark.toy
@pytest.mark.parametrize("rho", T1_RHOS)
@pytest.mark.parametrize("y_mult", [2.0, 0.5])
@pytest.mark.parametrize("s", [0.1, 0.2, 0.3])
def test_t1_ce_identity(rho, y_mult, s):
    m_c = M_C_REF
    Y0 = y_mult * m_c
    gamma, eps = s * m_c, EPS_FRAC * m_c
    assert np.all(Y0 >= gamma + 5 * eps)          # floor region inactive (spec condition)
    Y = jnp.asarray(Y0, jnp.float32)
    ce = np.asarray(certainty_equivalent(Y, Y, Y, jnp.asarray(gamma, jnp.float32),
                                         jnp.float32(rho), jnp.asarray(eps, jnp.float32)),
                    np.float64)
    # exact identity of the power mean with degenerate nodes: CE == gamma + smooth_surplus(Y0)
    ce_exact = gamma + smooth_surplus_np(Y0, gamma, eps)
    assert np.allclose(ce, ce_exact, rtol=2e-4)
    # CE == Y0 within 1e-3 rel; the smooth floor's own bias k^2/(4 d) relative to Y0 is
    # 1.1e-3 at (Y0 = 0.5 m_c, s = 0.3) (d = 9k), so that one case gets 2e-3
    floor_bias = np.max((ce_exact - Y0) / Y0)
    tol = 1e-3 if floor_bias < 5e-4 else 2e-3
    assert np.allclose(ce, Y0, rtol=tol), (ce / Y0 - 1)


# =====================================================================
# T2  continuity at the Taylor switch (|p| = P_TAYLOR  <=>  rho = 1 -+ P_TAYLOR)
# =====================================================================

def _grid(r0, side):
    """Four equally spaced rho values straddling the switch r0: two on the
    direct side (farther from 1), two on the Taylor side.  side = -1 for the
    lower switch (rho < 1), +1 for the upper one."""
    h = T2_H
    return [r0 + side * 3 * h, r0 + side * h, r0 - side * h, r0 - side * 3 * h]


def _t2_check(data):
    """Jump at the switch, trend-corrected: the CE change across the switch
    (r0 -+ h, direct -> Taylor branch) minus the change over the neighbouring
    interval of the same width on the direct branch.  A smooth function gives
    ~0 (the second difference is O(h^2)); a discontinuity shows up as the jump.
    Runs over every cell of `data` and names the worst one on failure."""
    for side, r0 in zip((-1, 1), T2_SWITCHES):
        ra, rb, rc, rd = _grid(r0, side)            # ra, rb direct; rc, rd Taylor
        ce = {r: _ce_jax(data, r) for r in (ra, rb, rc, rd)}
        jump = ((ce[rc] - ce[rb]) - (ce[rb] - ce[ra])) / ce[rb]
        assert np.all(np.isfinite(jump))
        assert np.all(np.abs(jump) < 1e-4), _worst_cell(jump, data, f"T2 jump at rho={r0:.4f}")
        # CE is monotone decreasing in rho on both sides of the switch (no sign flip of the trend)
        order = [ra, rb, rc, rd] if side < 0 else [rd, rc, rb, ra]      # increasing rho
        for lo, hi in zip(order[:-1], order[1:]):
            assert np.all(ce[lo] >= ce[hi] - 1e-5 * ce[lo]), \
                _worst_cell((ce[hi] - ce[lo]) / ce[lo], data, f"T2 monotone {lo:.5f}->{hi:.5f}")
    # both branch formulas agree at exactly |p| = P_TAYLOR (direct vs cumulant expansion):
    # at P_TAYLOR = 1e-3 the truncation error is ~1e-8 rel, float32 noise ~1e-6
    q = [jnp.asarray(v, jnp.float32) for v in _q(data)]
    g, e = _obs_gamma_eps(data)
    log_x = [jnp.log(smooth_surplus(y, jnp.asarray(g, jnp.float32), jnp.asarray(e, jnp.float32)))
             for y in five_points(*q)]
    for p in (P_TAYLOR, -P_TAYLOR):
        direct = np.asarray(log_power_mean(log_x, jnp.float32(p * 1.0001)), np.float64)
        taylor = np.asarray(log_power_mean(log_x, jnp.float32(p * 0.9999)), np.float64)
        rel = np.expm1(taylor - direct)
        assert np.all(np.abs(rel) < 2e-5), _worst_cell(rel, data, f"T2 branch agreement at p={p}")


@pytest.mark.toy
def test_t2_taylor_switch_continuity_toy(toy):
    _t2_check(toy)


@pytest.mark.realdata
def test_t2_taylor_switch_continuity_real(real_data):
    _t2_check(real_data)          # all 222K x 27 cells


def _loglik_by_country_f64(data, rho, s=0.3, beta=5.0):
    """Per-country summed categorical log-likelihood of the observed actions at
    rho_c = rho, s_c = s, beta_c = beta for every country, evaluated through
    the model's own compute_logits in float64 (jax.enable_x64 context, arrays
    cast) so the result carries no float32 summation noise (a float32 sum over
    40K obs has ~1e-2 nat noise, which would mask the quantity under test)."""
    n = data["N_country"]
    with jax.enable_x64(True):
        d64 = dict(data)
        for k in ("cf_log_q10", "cf_log_q50", "cf_log_q90", "q10", "q50", "q90", "m_c"):
            if k in d64:
                d64[k] = jnp.asarray(np.asarray(d64[k], np.float64))
        rho_c = jnp.full((n,), rho, jnp.float64)
        s_c = jnp.full((n,), s, jnp.float64)
        beta_c = jnp.full((n,), beta, jnp.float64)
        logits = compute_logits(rho_c, s_c, beta_c, d64)
        assert logits.dtype == jnp.float64
        logp = jax.nn.log_softmax(logits, axis=-1)
        lp = np.asarray(jnp.take_along_axis(logp, jnp.asarray(data["obs_action"])[:, None],
                                            axis=-1)[:, 0], np.float64)
    return np.bincount(np.asarray(data["obs_country_idx"]), weights=lp, minlength=n)


def _t2_loglik_check(data):
    """The quantity that matters for NUTS: the jump of the summed log-likelihood
    in rho_c across the branch switch, per country.  Second difference straddling
    the switch, D1 = f(r0 - s h) - 2 f(r0 + s h) + f(r0 + 3 s h) (s = side), minus
    the reference second difference two steps away on the direct branch (same
    spacing; removes the smooth curvature term): |D1 - D_ref| < 1e-2 nat."""
    h = T2_H
    for side, r0 in zip((-1, 1), T2_SWITCHES):
        rs = [r0 + side * k * h for k in (7, 5, 3, 1, -1)]     # 7,5,3 direct; 1 direct; -1 Taylor
        f = {r: _loglik_by_country_f64(data, r) for r in rs}
        d_switch = f[rs[4]] - 2 * f[rs[3]] + f[rs[2]]           # straddles the switch
        d_ref = f[rs[2]] - 2 * f[rs[1]] + f[rs[0]]              # all on the direct branch
        jump = d_switch - d_ref
        msg = (f"log-lik jump across rho={r0:.4f} per country "
               f"{dict(zip(data['countries'], np.round(jump, 5)))} nat "
               f"(second differences: switch {np.round(d_switch, 5)}, ref {np.round(d_ref, 5)})")
        assert np.all(np.isfinite(jump)), msg
        assert np.all(np.abs(jump) < T2_LOGLIK_TOL_NAT), msg


@pytest.mark.toy
def test_t2_loglik_jump_toy(toy):
    _t2_loglik_check(toy)


@pytest.mark.realdata
def test_t2_loglik_jump_real(real_data):
    _t2_loglik_check(real_data)   # all observations, beta = 5, s = 0.3


# =====================================================================
# T3  float64 numpy reference vs JAX float32 (1e-4 rel), every cell
# =====================================================================

def _ce_reference_blocked(q, g, rho, e, block=50_000):
    """ce_reference_np row-blocked so the (N, A, 5) float64 temporaries stay < 300 MB."""
    n = q[0].shape[0]
    out = np.empty(q[0].shape, np.float64)
    for i in range(0, n, block):
        sl = slice(i, i + block)
        out[sl] = ce_reference_np(q[0][sl], q[1][sl], q[2][sl], g[sl], rho, e[sl])
    return out


def _t3_check(data, rho):
    q = _q(data)
    g, e = _obs_gamma_eps(data)
    ref = _ce_reference_blocked(q, g, rho, e)
    ce = _ce_jax(data, rho)
    assert ce.shape == ref.shape
    assert np.all(np.isfinite(ce)), f"non-finite CE at rho={rho}"
    rel = ce / ref - 1
    assert np.all(np.abs(rel) < 1e-4), _worst_cell(rel, data, f"T3 rel error at rho={rho}")


@pytest.mark.toy
@pytest.mark.parametrize("rho", T3_RHOS)
def test_t3_float64_reference_toy(toy, rho):
    _t3_check(toy, rho)


@pytest.mark.realdata
@pytest.mark.parametrize("rho", T3_RHOS)
def test_t3_float64_reference_real(real_data, rho):
    _t3_check(real_data, rho)     # all 222K x 27 cells


# =====================================================================
# T4  finite gradients of the summed log-likelihood
# =====================================================================

def _t4_check(data, rho):
    n = data["N_country"]
    # (a) w.r.t. the latent means (mu_rho, mu_s, mu_lb); rho_c = 0.1 is the open lower
    #     bound of the sigmoid map, so the target is clipped to [0.1 + 1e-4, 5 - 1e-4]
    rho_t = float(np.clip(rho, RHO_LO + 1e-4, RHO_HI - 1e-4))
    mu = (jnp.float32(latent_for_rho(rho_t)), jnp.float32(latent_for_s(0.3)), jnp.float32(np.log(3.0)))
    f = lambda a, b, c: loglik_from_means(a, b, c, data)
    val = f(*mu)
    grads = jax.grad(f, argnums=(0, 1, 2))(*mu)
    assert np.isfinite(float(val))
    assert all(np.isfinite(float(g)) for g in grads), grads
    # (b) w.r.t. (rho_c, s_c, log beta_c) directly, at the exact rho incl. 0.1
    rho_c = jnp.full((n,), rho, jnp.float32)
    s_c = jnp.full((n,), 0.3, jnp.float32)
    lb = jnp.full((n,), np.log(3.0), jnp.float32)
    g2 = jax.grad(lambda r, s, l: log_likelihood(r, s, jnp.exp(l), data), argnums=(0, 1, 2))(
        rho_c, s_c, lb)
    assert all(bool(jnp.all(jnp.isfinite(g))) for g in g2)
    assert any(bool(jnp.any(g != 0)) for g in g2), (rho, g2)      # not an all-zero plateau
    # (c) s_c near its bounds and beta large: still finite, and not all-zero
    for s_val in (1e-3, S_MAX - 1e-3):
        g3 = jax.grad(lambda r, s, l: log_likelihood(r, s, jnp.exp(l), data), argnums=(0, 1, 2))(
            rho_c, jnp.full((n,), s_val, jnp.float32), jnp.full((n,), 5.0, jnp.float32))
        assert all(bool(jnp.all(jnp.isfinite(g))) for g in g3)
        assert any(bool(jnp.any(g != 0)) for g in g3), (rho, s_val, g3)
    # (d) log beta beyond the hard clip (lb = 7 > LOG_BETA_HI = 6): beta_c is clipped, so
    #     d/d lb is zero by design, but the likelihood gradient w.r.t. rho and s must stay
    #     finite and non-zero (the chain is not on a flat plateau in the other directions)
    f4 = lambda r, s, l: loglik_from_means(r, s, l, data)
    mu4 = (jnp.float32(latent_for_rho(rho_t)), jnp.float32(latent_for_s(0.3)), jnp.float32(7.0))
    assert np.isfinite(float(f4(*mu4)))
    g4 = jax.grad(f4, argnums=(0, 1, 2))(*mu4)
    assert all(np.isfinite(float(g)) for g in g4), g4
    assert float(g4[0]) != 0.0 and float(g4[1]) != 0.0, g4


@pytest.mark.toy
@pytest.mark.parametrize("rho", T4_RHOS)
def test_t4_finite_gradients_toy(toy, rho):
    _t4_check(toy, rho)


@pytest.mark.realdata
@pytest.mark.parametrize("rho", T4_RHOS)
def test_t4_finite_gradients_real(real_data, rho):
    _t4_check(real_data, rho)


# =====================================================================
# T5  logits finite, infeasible == -1e10, simulation respects feasibility
# =====================================================================

def _t5_check(data, seed=3):
    rho_c = jnp.array([0.5, 1.0, 2.0, 3.0, 4.5, 1.5], jnp.float32)
    s_c = jnp.array([0.1, 0.3, 0.59, 0.05, 0.4, 0.2], jnp.float32)
    beta_c = jnp.array([1.0, 3.0, 8.0, 0.5, 5.0, 2.0], jnp.float32)
    logits = np.asarray(compute_logits(rho_c, s_c, beta_c, data))
    mask = np.asarray(data["feasibility_mask"])[np.asarray(data["obs_cz_idx"])]
    assert logits.shape == (data["N_obs"], data["N_actions"])
    assert np.all(np.isfinite(logits))
    assert np.all(logits[~mask] == INFEASIBLE_LOGIT)
    assert np.all(np.abs(logits[mask]) < 1e6)
    row_means = np.where(mask, logits, 0).sum(1) / mask.sum(1)       # centred over feasible
    assert np.allclose(row_means, 0.0, atol=1e-2)
    acts = np.asarray(simulate_actions(jax.random.PRNGKey(seed), rho_c, s_c, beta_c, data))
    assert acts.shape == (data["N_obs"],) and acts.dtype == np.int32
    assert np.all(mask[np.arange(len(acts)), acts])
    ll = float(log_likelihood(rho_c, s_c, beta_c, data))
    assert np.isfinite(ll) and ll <= 0.0


@pytest.mark.toy
def test_t5_logits_and_simulation_toy():
    _t5_check(make_toy_data(n_obs=50, seed=5))


@pytest.mark.toy
def test_t5_precomputed_arrays_identical(toy):
    """compute_logits / log_likelihood give bit-identical results from the loader's
    precomputed q10/q50/q90/mask_obs and from the log arrays + mask (the loader
    and model_kwargs add the precomputed arrays; the toy dict has only the logs)."""
    from src.data_loader import model_kwargs
    rho_c = jnp.array([0.5, 1.0, 2.0, 3.0, 4.5, 1.5], jnp.float32)
    s_c = jnp.array([0.1, 0.3, 0.59, 0.05, 0.4, 0.2], jnp.float32)
    beta_c = jnp.array([1.0, 3.0, 8.0, 0.5, 5.0, 2.0], jnp.float32)
    pre = add_precomputed(toy)
    assert all(k in pre for k in ("q10", "q50", "q90", "mask_obs")) and "q10" not in toy
    mk = model_kwargs(toy)
    assert all(k in mk for k in ("q10", "q50", "q90", "mask_obs"))
    a = np.asarray(compute_logits(rho_c, s_c, beta_c, toy))
    b = np.asarray(compute_logits(rho_c, s_c, beta_c, pre))
    c = np.asarray(compute_logits(rho_c, s_c, beta_c, mk))
    assert np.array_equal(a, b) and np.array_equal(a, c)
    assert float(log_likelihood(rho_c, s_c, beta_c, toy)) == float(log_likelihood(rho_c, s_c, beta_c, pre))
    assert np.array_equal(np.asarray(pre["mask_obs"]),
                          np.asarray(toy["feasibility_mask"])[np.asarray(toy["obs_cz_idx"])])


@pytest.mark.realdata
def test_t5_logits_and_simulation_real(real_data):
    _t5_check(real_data)


# =====================================================================
# T6  loader m_c == independent pandas computation == the six spec values
# =====================================================================

@pytest.mark.realdata
def test_t6_loader_m_c(real_data):
    df = pd.read_parquet(DATA_DIR / "birl_sample.parquet", columns=["country", "action_id"])
    env = np.load(str(DATA_DIR / "env_model_output.npz"), allow_pickle=True)
    df["chosen_q50"] = env["q50"][np.arange(len(df)), df["action_id"].values]
    ref = df.groupby("country")["chosen_q50"].median().sort_index()
    assert list(ref.index) == real_data["countries"] == \
        ["Ethiopia", "Malawi", "Mali", "Nigeria", "Tanzania", "Uganda"]
    assert np.allclose(real_data["m_c_np"], ref.values, rtol=1e-6)
    assert np.allclose(np.asarray(real_data["m_c"]), ref.values, rtol=1e-5)
    assert np.allclose(real_data["m_c_np"], M_C_REF, atol=0.005), real_data["m_c_np"]
    assert real_data["N_country"] == 6 and real_data["N_obs"] == len(df)
    mask = np.asarray(real_data["feasibility_mask"])
    assert mask[np.asarray(real_data["obs_cz_idx"]), np.asarray(real_data["obs_action"])].all()


# =====================================================================
# T7  monotonicity: CE increasing in each node income
# =====================================================================

@pytest.mark.toy
@pytest.mark.parametrize("rho", T3_RHOS)
def test_t7_monotone_in_each_node(toy, rho):
    q = [jnp.asarray(v, jnp.float32) for v in _q(toy)]
    g, e = _obs_gamma_eps(toy)
    g, e = jnp.asarray(g, jnp.float32), jnp.asarray(e, jnp.float32)
    nodes = list(five_points(*q))
    base = np.asarray(ce_from_nodes(nodes, g, jnp.float32(rho), e), np.float64)
    nodes_np = [np.asarray(x, np.float64) for x in nodes]
    w = np.asarray(QUAD_W)
    for k in range(5):
        bumped = list(nodes)
        bumped[k] = nodes[k] * 1.10
        ce_b = np.asarray(ce_from_nodes(bumped, g, jnp.float32(rho), e), np.float64)
        assert np.all(ce_b >= base * (1 - 1e-6)), k              # never decreasing (float32)
        assert np.mean(ce_b) > np.mean(base)                      # strictly increasing on average
        # exact float64 reference: strictly increasing for every obs and node
        def ref(ns):
            x = smooth_surplus_np(np.stack(ns, -1), np.asarray(g)[..., None], np.asarray(e)[..., None])
            p = 1.0 - rho
            lc = np.sum(w * np.log(x), -1) if abs(p) < 1e-12 else np.log(np.sum(w * x ** p, -1)) / p
            return np.asarray(g) + np.exp(lc)          # g is (N, 1), lc is (N, A)
        b_np = list(nodes_np)
        b_np[k] = nodes_np[k] * 1.10
        assert np.all(ref(b_np) > ref(nodes_np))
    # and increasing in gamma-free income overall: scaling all q up raises CE
    ce_up = np.asarray(ce_from_nodes([x * 1.5 for x in nodes], g, jnp.float32(rho), e))
    assert np.all(ce_up > base)


# =====================================================================
# Guards: smooth floor, transforms
# =====================================================================

@pytest.mark.toy
def test_smooth_floor_properties():
    eps = 0.5
    Y = jnp.linspace(-50.0, 200.0, 2001)
    s = np.asarray(smooth_surplus(Y, jnp.float32(30.0), jnp.float32(eps)), np.float64)
    assert np.all(np.isfinite(s)) and np.all(s >= eps * (1 - 1e-6))
    assert np.all(np.diff(s) > 0)                                  # strictly increasing
    exact = np.maximum(np.asarray(Y, np.float64) - 30.0, eps)
    d = np.asarray(Y, np.float64) - 30.0 - eps
    far = np.abs(d) > 20 * eps
    # equals max(Y - gamma, eps) away from the kink up to the analytic bias
    # 0.5 (sqrt(d^2 + k^2) - |d|) <= k^2 / (4 |d|)  (1.25% of eps at |d| = 20 eps on the low side)
    assert np.all(np.abs(s[far] - exact[far]) <= eps * eps / (4 * np.abs(d[far])) + 1e-5 * exact[far])
    assert np.allclose(s[far & (d > 0)], exact[far & (d > 0)], rtol=1e-3)
    g = jax.grad(lambda y: smooth_surplus(y, 30.0, eps))
    assert all(np.isfinite(float(g(y))) for y in (-100.0, 29.5, 30.0, 30.5, 1e4))


@pytest.mark.toy
def test_transforms_and_derive():
    rng = np.random.default_rng(0)
    n = 200
    m_c = M_C_REF
    lat = {k: rng.normal(size=(n, 6)) * 3 for k in ("rho_lat_c", "s_lat_c", "lb_c")}
    p = derive_country_params(lat, m_c, s_max=S_MAX)
    assert p["rho_c"].shape == (n, 6)
    assert np.all((p["rho_c"] > RHO_LO) & (p["rho_c"] < RHO_HI))
    assert np.all((p["s_c"] > 0) & (p["s_c"] < S_MAX))
    assert np.allclose(p["gamma_c"], p["s_c"] * m_c[None, :])
    assert np.all(p["beta_c"] > 0) and np.all(p["beta_c"] <= np.exp(6.0) + 1e-3)
    # non-centred sites give the same numbers as their centred equivalents
    nc = {"mu_rho": np.zeros(n), "sigma_rho": np.ones(n), "rho_raw": lat["rho_lat_c"],
          "mu_s": np.zeros(n), "sigma_s": np.ones(n), "s_raw": lat["s_lat_c"],
          "mu_lb": np.zeros(n), "sigma_lb": np.ones(n), "lb_raw": lat["lb_c"]}
    p2 = derive_country_params(nc, m_c, s_max=S_MAX)
    assert np.allclose(p2["rho_c"], p["rho_c"]) and np.allclose(p2["beta_c"], p["beta_c"])
    # s_max flag changes the s_c bound; gfix fixes s_c
    p3 = derive_country_params(lat, m_c, s_max=0.8)
    assert np.all(p3["s_c"] < 0.8) and np.any(p3["s_c"] > S_MAX)
    p4 = derive_country_params(lat, m_c, s_fixed=0.3)
    assert np.all(p4["s_c"] == 0.3)
    # JAX transform agrees with numpy, and the inverse maps round-trip
    rj, sj, gj, bj = country_params_from_latents(jnp.asarray(lat["rho_lat_c"][0], jnp.float32),
                                                 jnp.asarray(lat["s_lat_c"][0], jnp.float32),
                                                 jnp.asarray(lat["lb_c"][0], jnp.float32),
                                                 jnp.asarray(m_c, jnp.float32), S_MAX)
    assert np.allclose(np.asarray(rj), p["rho_c"][0], rtol=1e-5)
    assert np.allclose(np.asarray(sj), p["s_c"][0], rtol=1e-5)
    assert np.allclose(np.asarray(bj), p["beta_c"][0], rtol=1e-4)
    for r in (0.1001, 1.0, 4.9):
        assert np.isclose(RHO_LO + (RHO_HI - RHO_LO) / (1 + np.exp(-latent_for_rho(r))), r)
    assert np.isclose(S_MAX / (1 + np.exp(-latent_for_s(0.3))), 0.3)


@pytest.mark.toy
def test_numpyro_models_trace(toy):
    """Both variants and all parameterisations trace on the toy data with the
    expected sample / deterministic sites."""
    from numpyro import handlers
    from src.models import v2_country, v2_country_gfix
    from src.data_loader import model_kwargs
    d = model_kwargs(toy)
    for kw, sampled, dets in [
        ({}, {"mu_rho", "sigma_rho", "mu_s", "sigma_s", "mu_lb", "sigma_lb",
              "rho_lat_c", "s_lat_c", "lb_c"}, {"rho_c", "s_c", "gamma_c", "beta_c"}),
        ({"noncentered": True}, {"mu_rho", "rho_raw", "s_raw", "lb_raw"},
         {"rho_lat_c", "s_lat_c", "lb_c", "rho_c", "s_c", "gamma_c", "beta_c"}),
        ({"flat_priors": True}, {"rho_lat_c", "s_lat_c", "lb_c"}, {"rho_c", "s_c", "gamma_c", "beta_c"}),
        ({"flat_priors": True, "s_max": 0.8}, {"rho_lat_c", "s_lat_c", "lb_c"}, {"s_c"}),
    ]:
        tr = handlers.trace(handlers.seed(v2_country, 0)).get_trace(d, **kw)
        names = {k for k, v in tr.items() if v["type"] == "sample" and not v.get("is_observed")}
        det = {k for k, v in tr.items() if v["type"] == "deterministic"}
        assert sampled <= names, (kw, names)
        assert dets <= det, (kw, det)
        assert "mu_s" not in names or not kw.get("flat_priors")
        assert tr["obs_action"]["fn"].log_prob(tr["obs_action"]["value"]).shape == (toy["N_obs"],)
        assert float(tr["s_c"]["value"].max()) < kw.get("s_max", S_MAX)
    tr = handlers.trace(handlers.seed(v2_country_gfix, 0)).get_trace(d, s_fixed=0.3)
    names = {k for k, v in tr.items() if v["type"] == "sample" and not v.get("is_observed")}
    assert "mu_s" not in names and "s_lat_c" not in names
    assert np.allclose(np.asarray(tr["s_c"]["value"]), 0.3)
    assert np.allclose(np.asarray(tr["gamma_c"]["value"]), 0.3 * np.asarray(toy["m_c"]))
