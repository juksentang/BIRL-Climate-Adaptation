"""The shared logits of the semi-parametric model equal the definition that was used
for the reported estimates (08_BIRL_v2/slurm/run_semipar.py before the package
refactor, reproduced verbatim below), element-wise, for centred (estimation) and
uncentred (counterfactual) use."""
import numpy as np
import pytest
import jax
import jax.numpy as jnp

from cropchoice.config import DATA_DIR  # noqa: F401  (device setup first)
from cropchoice.models import compute_logits_semipar, semipar_features, make_semipar_model, country_one_hot
from cropchoice.models_v2 import center_reward, INFEASIBLE_LOGIT
from cropchoice.fit_semipar import toy_model_kwargs

pytestmark = pytest.mark.toy
HI = jax.lax.Precision.HIGHEST


def _reference_logits(a, b, c, asc, mk, C, center=True):
    """Verbatim copy of the pre-refactor run_semipar.py definition."""
    q10, q50, q90 = mk["q10"], mk["q50"], mk["q90"]
    MU = jnp.log1p(jnp.maximum(q50, 0.0))
    SIG = jnp.clip((jnp.log1p(jnp.maximum(q90, 0.0)) - jnp.log1p(jnp.maximum(q10, 0.0))) / (2 * 1.2816), 0.01, 5.0)
    SIG2 = SIG * SIG
    MASK = mk["mask_obs"]; CI = mk["obs_country_idx"]
    OH = jax.nn.one_hot(CI, C, dtype=jnp.float32)
    per_obs = lambda v: jnp.dot(OH, v, precision=HI)
    V = per_obs(a)[:, None] * MU + per_obs(b)[:, None] * SIG + per_obs(c)[:, None] * SIG2
    alpha = per_obs(jnp.concatenate([jnp.zeros((C, 1)), asc], axis=1))
    Vc = center_reward(V, MASK) if center else V
    return jnp.where(MASK, Vc + alpha, INFEASIBLE_LOGIT)


@pytest.mark.parametrize("center", [True, False])
def test_logits_match_reference(center):
    mk, countries, _ = toy_model_kwargs(seed=5, Nt=300, At=9, C=len(["A", "B", "C"]))
    C = len(countries); A = mk["q50"].shape[1]
    rng = np.random.default_rng(11)
    a = jnp.asarray(rng.normal(0.5, 0.5, C), jnp.float32); b = jnp.asarray(rng.normal(-3, 1, C), jnp.float32)
    c = jnp.asarray(rng.normal(1, 0.3, C), jnp.float32); asc = jnp.asarray(rng.normal(0, 1, (C, A - 1)), jnp.float32)
    feat = semipar_features(mk); oh = country_one_hot(mk, C)
    new = compute_logits_semipar(a, b, c, asc, feat["MU"], feat["SIG"], feat["MASK"], oh, center=center)
    ref = _reference_logits(a, b, c, asc, mk, C, center=center)
    assert new.shape == ref.shape
    assert bool(jnp.array_equal(new, ref)), float(jnp.max(jnp.abs(new - ref)))


def test_full_asc_matrix_is_accepted():
    mk, countries, _ = toy_model_kwargs(seed=6, Nt=120, At=6)
    C = len(countries); A = mk["q50"].shape[1]
    rng = np.random.default_rng(2)
    a, b, c = (jnp.asarray(rng.normal(size=C), jnp.float32) for _ in range(3))
    asc = jnp.asarray(rng.normal(size=(C, A - 1)), jnp.float32)
    asc_full = jnp.concatenate([jnp.zeros((C, 1), jnp.float32), asc], axis=1)
    feat = semipar_features(mk); oh = country_one_hot(mk, C)
    l1 = compute_logits_semipar(a, b, c, asc, feat["MU"], feat["SIG"], feat["MASK"], oh, center=False)
    l2 = compute_logits_semipar(a, b, c, asc_full, feat["MU"], feat["SIG"], feat["MASK"], oh, center=False)
    assert bool(jnp.array_equal(l1, l2))


def test_model_traces_expected_sites():
    from numpyro import handlers
    mk, countries, _ = toy_model_kwargs(seed=7, Nt=100, At=6)
    C = len(countries); feat = semipar_features(mk); oh = country_one_hot(mk, C)
    model = make_semipar_model(feat, C, oh)
    tr = handlers.trace(handlers.seed(model, 0)).get_trace()
    assert {"a_c", "b_c", "c_c", "asc_c", "obs_action"} <= set(tr)
    assert tr["asc_c"]["value"].shape == (C, mk["q50"].shape[1] - 1)
