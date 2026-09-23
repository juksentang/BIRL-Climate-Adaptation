"""Semi-parametric structural crop-choice model (the estimated model of the paper).

  V_ia   = a_g mu_ia + b_g sigma_ia + c_g sigma_ia^2          g = country (or country x asset tercile)
  U_ia   = V_ia + ASC_{c,a}                                    ASC by country x action, action 0 = reference
  P(a|i) = softmax over the feasible actions of U_ia

`compute_logits_semipar` is the ONE definition used by NUTS fitting
(`cropchoice.fit_semipar*`, center=True as estimated) and by the counterfactual
(`cropchoice.counterfactual`, center=False).  Centring V over the feasible set
subtracts a per-observation constant, so choice probabilities are identical
either way; only the level of the logsum differs, which is why the money-metric
welfare in the counterfactual is stated relative to a per-observation constant
fixed by convention (see CHOICE_CF_SPEC.md).
"""
import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from numpyro import sample, plate

from cropchoice.broadcast import one_hot, per_obs
from cropchoice.models_v2 import center_reward, INFEASIBLE_LOGIT   # noqa: F401  (single definitions)
from cropchoice.quantiles import mu_sigma


def semipar_features(mk):
    """(MU, SIG, SIG2, MASK, ACT, CI) from a model_kwargs dict (or any dict with q10/q50/q90/mask_obs/...)."""
    mu, sig = mu_sigma(mk["q10"], mk["q50"], mk["q90"])
    return {"MU": mu, "SIG": sig, "SIG2": sig * sig, "MASK": mk["mask_obs"],
            "ACT": mk["obs_action"], "CI": mk["obs_country_idx"]}


def compute_logits_semipar(a, b, c, asc, mu, sig, mask, oh_c, oh_g=None, center=True):
    """Logits (N, A).  a, b, c: (G,) group coefficients broadcast with oh_g (N, G)
    (oh_g defaults to oh_c); asc: (C, A-1) with action 0 as the reference, broadcast
    with oh_c (N, C).  Infeasible actions get INFEASIBLE_LOGIT."""
    og = oh_c if oh_g is None else oh_g
    sig2 = sig * sig                                      # c * (sig*sig), as in the estimation runners
    V = per_obs(og, a)[:, None] * mu + per_obs(og, b)[:, None] * sig + per_obs(og, c)[:, None] * sig2
    C = oh_c.shape[1]
    A = mu.shape[1]
    asc_full = asc if asc.shape[-1] == A else jnp.concatenate([jnp.zeros((C, 1), asc.dtype), asc], axis=1)
    alpha = per_obs(oh_c, asc_full)                       # (N, A); asc may come with or without the reference column
    if center:
        V = center_reward(V, mask)
    return jnp.where(mask, V + alpha, INFEASIBLE_LOGIT)


def make_semipar_model(feat, C, oh_c, prior_sd=3.0, oh_g=None, G=None, center=True):
    """numpyro model closing over the (constant) data, as in the original runners.
    Sites: a_c/b_c/c_c (country level) or a_g/b_g/c_g (group level), asc_c (C, A-1)."""
    A = feat["MU"].shape[1]
    grouped = oh_g is not None
    n_par = G if grouped else C
    sfx = "g" if grouped else "c"

    def model(null=False):
        with plate("groups" if grouped else "countries", n_par):
            if null:
                a = b = c = jnp.zeros((n_par,))
            else:
                a = sample(f"a_{sfx}", dist.Normal(0.0, prior_sd))
                b = sample(f"b_{sfx}", dist.Normal(0.0, prior_sd))
                c = sample(f"c_{sfx}", dist.Normal(0.0, prior_sd))
        asc = sample("asc_c", dist.Normal(0.0, prior_sd).expand([C, A - 1]).to_event(2))
        logits = compute_logits_semipar(a, b, c, asc, feat["MU"], feat["SIG"], feat["MASK"], oh_c, oh_g, center=center)
        with plate("observations", logits.shape[0]):
            sample("obs_action", dist.Categorical(logits=logits), obs=feat["ACT"])
    return model


def make_probs_fn(feat, oh_c, oh_g=None, center=True):
    """jitted (a, b, c, asc) -> P (N, A) for PPCs."""
    def probs(a, b, c, asc):
        return jax.nn.softmax(compute_logits_semipar(a, b, c, asc, feat["MU"], feat["SIG"], feat["MASK"], oh_c, oh_g, center=center), axis=-1)
    return jax.jit(probs)


def country_one_hot(mk, C):
    return one_hot(mk["obs_country_idx"], C)
