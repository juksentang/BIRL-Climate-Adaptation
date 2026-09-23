"""Income quantiles -> log-income moments and the 5-node quadrature rule.

One definition for the whole pipeline (estimation in `cropchoice.models`,
counterfactual in `cropchoice.counterfactual`, tests):

  mu    = log1p(max(q50, 0))
  sigma = clip((log1p(max(q90, 0)) - log1p(max(q10, 0))) / (2 Z), 0.01, 5),  Z = 1.2816 = Phi^-1(0.9)
  nodes = (q10, (q10+q50)/2, q50, (q50+q90)/2, q90),  weights W5 = (.1, .2, .4, .2, .1)

The level-space helpers of the Stone-Geary model (q_from_log, q_arrays,
obs_mask, add_precomputed, five_points) live in `models_v2` and are re-exported
here so callers have a single import path.
"""
import numpy as np
import jax.numpy as jnp

from cropchoice.config import QUAD_W
from cropchoice.models_v2 import five_points, q_from_log, q_arrays, obs_mask, add_precomputed  # noqa: F401

Z = 1.2816                                       # norm.ppf(0.9)
W5 = np.array(QUAD_W, np.float32)                # (0.1, 0.2, 0.4, 0.2, 0.1)
SIGMA_LO, SIGMA_HI = 0.01, 5.0


def mu_sigma(q10, q50, q90):
    """(mu, sigma) of log income from the three quantiles, exactly as estimated."""
    mu = jnp.log1p(jnp.maximum(q50, 0.0))
    sig = (jnp.log1p(jnp.maximum(q90, 0.0)) - jnp.log1p(jnp.maximum(q10, 0.0))) / (2 * Z)
    return mu, jnp.clip(sig, SIGMA_LO, SIGMA_HI)


def five_nodes(q10, q50, q90):
    """(5, ...) income nodes of the quadrature rule (stacked on a new leading axis)."""
    return jnp.stack([q10, 0.5 * (q10 + q50), q50, 0.5 * (q50 + q90), q90], axis=0)


def e5(nodes):
    """Quadrature mean over the leading node axis."""
    return jnp.tensordot(jnp.asarray(W5), nodes, axes=(0, 0))
