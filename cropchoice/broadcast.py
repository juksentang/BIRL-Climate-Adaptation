"""Group -> observation broadcasting without gathers.

`x[idx]` for a few group parameters over 222K observations is a gather whose
transpose is a scatter-add into a handful of slots; on GPU that runs as
contended atomics (measured: 2.5 s per gradient vs 1.6 ms forward).  A one-hot
matmul has a GEMM transpose.  HIGHEST precision keeps the selection exact
(float32 matmuls default to TF32 on H100, ~1e-3 relative error).
"""
import jax
import jax.numpy as jnp

HI = jax.lax.Precision.HIGHEST


def one_hot(idx, n, dtype=jnp.float32):
    """(N, n) one-hot matrix of a group index."""
    return jax.nn.one_hot(jnp.asarray(idx), n, dtype=dtype)


def per_obs(oh, v):
    """Broadcast group values v (n,) or (n, k) to observations: (N,) or (N, k)."""
    return jnp.dot(oh, v, precision=HI)


def group_mean(oh, x_obs):
    """Mean of a per-observation array within each group -> (n,) or (n, k)."""
    return jnp.dot(oh.T, x_obs, precision=HI) / jnp.maximum(oh.sum(0), 1.0)[(slice(None),) + (None,) * (x_obs.ndim - 1)]
