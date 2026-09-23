"""Gradient-cost diagnostic: where does the time go?

Times, on the real data (prints after every step, flushed):
  1. one potential-energy evaluation (jit, compile separated)
  2. one gradient of the potential (jit, compile separated)
  3. gradient with the data passed as a traced ARGUMENT instead of a closure
     constant (what jit_model_args=True would do)
  4. gradient vmapped over 4 parameter sets (what chain_method='vectorized' does)
  5. 20 SVI steps (AutoMultivariateNormal), per-step wall time
  6. NUTS, 1 chain, max_tree_depth=3, 3 warmup + 2 samples (progress per draw)
Run from 08_BIRL_v2/:  python3 slurm/diag_grad.py
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("BIRL_HOST_DEVICES", "1")

from src.config import log  # noqa: E402  (sets host device count before jax import)
import jax, jax.numpy as jnp, numpy as np  # noqa: E402
import numpyro  # noqa: E402
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO  # noqa: E402
from numpyro.infer.autoguide import AutoMultivariateNormal  # noqa: E402
from numpyro.infer.util import initialize_model  # noqa: E402
from src.data_loader import load_data, model_kwargs  # noqa: E402
from src.models import v2_country  # noqa: E402


def say(msg):
    print(f"[diag {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def timed(label, fn, n=5):
    t0 = time.time(); out = fn(); jax.block_until_ready(out); t_first = time.time() - t0
    t0 = time.time()
    for _ in range(n):
        out = fn()
    jax.block_until_ready(out)
    per = (time.time() - t0) / n
    say(f"{label}: first call {t_first:.2f}s (incl. compile), then {per*1000:.1f} ms/call")
    return out


say(f"devices: {jax.devices()}")
from src.config import DATA_DIR
data = load_data(DATA_DIR)
mk = {"data": model_kwargs(data), "s_max": 0.6, "flat_priors": False, "noncentered": False}
say("data loaded")

rng = jax.random.PRNGKey(0)
info = initialize_model(rng, v2_country, model_kwargs=mk,
                       init_strategy=numpyro.infer.init_to_median())
z0 = info.param_info.z
say(f"latent sites: {[(k, np.shape(v)) for k, v in z0.items()]}")
pot = info.potential_fn

# 1. potential
pj = jax.jit(pot)
timed("1 potential (data as closure constant)", lambda: pj(z0))

# 2. gradient
g = jax.jit(jax.grad(pot))
timed("2 grad(potential)", lambda: g(z0))

# 3. gradient exactness check of the one-hot broadcast vs plain gather
from src.models import per_obs_params, EPS_FRAC  # noqa: E402
_rho = jnp.array([0.7, 1.3, 2.9, 4.1, 1.0, 3.3], jnp.float32)
_s = jnp.array([0.11, 0.23, 0.37, 0.05, 0.5, 0.42], jnp.float32)
_b = jnp.array([2.2, 7.7, 4.4, 1.1, 5.5, 3.3], jnp.float32)
_mc = mk["data"]["m_c"]; _ci = mk["data"]["obs_country_idx"]
_ro, _go, _eo, _mo, _bo = per_obs_params(_rho, _s, _b, _mc, _ci)
_ref = (_rho[_ci], (_s * _mc)[_ci], (EPS_FRAC * _mc)[_ci], _mc[_ci], _b[_ci])
_err = max(float(jnp.max(jnp.abs(a[:, 0] - r))) for a, r in zip((_ro, _go, _eo, _mo, _bo), _ref))
say(f"3 one-hot broadcast vs gather: max abs error {_err:.3e} (must be 0 or ~1e-6)")

# 4. vmapped over 4 chains
z4 = jax.tree_util.tree_map(lambda x: jnp.stack([x] * 4), z0)
g4 = jax.jit(jax.vmap(jax.grad(pot)))
timed("4 grad vmapped x4 (vectorized chains)", lambda: g4(z4))

# 5. SVI steps
guide = AutoMultivariateNormal(v2_country, init_loc_fn=numpyro.infer.init_to_median())
svi = SVI(v2_country, guide, numpyro.optim.Adam(1e-2), Trace_ELBO())
t0 = time.time(); state = svi.init(jax.random.PRNGKey(1), **mk); say(f"5 svi.init {time.time()-t0:.1f}s")
upd = jax.jit(lambda s: svi.update(s, **mk))
t0 = time.time(); state, loss = upd(state); jax.block_until_ready(loss); say(f"5 svi step 1 (compile) {time.time()-t0:.1f}s loss={float(loss):.1f}")
t0 = time.time()
for i in range(20):
    state, loss = upd(state)
jax.block_until_ready(loss)
say(f"5 svi 20 steps: {(time.time()-t0)/20*1000:.1f} ms/step, loss={float(loss):.1f}")

# 6. NUTS shallow trees, per-draw progress
for dense in (False, True):
    kern = NUTS(v2_country, max_tree_depth=3, dense_mass=dense,
                init_strategy=numpyro.infer.init_to_median())
    m = MCMC(kern, num_warmup=3, num_samples=2, num_chains=1, progress_bar=False)
    t0 = time.time()
    m.run(jax.random.PRNGKey(2), extra_fields=("num_steps",), **mk)
    steps = np.asarray(m.get_extra_fields()["num_steps"])
    say(f"6 NUTS depth<=3 dense_mass={dense}: 3+2 draws in {time.time()-t0:.1f}s (incl. compile), "
        f"leapfrogs per sampled draw {steps.tolist()}")
say("done")
