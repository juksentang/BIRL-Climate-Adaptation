"""Regression check: the packaged semi-parametric logits reproduce the reported
log-likelihood.  results.json['ll_semipar'] is the log-lik of the observed actions at
the PPC-averaged probabilities (100 draws picked with linspace over the flattened
posterior, exactly as fit_semipar does); recompute it with cropchoice.models and
compare.  Also reports the log-lik at the element-wise posterior median.
Run on the cluster from 08_BIRL_v2/:  python3 slurm/check_semipar_ll.py [posterior.npz results.json]
"""
import json, sys
import numpy as np, jax.numpy as jnp

from cropchoice.config import DATA_DIR, OUT_DIR
from cropchoice.data import load_data, model_kwargs
from cropchoice.models import semipar_features, make_probs_fn, country_one_hot

post = sys.argv[1] if len(sys.argv) > 1 else OUT_DIR / "semipar" / "posterior.npz"
res = sys.argv[2] if len(sys.argv) > 2 else OUT_DIR / "semipar" / "results.json"
data = load_data(DATA_DIR); mk = model_kwargs(data); C = len(data["countries"])
feat = semipar_features(mk); OH = country_one_hot(mk, C)
S = np.load(post); flat = {k: S[k].reshape((-1,) + S[k].shape[2:]) for k in ("a_c", "b_c", "c_c", "asc_c")}
probs = make_probs_fn(feat, OH)
n_ppc = min(100, flat["a_c"].shape[0]); pick = np.linspace(0, flat["a_c"].shape[0] - 1, n_ppc).astype(int)
N, A = feat["MU"].shape; act = np.asarray(feat["ACT"])
P = np.zeros((N, A), np.float64)
for j in pick:
    P += np.asarray(probs(jnp.asarray(flat["a_c"][j]), jnp.asarray(flat["b_c"][j]), jnp.asarray(flat["c_c"][j]), jnp.asarray(flat["asc_c"][j])))
P /= n_ppc
ll_ppc = float(np.sum(np.log(np.maximum(P[np.arange(N), act], 1e-12))))
med = {k: jnp.asarray(np.median(v, axis=0)) for k, v in flat.items()}
Pm = np.asarray(probs(med["a_c"], med["b_c"], med["c_c"], med["asc_c"]))
ll_med = float(np.sum(np.log(np.maximum(Pm[np.arange(N), act], 1e-12))))
ref = json.load(open(res))["ll_semipar"]
print(f"ll at PPC-averaged probs: new {ll_ppc:.4f}  reported {ref:.4f}  diff {ll_ppc - ref:+.4e}")
print(f"ll at posterior median params: {ll_med:.4f}")
print("PASS" if abs(ll_ppc - ref) < 1e-3 else "CHECK: differs by more than 1e-3")
