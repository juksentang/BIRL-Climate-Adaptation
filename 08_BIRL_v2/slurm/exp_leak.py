"""Leakage test for the environment model's sigma / mu.

Step 04 trained model_mu / model_sigma on 85% of households (trainval) and
then predicted the counterfactual matrix for ALL observations.  For trainval
rows the chosen cell is an in-sample fit (mu ~ realised y, sigma ~ in-sample
residual), for test rows it is out-of-sample.  If "choose the low-sigma
action" is a training artefact, the sigma effect must be much weaker on the
15% test households.  Fits (country x action ASCs, beta_c):
  logev    V = mu + sigma^2/2
  logfree  V = a mu + b sigma + c sigma^2
  free5    V = sum_k w_k q_k (level space)
separately on train rows and test rows, and reports coefficients, log-lik gain
per obs vs the ASC-only null, and descriptive sigma_chosen - median feasible.
Run from 08_BIRL_v2/:  python3 slurm/exp_leak.py [steps]
"""
import os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("BIRL_HOST_DEVICES", "1")

from src.config import log, DATA_DIR, OUT_DIR  # noqa: E402
import jax, jax.numpy as jnp, numpy as np  # noqa: E402
import numpyro  # noqa: E402
import numpyro.distributions as dist  # noqa: E402
from numpyro import sample, deterministic, plate  # noqa: E402
from numpyro.infer import SVI, Trace_ELBO  # noqa: E402
from numpyro.infer.autoguide import AutoMultivariateNormal  # noqa: E402
from numpyro.infer.util import log_density  # noqa: E402
from src.data_loader import load_data, model_kwargs  # noqa: E402
from src.models import center_reward, INFEASIBLE_LOGIT  # noqa: E402

STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
HI = jax.lax.Precision.HIGHEST
W5 = jnp.array([0.1, 0.2, 0.4, 0.2, 0.1], jnp.float32)


def say(msg):
    print(f"[leak {time.strftime('%H:%M:%S')}] {msg}", flush=True)


data = load_data(DATA_DIR)
countries = list(data["countries"]); C = len(countries)
mk_all = model_kwargs(data); m_c = jnp.asarray(data["m_c"], jnp.float32)
test = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "env_test_mask.npy"))
assert test.shape[0] == int(mk_all["obs_action"].shape[0])
say(f"devices {jax.devices()}  N {test.shape[0]}  test rows {int(test.sum())}  steps {STEPS}")

# descriptive: chosen-cell sigma and mu relative to the feasible set, train vs test
q10, q50, q90 = (np.asarray(mk_all[k], np.float64) for k in ("q10", "q50", "q90"))
mask = np.asarray(mk_all["mask_obs"]); act = np.asarray(mk_all["obs_action"]); ci = np.asarray(mk_all["obs_country_idx"])
mu = np.log1p(np.maximum(q50, 0)); sig = np.clip((np.log1p(np.maximum(q90, 0)) - np.log1p(np.maximum(q10, 0))) / (2 * 1.2816), 0.01, 5)
N = len(act); idx = np.arange(N)
sig_ch = sig[idx, act]; mu_ch = mu[idx, act]
sig_med = np.nanmedian(np.where(mask, sig, np.nan), 1); mu_med = np.nanmedian(np.where(mask, mu, np.nan), 1)
sig_rank = np.nanmean(np.where(mask, sig, np.nan) < sig_ch[:, None], 1)     # share of feasible actions with LOWER sigma than chosen
say("A. chosen cell vs feasible set (sigma of log income):")
say(f"   {'':10s} {'sig_ch-sig_med tr':>18s} {'test':>8s}   {'rank tr':>8s} {'test':>8s}   {'mu_ch-mu_med tr':>16s} {'test':>8s}")
desc = {}
for i, c in enumerate(countries):
    tr = (ci == i) & ~test; te = (ci == i) & test
    d = dict(sig_diff_train=float(np.mean(sig_ch[tr] - sig_med[tr])), sig_diff_test=float(np.mean(sig_ch[te] - sig_med[te])),
             rank_train=float(np.mean(sig_rank[tr])), rank_test=float(np.mean(sig_rank[te])),
             mu_diff_train=float(np.mean(mu_ch[tr] - mu_med[tr])), mu_diff_test=float(np.mean(mu_ch[te] - mu_med[te])))
    desc[c] = d
    say(f"   {c:10s} {d['sig_diff_train']:+18.3f} {d['sig_diff_test']:+8.3f}   {d['rank_train']:8.3f} {d['rank_test']:8.3f}   {d['mu_diff_train']:+16.3f} {d['mu_diff_test']:+8.3f}")


def subset(mk, sel):
    out = {}
    for k, v in mk.items():
        if hasattr(v, "shape") and v.ndim >= 1 and v.shape[0] == N:
            out[k] = v[jnp.asarray(np.flatnonzero(sel))]
        else:
            out[k] = v
    return out


def build(mk):
    q10, q50, q90 = mk["q10"], mk["q50"], mk["q90"]
    MU = jnp.log1p(jnp.maximum(q50, 0.0))
    SIG = jnp.clip((jnp.log1p(jnp.maximum(q90, 0.0)) - jnp.log1p(jnp.maximum(q10, 0.0))) / (2 * 1.2816), 0.01, 5.0)
    NODES = jnp.stack([q10, 0.5 * (q10 + q50), q50, 0.5 * (q50 + q90), q90], axis=0)
    MASK = mk["mask_obs"]; ACT = mk["obs_action"]; CI = mk["obs_country_idx"]
    n, A = q50.shape
    OH = jax.nn.one_hot(CI, C, dtype=jnp.float32)
    per_obs = lambda v: jnp.dot(OH, v, precision=HI)

    def model(family):
        with plate("countries", C):
            lb = sample("lb_c", dist.Normal(1.0, 2.0))
            if family == "logfree":
                wf = sample("wfree_c", dist.Normal(0.0, 2.0).expand([3]).to_event(1))
            if family == "free5":
                w5 = sample("w5_c", dist.Normal(0.0, 1.0).expand([5]).to_event(1))
        beta_c = deterministic("beta_c", jnp.exp(jnp.clip(lb, -4.0, 6.0)))
        if family == "null":
            V = jnp.zeros_like(MU)
        elif family == "logev":
            V = MU + 0.5 * SIG * SIG
        elif family == "logfree":
            deterministic("w_c", wf); w = per_obs(wf)
            V = w[:, 0:1] * MU + w[:, 1:2] * SIG + w[:, 2:3] * SIG * SIG
        elif family == "free5":
            deterministic("pi_c", w5)
            V = jnp.einsum("kna,nk->na", NODES, per_obs(w5)) / per_obs(m_c)[:, None]
        logits = per_obs(beta_c)[:, None] * center_reward(V, MASK)
        a = sample("asc_c", dist.Normal(0.0, 3.0).expand([C, A - 1]).to_event(2))
        logits = logits + per_obs(jnp.concatenate([jnp.zeros((C, 1)), a], axis=1))
        logits = jnp.where(MASK, logits, INFEASIBLE_LOGIT)
        with plate("observations", n):
            sample("obs_action", dist.Categorical(logits=logits), obs=ACT)
    return model, n


def fit(model, n, family, seed=0):
    guide = AutoMultivariateNormal(model, init_loc_fn=numpyro.infer.init_to_median())
    svi = SVI(model, guide, numpyro.optim.Adam(1e-2), Trace_ELBO())
    t0 = time.time()
    res = svi.run(jax.random.PRNGKey(seed), STEPS, progress_bar=False, family=family)
    med = guide.median(res.params)
    _, tr = log_density(numpyro.handlers.substitute(model, data=med), (), {"family": family}, {})
    ll = float(tr["obs_action"]["fn"].log_prob(tr["obs_action"]["value"]).sum())
    det = {k: np.asarray(v["value"]).tolist() for k, v in tr.items() if v["type"] == "deterministic"}
    say(f"   {family:8s} n={n:6d} log-lik {ll:12.1f}  ({time.time()-t0:.0f}s)")
    return {"loglik": ll, "n": n, "det": det}


out = {"desc": desc, "fits": {}}
for name, sel in (("train", ~test), ("test", test)):
    say(f"B. fits on {name} rows")
    model, n = build(subset(mk_all, sel))
    R = {fam: fit(model, n, fam) for fam in ("null", "logev", "logfree", "free5")}
    out["fits"][name] = R
    ll0 = R["null"]["loglik"]
    for fam in ("logev", "logfree", "free5"):
        say(f"   {name}/{fam}: gain/obs vs ASC-null = {(R[fam]['loglik'] - ll0) / n:.4f}")
    wf = np.array(R["logfree"]["det"]["w_c"])
    say(f"   {name}/logfree (a mu, b sig, c sig2) per country: " + " | ".join(" ".join(f"{x:+.2f}" for x in row) for row in wf))
    p5 = np.array(R["free5"]["det"]["pi_c"])
    say(f"   {name}/free5 pi(q10..q90) per country: " + " | ".join(" ".join(f"{x:+.2f}" for x in row) for row in p5))
say(f"countries: {countries}")
o = OUT_DIR / "exp_leak"; o.mkdir(parents=True, exist_ok=True)
json.dump(out, open(o / "exp_leak.json", "w"), indent=2)
say(f"written {o/'exp_leak.json'}")
