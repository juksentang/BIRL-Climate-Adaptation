"""LOG-SPACE utility-family experiments (SVI point estimates + log-lik).

free5 showed weights (+ on q10, - on q50): preference for low q10/q50 ratio =
low sigma of log income.  Under the lognormal env model CRRA has the closed form
log CE = mu - (rho-1) sigma^2 / 2, no quadrature, no floor.  Families (reward in
log units, country x action ASCs, beta_c):
  logmed   V = mu                        logev    V = mu + sigma^2/2 (risk-neutral)
  logcrra  V = mu - th_c sigma^2/2       (th = rho - 1, free real)
  logmsd   V = mu - th_c sigma           logfree  V = a mu + b sigma + c sigma^2
ORIGINAL DOC:

Motivation (exp_risk_signal): choices load on the predicted 10th percentile
far more than on the median / 90th (w10 >> w90 in 5/6 countries), which the
Stone-Geary CRRA family can only mimic with rho -> 5.  Here every family uses
the same 5-node quadrature (q10, (q10+q50)/2, q50, (q50+q90)/2, q90; base
probabilities .1 .2 .4 .2 .1), country x action ASCs and a per-country
beta on value / m_c.  Families:
  ev          risk-neutral: V = sum_k w_k y_k                         (reference)
  rdeu_pow    rank-dependent, w(p) = p^delta, delta_c = exp(th_c):
              delta < 1 overweights the worst outcomes (pessimism)
  rdeu_prelec rank-dependent, w(p) = exp(-(-ln p)^alpha), alpha_c = exp(th_c)
  es          expected shortfall: V = E[y] - lam_c * E[(tau_c - y)+],
              tau_c = s_c * m_c, s_c in (0, 0.6), lam_c = exp(llam_c)
  free5       5 free node weights per country (unrestricted benchmark; RDEU
              families are nested in it)
Prints log-lik, gain/obs vs logev, per-country parameters.  Writes
outputs/exp_utility/exp_utility_sf.json.
Run from 08_BIRL_v2/:  python3 slurm/exp_utility.py [steps]
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
F5 = jnp.array([0.1, 0.3, 0.7, 0.9, 1.0], jnp.float32)        # cumulative from the worst node


def say(msg):
    print(f"[util {time.strftime('%H:%M:%S')}] {msg}", flush=True)


if os.environ.get("EXP_TOY"):          # tiny synthetic dataset: end-to-end check without real data
    rng = np.random.default_rng(0); Nt, At, C = 300, 6, 3
    countries = ["A", "B", "C"]; m_np = np.array([20.0, 100.0, 30.0], np.float32)
    _q50 = rng.lognormal(3, 0.7, (Nt, At)); _rs = rng.lognormal(0, 0.6, (Nt, At))
    _m = rng.random((Nt, At)) > 0.2; _m[:, 0] = True
    mk = {"q10": jnp.asarray(_q50 * np.maximum(1 - 0.5 * _rs, 0.05), jnp.float32), "q50": jnp.asarray(_q50, jnp.float32),
          "q90": jnp.asarray(_q50 * (1 + 0.8 * _rs), jnp.float32), "mask_obs": jnp.asarray(_m),
          "obs_action": jnp.asarray(np.array([rng.choice(np.flatnonzero(r)) for r in _m]), jnp.int32),
          "obs_country_idx": jnp.asarray(rng.integers(0, C, Nt), jnp.int32)}
    STEPS = 50
else:
    data = load_data(DATA_DIR)
    countries = list(data["countries"]); C = len(countries)
    mk = model_kwargs(data); m_np = np.asarray(data["m_c"], np.float32)
m_c = jnp.asarray(m_np, jnp.float32)
q10, q50, q90 = mk["q10"], mk["q50"], mk["q90"]
MU = jnp.log1p(jnp.maximum(q50, 0.0))
SIG = jnp.clip((jnp.log1p(jnp.maximum(q90, 0.0)) - jnp.log1p(jnp.maximum(q10, 0.0))) / (2 * 1.2816), 0.01, 5.0)
SIG2 = SIG * SIG
MASK = mk["mask_obs"]; ACT = mk["obs_action"]; CI = mk["obs_country_idx"]
N, A = q50.shape
OH = jax.nn.one_hot(CI, C, dtype=jnp.float32)
say(f"devices {jax.devices()}  N {N}  A {A}  steps {STEPS}")


def per_obs(vec_c):                      # (C,) or (C,k) -> (N,) or (N,k), no scatter in the backward pass
    return jnp.dot(OH, vec_c, precision=HI)


from jax.scipy.stats import norm as _norm  # noqa: E402


def model(family, asc=True):
    with plate("countries", C):
        lb = sample("lb_c", dist.Normal(1.0, 2.0))
        if family in ("roy", "ruin"):
            s_lat = sample("s_lat_c", dist.Normal(-1.0, 1.5))
        if family == "ruin":
            llam = sample("llam_c", dist.Normal(0.0, 1.5))
        if family == "kfree":
            wk = sample("wk_c", dist.Normal(0.0, 2.0).expand([2]).to_event(1))
        if family == "logfree":
            wf = sample("wfree_c", dist.Normal(0.0, 2.0).expand([3]).to_event(1))
    beta_c = deterministic("beta_c", jnp.exp(jnp.clip(lb, -4.0, 6.0)))
    if family == "logev":
        V = MU + 0.5 * SIG2
    elif family == "roy":              # safety-first: max (mu - log tau) / sigma  <=> min P(y < tau)
        s_c = deterministic("s_c", 3.0 * jax.nn.sigmoid(s_lat))
        ltau = per_obs(jnp.log(s_c * m_c))[:, None]
        V = (MU - ltau) / SIG
    elif family == "ruin":             # mu - lam * P(y < tau)
        s_c = deterministic("s_c", 3.0 * jax.nn.sigmoid(s_lat))
        lam = deterministic("lam_c", jnp.exp(llam))
        ltau = per_obs(jnp.log(s_c * m_c))[:, None]
        V = MU - per_obs(lam)[:, None] * _norm.cdf((ltau - MU) / SIG)
    elif family == "kfree":            # a mu + b sigma (Kataoka-type, free ratio)
        deterministic("w_c", wk); w = per_obs(wk)
        V = w[:, 0:1] * MU + w[:, 1:2] * SIG
    elif family == "logfree":
        deterministic("w_c", wf); w = per_obs(wf)
        V = w[:, 0:1] * MU + w[:, 1:2] * SIG + w[:, 2:3] * SIG2
    else:
        raise ValueError(family)
    logits = per_obs(beta_c)[:, None] * center_reward(V, MASK)
    if asc:
        a = sample("asc_c", dist.Normal(0.0, 3.0).expand([C, A - 1]).to_event(2))
        alpha_c = jnp.concatenate([jnp.zeros((C, 1)), a], axis=1)
        logits = logits + per_obs(alpha_c)
    logits = jnp.where(MASK, logits, INFEASIBLE_LOGIT)
    with plate("observations", N):
        sample("obs_action", dist.Categorical(logits=logits), obs=ACT)


def fit(family, asc=True, seed=0):
    guide = AutoMultivariateNormal(model, init_loc_fn=numpyro.infer.init_to_median())
    svi = SVI(model, guide, numpyro.optim.Adam(1e-2), Trace_ELBO())
    t0 = time.time()
    res = svi.run(jax.random.PRNGKey(seed), STEPS, progress_bar=False, family=family, asc=asc)
    losses = np.asarray(res.losses)
    med = guide.median(res.params)
    _, tr = log_density(numpyro.handlers.substitute(model, data=med), (), {"family": family, "asc": asc}, {})
    ll = float(tr["obs_action"]["fn"].log_prob(tr["obs_action"]["value"]).sum())
    det = {k: np.asarray(v["value"]) for k, v in tr.items() if v["type"] == "deterministic"}
    say(f"{family:12s} asc={asc}  {STEPS} steps {time.time()-t0:4.0f}s  loss(last500) {losses[-500:].mean():.0f}  log-lik {ll:.1f}")
    return {"loglik": ll, "loss": float(losses[-500:].mean()), "det": {k: v.tolist() for k, v in det.items()}}


R = {}
for fam in ("logev", "kfree", "roy", "ruin", "logfree"):
    R[fam] = fit(fam)
R["roy_noasc"] = fit("roy", asc=False)

say("")
say(f"{'family':14s} {'log-lik':>11s} {'gain/obs vs logev':>15s}   per-country parameters")
for fam, r in R.items():
    g = (r["loglik"] - R["logev"]["loglik"]) / N
    d = r["det"]
    if "lam_c" in d:
        par = "lam=" + " ".join(f"{x:.2f}" for x in d["lam_c"]) + "  tau/m=" + " ".join(f"{x:.2f}" for x in d["s_c"])
    elif "s_c" in d:
        par = "tau/m=" + " ".join(f"{x:.2f}" for x in d["s_c"])
    elif "w_c" in d:
        par = "w=" + " | ".join(" ".join(f"{x:+.2f}" for x in row) for row in d["w_c"])
    else:
        par = ""
    say(f"{fam:14s} {r['loglik']:11.1f} {g:15.4f}   beta=" + " ".join(f"{x:.2f}" for x in d["beta_c"]) + "   " + par)
say(f"countries: {countries}   m_c: {np.round(m_np, 1).tolist()}")
o = OUT_DIR / "exp_utility_sf"; o.mkdir(parents=True, exist_ok=True)
json.dump({"countries": countries, "N": N, "steps": STEPS, "results": R}, open(o / "exp_utility_sf.json", "w"), indent=2)
say(f"written {o/'exp_utility_sf.json'}")
