"""Utility-family experiments on the REAL data (SVI point estimates + log-lik).

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
Prints log-lik, gain/obs vs ev, per-country parameters.  Writes
outputs/exp_utility/exp_utility.json.
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
NODES = jnp.stack([q10, 0.5 * (q10 + q50), q50, 0.5 * (q50 + q90), q90], axis=0)   # (5, N, A)
MASK = mk["mask_obs"]; ACT = mk["obs_action"]; CI = mk["obs_country_idx"]
N, A = q50.shape
OH = jax.nn.one_hot(CI, C, dtype=jnp.float32)
say(f"devices {jax.devices()}  N {N}  A {A}  steps {STEPS}")


def per_obs(vec_c):                      # (C,) or (C,k) -> (N,) or (N,k), no scatter in the backward pass
    return jnp.dot(OH, vec_c, precision=HI)


def model(family, asc=True):
    with plate("countries", C):
        lb = sample("lb_c", dist.Normal(1.0, 2.0))
        th = sample("th_c", dist.Normal(0.0, 1.0)) if family in ("rdeu_pow", "rdeu_prelec") else None
        if family == "es":
            llam = sample("llam_c", dist.Normal(0.0, 1.5))
            s_lat = sample("s_lat_c", dist.Normal(-1.0, 1.5))
        if family == "free5":
            wfree = sample("wfree_c", dist.Normal(0.0, 1.0).expand([5]).to_event(1))   # (C,5)
    beta_c = deterministic("beta_c", jnp.exp(jnp.clip(lb, -4.0, 6.0)))

    if family == "ev":
        pi = jnp.broadcast_to(W5, (C, 5))
    elif family == "rdeu_pow":
        delta = deterministic("delta_c", jnp.exp(th))                                   # (C,)
        Wc = F5[None, :] ** delta[:, None]                                               # (C,5)
        pi = jnp.concatenate([Wc[:, :1], Wc[:, 1:] - Wc[:, :-1]], axis=1)
    elif family == "rdeu_prelec":
        alpha = deterministic("alpha_c", jnp.exp(th))
        nl = -jnp.log(F5)[None, :]                                                       # (1,5), last = 0
        Wc = jnp.exp(-(jnp.maximum(nl, 1e-12) ** alpha[:, None]))
        Wc = Wc.at[:, -1].set(1.0)
        pi = jnp.concatenate([Wc[:, :1], Wc[:, 1:] - Wc[:, :-1]], axis=1)
    elif family == "free5":
        pi = deterministic("pi_c", wfree)                                                # free signed weights
    elif family == "es":
        lam = deterministic("lam_c", jnp.exp(llam))
        s_c = deterministic("s_c", 0.6 * jax.nn.sigmoid(s_lat))
        tau_obs = per_obs(s_c * m_c)[:, None]                                            # (N,1)
        lam_obs = per_obs(lam)[:, None]
        pi = jnp.broadcast_to(W5, (C, 5))
    else:
        raise ValueError(family)

    pi_obs = per_obs(pi)                                                                 # (N,5)
    V = jnp.einsum("kna,nk->na", NODES, pi_obs)                                          # (N,A)
    if family == "es":
        short = jnp.einsum("kna,k->na", jnp.maximum(tau_obs[None] - NODES, 0.0), W5)
        V = V - lam_obs * short
    reward = V / per_obs(m_c)[:, None]
    logits = per_obs(beta_c)[:, None] * center_reward(reward, MASK)
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
for fam in ("ev", "rdeu_pow", "rdeu_prelec", "es", "free5"):
    R[fam] = fit(fam)
R["rdeu_pow_noasc"] = fit("rdeu_pow", asc=False)
R["es_noasc"] = fit("es", asc=False)

say("")
say(f"{'family':14s} {'log-lik':>11s} {'gain/obs vs ev':>15s}   per-country parameters")
for fam, r in R.items():
    g = (r["loglik"] - R["ev"]["loglik"]) / N
    d = r["det"]
    if "delta_c" in d:
        par = "delta=" + " ".join(f"{x:.2f}" for x in d["delta_c"])
    elif "alpha_c" in d:
        par = "alpha=" + " ".join(f"{x:.2f}" for x in d["alpha_c"])
    elif "lam_c" in d:
        par = "lam=" + " ".join(f"{x:.2f}" for x in d["lam_c"]) + "  s=" + " ".join(f"{x:.2f}" for x in d["s_c"])
    elif "pi_c" in d:
        par = "pi(q10..q90)=" + " | ".join(" ".join(f"{x:+.2f}" for x in row) for row in d["pi_c"])
    else:
        par = ""
    say(f"{fam:14s} {r['loglik']:11.1f} {g:15.4f}   beta=" + " ".join(f"{x:.2f}" for x in d["beta_c"]) + "   " + par)
say(f"countries: {countries}   m_c: {np.round(m_np, 1).tolist()}")
o = OUT_DIR / "exp_utility"; o.mkdir(parents=True, exist_ok=True)
json.dump({"countries": countries, "N": N, "steps": STEPS, "results": R}, open(o / "exp_utility.json", "w"), indent=2)
say(f"written {o/'exp_utility.json'}")
