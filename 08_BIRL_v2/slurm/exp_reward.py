"""Reward-specification experiments on the REAL data (SVI point estimates).

Why: the v2 baseline gives beta ~ 0.05 and only 0.012 nat/obs over a uniform
choice, i.e. the CE ranking of actions from the environment model does not
predict observed crop choices.  Two candidate reasons, both economic:
  (a) Y is harvest VALUE, not profit: high-intensity actions carry seed,
      fertilizer and hired-labour costs that are never subtracted;
  (b) no alternative-specific constants (ASC): unmodelled costs / access /
      habit are forced onto beta and rho.
Variants (all SVI, AutoMultivariateNormal, Adam 1e-2, STEPS steps):
  base            v2_country as is
  asc_g           + global ASC (27 actions, ref = action 0)
  asc_c           + country x action ASC
  cost            Y_net = Y - median input cost per (country, action)
  cost_asc_g      both
  null_uniform    beta = 0, no ASC   (log-lik of uniform over feasible)
  null_asc_g      beta = 0, global ASC  (pure choice-frequency model)
  null_asc_c      beta = 0, country ASC
Prints per variant: log-lik, gain/obs vs the matching null, and the country
table (rho, s, gamma, beta).  Writes outputs/exp_reward/exp_reward.json.
Run from 08_BIRL_v2/:  python3 slurm/exp_reward.py [steps]
"""
import os, sys, time, json
os.environ.setdefault("BIRL_HOST_DEVICES", "1")

from cropchoice.config import log, DATA_DIR, OUT_DIR  # noqa: E402
import jax, jax.numpy as jnp, numpy as np, pandas as pd  # noqa: E402
import numpyro  # noqa: E402
import numpyro.distributions as dist  # noqa: E402
from numpyro.infer import SVI, Trace_ELBO  # noqa: E402
from numpyro.infer.autoguide import AutoMultivariateNormal  # noqa: E402
from numpyro.infer.util import log_density  # noqa: E402
from cropchoice.data import load_data, model_kwargs  # noqa: E402
from cropchoice.models_v2 import v2_country, derive_country_params  # noqa: E402

STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
COST_COLS = ["seed_value_USD", "inorganic_fertilizer_value_USD", "hired_labor_value_USD"]


def say(msg):
    print(f"[exp {time.strftime('%H:%M:%S')}] {msg}", flush=True)


data = load_data(DATA_DIR)
countries = list(data["countries"])
m_c = np.asarray(data["m_c"])
base_mk = model_kwargs(data)
N = int(base_mk["obs_action"].shape[0]); A = int(base_mk["q50"].shape[1])
say(f"devices {jax.devices()}  N_obs {N}  A {A}  steps {STEPS}")

# ── cost matrix: median input cost per (country, action), fallbacks crop -> country ──
env = np.load(DATA_DIR / "env_model_output.npz", allow_pickle=True)
action_ids = np.asarray(env["action_ids"])
df = pd.read_parquet(DATA_DIR / "birl_sample.parquet",
                     columns=["country", "action_id", "action_crop"] + COST_COLS)
assert len(df) == N
cost = df[COST_COLS].fillna(0.0).sum(axis=1).astype(float)
df["cost"] = cost
col_of = {int(a): i for i, a in enumerate(action_ids)}
df["a"] = df["action_id"].map(col_of)
crop_of_action = df.groupby("a")["action_crop"].agg(lambda s: s.mode().iloc[0]).to_dict()
cost_mat = np.zeros((len(countries), A), np.float32)
n_cell = np.zeros((len(countries), A), int)
med_ca = df.groupby(["country", "a"])["cost"].median()
cnt_ca = df.groupby(["country", "a"])["cost"].size()
med_ccrop = df.groupby(["country", "action_crop"])["cost"].median()
med_c = df.groupby("country")["cost"].median()
for ci, c in enumerate(countries):
    for a in range(A):
        n = int(cnt_ca.get((c, a), 0)); n_cell[ci, a] = n
        if n >= 30:
            cost_mat[ci, a] = med_ca[(c, a)]
        elif (c, crop_of_action.get(a)) in med_ccrop.index:
            cost_mat[ci, a] = med_ccrop[(c, crop_of_action[a])]
        else:
            cost_mat[ci, a] = med_c[c]
say("median input cost (USD) per country: " +
    ", ".join(f"{c}={np.median(cost_mat[i]):.1f} (chosen-cell median {df[df.country==c]['cost'].median():.1f}, m_c {m_c[i]:.1f})"
              for i, c in enumerate(countries)))
cost_obs = cost_mat[np.asarray(base_mk["obs_country_idx"])]                     # (N, A)


def with_cost(mk):
    out = dict(mk)
    for k in ("q10", "q50", "q90"):
        out[k] = jnp.asarray(np.asarray(mk[k]) - cost_obs, jnp.float32)
    return out


cost_mk = with_cost(base_mk)


def fit(name, mk, asc=None, beta_fixed=None):
    kw = {"data": mk, "s_max": 0.6, "flat_priors": False, "noncentered": False,
          "asc": asc, "beta_fixed": beta_fixed}
    guide = AutoMultivariateNormal(v2_country, init_loc_fn=numpyro.infer.init_to_median())
    svi = SVI(v2_country, guide, numpyro.optim.Adam(1e-2), Trace_ELBO())
    t0 = time.time()
    res = svi.run(jax.random.PRNGKey(0), STEPS, progress_bar=False, **kw)
    losses = np.asarray(res.losses)
    med = guide.median(res.params)
    # log-likelihood of the observed actions at the guide median (all sites conditioned)
    ld, tr = log_density(numpyro.handlers.substitute(v2_country, data=med), (), kw, {})
    ll = float(tr["obs_action"]["fn"].log_prob(tr["obs_action"]["value"]).sum())
    est = derive_country_params({k: np.asarray(v) for k, v in med.items()}, m_c, s_max=0.6)
    if beta_fixed is not None:
        est["beta_c"] = np.full(len(m_c), float(beta_fixed))
    say(f"{name:12s} {STEPS} steps {time.time()-t0:4.0f}s  loss {losses[-500:].mean():.0f}  log-lik {ll:.1f}")
    return {"loglik": ll, "loss_last500": float(losses[-500:].mean()),
            **{k: np.asarray(v).tolist() for k, v in est.items()}}


R = {}
R["null_uniform"] = fit("null_uniform", base_mk, beta_fixed=0.0)
R["null_asc_g"] = fit("null_asc_g", base_mk, asc="global", beta_fixed=0.0)
R["null_asc_c"] = fit("null_asc_c", base_mk, asc="country", beta_fixed=0.0)
R["base"] = fit("base", base_mk)
R["asc_g"] = fit("asc_g", base_mk, asc="global")
R["asc_c"] = fit("asc_c", base_mk, asc="country")
R["cost"] = fit("cost", cost_mk)
R["cost_asc_g"] = fit("cost_asc_g", cost_mk, asc="global")
R["cost_asc_c"] = fit("cost_asc_c", cost_mk, asc="country")

NULL_OF = {"base": "null_uniform", "cost": "null_uniform", "asc_g": "null_asc_g",
           "cost_asc_g": "null_asc_g", "asc_c": "null_asc_c", "cost_asc_c": "null_asc_c"}
say("")
say(f"{'variant':12s} {'log-lik':>11s} {'gain/obs vs null':>17s}   rho (6 countries)                     s (6)                              beta (6)")
for k in ["null_uniform", "null_asc_g", "null_asc_c", "base", "asc_g", "asc_c", "cost", "cost_asc_g", "cost_asc_c"]:
    r = R[k]; g = (r["loglik"] - R[NULL_OF[k]]["loglik"]) / N if k in NULL_OF else float("nan")
    fmt = lambda v, w: " ".join(f"{x:{w}.2f}" for x in v)
    say(f"{k:12s} {r['loglik']:11.1f} {g:17.4f}   {fmt(r['rho_c'],5)}   {fmt(r['s_c'],5)}   {fmt(r['beta_c'],5)}")
say(f"countries: {countries}")
out = OUT_DIR / "exp_reward"; out.mkdir(parents=True, exist_ok=True)
json.dump({"countries": countries, "m_c": m_c.tolist(), "N": N, "steps": STEPS,
           "cost_mat": cost_mat.tolist(), "n_cell": n_cell.tolist(), "results": R},
          open(out / "exp_reward.json", "w"), indent=2)
say(f"written {out/'exp_reward.json'}")
