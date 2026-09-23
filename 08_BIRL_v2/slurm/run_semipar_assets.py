"""NUTS estimation of the semi-parametric crop-choice model with household-asset heterogeneity.

  V_ia   = a_g * mu_ia + b_g * sigma_ia + c_g * sigma_ia^2,   g = country x within-country asset tercile
  logits = center_feasible(V) + ASC_{c,a}   (ASC by country x action, action 0 = reference; same as run_semipar.py)
  priors: a, b, c ~ N(0, 3) per group;  ASC ~ N(0, 3).  No beta.

Terciles are cut within country on hh_asset_index (33.3 / 66.7 percentiles of the observations).
Observations with a missing asset index form a separate "na" group per country when the country has
>= 500 such observations, otherwise they are dropped (counts are reported).  Centring of V follows
run_semipar.py (center_reward over the feasible set) so the two runs are directly comparable; centring
does not change choice probabilities.

Outputs (outputs/semipar_assets/): summary.csv, derived.csv, convergence.txt, results.json, posterior.npz, run.log.
Run from 08_BIRL_v2/:  python3 slurm/run_semipar_assets.py [--warmup W --samples S --chains K --skip-timing --dense]
"""
import os, sys, time, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("BIRL_HOST_DEVICES", "1")

ap = argparse.ArgumentParser()
ap.add_argument("--warmup", type=int, default=1000); ap.add_argument("--samples", type=int, default=1000)
ap.add_argument("--chains", type=int, default=4); ap.add_argument("--skip-timing", action="store_true")
ap.add_argument("--dense", action="store_true"); ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--min-na", type=int, default=500, help="keep a per-country 'na' asset group only above this size")
args = ap.parse_args()

from src.config import log, DATA_DIR, OUT_DIR, add_file_log  # noqa: E402
import jax, jax.numpy as jnp, numpy as np, pandas as pd  # noqa: E402
import numpyro  # noqa: E402
import numpyro.distributions as dist  # noqa: E402
from numpyro import sample, deterministic, plate  # noqa: E402
from numpyro.infer import MCMC, NUTS  # noqa: E402
from src.data_loader import load_data, model_kwargs  # noqa: E402
from src.models import center_reward, INFEASIBLE_LOGIT  # noqa: E402
from src.diagnostics import rhat_rank, ess_bulk, ess_tail  # noqa: E402

HI = jax.lax.Precision.HIGHEST
RUN_DIR = OUT_DIR / "semipar_assets"; RUN_DIR.mkdir(parents=True, exist_ok=True)
add_file_log(RUN_DIR / "run.log")
TERCILE_NAMES = ["T1_poor", "T2_mid", "T3_rich"]


def say(msg):
    log.info(msg)


TOY = bool(os.environ.get("EXP_TOY"))
if TOY:
    rng = np.random.default_rng(0); Nt, At, C = 600, 6, 3
    countries = ["A", "B", "C"]
    _q50 = rng.lognormal(3, 0.7, (Nt, At)); _rs = rng.lognormal(0, 0.6, (Nt, At))
    _m = rng.random((Nt, At)) > 0.2; _m[:, 0] = True
    mk = {"q10": jnp.asarray(_q50 * np.maximum(1 - 0.5 * _rs, 0.05), jnp.float32), "q50": jnp.asarray(_q50, jnp.float32),
          "q90": jnp.asarray(_q50 * (1 + 0.8 * _rs), jnp.float32), "mask_obs": jnp.asarray(_m),
          "obs_action": jnp.asarray(np.array([rng.choice(np.flatnonzero(r)) for r in _m]), jnp.int32),
          "obs_country_idx": jnp.asarray(rng.integers(0, C, Nt), jnp.int32)}
    asset = rng.normal(size=Nt); asset[rng.random(Nt) < 0.05] = np.nan
    args.warmup, args.samples = 30, 30
else:
    data = load_data(DATA_DIR); countries = list(data["countries"]); C = len(countries)
    mk = model_kwargs(data)
    df = pd.read_parquet(DATA_DIR / "birl_sample.parquet", columns=["hh_asset_index"])
    asset = pd.to_numeric(df["hh_asset_index"], errors="coerce").values.astype(float)
    assert len(asset) == int(mk["obs_action"].shape[0])

# ── groups: country x within-country asset tercile (+ optional per-country 'na') ──
ci_np = np.asarray(mk["obs_country_idx"]); N_all = len(ci_np)
terc = np.full(N_all, -1, int)
for i in range(C):
    sel = (ci_np == i) & np.isfinite(asset)
    if sel.sum() == 0:
        continue
    lo, hi = np.nanpercentile(asset[sel], [100 / 3, 200 / 3])
    terc[sel] = np.where(asset[sel] <= lo, 0, np.where(asset[sel] <= hi, 1, 2))
group_names, gidx = [], np.full(N_all, -1, int)
for i, cn in enumerate(countries):
    for t in range(3):
        sel = (ci_np == i) & (terc == t)
        gidx[sel] = len(group_names); group_names.append((cn, TERCILE_NAMES[t]))
    sel_na = (ci_np == i) & (terc == -1)
    if sel_na.sum() >= args.min_na:
        gidx[sel_na] = len(group_names); group_names.append((cn, "na"))
keep = gidx >= 0
n_drop = int((~keep).sum())
G = len(group_names)
say(f"groups G={G}: " + ", ".join(f"{c}/{t}={int(((gidx == g)).sum())}" for g, (c, t) in enumerate(group_names)) + f"; dropped (asset missing, small) {n_drop}")

def sub(x):
    return x[jnp.asarray(np.flatnonzero(keep))] if hasattr(x, "shape") and x.shape[0] == N_all else x

q10, q50, q90 = sub(mk["q10"]), sub(mk["q50"]), sub(mk["q90"])
MASK = sub(mk["mask_obs"]); ACT = sub(mk["obs_action"]); CI = sub(mk["obs_country_idx"])
GI = jnp.asarray(gidx[keep], jnp.int32)
MU = jnp.log1p(jnp.maximum(q50, 0.0))
SIG = jnp.clip((jnp.log1p(jnp.maximum(q90, 0.0)) - jnp.log1p(jnp.maximum(q10, 0.0))) / (2 * 1.2816), 0.01, 5.0)
SIG2 = SIG * SIG
N, A = q50.shape
OH_C = jax.nn.one_hot(CI, C, dtype=jnp.float32)
OH_G = jax.nn.one_hot(GI, G, dtype=jnp.float32)
per_c = lambda v: jnp.dot(OH_C, v, precision=HI)
per_g = lambda v: jnp.dot(OH_G, v, precision=HI)
say(f"semipar_assets  N={N} (of {N_all}) A={A} C={C} G={G}  devices={jax.devices()}  toy={TOY}")


def model():
    with plate("groups", G):
        a = sample("a_g", dist.Normal(0.0, 3.0)); b = sample("b_g", dist.Normal(0.0, 3.0)); c = sample("c_g", dist.Normal(0.0, 3.0))
    asc = sample("asc_c", dist.Normal(0.0, 3.0).expand([C, A - 1]).to_event(2))
    V = per_g(a)[:, None] * MU + per_g(b)[:, None] * SIG + per_g(c)[:, None] * SIG2
    alpha = per_c(jnp.concatenate([jnp.zeros((C, 1)), asc], axis=1))
    logits = jnp.where(MASK, center_reward(V, MASK) + alpha, INFEASIBLE_LOGIT)
    with plate("observations", N):
        sample("obs_action", dist.Categorical(logits=logits), obs=ACT)


def kernel(dense):
    return NUTS(model, target_accept_prob=0.8, max_tree_depth=10, dense_mass=dense,
                init_strategy=numpyro.infer.init_to_median())


def hpdi(x, prob=0.89):
    x = np.sort(np.asarray(x).ravel()); n = len(x); k = max(int(np.floor(prob * n)), 1)
    w = x[k:] - x[:n - k]; i = int(np.argmin(w)); return float(x[i]), float(x[i + k])


# ── [1] timing ──
dense = args.dense; warm, samp = args.warmup, args.samples
if not args.skip_timing and not TOY:
    m = MCMC(kernel(dense), num_warmup=50, num_samples=20, num_chains=args.chains, chain_method="vectorized", progress_bar=False)
    t0 = time.time(); m.warmup(jax.random.PRNGKey(1), extra_fields=("num_steps",), collect_warmup=True); tw = time.time() - t0
    lw = float(np.mean(np.asarray(m.get_extra_fields()["num_steps"])))
    t0 = time.time(); m.run(jax.random.PRNGKey(2), extra_fields=("num_steps",)); ts = time.time() - t0
    ls = float(np.mean(np.asarray(m.get_extra_fields()["num_steps"])))
    proj_h = (tw / 50 * warm + ts / 20 * samp) / 3600
    say(f"[TIMING] warmup {tw/50:.2f} s/draw (leapfrog {lw:.0f}, incl compile), sampling {ts/20:.2f} s/draw (leapfrog {ls:.0f}); projected {proj_h:.2f} h")
    if proj_h > 2:
        dense, warm, samp = True, 600, 600
        say("[TIMING] projected > 2 h: switching to dense_mass=True and 600+600")

# ── [2] full run ──
m = MCMC(kernel(dense), num_warmup=warm, num_samples=samp, num_chains=args.chains, chain_method="vectorized", progress_bar=False)
t0 = time.time(); m.run(jax.random.PRNGKey(args.seed), extra_fields=("diverging", "num_steps"))
say(f"[RUN] {args.chains} x ({warm}+{samp}) dense={dense}: {(time.time()-t0)/60:.1f} min")
S = {k: np.asarray(v) for k, v in m.get_samples(group_by_chain=True).items()}
ex = m.get_extra_fields(group_by_chain=True); div = np.asarray(ex["diverging"]); nsteps = np.asarray(ex["num_steps"])
np.savez_compressed(RUN_DIR / "posterior.npz", **S, diverging=div, num_steps=nsteps, group_country=np.array([g[0] for g in group_names]),
                    group_tercile=np.array([g[1] for g in group_names]))
say(f"divergences {int(div.sum())}/{div.size} ({div.mean():.4f}); mean leapfrog/draw {nsteps.mean():.1f}")

# ── summary per group ──
n_g = np.bincount(gidx[keep], minlength=G)
rows = []
for site, name in (("a_g", "a"), ("b_g", "b"), ("c_g", "c")):
    for g, (cn, tn) in enumerate(group_names):
        x = S[site][:, :, g]; lo, hi = hpdi(x)
        rows.append(dict(param=name, country=cn, tercile=tn, n_obs=int(n_g[g]), median=float(np.median(x)), sd=float(x.std()),
                         hpdi_lo=lo, hpdi_hi=hi, r_hat=float(rhat_rank(x)), ess_bulk=float(ess_bulk(x)), ess_tail=float(ess_tail(x))))
summ = pd.DataFrame(rows); summ.to_csv(RUN_DIR / "summary.csv", index=False)
asc_rhat = np.array([[rhat_rank(S["asc_c"][:, :, i, j]) for j in range(A - 1)] for i in range(C)])
asc_ess = np.array([[ess_bulk(S["asc_c"][:, :, i, j]) for j in range(A - 1)] for i in range(C)])

# ── derived: poor vs rich contrasts, marginal sigma effect at 0.5 ──
a, b, c = (S[k].reshape(-1, G) for k in ("a_g", "b_g", "c_g"))
drows, mrows = [], []
for g, (cn, tn) in enumerate(group_names):
    me = b[:, g] + 2 * c[:, g] * 0.5; lo, hi = hpdi(me)
    mrows.append(dict(country=cn, tercile=tn, n_obs=int(n_g[g]), marg_sigma_05=float(np.median(me)), lo=lo, hi=hi,
                      a=float(np.median(a[:, g])), b=float(np.median(b[:, g])), c=float(np.median(c[:, g]))))
for cn in countries:
    idx = {tn: g for g, (c_, tn) in enumerate(group_names) if c_ == cn}
    if "T1_poor" not in idx or "T3_rich" not in idx:
        continue
    gp, gr = idx["T1_poor"], idx["T3_rich"]
    d = {"country": cn}
    for nm, arr in (("a", a), ("b", b), ("c", c)):
        diff = arr[:, gp] - arr[:, gr]; lo, hi = hpdi(diff)
        d[f"d{nm}_poor_minus_rich"] = float(np.median(diff)); d[f"d{nm}_lo"] = lo; d[f"d{nm}_hi"] = hi; d[f"p_d{nm}_gt0"] = float(np.mean(diff > 0))
    mp = b[:, gp] + c[:, gp]; mr = b[:, gr] + c[:, gr]; diff = mp - mr; lo, hi = hpdi(diff)
    d.update(dmarg05_poor_minus_rich=float(np.median(diff)), dmarg05_lo=lo, dmarg05_hi=hi, p_dmarg05_gt0=float(np.mean(diff > 0)))
    drows.append(d)
pd.DataFrame(mrows).to_csv(RUN_DIR / "marginal_by_group.csv", index=False)
pd.DataFrame(drows).to_csv(RUN_DIR / "derived.csv", index=False)

# ── log-lik at the posterior median (all sites) and structural gain vs the country-level semipar run ──
def loglik(a_, b_, c_, asc_):
    V = per_g(a_)[:, None] * MU + per_g(b_)[:, None] * SIG + per_g(c_)[:, None] * SIG2
    alpha = per_c(jnp.concatenate([jnp.zeros((C, 1)), asc_], axis=1))
    logp = jax.nn.log_softmax(jnp.where(MASK, center_reward(V, MASK) + alpha, INFEASIBLE_LOGIT), axis=-1)
    return jnp.sum(jnp.take_along_axis(logp, ACT[:, None], axis=-1))


med = {k: jnp.asarray(np.median(v.reshape((-1,) + v.shape[2:]), axis=0)) for k, v in S.items()}
ll = float(jax.jit(loglik)(med["a_g"], med["b_g"], med["c_g"], med["asc_c"]))
ref = OUT_DIR / "semipar" / "results.json"
ref_ll = ref_null = float("nan")
if ref.exists():
    rj = json.load(open(ref)); ref_ll, ref_null = rj.get("ll_semipar", float("nan")), rj.get("ll_null", float("nan"))
say(f"log-lik at posterior median {ll:.1f} on N={N}; country-level semipar (N={N_all}) {ref_ll:.1f}; ASC null {ref_null:.1f}")

# ── convergence.txt ──
with open(RUN_DIR / "convergence.txt", "w") as f:
    f.write(f"semipar_assets: {args.chains} x ({warm}+{samp}) dense={dense}; N={N} (dropped {n_drop}); G={G}\n")
    f.write(f"Divergences: {int(div.sum())}/{div.size} ({div.mean():.4f})  [{'PASS' if div.mean() < 0.01 else 'FAIL'}]\n")
    f.write(f"R-hat max (a,b,c): {summ['r_hat'].max():.4f}   ESS bulk min: {summ['ess_bulk'].min():.0f}   ASC r_hat max {np.nanmax(asc_rhat):.4f}, ess min {np.nanmin(asc_ess):.0f}\n\n")
    for r in rows:
        f.write(f"  {r['param']}[{r['country']:9s} {r['tercile']:8s} n={r['n_obs']:6d}] {r['median']:+8.3f} [{r['hpdi_lo']:+8.3f}, {r['hpdi_hi']:+8.3f}]  r_hat={r['r_hat']:.3f} ess={r['ess_bulk']:.0f}\n")
    f.write("\npoor - rich contrasts (median [89% HPDI], P(>0)):\n")
    for d in drows:
        f.write(f"  {d['country']:9s} da={d['da_poor_minus_rich']:+.3f} [{d['da_lo']:+.3f},{d['da_hi']:+.3f}] P={d['p_da_gt0']:.2f}  "
                f"db={d['db_poor_minus_rich']:+.3f} P={d['p_db_gt0']:.2f}  dc={d['dc_poor_minus_rich']:+.3f} P={d['p_dc_gt0']:.2f}  "
                f"dmarg(0.5)={d['dmarg05_poor_minus_rich']:+.3f} [{d['dmarg05_lo']:+.3f},{d['dmarg05_hi']:+.3f}] P={d['p_dmarg05_gt0']:.2f}\n")
for line in open(RUN_DIR / "convergence.txt"):
    say(line.rstrip())

json.dump({"countries": countries, "groups": group_names, "n_obs_group": n_g.tolist(), "N": N, "N_all": N_all, "dropped": n_drop,
           "chains": args.chains, "warmup": warm, "samples": samp, "dense_mass": dense,
           "divergence_rate": float(div.mean()), "mean_leapfrog": float(nsteps.mean()),
           "r_hat_max_abc": float(summ["r_hat"].max()), "ess_min_abc": float(summ["ess_bulk"].min()),
           "asc_r_hat_max": float(np.nanmax(asc_rhat)), "asc_ess_min": float(np.nanmin(asc_ess)),
           "ll_median_params": ll, "ll_semipar_country": ref_ll, "ll_null_country": ref_null,
           "gain_vs_country_semipar_per_obs": (ll - ref_ll) / N if np.isfinite(ref_ll) else None,
           "summary": rows, "derived": drows, "marginal_by_group": mrows}, open(RUN_DIR / "results.json", "w"), indent=2)
say(f"written {RUN_DIR}")
