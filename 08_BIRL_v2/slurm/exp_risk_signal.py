"""Existence test: is there ANY risk signal in crop choice?

Part A (reduced form).  For each plot-observation, the riskiness of the chosen
action = percentile rank, among the feasible actions of that observation, of
the predicted relative spread (q90 - q10) / q50 from the environment model
(0 = safest available, 1 = riskiest available).  Regress it on long-run
rainfall CV, last-wave rainfall anomaly, reported shocks and household
controls, with country x wave fixed effects, then with household fixed
effects.  Cluster-robust SE by household.

Part B (model-consistent).  Mean-variance logit with country x action ASCs:
  logit_ia = asc[c,a] + b1[c] * q50_ia / m_c + b2[c] * (q90 - q10)_ia / m_c
MLE by BFGS, SE from the inverse Hessian.  b2 < 0 = aversion to spread beyond
what the mean explains; this is the linearised version of what rho would do.
Then b2 = g0 + g1 * z(rain CV) + g2 * z(asset index): risk aversion should be
stronger where rainfall is more variable and for poorer households.

Run from 08_BIRL_v2/:  python3 slurm/exp_risk_signal.py
"""
import os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("BIRL_HOST_DEVICES", "1")

from src.config import log, DATA_DIR, OUT_DIR  # noqa: E402
import jax, jax.numpy as jnp, numpy as np, pandas as pd  # noqa: E402
from jax.flatten_util import ravel_pytree  # noqa: E402
from jax.scipy.optimize import minimize as jsp_minimize  # noqa: E402
from src.data_loader import load_data, model_kwargs  # noqa: E402


def say(msg):
    print(f"[risk {time.strftime('%H:%M:%S')}] {msg}", flush=True)


data = load_data(DATA_DIR)
countries = list(data["countries"]); C = len(countries)
mk = model_kwargs(data)
q10, q50, q90 = (np.asarray(mk[k], np.float64) for k in ("q10", "q50", "q90"))
mask = np.asarray(mk["mask_obs"], bool)
ci = np.asarray(mk["obs_country_idx"]); act = np.asarray(mk["obs_action"])
m_c = np.asarray(data["m_c"], np.float64)
N, A = q50.shape
say(f"N {N} A {A} devices {jax.devices()}")

cols = ["country", "hh_id_merge", "wave", "year", "rainfall_10yr_cv_final", "rainfall_10yr_mean_final",
        "rainfall_growing_sum_final", "rain_shock", "drought_shock", "flood_shock", "hh_shock",
        "hh_asset_index", "ag_asset_index", "farm_size", "irrigated", "dist_market", "hh_size",
        "agro_ecological_zone"]
df = pd.read_parquet(DATA_DIR / "birl_sample.parquet", columns=cols)
assert len(df) == N
for c in cols[2:]:
    if c != "agro_ecological_zone":
        df[c] = pd.to_numeric(df[c], errors="coerce")

# ── riskiness of the chosen action ──
rel = (q90 - q10) / np.maximum(q50, 1e-3)
rel = np.where(mask, rel, np.nan)
chosen_rel = rel[np.arange(N), act]
pct = np.nanmean(rel < chosen_rel[:, None], axis=1)            # share of feasible actions safer than chosen
nfeas = mask.sum(1)
pct = np.where(nfeas > 1, pct * nfeas / (nfeas - 1), np.nan)   # rescale to [0,1] excluding self
df["risk_pct"] = pct
df["chosen_rel"] = chosen_rel
df["feas_med_rel"] = np.nanmedian(rel, axis=1)
# mean-rank of the chosen action too (does the mean explain choice?)
q50m = np.where(mask, q50, np.nan)
mean_pct = np.nanmean(q50m < q50[np.arange(N), act][:, None], axis=1)
df["mean_pct"] = np.where(nfeas > 1, mean_pct * nfeas / (nfeas - 1), np.nan)
say(f"risk_pct: mean {np.nanmean(pct):.3f} (0.5 = random w.r.t. spread); mean_pct: {np.nanmean(df['mean_pct']):.3f}")

# ── last-wave rainfall anomaly (within household) ──
df["rain_anom"] = df["rainfall_growing_sum_final"] / df["rainfall_10yr_mean_final"] - 1.0
hw = df.groupby(["hh_id_merge", "wave"])["rain_anom"].mean().reset_index().sort_values(["hh_id_merge", "wave"])
hw["rain_anom_prev"] = hw.groupby("hh_id_merge")["rain_anom"].shift(1)
df = df.merge(hw[["hh_id_merge", "wave", "rain_anom_prev"]], on=["hh_id_merge", "wave"], how="left")

# ── descriptive: risk_pct by country x rainfall-CV tercile, and by drought shock ──
say("Part A0: mean risk_pct of chosen action (0=safest feasible, 1=riskiest feasible)")
rows = []
for c in countries:
    d = df[df.country == c].dropna(subset=["risk_pct", "rainfall_10yr_cv_final"])
    if len(d) < 100:
        continue
    t = pd.qcut(d["rainfall_10yr_cv_final"], 3, labels=["lowCV", "midCV", "highCV"], duplicates="drop")
    g = d.groupby(t, observed=True)["risk_pct"].mean()
    ds = d.groupby(d["drought_shock"].fillna(0) > 0)["risk_pct"].mean()
    rows.append((c, len(d), *[g.get(k, np.nan) for k in ["lowCV", "midCV", "highCV"]],
                 ds.get(False, np.nan), ds.get(True, np.nan)))
    say(f"  {c:10s} n={len(d):6d}  CV tercile low/mid/high = {g.get('lowCV', np.nan):.3f} / {g.get('midCV', np.nan):.3f} / {g.get('highCV', np.nan):.3f}"
        f"   drought no/yes = {ds.get(False, np.nan):.3f} / {ds.get(True, np.nan):.3f}")


# ── OLS with FE and cluster-robust SE ──
def zs(x):
    x = np.asarray(x, np.float64); return (x - np.nanmean(x)) / np.nanstd(x)


def demean(M, groups):
    g, inv = np.unique(groups, return_inverse=True)
    out = M.copy()
    for j in range(M.shape[1]):
        s = np.bincount(inv, M[:, j]); n = np.bincount(inv)
        out[:, j] = M[:, j] - (s / n)[inv]
    return out


def ols_cluster(y, X, cluster, names):
    XtX = X.T @ X; b = np.linalg.solve(XtX, X.T @ y); e = y - X @ b
    g, inv = np.unique(cluster, return_inverse=True)
    S = np.zeros_like(XtX)
    Xe = X * e[:, None]
    for j in range(X.shape[1]):
        pass
    # sum over clusters of (X_g' e_g)(X_g' e_g)'
    agg = np.zeros((len(g), X.shape[1]))
    for j in range(X.shape[1]):
        agg[:, j] = np.bincount(inv, Xe[:, j], minlength=len(g))
    S = agg.T @ agg
    XtXi = np.linalg.inv(XtX)
    V = XtXi @ S @ XtXi * (len(g) / (len(g) - 1))
    se = np.sqrt(np.diag(V))
    return [(n, float(bb), float(ss), float(bb / ss)) for n, bb, ss in zip(names, b, se)]


def report(title, res, n, ncl):
    say(f"{title}  (N={n}, clusters={ncl})")
    for n_, b, s, t in res:
        flag = "***" if abs(t) > 3.29 else "**" if abs(t) > 2.58 else "*" if abs(t) > 1.96 else ""
        say(f"    {n_:22s} b={b:+.4f}  se={s:.4f}  t={t:+6.2f} {flag}")


REG = ["z_raincv", "rain_anom_prev", "drought_shock", "flood_shock", "z_asset", "log_farm", "irrigated", "log_distmkt"]
df["z_raincv"] = zs(df["rainfall_10yr_cv_final"])
df["z_asset"] = zs(df["hh_asset_index"])
df["log_farm"] = np.log1p(df["farm_size"].clip(lower=0))
df["log_distmkt"] = np.log1p(df["dist_market"].clip(lower=0))
df["drought_shock"] = (df["drought_shock"].fillna(0) > 0).astype(float)
df["flood_shock"] = (df["flood_shock"].fillna(0) > 0).astype(float)
df["irrigated"] = (df["irrigated"].fillna(0) > 0).astype(float)
df["cw"] = df["country"].astype(str) + "_" + df["wave"].astype(str)

out = {"A0": rows, "A1": {}, "A2": {}, "A1_country": {}}
d = df.dropna(subset=["risk_pct"] + REG).copy()
y = demean(d[["risk_pct"]].values.astype(float), d["cw"].values)[:, 0]
X = demean(d[REG].values.astype(float), d["cw"].values)
res = ols_cluster(y, X, d["hh_id_merge"].values, REG)
report("Part A1: risk_pct ~ X, country x wave FE, cluster hh", res, len(d), d["hh_id_merge"].nunique())
out["A1"] = res
# per-country
for c in countries:
    dc = d[d.country == c]
    if len(dc) < 500:
        continue
    yc = demean(dc[["risk_pct"]].values.astype(float), dc["cw"].values)[:, 0]
    Xc = demean(dc[REG].values.astype(float), dc["cw"].values)
    keep = Xc.std(0) > 1e-9
    r = ols_cluster(yc, Xc[:, keep], dc["hh_id_merge"].values, [n for n, k in zip(REG, keep) if k])
    out["A1_country"][c] = r
    say(f"  {c:10s} " + "  ".join(f"{n}={b:+.3f}(t{t:+.1f})" for n, b, s, t in r if n in ("z_raincv", "rain_anom_prev", "drought_shock", "z_asset")))
# household FE (time-varying regressors only) + wave FE
REG2 = ["rain_anom_prev", "drought_shock", "flood_shock", "z_asset", "log_farm", "irrigated"]
d2 = df.dropna(subset=["risk_pct"] + REG2).copy()
d2 = d2[d2.groupby("hh_id_merge")["wave"].transform("nunique") >= 2]
y2 = demean(demean(d2[["risk_pct"]].values.astype(float), d2["hh_id_merge"].values), d2["cw"].values)[:, 0]
X2 = demean(demean(d2[REG2].values.astype(float), d2["hh_id_merge"].values), d2["cw"].values)
res2 = ols_cluster(y2, X2, d2["hh_id_merge"].values, REG2)
report("Part A2: risk_pct ~ X, household FE + country x wave FE, cluster hh", res2, len(d2), d2["hh_id_merge"].nunique())
out["A2"] = res2

# ── Part B: mean-variance logit with country x action ASCs, MLE + Hessian SE ──
mo = m_c[ci]
Xmean = jnp.asarray(q50 / mo[:, None], jnp.float32)
Xspr = jnp.asarray((q90 - q10) / mo[:, None], jnp.float32)
X10 = jnp.asarray(q10 / mo[:, None], jnp.float32); X90 = jnp.asarray(q90 / mo[:, None], jnp.float32)
maskj = jnp.asarray(mask); actj = jnp.asarray(act); cij = jnp.asarray(ci)
oh = jax.nn.one_hot(cij, C, dtype=jnp.float32)
zcv = jnp.asarray(np.nan_to_num(zs(df["rainfall_10yr_cv_final"]), nan=0.0), jnp.float32)
zas = jnp.asarray(np.nan_to_num(zs(df["hh_asset_index"]), nan=0.0), jnp.float32)


ADAM_STEPS = 5000
ADAM_LR = 0.05


def adam(loss, gradf, x, steps, lr, b1=0.9, b2=0.999, eps=1e-8):
    """Plain Adam on a flat parameter vector (jax.scipy BFGS is unreliable in float32)."""
    @jax.jit
    def step(x, m, v, t):
        g = gradf(x)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        mh = m / (1 - b1 ** t); vh = v / (1 - b2 ** t)
        return x - lr * mh / (jnp.sqrt(vh) + eps), m, v
    m = jnp.zeros_like(x); v = jnp.zeros_like(x)
    for t in range(1, steps + 1):
        x, m, v = step(x, m, v, float(t))
    return x


def make_loglik(spec):
    def ll(p):
        asc = jnp.concatenate([jnp.zeros((C, 1)), p["asc"]], axis=1)         # (C, A)
        alpha = jnp.dot(oh, asc, precision=jax.lax.Precision.HIGHEST)          # (N, A)
        b1 = jnp.dot(oh, p["b1"], precision=jax.lax.Precision.HIGHEST)[:, None]
        if spec in ("q3", "q3_het"):
            w10 = jnp.dot(oh, p["w10"], precision=jax.lax.Precision.HIGHEST)
            w90 = jnp.dot(oh, p["w90"], precision=jax.lax.Precision.HIGHEST)
            if spec == "q3_het":   # downside weight shifts with rain CV / assets
                w10 = w10 + p["h1"] * zcv + p["h2"] * zas
            logits = jnp.where(maskj, alpha + w10[:, None] * X10 + b1 * Xmean + w90[:, None] * X90, -1e10)
            logp = jax.nn.log_softmax(logits, axis=-1)
            return jnp.sum(jnp.take_along_axis(logp, actj[:, None], axis=-1))
        if spec == "mean":
            b2 = 0.0
        elif spec == "mv":
            b2 = jnp.dot(oh, p["b2"], precision=jax.lax.Precision.HIGHEST)[:, None]
        else:  # "mv_het": b2 = g0_c + g1 z_cv + g2 z_asset
            b2 = (jnp.dot(oh, p["b2"], precision=jax.lax.Precision.HIGHEST) + p["g1"] * zcv + p["g2"] * zas)[:, None]
        logits = jnp.where(maskj, alpha + b1 * Xmean + b2 * Xspr, -1e10)
        logp = jax.nn.log_softmax(logits, axis=-1)
        return jnp.sum(jnp.take_along_axis(logp, actj[:, None], axis=-1))
    return ll


def fit(spec, init=None):
    p0 = {"asc": jnp.zeros((C, A - 1)), "b1": jnp.zeros((C,))}
    if spec != "mean":
        p0["b2"] = jnp.zeros((C,))
    if spec == "mv_het":
        p0["g1"] = jnp.zeros(()); p0["g2"] = jnp.zeros(())
    if spec in ("q3", "q3_het"):
        p0.pop("b2", None); p0["w10"] = jnp.zeros((C,)); p0["w90"] = jnp.zeros((C,))
    if spec == "q3_het":
        p0["h1"] = jnp.zeros(()); p0["h2"] = jnp.zeros(())
    if init is not None:
        for k in init:
            if k in p0:
                p0[k] = init[k]
    flat0, unravel = ravel_pytree(p0)
    ll = make_loglik(spec)
    loss = jax.jit(lambda f: -ll(unravel(f)) / N + 1e-4 * jnp.sum(unravel(f)["asc"] ** 2) / N)
    gradf = jax.jit(jax.grad(loss))
    t0 = time.time()
    f = adam(loss, gradf, flat0, steps=ADAM_STEPS, lr=ADAM_LR)
    gnorm = float(jnp.linalg.norm(gradf(f)))
    r = type("R", (), {"success": gnorm < 1e-4, "nit": ADAM_STEPS})()
    nll = lambda f: -ll(unravel(f))
    g = jax.jit(jax.grad(nll))
    hvp = jax.jit(lambda f, v: jax.jvp(g, (f,), (v,))[1])       # one gradient pass per direction (low memory)
    eye = jnp.eye(f.shape[0], dtype=f.dtype)
    H = jnp.stack([hvp(f, eye[i]) for i in range(f.shape[0])])
    cov = jnp.linalg.pinv(0.5 * (H + H.T)); se = unravel(jnp.sqrt(jnp.clip(jnp.diag(cov), 0)))
    est = unravel(f); llv = float(ll(est))
    say(f"Part B [{spec}]: grad-norm {gnorm:.2e} (converged={bool(r.success)}) steps={int(r.nit)} log-lik {llv:.1f}  ({time.time()-t0:.0f}s)")
    return est, se, llv


estM, seM, llM = fit("mean")
estV, seV, llV = fit("mv", init=estM)
estH, seH, llH = fit("mv_het", init=estV)
say(f"  LR test spread terms (6 df): 2*dLL = {2*(llV-llM):.1f};  heterogeneity (2 df): {2*(llH-llV):.1f}")
say(f"  {'country':10s} {'b1 mean':>14s} {'b2 spread':>16s}")
B = {}
for i, c in enumerate(countries):
    say(f"  {c:10s} {float(estV['b1'][i]):+7.3f}±{float(seV['b1'][i]):.3f}   {float(estV['b2'][i]):+7.3f}±{float(seV['b2'][i]):.3f}  t={float(estV['b2'][i]/max(float(seV['b2'][i]),1e-9)):+.1f}")
    B[c] = {"b1": float(estV["b1"][i]), "b1_se": float(seV["b1"][i]), "b2": float(estV["b2"][i]), "b2_se": float(seV["b2"][i])}
say(f"  heterogeneity: g1 (z rain CV) = {float(estH['g1']):+.4f}±{float(seH['g1']):.4f}   g2 (z asset) = {float(estH['g2']):+.4f}±{float(seH['g2']):.4f}")
estQ, seQ, llQ = fit("q3", init=estM)
estQH, seQH, llQH = fit("q3_het", init=estQ)
say(f"Part B [q3]: weights on q10 / q50 / q90 per m_c (risk-neutral 5-point rule = 0.2 / 0.6 / 0.2 x beta; aversion => w10 > w90).  LR vs mean-only (12 df): {2*(llQ-llM):.1f}; het (2 df): {2*(llQH-llQ):.1f}")
Q = {}
for i, c in enumerate(countries):
    w10, w50, w90 = float(estQ["w10"][i]), float(estQ["b1"][i]), float(estQ["w90"][i])
    s10, s50, s90 = float(seQ["w10"][i]), float(seQ["b1"][i]), float(seQ["w90"][i])
    d = w10 - w90; sd = (s10**2 + s90**2) ** 0.5
    say(f"  {c:10s} w10={w10:+.3f}±{s10:.3f}  w50={w50:+.3f}±{s50:.3f}  w90={w90:+.3f}±{s90:.3f}   w10-w90={d:+.3f} (t={d/max(sd,1e-9):+.1f})")
    Q[c] = {"w10": w10, "w50": w50, "w90": w90, "se": [s10, s50, s90], "diff_t": d / max(sd, 1e-9)}
say(f"  heterogeneity of w10: h1 (z rain CV) = {float(estQH['h1']):+.4f}±{float(seQH['h1']):.4f}   h2 (z asset) = {float(estQH['h2']):+.4f}±{float(seQH['h2']):.4f}")
out["B_q3"] = {"ll_q3": llQ, "ll_q3_het": llQH, "per_country": Q, "h1": float(estQH["h1"]), "h1_se": float(seQH["h1"]), "h2": float(estQH["h2"]), "h2_se": float(seQH["h2"])}
out["B"] = {"ll_mean": llM, "ll_mv": llV, "ll_mv_het": llH, "per_country": B,
            "g1": float(estH["g1"]), "g1_se": float(seH["g1"]), "g2": float(estH["g2"]), "g2_se": float(seH["g2"])}
o = OUT_DIR / "exp_risk_signal"; o.mkdir(parents=True, exist_ok=True)
json.dump(out, open(o / "exp_risk_signal.json", "w"), indent=2, default=float)
say(f"written {o/'exp_risk_signal.json'}")
