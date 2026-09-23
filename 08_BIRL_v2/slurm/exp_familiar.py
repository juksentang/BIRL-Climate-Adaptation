"""Is sigma (predicted log-income dispersion) RISK or FAMILIARITY?

Part 1 (descriptive, per observation within its feasible set): Spearman of
  sigma_a with the local choice frequency of a (country x zone), Spearman of
  sigma_a with mu_a, and the within-set dispersion of sigma vs mu.
Part 2 (decisive): logfree V = a mu + b sigma + c sigma^2 with three ASC
  structures: country x action (baseline), country-zone x action (absorbs the
  local planting frequency), country-zone x action + g_c * log(local freq).
  If b, c shrink a lot under zone ASCs, sigma is mostly a familiarity proxy.
Part 3: logfree (country x action ASC) fitted separately on observations
  whose chosen action is locally common vs rare.
Run from 08_BIRL_v2/:  python3 slurm/exp_familiar.py [steps]   (EXP_TOY=1 for a synthetic check)
"""
import os, sys, time, json
os.environ.setdefault("BIRL_HOST_DEVICES", "1")

from cropchoice.config import log, DATA_DIR, OUT_DIR  # noqa: E402
import jax, jax.numpy as jnp, numpy as np, pandas as pd  # noqa: E402
import numpyro  # noqa: E402
import numpyro.distributions as dist  # noqa: E402
from numpyro import sample, deterministic, plate  # noqa: E402
from numpyro.infer import SVI, Trace_ELBO  # noqa: E402
from numpyro.infer.autoguide import AutoNormal  # noqa: E402
from numpyro.infer.util import log_density  # noqa: E402
from cropchoice.data import load_data, model_kwargs  # noqa: E402
from cropchoice.models_v2 import center_reward, INFEASIBLE_LOGIT  # noqa: E402

STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
HI = jax.lax.Precision.HIGHEST


def say(msg):
    print(f"[fam {time.strftime('%H:%M:%S')}] {msg}", flush=True)


if os.environ.get("EXP_TOY"):
    rng = np.random.default_rng(0); Nt, At, C, NCZ = 400, 6, 3, 5
    countries = ["A", "B", "C"]; m_np = np.array([20.0, 100.0, 30.0], np.float32)
    _q50 = rng.lognormal(3, 0.7, (Nt, At)); _rs = rng.lognormal(0, 0.6, (Nt, At))
    _m = rng.random((Nt, At)) > 0.2; _m[:, 0] = True
    _ci = rng.integers(0, C, Nt); _cz = _ci * 2 + rng.integers(0, 2, Nt)   # cz nested in country
    mk = {"q10": jnp.asarray(_q50 * np.maximum(1 - 0.5 * _rs, 0.05), jnp.float32), "q50": jnp.asarray(_q50, jnp.float32),
          "q90": jnp.asarray(_q50 * (1 + 0.8 * _rs), jnp.float32), "mask_obs": jnp.asarray(_m),
          "obs_action": jnp.asarray(np.array([rng.choice(np.flatnonzero(r)) for r in _m]), jnp.int32),
          "obs_country_idx": jnp.asarray(_ci, jnp.int32), "obs_cz_idx": jnp.asarray(_cz, jnp.int32)}
    NCZ = 6; STEPS = 40
else:
    data = load_data(DATA_DIR)
    countries = list(data["countries"]); C = len(countries)
    mk = model_kwargs(data); m_np = np.asarray(data["m_c"], np.float32)
    NCZ = int(np.asarray(mk["obs_cz_idx"]).max()) + 1
m_c = jnp.asarray(m_np, jnp.float32)
N, A = mk["q50"].shape
ci = np.asarray(mk["obs_country_idx"]); cz = np.asarray(mk["obs_cz_idx"]); act = np.asarray(mk["obs_action"])
mask = np.asarray(mk["mask_obs"], bool)
say(f"devices {jax.devices()}  N {N}  A {A}  C {C}  n_cz {NCZ}  steps {STEPS}")

# ── local choice frequency per (country-zone, action) ──
cnt = np.zeros((NCZ, A)); np.add.at(cnt, (cz, act), 1.0)
freq_cz = cnt / np.maximum(cnt.sum(1, keepdims=True), 1)
freq_obs = freq_cz[cz]                                                # (N, A)
q10, q50, q90 = (np.asarray(mk[k], np.float64) for k in ("q10", "q50", "q90"))
mu = np.log1p(np.maximum(q50, 0)); sig = np.clip((np.log1p(np.maximum(q90, 0)) - np.log1p(np.maximum(q10, 0))) / (2 * 1.2816), 0.01, 5)


def row_spearman(X, Y, m):
    """Spearman between X and Y within the feasible set of each row (NaN if < 3 feasible)."""
    Xm = np.where(m, X, np.nan); Ym = np.where(m, Y, np.nan)
    rx = pd.DataFrame(Xm).rank(axis=1).values; ry = pd.DataFrame(Ym).rank(axis=1).values
    n = m.sum(1)
    mx = np.nanmean(rx, 1, keepdims=True); my = np.nanmean(ry, 1, keepdims=True)
    dx = np.where(m, rx - mx, 0); dy = np.where(m, ry - my, 0)
    num = (dx * dy).sum(1); den = np.sqrt((dx ** 2).sum(1) * (dy ** 2).sum(1))
    r = np.where((n >= 3) & (den > 0), num / np.where(den > 0, den, 1), np.nan)
    return r


r_sf = row_spearman(sig, freq_obs, mask)
r_sm = row_spearman(sig, mu, mask)
sd_sig = np.nanstd(np.where(mask, sig, np.nan), 1); sd_mu = np.nanstd(np.where(mask, mu, np.nan), 1)
say("Part 1: within-feasible-set correlations (per-obs Spearman, country means; q25/q75 in brackets)")
say(f"   {'country':10s} {'rho(sig,freq)':>22s} {'rho(sig,mu)':>22s} {'sd(sig)':>8s} {'sd(mu)':>8s} {'n_feas':>7s}")
P1 = {}
for i, c in enumerate(countries):
    s = ci == i
    a = np.nanmean(r_sf[s]); b = np.nanmean(r_sm[s])
    qa = np.nanpercentile(r_sf[s], [25, 75]); qb = np.nanpercentile(r_sm[s], [25, 75])
    P1[c] = {"rho_sig_freq": float(a), "rho_sig_mu": float(b), "sd_sig": float(np.nanmean(sd_sig[s])),
             "sd_mu": float(np.nanmean(sd_mu[s])), "n_feas": float(mask[s].sum(1).mean())}
    say(f"   {c:10s} {a:+8.3f} [{qa[0]:+.2f},{qa[1]:+.2f}]   {b:+8.3f} [{qb[0]:+.2f},{qb[1]:+.2f}]   {P1[c]['sd_sig']:8.3f} {P1[c]['sd_mu']:8.3f} {P1[c]['n_feas']:7.1f}")
say(f"   ALL        rho(sig,freq)={np.nanmean(r_sf):+.3f}  rho(sig,mu)={np.nanmean(r_sm):+.3f}")


# ── Part 2 / 3: logfree with different ASC structures ──
def subset(d, sel):
    idx = jnp.asarray(np.flatnonzero(sel))
    return {k: (v[idx] if hasattr(v, "shape") and v.ndim >= 1 and v.shape[0] == N else v) for k, v in d.items()}


def build(d):
    q10, q50, q90 = d["q10"], d["q50"], d["q90"]
    MU = jnp.log1p(jnp.maximum(q50, 0.0))
    SIG = jnp.clip((jnp.log1p(jnp.maximum(q90, 0.0)) - jnp.log1p(jnp.maximum(q10, 0.0))) / (2 * 1.2816), 0.01, 5.0)
    MASK = d["mask_obs"]; ACT = d["obs_action"]; CI = d["obs_country_idx"]; CZ = d["obs_cz_idx"]
    LFREQ = jnp.asarray(np.log(np.asarray(d["freq_obs"]) + 1e-4), jnp.float32)
    n = int(q50.shape[0])
    OH = jax.nn.one_hot(CI, C, dtype=jnp.float32); OHZ = jax.nn.one_hot(CZ, NCZ, dtype=jnp.float32)
    per_obs = lambda v: jnp.dot(OH, v, precision=HI)
    per_cz = lambda v: jnp.dot(OHZ, v, precision=HI)

    def model(asc, null=False, lfreq=False):
        with plate("countries", C):
            lb = sample("lb_c", dist.Normal(1.0, 2.0))
            wf = sample("wfree_c", dist.Normal(0.0, 2.0).expand([3]).to_event(1))
            g = sample("g_c", dist.Normal(0.0, 2.0)) if lfreq else None
        beta_c = deterministic("beta_c", jnp.exp(jnp.clip(lb, -4.0, 6.0)))
        deterministic("w_c", wf); w = per_obs(wf)
        V = w[:, 0:1] * MU + w[:, 1:2] * SIG + w[:, 2:3] * SIG * SIG
        logits = jnp.zeros_like(V) if null else per_obs(beta_c)[:, None] * center_reward(V, MASK)
        if lfreq:
            logits = logits + per_obs(g)[:, None] * center_reward(LFREQ, MASK)
        if asc == "c":
            a = sample("asc_c", dist.Normal(0.0, 3.0).expand([C, A - 1]).to_event(2))
            logits = logits + per_obs(jnp.concatenate([jnp.zeros((C, 1)), a], axis=1))
        elif asc == "cz":
            a = sample("asc_cz", dist.Normal(0.0, 3.0).expand([NCZ, A - 1]).to_event(2))
            logits = logits + per_cz(jnp.concatenate([jnp.zeros((NCZ, 1)), a], axis=1))
        logits = jnp.where(MASK, logits, INFEASIBLE_LOGIT)
        with plate("observations", n):
            sample("obs_action", dist.Categorical(logits=logits), obs=ACT)
    return model, n


def fit(model, n, label, seed=0, **kw):
    guide = AutoNormal(model, init_loc_fn=numpyro.infer.init_to_median())
    svi = SVI(model, guide, numpyro.optim.Adam(1e-2), Trace_ELBO())
    t0 = time.time()
    res = svi.run(jax.random.PRNGKey(seed), STEPS, progress_bar=False, **kw)
    med = guide.median(res.params)
    _, tr = log_density(numpyro.handlers.substitute(model, data=med), (), kw, {})
    ll = float(tr["obs_action"]["fn"].log_prob(tr["obs_action"]["value"]).sum())
    det = {k: np.asarray(v["value"]).tolist() for k, v in tr.items() if v["type"] == "deterministic"}
    if "g_c" in med:
        det["g_c"] = np.asarray(med["g_c"]).tolist()
    say(f"   {label:24s} n={n:6d} log-lik {ll:12.1f}  ({time.time()-t0:.0f}s)")
    return {"loglik": ll, "n": n, "det": det}


def wtab(det):
    return " | ".join(" ".join(f"{x:+.2f}" for x in row) for row in det["w_c"])


mk = dict(mk); mk["freq_obs"] = jnp.asarray(freq_obs, jnp.float32)
model, n = build(mk)
say("Part 2: logfree V = a mu + b sigma + c sigma^2 under three ASC structures (w per country: a b c)")
P2 = {}
for asc, lf in (("c", False), ("cz", False), ("cz", True)):
    lab = f"asc_{asc}" + ("_lfreq" if lf else "")
    r0 = fit(model, n, lab + "/null", asc=asc, null=True, lfreq=lf)
    r1 = fit(model, n, lab + "/logfree", asc=asc, null=False, lfreq=lf)
    gain = (r1["loglik"] - r0["loglik"]) / n
    P2[lab] = {"null": r0["loglik"], "logfree": r1["loglik"], "gain_per_obs": gain, "w_c": r1["det"]["w_c"],
               "beta_c": r1["det"]["beta_c"], "g_c": r1["det"].get("g_c")}
    say(f"   {lab:12s} gain/obs vs its null = {gain:.4f}   w = {wtab(r1['det'])}   beta = " +
        " ".join(f"{x:.2f}" for x in r1["det"]["beta_c"]) + (f"   g(log freq) = " + " ".join(f"{x:.2f}" for x in r1["det"]["g_c"]) if lf else ""))

say("Part 3: logfree (asc_c) on observations whose CHOSEN action is locally common vs rare (split at the cz median chosen-frequency)")
f_ch = freq_obs[np.arange(N), act]
thr = np.zeros(N)
for z in range(NCZ):
    s = cz == z
    if s.any():
        thr[s] = np.median(f_ch[s])
common = f_ch >= thr
P3 = {}
for lab, sel in (("common", common), ("rare", ~common)):
    if sel.sum() < 50:
        continue
    m_sub, n_sub = build(subset(mk, sel))
    r0 = fit(m_sub, n_sub, lab + "/null", asc="c", null=True)
    r1 = fit(m_sub, n_sub, lab + "/logfree", asc="c", null=False)
    gain = (r1["loglik"] - r0["loglik"]) / n_sub
    P3[lab] = {"n": n_sub, "gain_per_obs": gain, "w_c": r1["det"]["w_c"], "beta_c": r1["det"]["beta_c"]}
    say(f"   {lab:7s} n={n_sub:6d} gain/obs = {gain:.4f}   w = {wtab(r1['det'])}")

say(f"countries: {countries}")
o = OUT_DIR / "exp_familiar"; o.mkdir(parents=True, exist_ok=True)
json.dump({"countries": countries, "N": N, "n_cz": NCZ, "steps": STEPS, "part1": P1, "part2": P2, "part3": P3},
          open(o / "exp_familiar.json", "w"), indent=2)
say(f"written {o/'exp_familiar.json'}")
