"""Stage 3 (rewritten): choice-model counterfactual and welfare on the Step 08
semi-parametric choice model.  Spec: ../CHOICE_CF_SPEC.md.

  climate  in {baseline, ssp245, ssp585}   (stage 2 quantile matrices)
  policy   in {none, transfer, contraction, safety_net, insurance, both}
Outputs (results/choice_cf/):
  metrics.parquet      long: climate, policy, country, draw, metric, value
  crop_shares.parquet  long: climate, policy, country, draw, action, share
  run_info.json        parameters, K, timing, git commit, PPC summary
Usage (from 07_2050_Counter_Fact/):
  python3 scripts/03_choice_counterfactual.py [--K 200] [--f 0.5 --t 0.1 ...]
  python3 scripts/03_choice_counterfactual.py --toy        # synthetic, local check
"""
import argparse, json, os, sys, time, subprocess, importlib.util
from pathlib import Path

STEP07 = Path(__file__).resolve().parent.parent
STEP08 = STEP07.parent / "08_BIRL_v2"

parser = argparse.ArgumentParser()
parser.add_argument("--toy", action="store_true", help="synthetic data, no real files (local check)")
parser.add_argument("--K", type=int, default=200, help="posterior draws after thinning")
parser.add_argument("--batch", type=int, default=10, help="draws per device batch")
parser.add_argument("--posterior", default=None, help="path to 08 semipar posterior.npz")
parser.add_argument("--out", default=None, help="results directory")
parser.add_argument("--climates", default="baseline,ssp245,ssp585")
parser.add_argument("--policies", default="none,transfer,contraction,safety_net,insurance,both")
for k, v in dict(t=0.10, lam=0.5, f=0.5, f_both=0.3, trigger=0.5, basis=0.3, loading=0.2, theta=0.3).items():
    parser.add_argument(f"--{k}", type=float, default=v)
args = parser.parse_args()
PARAMS = {k: getattr(args, k) for k in ("t", "lam", "f", "f_both", "trigger", "basis", "loading", "theta")}

os.environ.setdefault("BIRL_HOST_DEVICES", "1")
if args.toy:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("BIRL_OUT_DIR", str(STEP07 / "results" / "choice_cf_toy" / "_08_out"))
sys.path.insert(0, str(STEP08))            # `src` -> 08_BIRL_v2/src (loader, config)
import numpy as np                          # noqa: E402
import pandas as pd                         # noqa: E402
import jax, jax.numpy as jnp                # noqa: E402

spec = importlib.util.spec_from_file_location("choice_engine", STEP07 / "src" / "choice_engine.py")
ce = importlib.util.module_from_spec(spec); spec.loader.exec_module(ce)

t_start = time.time()
def say(msg):
    print(f"[cf {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ─────────────────────────────────────────────────────────────── inputs ──
CLIMATES = [c for c in args.climates.split(",") if c]
POLICIES = [p for p in args.policies.split(",") if p]
for p in POLICIES:
    assert p in ce.POLICIES, p

if args.toy:
    rng = np.random.default_rng(0); N, A, C = 500, 27, 6
    countries = ["Ethiopia", "Malawi", "Mali", "Nigeria", "Tanzania", "Uganda"]
    m_c = np.array([19.9, 19.1, 130.0, 102.7, 26.0, 25.5], np.float32)
    ci = rng.integers(0, C, N).astype(np.int32)
    base = rng.lognormal(np.log(m_c[ci])[:, None], 0.8, (N, A)).astype(np.float32)
    rs = rng.lognormal(0, 0.5, (N, A)).astype(np.float32)
    def mk_q(scale):
        q50 = base * scale; return (q50 * np.maximum(1 - 0.5 * rs, 0.05)).astype(np.float32), q50.astype(np.float32), (q50 * (1 + 0.8 * rs)).astype(np.float32)
    quant = {"baseline": mk_q(1.0), "ssp245": mk_q(0.95), "ssp585": mk_q(0.85)}
    mask = rng.random((N, A)) > 0.2; mask[:, 0] = True
    obs_action = np.array([rng.choice(np.flatnonzero(r)) for r in mask], np.int32)
    K = min(args.K, 8)
    draws = {"a": rng.normal(0.8, 0.2, (K, C)).astype(np.float32), "b": rng.normal(-3, 0.3, (K, C)).astype(np.float32),
             "c": rng.normal(1, 0.1, (K, C)).astype(np.float32), "asc": np.concatenate([np.zeros((K, C, 1)), rng.normal(0, 0.5, (K, C, A - 1))], 2).astype(np.float32),
             "n_total": K, "idx": np.arange(K)}
    draws["a"][:, 4] = -0.05; draws["a"][:, 1] = -0.6                    # Tanzania / Malawi: a <= 0 as in the real posterior
    out_dir = Path(args.out) if args.out else STEP07 / "results" / "choice_cf_toy"
    hh = rng.integers(0, 100, N); labels = [f"a{i}" for i in range(A)]
else:
    from src.config import DATA_DIR                                        # 08's data dir (BIRL_DATA_DIR)
    from src.data_loader import load_data, model_kwargs
    data = load_data(DATA_DIR); mk = model_kwargs(data)
    countries = list(data["countries"]); C = len(countries)
    m_c = np.asarray(data["m_c"], np.float32)
    ci = np.asarray(mk["obs_country_idx"]); obs_action = np.asarray(mk["obs_action"])
    mask = np.asarray(mk["mask_obs"]); N, A = mask.shape
    env = np.load(DATA_DIR / "env_model_output.npz", allow_pickle=True)
    assert np.array_equal(np.asarray(env["action_ids"]), np.arange(A)), "action_ids must be arange(A)"
    quant = {"baseline": (np.asarray(mk["q10"]), np.asarray(mk["q50"]), np.asarray(mk["q90"]))}
    for clim in ("ssp245", "ssp585"):
        if clim in CLIMATES:
            d = np.load(STEP07 / "data" / f"{clim}_cf.npz")
            q = tuple(np.asarray(d[k], np.float32) for k in ("q10", "q50", "q90"))
            assert q[1].shape == (N, A), (clim, q[1].shape, (N, A))
            quant[clim] = q
    post = Path(args.posterior) if args.posterior else STEP08 / "outputs" / "semipar" / "posterior.npz"
    draws = ce.load_posterior_thinned(post, args.K); K = draws["a"].shape[0]
    assert draws["asc"].shape[1:] == (C, A)
    out_dir = Path(args.out) if args.out else STEP07 / "results" / "choice_cf"
    al = data["action_labels"]; labels = [al[str(i)] if isinstance(al, dict) else al[i] for i in range(A)]
out_dir.mkdir(parents=True, exist_ok=True)
say(f"N={N} A={A} C={C} K={K} climates={CLIMATES} policies={POLICIES} devices={jax.devices()} toy={args.toy}")
say(f"params {PARAMS}")

m_obs = jnp.asarray(m_c[ci]); mask_j = jnp.asarray(mask); ci_j = jnp.asarray(ci)
b10, b50, b90 = (jnp.asarray(x) for x in quant["baseline"])
ctx = ce.cost_context(b10, b50, b90, jnp.asarray(obs_action), ci_j, C)    # baseline chosen-action nodes for costs

# ─────────────────────────────────────────────────────────────── loop ──
res = {}          # (climate, policy) -> dict of arrays
info_all = {}
for clim in CLIMATES:
    c10, c50, c90 = (jnp.asarray(x) for x in quant[clim])
    for pol in POLICIES:
        t0 = time.time()
        p10, p50, p90, info = ce.apply_policy(c10, c50, c90, pol, PARAMS, m_obs, ctx)
        out = ce.scenario_metrics(p10, p50, p90, c10, c50, c90, draws, mask_j, ci, C, m_obs, PARAMS,
                                  pol, batch=args.batch, info=info)
        out["cost"] = np.broadcast_to(np.asarray(info["cost"])[None, :], (K, C)).copy()
        out["payout"] = np.broadcast_to(np.asarray(info["payout"])[None, :], (K, C)).copy()
        out["premium"] = np.broadcast_to(np.asarray(info["premium"])[None, :], (K, C)).copy()
        if pol in ("transfer",):
            out["cost_behav"] = out["cost"].copy()
        res[(clim, pol)] = out
        info_all[f"{clim}/{pol}"] = {k: np.asarray(v).round(4).tolist() for k, v in info.items()}
        say(f"{clim:8s} {pol:12s} {time.time()-t0:5.1f}s  E[y] {np.median(out['exp_income'], 0).round(1).tolist()}")

# derived: CV (money metric, a > 0 only), switch share vs same-climate 'none'
a = draws["a"]                                                              # (K, C)
ls0 = res[("baseline", "none")]["logsum"] if ("baseline", "none") in res else None
rows, share_rows = [], []
for (clim, pol), out in res.items():
    if ls0 is not None:
        cv_log = np.where(a > 0, (out["logsum"] - ls0) / np.where(a > 0, a, 1.0), np.nan)
        out["cv_log"] = cv_log; out["cv_pct"] = np.expm1(cv_log)
    base_sh = res[(clim, "none")]["shares"]
    out["switch_share"] = 0.5 * np.abs(out["shares"] - base_sh).sum(-1)     # (K, C)
    for metric, arr in out.items():
        if metric == "shares":
            continue
        for k in range(K):
            for c in range(C):
                rows.append((clim, pol, countries[c], k, metric, float(arr[k, c])))
    sh = out["shares"]
    for k in range(K):
        for c in range(C):
            for aa in range(A):
                share_rows.append((clim, pol, countries[c], k, aa, float(sh[k, c, aa])))

metrics = pd.DataFrame(rows, columns=["climate", "policy", "country", "draw", "metric", "value"])
shares = pd.DataFrame(share_rows, columns=["climate", "policy", "country", "draw", "action", "share"])
metrics.to_parquet(out_dir / "metrics.parquet", index=False)
shares.to_parquet(out_dir / "crop_shares.parquet", index=False)

# PPC: baseline/none predicted vs observed shares
obs_sh = ce.observed_shares(obs_action, ci, C, A)
pred_sh = np.median(res[("baseline", "none")]["shares"], axis=0) if ("baseline", "none") in res else None
ppc = {countries[c]: float(np.abs(pred_sh[c] - obs_sh[c]).max()) for c in range(C)} if pred_sh is not None else {}
try:
    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=STEP07, text=True).strip()
except Exception:
    commit = None
run_info = {"toy": args.toy, "N": int(N), "A": int(A), "C": int(C), "K": int(K), "n_posterior_total": int(draws["n_total"]),
            "climates": CLIMATES, "policies": POLICIES, "params": PARAMS, "countries": countries,
            "m_c": {countries[c]: float(m_c[c]) for c in range(C)}, "m_c_list": m_c.tolist(), "labels": labels, "action_labels": labels, "policy_info": info_all,
            "ppc_max_abs_share_diff": ppc, "share_draws_a_positive": {countries[c]: float((a[:, c] > 0).mean()) for c in range(C)},
            "device": str(jax.devices()[0]), "elapsed_s": round(time.time() - t_start, 1), "git_commit": commit,
            "metric_names": sorted(metrics.metric.unique().tolist())}
json.dump(run_info, open(out_dir / "run_info.json", "w"), indent=2)
say(f"PPC max |pred-obs| share by country: { {k: round(v, 4) for k, v in ppc.items()} }")
say(f"wrote {out_dir}/metrics.parquet ({len(metrics)} rows), crop_shares.parquet ({len(shares)} rows), run_info.json; {time.time()-t_start:.0f}s")
