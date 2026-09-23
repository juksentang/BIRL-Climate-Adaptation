"""Choice-model counterfactual (Step 07 stage 3): climate x policy scenarios on the
semi-parametric choice model.  Spec: 07_2050_Counter_Fact/CHOICE_CF_SPEC.md.

  draws  = load_posterior_thinned(posterior.npz, K)
  q'     = policies.apply_policy(q10, q50, q90, policy, params, m_obs, ctx)
  out    = scenario_metrics(q10', q50', q90', q10, q50, q90, draws, mask, ci, C, m_obs, params, policy, info=info)
  run_scenarios(...) loops climates x policies and returns the two long tables.

Outcome metrics per (draw, country): crop shares, expected income (5-node mean),
expected median, exact downside probability P(y' < theta) (inverse-transformed
threshold on the pre-policy lognormal), 5-node expected shortfall E[(theta - y')+],
logsum, behaviour-adjusted policy cost, calibrated-rho certainty equivalents.
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from jax.scipy.stats import norm as jnorm

from cropchoice.broadcast import HI, one_hot, per_obs
from cropchoice.models import compute_logits_semipar
from cropchoice.models_v2 import INFEASIBLE_LOGIT
from cropchoice.policies import (POLICIES, DEFAULT_PARAMS, POLICY_KIND, cost_context, apply_policy,
                                 inverse_threshold)
from cropchoice.quantiles import mu_sigma, five_nodes, e5

INFEASIBLE = INFEASIBLE_LOGIT
METRIC_NAMES = ("exp_income", "exp_median", "p_below_theta", "exp_shortfall", "logsum", "cv_log", "cv_pct",
                "ce_rho1.5", "ce_rho2.5", "ce_rho3.5", "cost", "cost_behav", "payout", "premium", "switch_share")


# ─────────────────────────────────────────────────────────────── posterior ──
def load_posterior_thinned(path, K=200):
    """Flatten chains, keep every (n_total // K)-th draw.  Returns dict of numpy:
    a, b, c: (K, C);  asc: (K, C, A) with the reference action's 0 prepended."""
    d = np.load(path)
    a = d["a_c"].reshape(-1, d["a_c"].shape[-1])
    b = d["b_c"].reshape(-1, d["b_c"].shape[-1])
    c = d["c_c"].reshape(-1, d["c_c"].shape[-1])
    asc = d["asc_c"].reshape(-1, *d["asc_c"].shape[-2:])
    n = a.shape[0]
    step = max(n // K, 1)
    idx = np.arange(0, n, step)[:K]
    asc_full = np.concatenate([np.zeros((len(idx), asc.shape[1], 1), asc.dtype), asc[idx]], axis=2)
    return {"a": a[idx].astype(np.float32), "b": b[idx].astype(np.float32),
            "c": c[idx].astype(np.float32), "asc": asc_full.astype(np.float32),
            "n_total": int(n), "idx": idx}


# ─────────────────────────────────────────────────────────────── metrics ──
def _softmax_feasible(U, mask):
    U = jnp.where(mask, U, INFEASIBLE)
    m = jnp.max(U, axis=-1, keepdims=True)
    e = jnp.where(mask, jnp.exp(U - m), 0.0)
    s = jnp.sum(e, axis=-1, keepdims=True)
    return e / s, (jnp.log(s) + m)[..., 0]                 # P (..., N, A), logsum (..., N)


def _batch_metrics(q10, q50, q90, q10c, q50c, q90c, a, b, c, asc, mask, oh, m_obs, theta_obs,
                   F_obs, T_obs, basis, shift_obs, prem_obs, lam, rhos, policy_kind):
    """One batch of draws.  q*  : post-policy quantiles (N, A)
                            q*c : climate (pre-policy) quantiles
                            a,b,c: (Kb, C); asc: (Kb, C, A); oh: (N, C) one-hot.
    Returns per-country arrays (Kb, C, ...)."""
    mu, sig = mu_sigma(q10, q50, q90)
    nodes = five_nodes(q10, q50, q90)                                       # (5, N, A)
    ey = e5(nodes)                                                          # (N, A)
    mu_c, sig_c = mu_sigma(q10c, q50c, q90c)
    ystar = inverse_threshold(theta_obs, policy_kind, q50c, F_obs, T_obs, prem_obs, basis, shift_obs, lam)
    pb = jnp.where(ystar > 0, jnorm.cdf((jnp.log1p(jnp.maximum(ystar, 0.0)) - mu_c) / sig_c), 0.0)   # (N, A)
    es = e5(jnp.maximum(theta_obs[None, :, None] - nodes, 0.0))                          # E[(theta - y')+], exact on transformed nodes
    ce = {r: jnp.exp(mu + 0.5 * (1.0 - r) * sig * sig) - 1.0 for r in rhos}
    nodes_c = five_nodes(q10c, q50c, q90c)
    if policy_kind == "floor":
        cb = e5(jnp.maximum(F_obs[None, :, None] - nodes_c, 0.0))            # (N, A)
    elif policy_kind == "insurance":
        cb = e5((1.0 - basis) * jnp.maximum(T_obs[None, :, None] - nodes_c, 0.0))
    elif policy_kind == "both":
        cb = (e5(jnp.maximum(F_obs[None, :, None] - nodes_c, 0.0))
              + e5((1.0 - basis) * jnp.maximum(T_obs[None, :, None] - jnp.maximum(nodes_c, F_obs[None, :, None]), 0.0)))
    else:
        cb = jnp.zeros_like(ey)
    n_c = jnp.maximum(oh.sum(0), 1.0)                                       # (C,)

    def per_draw(a_k, b_k, c_k, asc_k):
        U = compute_logits_semipar(a_k, b_k, c_k, asc_k, mu, sig, mask, oh, center=False)   # (N, A), uncentred
        P, ls = _softmax_feasible(U, mask)
        cm = lambda v: jnp.dot(oh.T, v, precision=HI) / n_c                 # (N,) -> (C,)
        out = {
            "shares": jnp.dot(oh.T, P, precision=HI) / n_c[:, None],        # (C, A)
            "exp_income": cm(jnp.sum(P * ey, -1)),
            "exp_median": cm(jnp.sum(P * q50, -1)),
            "p_below_theta": cm(jnp.sum(P * pb, -1)),
            "exp_shortfall": cm(jnp.sum(P * es, -1)),
            "logsum": cm(ls),
            "cost_behav": cm(jnp.sum(P * cb, -1)),
        }
        for r in rhos:
            out[f"ce_rho{r}"] = cm(jnp.sum(P * ce[r], -1))
        return out

    return jax.vmap(per_draw)(a, b, c, asc)


_batch_metrics_jit = jax.jit(_batch_metrics, static_argnames=("rhos", "policy_kind"))


def scenario_metrics(q10, q50, q90, q10c, q50c, q90c, draws, mask, ci, C, m_obs, params,
                     policy, batch=10, rhos=(1.5, 2.5, 3.5), info=None):
    """Loop the draws in batches on the device.  Returns dict of numpy arrays:
    shares (K, C, A); exp_income, exp_median, p_below_theta, exp_shortfall, logsum,
    cost_behav, ce_rho* : (K, C)."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    kind = POLICY_KIND.get(policy, "none")
    oh = one_hot(ci, C)
    theta_obs = p["theta"] * m_obs
    F_obs = (p["f_both"] if policy == "both" else p["f"]) * m_obs
    T_obs = p["trigger"] * m_obs
    shift_obs = (p["t"] if policy == "transfer" else 0.0) * m_obs
    prem_c = jnp.asarray(info["premium"], jnp.float32) if info is not None else jnp.zeros((C,), jnp.float32)
    prem_obs = per_obs(oh, prem_c)
    lam = float(p["lam"]) if policy == "contraction" else 1.0
    K = draws["a"].shape[0]
    outs = []
    for s in range(0, K, batch):
        sl = slice(s, s + batch)
        o = _batch_metrics_jit(q10, q50, q90, q10c, q50c, q90c,
                               jnp.asarray(draws["a"][sl]), jnp.asarray(draws["b"][sl]),
                               jnp.asarray(draws["c"][sl]), jnp.asarray(draws["asc"][sl]),
                               mask, oh, m_obs, theta_obs, F_obs, T_obs, float(p["basis"]),
                               shift_obs, prem_obs, lam, tuple(rhos), kind)
        outs.append({k: np.asarray(v) for k, v in o.items()})
    return {k: np.concatenate([o[k] for o in outs], axis=0) for k in outs[0]}


def observed_shares(obs_action, ci, C, A):
    """Observed action frequencies per country (C, A) for the PPC."""
    out = np.zeros((C, A), np.float64)
    for c in range(C):
        sel = np.asarray(ci) == c
        out[c] = np.bincount(np.asarray(obs_action)[sel], minlength=A) / max(sel.sum(), 1)
    return out


# ─────────────────────────────────────────────────────────────── driver ──
def toy_inputs(K=8, N=500, A=27, C=6, seed=0):
    """Synthetic inputs with the real m_c values (local checks; no data files)."""
    rng = np.random.default_rng(seed)
    countries = ["Ethiopia", "Malawi", "Mali", "Nigeria", "Tanzania", "Uganda"][:C]
    m_c = np.array([19.9, 19.1, 130.0, 102.7, 26.0, 25.5], np.float32)[:C]
    ci = rng.integers(0, C, N).astype(np.int32)
    base = rng.lognormal(np.log(m_c[ci])[:, None], 0.8, (N, A)).astype(np.float32)
    rs = rng.lognormal(0, 0.5, (N, A)).astype(np.float32)

    def mk_q(scale):
        q50 = base * scale
        return (q50 * np.maximum(1 - 0.5 * rs, 0.05)).astype(np.float32), q50.astype(np.float32), (q50 * (1 + 0.8 * rs)).astype(np.float32)
    quant = {"baseline": mk_q(1.0), "ssp245": mk_q(0.95), "ssp585": mk_q(0.85)}
    mask = rng.random((N, A)) > 0.2; mask[:, 0] = True
    obs_action = np.array([rng.choice(np.flatnonzero(r)) for r in mask], np.int32)
    draws = {"a": rng.normal(0.8, 0.2, (K, C)).astype(np.float32), "b": rng.normal(-3, 0.3, (K, C)).astype(np.float32),
             "c": rng.normal(1, 0.1, (K, C)).astype(np.float32),
             "asc": np.concatenate([np.zeros((K, C, 1)), rng.normal(0, 0.5, (K, C, A - 1))], 2).astype(np.float32),
             "n_total": K, "idx": np.arange(K)}
    if C > 4:
        draws["a"][:, 4] = -0.05; draws["a"][:, 1] = -0.6                    # Tanzania / Malawi: a <= 0 as in the real posterior
    labels = [f"a{i}" for i in range(A)]
    return dict(countries=countries, m_c=m_c, ci=ci, quant=quant, mask=mask, obs_action=obs_action,
                draws=draws, labels=labels)


def real_inputs(climates, K, posterior=None, data_dir=None, step07_dir=None):
    """Loader dict + stage-2 quantile matrices + thinned posterior for the real run."""
    from cropchoice.config import DATA_DIR, OUT_DIR, STEP07_DIR
    from cropchoice.data import load_data, model_kwargs
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    step07 = Path(step07_dir) if step07_dir else STEP07_DIR
    data = load_data(data_dir); mk = model_kwargs(data)
    countries = list(data["countries"]); C = len(countries)
    m_c = np.asarray(data["m_c"], np.float32)
    ci = np.asarray(mk["obs_country_idx"]); obs_action = np.asarray(mk["obs_action"])
    mask = np.asarray(mk["mask_obs"]); N, A = mask.shape
    env = np.load(data_dir / "env_model_output.npz", allow_pickle=True)
    assert np.array_equal(np.asarray(env["action_ids"]), np.arange(A)), "action_ids must be arange(A)"
    quant = {"baseline": (np.asarray(mk["q10"]), np.asarray(mk["q50"]), np.asarray(mk["q90"]))}
    for clim in climates:
        if clim == "baseline":
            continue
        d = np.load(step07 / "data" / f"{clim}_cf.npz")
        q = tuple(np.asarray(d[k], np.float32) for k in ("q10", "q50", "q90"))
        assert q[1].shape == (N, A), (clim, q[1].shape, (N, A))
        quant[clim] = q
    post = Path(posterior) if posterior else OUT_DIR / "semipar" / "posterior.npz"
    draws = load_posterior_thinned(post, K)
    assert draws["asc"].shape[1:] == (C, A)
    al = data["action_labels"]; labels = [al[str(i)] if isinstance(al, dict) else al[i] for i in range(A)]
    return dict(countries=countries, m_c=m_c, ci=ci, quant=quant, mask=mask, obs_action=obs_action,
                draws=draws, labels=labels)


def run_scenarios(inp, climates, policies, params, batch=10, say=print):
    """Loop climates x policies.  Returns (res, info_all): res[(climate, policy)] -> dict of (K, C[, A]) arrays
    including cost/payout/premium, cv_log/cv_pct (relative to baseline/none, NaN where a <= 0) and
    switch_share (vs the same-climate 'none')."""
    countries, m_c, ci, mask, obs_action, draws = (inp[k] for k in ("countries", "m_c", "ci", "mask", "obs_action", "draws"))
    C = len(countries); K = draws["a"].shape[0]
    m_obs = jnp.asarray(m_c[ci]); mask_j = jnp.asarray(mask); ci_j = jnp.asarray(ci)
    b10, b50, b90 = (jnp.asarray(x) for x in inp["quant"]["baseline"])
    ctx = cost_context(b10, b50, b90, jnp.asarray(obs_action), ci_j, C)
    res, info_all = {}, {}
    for clim in climates:
        c10, c50, c90 = (jnp.asarray(x) for x in inp["quant"][clim])
        for pol in policies:
            t0 = time.time()
            p10, p50, p90, info = apply_policy(c10, c50, c90, pol, params, m_obs, ctx)
            out = scenario_metrics(p10, p50, p90, c10, c50, c90, draws, mask_j, ci, C, m_obs, params,
                                   pol, batch=batch, info=info)
            out["cost"] = np.broadcast_to(np.asarray(info["cost"])[None, :], (K, C)).copy()
            out["payout"] = np.broadcast_to(np.asarray(info["payout"])[None, :], (K, C)).copy()
            out["premium"] = np.broadcast_to(np.asarray(info["premium"])[None, :], (K, C)).copy()
            if pol in ("transfer",):
                out["cost_behav"] = out["cost"].copy()
            res[(clim, pol)] = out
            info_all[f"{clim}/{pol}"] = {k: np.asarray(v).round(4).tolist() for k, v in info.items()}
            say(f"{clim:8s} {pol:12s} {time.time()-t0:5.1f}s  E[y] {np.median(out['exp_income'], 0).round(1).tolist()}")
    a = draws["a"]
    ls0 = res[("baseline", "none")]["logsum"] if ("baseline", "none") in res else None
    for (clim, pol), out in res.items():
        if ls0 is not None:
            cv_log = np.where(a > 0, (out["logsum"] - ls0) / np.where(a > 0, a, 1.0), np.nan)
            out["cv_log"] = cv_log; out["cv_pct"] = np.expm1(cv_log)
        base_sh = res[(clim, "none")]["shares"]
        out["switch_share"] = 0.5 * np.abs(out["shares"] - base_sh).sum(-1)
    return res, info_all


def long_tables(res, countries):
    """(metrics, crop_shares) long DataFrames from run_scenarios' result."""
    rows, share_rows = [], []
    for (clim, pol), out in res.items():
        K, C = out["exp_income"].shape
        for metric, arr in out.items():
            if metric == "shares":
                continue
            for k in range(K):
                for c in range(C):
                    rows.append((clim, pol, countries[c], k, metric, float(arr[k, c])))
        sh = out["shares"]; A = sh.shape[-1]
        for k in range(K):
            for c in range(C):
                for aa in range(A):
                    share_rows.append((clim, pol, countries[c], k, aa, float(sh[k, c, aa])))
    metrics = pd.DataFrame(rows, columns=["climate", "policy", "country", "draw", "metric", "value"])
    shares = pd.DataFrame(share_rows, columns=["climate", "policy", "country", "draw", "action", "share"])
    return metrics, shares


def main(argv=None):
    """CLI of stage 3 (python -m cropchoice counterfactual / scripts/03_choice_counterfactual.py)."""
    import argparse, os
    parser = argparse.ArgumentParser(description="choice-model counterfactual (Step 07 stage 3)")
    parser.add_argument("--toy", action="store_true", help="synthetic data, no real files (local check)")
    parser.add_argument("--K", type=int, default=200, help="posterior draws after thinning")
    parser.add_argument("--batch", type=int, default=10, help="draws per device batch")
    parser.add_argument("--posterior", default=None, help="path to the semipar posterior.npz")
    parser.add_argument("--out", default=None, help="results directory")
    parser.add_argument("--climates", default="baseline,ssp245,ssp585")
    parser.add_argument("--policies", default=",".join(POLICIES))
    for k, v in DEFAULT_PARAMS.items():
        parser.add_argument(f"--{k}", type=float, default=v)
    args = parser.parse_args(argv)
    params = {k: getattr(args, k) for k in DEFAULT_PARAMS}
    os.environ.setdefault("BIRL_HOST_DEVICES", "1")
    if args.toy:
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
    from cropchoice.config import STEP07_DIR   # noqa: F811  (sets host devices before jax use)
    t_start = time.time()

    def say(msg):
        print(f"[cf {time.strftime('%H:%M:%S')}] {msg}", flush=True)

    climates = [c for c in args.climates.split(",") if c]
    policies = [p for p in args.policies.split(",") if p]
    for p in policies:
        assert p in POLICIES, p
    if args.toy:
        inp = toy_inputs(K=min(args.K, 8))
        out_dir = Path(args.out) if args.out else STEP07_DIR / "results" / "choice_cf_toy"
    else:
        inp = real_inputs(climates, args.K, posterior=args.posterior)
        out_dir = Path(args.out) if args.out else STEP07_DIR / "results" / "choice_cf"
    out_dir.mkdir(parents=True, exist_ok=True)
    countries, m_c, ci, obs_action, draws = (inp[k] for k in ("countries", "m_c", "ci", "obs_action", "draws"))
    C = len(countries); N, A = inp["mask"].shape; K = draws["a"].shape[0]
    say(f"N={N} A={A} C={C} K={K} climates={climates} policies={policies} devices={jax.devices()} toy={args.toy}")
    say(f"params {params}")
    res, info_all = run_scenarios(inp, climates, policies, params, batch=args.batch, say=say)
    metrics, shares = long_tables(res, countries)
    metrics.to_parquet(out_dir / "metrics.parquet", index=False)
    shares.to_parquet(out_dir / "crop_shares.parquet", index=False)
    obs_sh = observed_shares(obs_action, ci, C, A)
    pred_sh = np.median(res[("baseline", "none")]["shares"], axis=0) if ("baseline", "none") in res else None
    ppc = {countries[c]: float(np.abs(pred_sh[c] - obs_sh[c]).max()) for c in range(C)} if pred_sh is not None else {}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=STEP07_DIR, text=True).strip()
    except Exception:
        commit = None
    a = draws["a"]
    run_info = {"toy": args.toy, "N": int(N), "A": int(A), "C": int(C), "K": int(K), "n_posterior_total": int(draws["n_total"]),
                "climates": climates, "policies": policies, "params": params, "countries": countries,
                "m_c": {countries[c]: float(m_c[c]) for c in range(C)}, "m_c_list": m_c.tolist(),
                "labels": inp["labels"], "action_labels": inp["labels"], "policy_info": info_all,
                "ppc_max_abs_share_diff": ppc, "share_draws_a_positive": {countries[c]: float((a[:, c] > 0).mean()) for c in range(C)},
                "device": str(jax.devices()[0]), "elapsed_s": round(time.time() - t_start, 1), "git_commit": commit,
                "metric_names": sorted(metrics.metric.unique().tolist())}
    json.dump(run_info, open(out_dir / "run_info.json", "w"), indent=2)
    say(f"PPC max |pred-obs| share by country: { {k: round(v, 4) for k, v in ppc.items()} }")
    say(f"wrote {out_dir}/metrics.parquet ({len(metrics)} rows), crop_shares.parquet ({len(shares)} rows), run_info.json; {time.time()-t_start:.0f}s")
    return out_dir
