"""Sobol global sensitivity of the choice counterfactual to the policy-intensity
parameters (Saltelli design, Jansen estimators, bootstrap CIs).  Step 07 stage 5.

Parameters and boxes (shares of the country median income m_c unless noted):
  t       transfer size            [0.02, 0.30]
  lam     contraction factor       [0.10, 0.90]   (q_k' = q50 + lam (q_k - q50))
  f       safety-net floor         [0.20, 0.80]
  f_both  floor inside 'both'      [0.10, 0.50]
  trigger insurance trigger        [0.30, 0.80]
  basis   basis risk               [0.00, 0.50]
  loading premium loading          [0.00, 0.50]
  theta   downside threshold       [0.20, 0.50]
Each design point: one climate (default ssp585), the six policies, the posterior
MEDIAN parameters as a single draw.  Per country outputs: headline_ratio
(= switch_contraction / switch_transfer), switch_<policy>, d_pbelow_<policy>,
d_expinc_<policy>, cost_<policy>, pd_<policy> (= -d_pbelow / public cost),
pdr_<policy> (same with the resource cost), sn_better.  The base point is re-run
with K posterior draws to compare posterior and parameter uncertainty.
Outputs (results/choice_cf_sobol/): points.parquet, sobol_indices.csv, summary.json, figures/.

  python -m cropchoice sobol [--n-base 128] [--seed 0] [--climate ssp585]
  python -m cropchoice sobol --toy --n-base 4
  python -m cropchoice sobol --figures-only        # local: figures from an existing results dir
"""
import argparse, json, os, time
from pathlib import Path

import numpy as np
import pandas as pd

BOX = {"t": (0.02, 0.30), "lam": (0.10, 0.90), "f": (0.20, 0.80), "f_both": (0.10, 0.50),
       "trigger": (0.30, 0.80), "basis": (0.0, 0.5), "loading": (0.0, 0.5), "theta": (0.20, 0.50)}
PNAMES = list(BOX); D = len(PNAMES)
COST_POL = ("safety_net", "insurance", "both")


def make_figures(out_dir):
    """Two figures from points.parquet / sobol_indices.csv / summary.json (needs matplotlib)."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    out_dir = Path(out_dir); (out_dir / "figures").mkdir(exist_ok=True)
    S = json.load(open(out_dir / "summary.json")); sob = pd.read_csv(out_dir / "sobol_indices.csv")
    pts = pd.read_parquet(out_dir / "points.parquet")
    countries_ = S["countries"]; mc = S["m_c"]; pn = S["params"]; C_ = len(countries_)
    ordc = sorted(countries_, key=lambda c: mc[c])
    fig, axes = plt.subplots(1, C_, figsize=(1.6 * C_ + 1, 2.6), sharey=True)
    for ax, cname in zip(np.atleast_1d(axes), ordc):
        t = sob[(sob.country == cname) & (sob.metric == "headline_ratio")].set_index("param").reindex(pn)
        ax.barh(pn, t["ST"].values, xerr=[np.clip(t["ST"] - t["ST_lo"], 0, None), np.clip(t["ST_hi"] - t["ST"], 0, None)],
                color="#4C7A3F", alpha=0.85, error_kw=dict(lw=0.6))
        ax.set_title(f"{cname}\n({mc[cname]:.0f} USD)", fontsize=8); ax.set_xlim(0, max(1.0, float(np.nanmax(t["ST_hi"])) * 1.05))
    np.atleast_1d(axes)[0].set_ylabel("policy parameter")
    fig.suptitle("Total Sobol index of the headline ratio (contraction / transfer switching)", fontsize=9)
    fig.tight_layout(); fig.savefig(out_dir / "figures" / "fig_sobol_ST_headline.pdf"); plt.close(fig)
    hr = pts[pts.metric == "headline_ratio"]
    dat = [hr[hr.country == c].value.values for c in ordc]; dat = [d[np.isfinite(d)] for d in dat]
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ax.violinplot(dat, showmedians=True); ax.axhline(1, color="grey", lw=0.8, ls="--")
    ax.set_xticks(range(1, C_ + 1)); ax.set_xticklabels([f"{c}\n({mc[c]:.0f})" for c in ordc])
    ax.set_yscale("log"); ax.set_ylabel("headline ratio over the parameter box")
    ax.set_title(f"{S['climate']}: {S['n_eval']} Saltelli points, posterior-median parameters", fontsize=9)
    fig.tight_layout(); fig.savefig(out_dir / "figures" / "fig_headline_ratio_box.pdf"); plt.close(fig)
    return out_dir / "figures"


def jansen(YA, YB, YAB):
    """YA, YB: (n,), YAB: (D, n).  Returns S1 (D,), ST (D,)."""
    Y = np.concatenate([YA, YB]); V = np.var(Y, ddof=1)
    if not np.isfinite(V) or V <= 0:
        return np.full(D, np.nan), np.full(D, np.nan)
    S1 = np.array([np.mean(YB * (YAB[i] - YA)) / V for i in range(D)])
    ST = np.array([0.5 * np.mean((YA - YAB[i]) ** 2) / V for i in range(D)])
    return S1, ST


def main(argv=None):
    parser = argparse.ArgumentParser(description="Sobol sweep of the policy parameters (Step 07 stage 5)")
    parser.add_argument("--toy", action="store_true")
    parser.add_argument("--n-base", type=int, default=128, dest="n_base")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--climate", default="ssp585")
    parser.add_argument("--K", type=int, default=200, help="posterior draws for the base-point comparison")
    parser.add_argument("--posterior", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--n-boot", type=int, default=200, dest="n_boot")
    parser.add_argument("--figures-only", action="store_true", dest="figures_only", help="only draw figures from an existing results dir")
    args = parser.parse_args(argv)
    os.environ.setdefault("BIRL_HOST_DEVICES", "1")
    if args.toy:
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
    from cropchoice.config import STEP07_DIR
    if args.figures_only:
        fdir = make_figures(Path(args.out) if args.out else STEP07_DIR / "results" / "choice_cf_sobol")
        print(f"[sobol] figures -> {fdir}"); return 0

    import jax, jax.numpy as jnp
    from scipy.stats import qmc, spearmanr
    from cropchoice.policies import POLICIES, DEFAULT_PARAMS, cost_context, apply_policy
    from cropchoice.counterfactual import scenario_metrics, toy_inputs, real_inputs
    from cropchoice.diagnostics import hpdi
    t_start = time.time()

    def say(msg):
        print(f"[sobol {time.strftime('%H:%M:%S')}] {msg}", flush=True)

    policies = list(POLICIES)
    if args.toy:
        inp = toy_inputs(K=min(args.K, 8))
        out_dir = Path(args.out) if args.out else STEP07_DIR / "results" / "choice_cf_toy" / "sobol"
    else:
        inp = real_inputs([args.climate], args.K, posterior=args.posterior)
        out_dir = Path(args.out) if args.out else STEP07_DIR / "results" / "choice_cf_sobol"
    out_dir.mkdir(parents=True, exist_ok=True); (out_dir / "figures").mkdir(exist_ok=True)
    countries, m_c, ci, mask, obs_action, draws, quant = (inp[k] for k in ("countries", "m_c", "ci", "mask", "obs_action", "draws", "quant"))
    C = len(countries); N, A = mask.shape

    med = {k: np.median(draws[k], axis=0, keepdims=True).astype(np.float32) for k in ("a", "b", "c", "asc")}
    a_med = med["a"][0]
    say(f"N={N} A={A} C={C} climate={args.climate} n_base={args.n_base} D={D} -> {args.n_base*(D+2)} points x {len(policies)} policies; devices={jax.devices()} toy={args.toy}")
    say(f"posterior-median a: {dict(zip(countries, a_med.round(3).tolist()))}")

    m_obs = jnp.asarray(m_c[ci]); mask_j = jnp.asarray(mask); ci_j = jnp.asarray(ci)
    b10, b50, b90 = (jnp.asarray(x) for x in quant["baseline"])
    ctx = cost_context(b10, b50, b90, jnp.asarray(obs_action), ci_j, C)
    c10, c50, c90 = (jnp.asarray(x) for x in quant[args.climate])

    def evaluate(params, drw, batch=10):
        res = {}
        for pol in policies:
            p10, p50, p90, info = apply_policy(c10, c50, c90, pol, params, m_obs, ctx)
            out = scenario_metrics(p10, p50, p90, c10, c50, c90, drw, mask_j, ci, C, m_obs, params, pol,
                                   batch=batch, rhos=(2.5,), info=info)
            Kd = out["exp_income"].shape[0]
            out["cost"] = np.broadcast_to(np.asarray(info["cost"])[None, :], (Kd, C)).copy()
            out["payout"] = np.broadcast_to(np.asarray(info["payout"])[None, :], (Kd, C)).copy()
            if pol == "safety_net":
                out["rcost"] = out["cost"].copy()
            elif pol == "insurance":
                out["rcost"] = out["payout"].copy()
            elif pol == "both":
                out["rcost"] = out["cost"] - params.get("loading", DEFAULT_PARAMS["loading"]) * out["payout"] + out["payout"]
            else:
                out["rcost"] = out["cost"].copy()
            res[pol] = out
        return res

    def point_outputs(res):
        none = res["none"]; o = {}
        for pol in policies[1:]:
            r = res[pol]
            o[f"switch_{pol}"] = 0.5 * np.abs(r["shares"] - none["shares"]).sum(-1)
            o[f"d_pbelow_{pol}"] = r["p_below_theta"] - none["p_below_theta"]
            o[f"d_expinc_{pol}"] = r["exp_income"] - none["exp_income"]
            o[f"cost_{pol}"] = r["cost"]
            if pol in COST_POL:
                with np.errstate(divide="ignore", invalid="ignore"):
                    o[f"pd_{pol}"] = np.where(r["cost"] > 1e-9, -o[f"d_pbelow_{pol}"] / r["cost"], np.nan)
                    o[f"pdr_{pol}"] = np.where(r["rcost"] > 1e-9, -o[f"d_pbelow_{pol}"] / r["rcost"], np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            o["headline_ratio"] = o["switch_contraction"] / o["switch_transfer"]
        o["sn_better"] = (o["pdr_safety_net"] > o["pdr_insurance"]).astype(float)
        o["pbelow_none"] = none["p_below_theta"]
        return o

    # ── Saltelli design ──
    sampler = qmc.Sobol(d=2 * D, scramble=True, seed=args.seed)
    U = sampler.random(args.n_base)
    lo = np.array([BOX[k][0] for k in PNAMES]); hi = np.array([BOX[k][1] for k in PNAMES])
    A_ = lo + (hi - lo) * U[:, :D]; B_ = lo + (hi - lo) * U[:, D:]
    design = [("A", None, A_), ("B", None, B_)]
    for i in range(D):
        AB = A_.copy(); AB[:, i] = B_[:, i]; design.append(("AB", i, AB))
    rows, mats = [], {}
    t0 = time.time(); n_eval = 0
    for block, i, M in design:
        acc = {}
        for j in range(M.shape[0]):
            params = {k: float(M[j, kk]) for kk, k in enumerate(PNAMES)}
            o = point_outputs(evaluate(params, med, batch=1)); n_eval += 1
            for metric, arr in o.items():
                acc.setdefault(metric, []).append(arr[0])
                for c in range(C):
                    rows.append({"block": block, "i": -1 if i is None else i, "j": j, "country": countries[c],
                                 "metric": metric, "value": float(arr[0, c]), **params})
        mats[(block, i)] = {m: np.stack(v) for m, v in acc.items()}
        if block != "AB" or i == D - 1:
            say(f"block {block}{'' if i is None else i} done: {n_eval} points, {time.time()-t0:.0f}s ({(time.time()-t0)/max(n_eval,1):.2f} s/point)")
    points = pd.DataFrame(rows)
    points["point"] = points["block"] + points["i"].astype(str) + "_" + points["j"].astype(str)
    points.to_parquet(out_dir / "points.parquet", index=False)
    say(f"points.parquet: {len(points)} rows, {n_eval} evaluations, {time.time()-t0:.0f}s")

    # ── Sobol indices ──
    METRICS = [m for m in mats[("A", None)] if not m.startswith("cost_")]
    rng_b = np.random.default_rng(args.seed + 1)
    idx_rows = []
    for c in range(C):
        for metric in METRICS:
            YA = mats[("A", None)][metric][:, c]; YB = mats[("B", None)][metric][:, c]
            YAB = np.stack([mats[("AB", i)][metric][:, c] for i in range(D)])
            ok = np.isfinite(YA) & np.isfinite(YB) & np.isfinite(YAB).all(0)
            if ok.sum() < 4:
                continue
            S1, ST = jansen(YA[ok], YB[ok], YAB[:, ok])
            boots = []; n_ok = int(ok.sum())
            for _ in range(args.n_boot):
                r = rng_b.integers(0, n_ok, n_ok)
                boots.append(jansen(YA[ok][r], YB[ok][r], YAB[:, ok][:, r]))
            b1 = np.array([b[0] for b in boots]); bt = np.array([b[1] for b in boots])
            for i, pn in enumerate(PNAMES):
                idx_rows.append({"country": countries[c], "metric": metric, "param": pn,
                                 "S1": S1[i], "S1_lo": np.nanpercentile(b1[:, i], 5.5), "S1_hi": np.nanpercentile(b1[:, i], 94.5),
                                 "ST": ST[i], "ST_lo": np.nanpercentile(bt[:, i], 5.5), "ST_hi": np.nanpercentile(bt[:, i], 94.5),
                                 "n_ok": n_ok})
    sob = pd.DataFrame(idx_rows, columns=["country", "metric", "param", "S1", "S1_lo", "S1_hi", "ST", "ST_lo", "ST_hi", "n_ok"])
    sob.to_csv(out_dir / "sobol_indices.csv", index=False)

    # ── ranking robustness ──
    AB_all = np.concatenate([mats[("A", None)]["headline_ratio"], mats[("B", None)]["headline_ratio"]] +
                            [mats[("AB", i)]["headline_ratio"] for i in range(D)], axis=0)
    rho_arr = np.array([spearmanr(row, -m_c).correlation for row in AB_all if np.isfinite(row).all()])
    share_rank = float(np.mean(rho_arr >= 0.8)) if len(rho_arr) else float("nan")
    low4 = [countries.index(x) for x in ("Malawi", "Ethiopia", "Uganda", "Tanzania") if x in countries]
    mali = countries.index("Mali") if "Mali" in countries else None
    ok_rows = np.isfinite(AB_all).all(1)
    cond = np.all(AB_all[ok_rows][:, low4] > 1, axis=1) if low4 else np.zeros(int(ok_rows.sum()), bool)
    if mali is not None:
        cond &= AB_all[ok_rows][:, mali] < 1
    share_pattern = float(np.mean(cond)) if ok_rows.any() else float("nan")
    snb = np.concatenate([mats[("A", None)]["sn_better"], mats[("B", None)]["sn_better"]] +
                         [mats[("AB", i)]["sn_better"] for i in range(D)], axis=0)
    share_sn = {countries[c]: float(np.nanmean(snb[:, c])) for c in range(C)}
    top2 = {}
    for metric in ("headline_ratio", "switch_safety_net", "switch_insurance", "pdr_safety_net", "pdr_insurance", "d_pbelow_safety_net", "d_pbelow_insurance"):
        top2[metric] = {}
        for c in countries:
            s = sob[(sob.country == c) & (sob.metric == metric)].sort_values("ST", ascending=False)
            top2[metric][c] = [(r.param, round(float(r.ST), 3)) for r in s.head(2).itertuples()]

    # ── base point: posterior vs parameter uncertainty ──
    say("base point with posterior draws ...")
    base = point_outputs(evaluate(dict(DEFAULT_PARAMS), draws, batch=10))
    hr_post = base["headline_ratio"]
    unc = {}
    for c in range(C):
        col_post = hr_post[:, c][np.isfinite(hr_post[:, c])]
        lo_, hi_ = hpdi(col_post) if len(col_post) >= 4 else (np.nan, np.nan)
        col = AB_all[:, c]; col = col[np.isfinite(col)]
        unc[countries[c]] = {"posterior_median": float(np.nanmedian(hr_post[:, c])), "posterior_hpdi89": [lo_, hi_],
                             "posterior_hpdi_width": float(hi_ - lo_) if np.isfinite(hi_) else None,
                             "param_box_median": float(np.median(col)) if len(col) else None,
                             "param_box_iqr": float(np.percentile(col, 75) - np.percentile(col, 25)) if len(col) else None,
                             "param_box_q05_q95": [float(np.percentile(col, 5)), float(np.percentile(col, 95))] if len(col) else None,
                             "share_points_ratio_gt1": float(np.mean(col > 1)) if len(col) else None}
    summary = {"toy": args.toy, "climate": args.climate, "n_base": args.n_base, "D": D, "params": PNAMES, "box": BOX,
               "n_eval": n_eval, "elapsed_s": round(time.time() - t_start, 1), "K_base": int(hr_post.shape[0]),
               "countries": countries, "m_c": {countries[c]: float(m_c[c]) for c in range(C)},
               "share_points_rank_spearman_ge_0.8": share_rank, "spearman_median": float(np.nanmedian(rho_arr)) if len(rho_arr) else None,
               "share_points_low4_gt1_and_mali_lt1": share_pattern,
               "share_points_safety_net_better_per_resource_dollar": share_sn,
               "top2_total_sobol": top2, "headline_uncertainty": unc,
               "posterior_median_a": dict(zip(countries, a_med.round(4).tolist()))}
    json.dump(summary, open(out_dir / "summary.json", "w"), indent=2, default=float)
    try:
        fdir = make_figures(out_dir); say(f"figures -> {fdir}")
    except ModuleNotFoundError as e:
        say(f"figures skipped ({e}); run --figures-only --out {out_dir} locally")
    say(f"rank share (Spearman>=0.8): {share_rank:.3f}; low4>1 & Mali<1: {share_pattern:.3f}; SN better per resource $: { {k: round(v,2) for k,v in share_sn.items()} }")
    say(f"wrote {out_dir}; total {time.time()-t_start:.0f}s")
    return out_dir
