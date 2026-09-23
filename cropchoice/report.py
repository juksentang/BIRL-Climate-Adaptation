"""Tables and figures of the choice counterfactual (Step 07 stage 4).

Reads results/choice_cf/{metrics,crop_shares}.parquet + run_info.json and writes
tables/*.csv (median, 89% HPDI) and figures/*.pdf.  `--toy` builds synthetic inputs
in a temporary directory and touches nothing under results/.
"""
from __future__ import annotations

import argparse
import json
import os, tempfile
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
from cropchoice.config import STEP07_DIR, DATA_DIR, HPDI_PROB
from cropchoice.diagnostics import hpdi
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS_DEFAULT = STEP07_DIR / "results" / "choice_cf"
ACTION_CFG = DATA_DIR / "action_space_config.json"
BIRL_SAMPLE = DATA_DIR / "birl_sample.parquet"

CLIMATES = ["baseline", "ssp245", "ssp585"]
POLICIES = ["none", "transfer", "contraction", "safety_net", "insurance", "both"]
COUNTRIES = ["Ethiopia", "Malawi", "Mali", "Nigeria", "Tanzania", "Uganda"]
CROPS = ["maize", "tree_crops", "tubers", "legumes", "sorghum_millet", "teff", "other", "rice", "wheat_barley"]
INTENS = ["low", "medium", "high"]
# country median of the chosen-action q50 (USD, baseline); overridden by run_info["m_c"] when present
M_C_FALLBACK = {"Ethiopia": 19.86, "Malawi": 19.12, "Mali": 129.96, "Nigeria": 102.74, "Tanzania": 26.03, "Uganda": 25.46}
METRICS = ["exp_income", "exp_median", "p_below_theta", "logsum", "cv_log", "cv_pct",
           "ce_rho1.5", "ce_rho2.5", "ce_rho3.5", "cost", "cost_behav", "payout", "premium", "switch_share"]

# ── figure style (mirrors paper/visualization/config.py, not imported) ──
SINGLE_COL, DOUBLE_COL = 3.5, 7.08
RC = {"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
      "font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7, "xtick.labelsize": 6,
      "ytick.labelsize": 6, "legend.fontsize": 6, "figure.dpi": 150, "savefig.dpi": 300,
      "savefig.bbox": "tight", "savefig.pad_inches": 0.05, "axes.linewidth": 0.5,
      "xtick.major.width": 0.5, "ytick.major.width": 0.5, "lines.linewidth": 0.8}
POLICY_COLORS = {"none": "#7f7f7f", "transfer": "#d95f02", "contraction": "#1b9e77",
                 "safety_net": "#7570b3", "insurance": "#e7298a", "both": "#66a61e"}
CLIMATE_COLORS = {"baseline": "#4d4d4d", "ssp245": "#fdae61", "ssp585": "#d7191c"}


# ── helpers ──
def summarise(g, col="value"):
    x = g[col].to_numpy(float)
    lo, hi = hpdi(x, HPDI_PROB)
    return pd.Series({"median": float(np.nanmedian(x)) if np.isfinite(x).any() else np.nan,
                      "hpdi_lo": lo, "hpdi_hi": hi, "n_draws": int(len(x)),
                      "share_finite": float(np.isfinite(x).mean()) if len(x) else np.nan})


def action_labels():
    if ACTION_CFG.exists():
        al = json.load(open(ACTION_CFG))["action_labels"]
        return [al[str(i)] if isinstance(al, dict) else al[i] for i in range(27)]
    return [f"{c}_{s}" for c in CROPS for s in INTENS]


def wide(metrics: pd.DataFrame, metric: str) -> pd.DataFrame:
    """metric values as a frame indexed by (climate, policy, country, draw)."""
    m = metrics[metrics.metric == metric]
    return m.set_index(["climate", "policy", "country", "draw"])["value"].sort_index()


def paired_delta(w: pd.Series, a, b) -> pd.DataFrame:
    """w[a] - w[b] paired by (country, draw); a, b are (climate, policy) tuples."""
    xa = w.loc[a[0], a[1]]; xb = w.loc[b[0], b[1]]
    d = (xa - xb).dropna().rename("delta").reset_index()
    return d


# ── toy data ──
def make_toy(out_dir: Path, K=60, seed=0):
    rng = np.random.default_rng(seed)
    labels = action_labels()
    m_c = M_C_FALLBACK
    a_true = {"Ethiopia": 0.41, "Malawi": -0.63, "Mali": 1.34, "Nigeria": 1.71, "Tanzania": -0.09, "Uganda": 0.91}
    rows, shares = [], []
    for c in COUNTRIES:
        base = rng.dirichlet(np.full(27, 0.6))
        for cl in CLIMATES:
            cl_shift = {"baseline": 0.0, "ssp245": -0.06, "ssp585": -0.12}[cl]
            for pol in POLICIES:
                pol_lvl = {"none": 0, "transfer": 0.10, "contraction": 0.0, "safety_net": 0.06, "insurance": 0.03, "both": 0.09}[pol]
                pol_var = {"none": 0, "transfer": 0.0, "contraction": 0.5, "safety_net": 0.25, "insurance": 0.35, "both": 0.5}[pol]
                cost = {"none": 0, "transfer": 0.10, "contraction": 0.0, "safety_net": 0.04, "insurance": 0.02, "both": 0.06}[pol] * m_c[c]
                a_c = a_true[c]
                resp = 0.15 * max(a_c, 0) * pol_lvl * 10 + 0.4 * pol_var          # switching intensity
                for k in range(K):
                    ak = a_c + rng.normal(0, 0.05)
                    inc = m_c[c] * (1 + cl_shift + pol_lvl + rng.normal(0, 0.02))
                    p_theta = float(np.clip(0.25 - cl_shift * 0.8 - pol_var * 0.15 - pol_lvl * 0.2 + rng.normal(0, 0.01), 0.01, 0.9))
                    dW = (cl_shift + pol_lvl + 0.3 * pol_var) * max(ak, 0) + rng.normal(0, 0.01)
                    cv_log = dW / ak if ak > 0 else np.nan
                    vals = {"exp_income": inc, "exp_median": inc * 0.85, "p_below_theta": p_theta,
                            "logsum": 1.2 + dW, "cv_log": cv_log,
                            "cv_pct": (np.exp(cv_log) - 1) if np.isfinite(cv_log) else np.nan,
                            "ce_rho1.5": inc * 0.9, "ce_rho2.5": inc * 0.8, "ce_rho3.5": inc * 0.7,
                            "cost": cost, "cost_behav": cost * 0.95, "payout": cost * 0.8 if pol in ("insurance", "both") else 0.0,
                            "premium": cost * 1.2 if pol in ("insurance", "both") else 0.0,
                            "switch_share": 0.0 if pol == "none" else float(np.clip(resp + rng.normal(0, 0.01), 0, 1))}
                    for mname, v in vals.items():
                        rows.append((cl, pol, c, k, mname, float(v)))
                    sh = base * np.exp(rng.normal(0, 0.05, 27) + resp * rng.normal(0, 0.6, 27) + cl_shift * rng.normal(0, 1, 27))
                    sh = sh / sh.sum()
                    shares.extend((cl, pol, c, k, a, float(sh[a])) for a in range(27))
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["climate", "policy", "country", "draw", "metric", "value"]).to_parquet(out_dir / "metrics.parquet", index=False)
    pd.DataFrame(shares, columns=["climate", "policy", "country", "draw", "action", "share"]).to_parquet(out_dir / "crop_shares.parquet", index=False)
    # synthetic observed shares (toy only: never read real data here)
    obs = []
    for c in COUNTRIES:
        o = rng.dirichlet(np.full(27, 0.6)); obs.extend((c, a, float(o[a])) for a in range(27))
    pd.DataFrame(obs, columns=["country", "action", "obs_share"]).to_parquet(out_dir / "observed_shares_toy.parquet", index=False)
    json.dump({"toy": True, "K": K, "m_c": m_c, "labels": labels}, open(out_dir / "run_info.json", "w"), indent=2)
    return out_dir


def observed_shares(results_dir: Path, toy: bool) -> pd.DataFrame:
    if toy:
        return pd.read_parquet(results_dir / "observed_shares_toy.parquet")
    df = pd.read_parquet(BIRL_SAMPLE, columns=["country", "action_id"])
    tab = (df.groupby(["country", "action_id"]).size().rename("n").reset_index())
    tab["obs_share"] = tab["n"] / tab.groupby("country")["n"].transform("sum")
    tab = tab.rename(columns={"action_id": "action"})[["country", "action", "obs_share"]]
    full = pd.MultiIndex.from_product([COUNTRIES, range(27)], names=["country", "action"]).to_frame(index=False)
    return full.merge(tab, how="left").fillna({"obs_share": 0.0})


# ── tables ──
def build_tables(metrics, shares, obs, m_c, labels, tdir: Path):
    tdir.mkdir(parents=True, exist_ok=True)
    # 1. long summary
    summ = (metrics.groupby(["climate", "policy", "country", "metric"])["value"].apply(lambda x: summarise(x.to_frame("value"))).unstack().reset_index())
    summ.to_csv(tdir / "metrics_summary.csv", index=False)

    # 2. climate loss: (ssp, none) - (baseline, none), paired by draw
    rows = []
    for ssp in ["ssp245", "ssp585"]:
        for met in ["exp_income", "p_below_theta", "cv_pct"]:
            w = wide(metrics, met)
            if met == "cv_pct":            # already relative to (baseline, none)
                d = w.loc[ssp, "none"].rename("delta").reset_index()
            else:
                d = paired_delta(w, (ssp, "none"), ("baseline", "none"))
            base = w.loc["baseline", "none"]
            for c, g in d.groupby("country"):
                lo, hi = hpdi(g["delta"]); dv = g["delta"].to_numpy(float)
                med = float(np.nanmedian(dv)) if np.isfinite(dv).any() else np.nan
                bx = base.loc[c].to_numpy(float) if c in base.index.get_level_values(0) else np.array([])
                b = float(np.nanmedian(bx)) if np.isfinite(bx).any() else np.nan
                rows.append({"climate": ssp, "metric": met, "country": c, "delta_median": med,
                             "delta_lo": lo, "delta_hi": hi, "baseline_median": b,
                             "pct_change": med / b * 100 if (met == "exp_income" and b) else np.nan})
    pd.DataFrame(rows).to_csv(tdir / "climate_loss.csv", index=False)

    # 3. policy effects within climate, relative to none, and per dollar of public cost
    rows = []
    cost_w = wide(metrics, "cost")
    for cl in CLIMATES:
        for pol in POLICIES[1:]:
            for met in ["exp_income", "p_below_theta", "switch_share", "cv_log", "cv_pct", "ce_rho2.5"]:
                w = wide(metrics, met)
                try:
                    if met == "switch_share":
                        d = w.loc[cl, pol].rename("delta").reset_index()
                    elif met == "cv_pct":      # per cent effect of the policy: exp(dlog) - 1
                        dl = paired_delta(wide(metrics, "cv_log"), (cl, pol), (cl, "none"))
                        d = dl.assign(delta=np.exp(dl["delta"]) - 1)
                    else:
                        d = paired_delta(w, (cl, pol), (cl, "none"))
                except KeyError:
                    continue
                cst = cost_w.loc[cl, pol].rename("cost").reset_index()
                d = d.merge(cst, on=["country", "draw"], how="left")
                d["per_dollar"] = np.where(d["cost"] > 0, d["delta"] / d["cost"], np.nan)
                for c, g in d.groupby("country"):
                    lo, hi = hpdi(g["delta"]); plo, phi = hpdi(g["per_dollar"])
                    rows.append({"climate": cl, "policy": pol, "metric": met, "country": c,
                                 "delta_median": float(np.nanmedian(g["delta"])), "delta_lo": lo, "delta_hi": hi,
                                 "cost_median": float(np.nanmedian(g["cost"])),
                                 "per_dollar_median": float(np.nanmedian(g["per_dollar"])) if np.isfinite(g["per_dollar"]).any() else np.nan,
                                 "per_dollar_lo": plo, "per_dollar_hi": phi,
                                 "share_finite": float(np.isfinite(g["delta"]).mean())})
    pd.DataFrame(rows).to_csv(tdir / "policy_effects.csv", index=False)

    # 4. headline: contraction vs transfer switching, ratio per draw
    rows = []
    w = wide(metrics, "switch_share")
    for cl in CLIMATES:
        con = w.loc[cl, "contraction"]; tr = w.loc[cl, "transfer"]
        j = pd.concat([con.rename("contraction"), tr.rename("transfer")], axis=1).dropna().reset_index()
        j["ratio"] = j["contraction"] / j["transfer"].replace(0, np.nan)
        j["log_ratio"] = np.log(j["ratio"])
        for c, g in j.groupby("country"):
            rlo, rhi = hpdi(g["ratio"]); clo, chi = hpdi(g["contraction"]); tlo, thi = hpdi(g["transfer"])
            rows.append({"climate": cl, "country": c, "m_c": m_c.get(c, np.nan),
                         "switch_contraction": float(np.nanmedian(g["contraction"])), "contraction_lo": clo, "contraction_hi": chi,
                         "switch_transfer": float(np.nanmedian(g["transfer"])), "transfer_lo": tlo, "transfer_hi": thi,
                         "ratio_median": float(np.nanmedian(g["ratio"])), "ratio_lo": rlo, "ratio_hi": rhi,
                         "p_ratio_gt1": float(np.nanmean(g["ratio"] > 1))})
    hr = pd.DataFrame(rows).sort_values(["climate", "m_c"])
    hr.to_csv(tdir / "headline_ratio.csv", index=False)

    # 5. PPC: predicted (baseline, none) shares vs observed
    pred = (shares[(shares.climate == "baseline") & (shares.policy == "none")]
            .groupby(["country", "action"])["share"].agg(pred_median="median", pred_lo=lambda x: hpdi(x)[0], pred_hi=lambda x: hpdi(x)[1]).reset_index())
    ppc = pred.merge(obs, on=["country", "action"], how="left")
    ppc["label"] = ppc["action"].map(lambda a: labels[a])
    ppc["abs_diff"] = (ppc["pred_median"] - ppc["obs_share"]).abs()
    ppc.to_csv(tdir / "ppc_crop_shares.csv", index=False)
    (ppc.groupby("country")["abs_diff"].agg(l1_half=lambda x: x.sum() / 2, max_abs="max").reset_index()
        .to_csv(tdir / "ppc_crop_shares_by_country.csv", index=False))

    # 6. crop shifts by crop (9), relative to (baseline, none), paired by draw
    sh = shares.copy(); sh["crop"] = sh["action"] // 3
    bycrop = sh.groupby(["climate", "policy", "country", "draw", "crop"])["share"].sum()
    base = bycrop.loc["baseline", "none"]
    rows = []
    for (cl, pol), g in bycrop.groupby(level=[0, 1]):
        if (cl, pol) == ("baseline", "none"):
            continue
        d = (g.droplevel([0, 1]) - base).dropna().rename("delta").reset_index()
        for (c, cr), gg in d.groupby(["country", "crop"]):
            lo, hi = hpdi(gg["delta"])
            rows.append({"climate": cl, "policy": pol, "country": c, "crop": CROPS[cr],
                         "delta_share_median": float(np.nanmedian(gg["delta"])), "delta_lo": lo, "delta_hi": hi,
                         "base_share_median": float(np.nanmedian(base.loc[c].xs(cr, level="crop"))) if c in base.index.get_level_values(0) else np.nan})
    pd.DataFrame(rows).to_csv(tdir / "crop_shifts.csv", index=False)
    return summ, hr


# ── figures ──
def fig_headline(hr, m_c, fdir):
    plt.rcParams.update(RC)
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.4))
    order = sorted(COUNTRIES, key=lambda c: m_c.get(c, 0))
    x = np.arange(len(order)); off = {"baseline": -0.22, "ssp245": 0.0, "ssp585": 0.22}
    for cl in CLIMATES:
        d = hr[hr.climate == cl].set_index("country").reindex(order)
        ax.errorbar(x + off[cl], d["ratio_median"], yerr=[d["ratio_median"] - d["ratio_lo"], d["ratio_hi"] - d["ratio_median"]],
                    fmt="o", ms=3, lw=0.7, capsize=1.5, color=CLIMATE_COLORS[cl], label=cl)
    ax.axhline(1, color="#999999", lw=0.5, ls="--")
    ax.set_xticks(x); ax.set_xticklabels([f"{c}\n({m_c.get(c, float('nan')):.0f} USD)" for c in order])
    ax.set_yscale("log"); ax.set_ylabel("switching: variance-cut / income transfer")
    ax.set_title("Level vs variance responsiveness (countries by median income)")
    ax.legend(frameon=False, ncol=3, loc="upper right")
    fig.savefig(fdir / "fig_headline_ratio.pdf"); plt.close(fig)


def fig_switch(summ, fdir, climate="ssp585"):
    plt.rcParams.update(RC)
    d = summ[(summ.metric == "switch_share") & (summ.climate == climate) & (summ.policy != "none")]
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 2.6))
    x = np.arange(len(COUNTRIES)); w = 0.15
    for j, pol in enumerate(POLICIES[1:]):
        s = d[d.policy == pol].set_index("country").reindex(COUNTRIES)
        ax.bar(x + (j - 2) * w, s["median"], w, color=POLICY_COLORS[pol], label=pol,
               yerr=[s["median"] - s["hpdi_lo"], s["hpdi_hi"] - s["median"]], error_kw={"lw": 0.5, "capsize": 1})
    ax.set_xticks(x); ax.set_xticklabels(COUNTRIES); ax.set_ylabel("share of plots switching crop x intensity")
    ax.set_title(f"Choice response to policies under {climate} (2050)"); ax.legend(frameon=False, ncol=5)
    fig.savefig(fdir / f"fig_switch_{climate}.pdf"); plt.close(fig)


def fig_downside(summ, fdir):
    plt.rcParams.update(RC)
    pols = ["none", "safety_net", "insurance"]
    d = summ[(summ.metric == "p_below_theta") & (summ.policy.isin(pols))]
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, 2.4), sharey=True)
    for ax, cl in zip(axes, CLIMATES):
        x = np.arange(len(COUNTRIES)); w = 0.25
        for j, pol in enumerate(pols):
            s = d[(d.climate == cl) & (d.policy == pol)].set_index("country").reindex(COUNTRIES)
            ax.bar(x + (j - 1) * w, s["median"], w, color=POLICY_COLORS[pol], label=pol,
                   yerr=[s["median"] - s["hpdi_lo"], s["hpdi_hi"] - s["median"]], error_kw={"lw": 0.5, "capsize": 1})
        ax.set_title(cl); ax.set_xticks(x); ax.set_xticklabels(COUNTRIES, rotation=45, ha="right")
    axes[0].set_ylabel("P(income < 0.3 x median)"); axes[0].legend(frameon=False)
    fig.savefig(fdir / "fig_downside.pdf"); plt.close(fig)


def fig_cv(summ, fdir):
    plt.rcParams.update(RC)
    d = summ[(summ.metric == "cv_pct") & (summ.share_finite >= 0.5)]
    ctry = [c for c in COUNTRIES if c in set(d.country)]
    if not ctry:
        return
    fig, axes = plt.subplots(1, len(CLIMATES), figsize=(DOUBLE_COL, 2.4), sharey=True)
    for ax, cl in zip(np.atleast_1d(axes), CLIMATES):
        x = np.arange(len(ctry)); w = 0.14
        for j, pol in enumerate(POLICIES):
            s = d[(d.climate == cl) & (d.policy == pol)].set_index("country").reindex(ctry)
            ax.bar(x + (j - 2.5) * w, 100 * s["median"], w, color=POLICY_COLORS[pol], label=pol,
                   yerr=[100 * (s["median"] - s["hpdi_lo"]), 100 * (s["hpdi_hi"] - s["median"])], error_kw={"lw": 0.5, "capsize": 1})
        ax.axhline(0, color="#999999", lw=0.5); ax.set_title(cl); ax.set_xticks(x); ax.set_xticklabels(ctry)
    np.atleast_1d(axes)[0].set_ylabel("compensating variation, % of income (vs baseline / none)")
    np.atleast_1d(axes)[0].legend(frameon=False, ncol=2)
    fig.suptitle("Logsum money metric, countries with a > 0 only", fontsize=8)
    fig.savefig(fdir / "fig_cv_pct.pdf"); plt.close(fig)


def fig_heatmap(tdir, fdir):
    plt.rcParams.update(RC)
    cs = pd.read_csv(tdir / "crop_shifts.csv")
    d = cs[(cs.climate == "ssp585") & (cs.policy == "none")].pivot(index="country", columns="crop", values="delta_share_median").reindex(index=COUNTRIES, columns=CROPS)
    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.4, 2.4))
    v = np.nanmax(np.abs(d.to_numpy())) if np.isfinite(d.to_numpy()).any() else 0.01
    im = ax.imshow(d.to_numpy() * 100, cmap="RdBu_r", vmin=-100 * v, vmax=100 * v, aspect="auto")
    ax.set_xticks(range(len(CROPS))); ax.set_xticklabels(CROPS, rotation=45, ha="right")
    ax.set_yticks(range(len(COUNTRIES))); ax.set_yticklabels(COUNTRIES)
    for i in range(len(COUNTRIES)):
        for j in range(len(CROPS)):
            val = d.iloc[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{100 * val:+.1f}", ha="center", va="center", fontsize=5)
    plt.colorbar(im, ax=ax, label="crop share change, pp (ssp585 vs baseline, no policy)")
    ax.set_title("Predicted 2050 crop shifts")
    fig.savefig(fdir / "fig_crop_shift_heatmap.pdf"); plt.close(fig)


# ── main ──
def main(argv=None):
    ap = argparse.ArgumentParser(description="tables and figures of the choice counterfactual (Step 07 stage 4)")
    ap.add_argument("--results", type=Path, default=RESULTS_DEFAULT)
    ap.add_argument("--out", type=Path, default=None, help="tables/ and figures/ go here (default: --results)")
    ap.add_argument("--toy", action="store_true", help="synthetic inputs in the scratchpad; touches nothing under results/")
    args = ap.parse_args(argv)
    if args.toy:
        scratch = Path(os.environ.get("CLAUDE_SCRATCHPAD", tempfile.gettempdir())) / "choice_cf_toy"
        args.results = make_toy(scratch); args.out = scratch
    out = args.out or args.results
    tdir, fdir = out / "tables", out / "figures"; fdir.mkdir(parents=True, exist_ok=True)

    metrics = pd.read_parquet(args.results / "metrics.parquet")
    shares = pd.read_parquet(args.results / "crop_shares.parquet")
    info = json.load(open(args.results / "run_info.json")) if (args.results / "run_info.json").exists() else {}
    m_c = {k: float(v) for k, v in info.get("m_c", M_C_FALLBACK).items()} if isinstance(info.get("m_c", None), dict) else M_C_FALLBACK
    labels = info.get("labels") or action_labels()
    obs = observed_shares(args.results, args.toy)
    missing = sorted(set(METRICS) - set(metrics.metric.unique()))
    if missing:
        print(f"note: metrics absent from input: {missing}", file=sys.stderr)

    summ, hr = build_tables(metrics, shares, obs, m_c, labels, tdir)
    fig_headline(hr, m_c, fdir); fig_switch(summ, fdir); fig_downside(summ, fdir); fig_cv(summ, fdir); fig_heatmap(tdir, fdir)
    print(f"tables -> {tdir}\nfigures -> {fdir}")
    print(hr[hr.climate == "ssp585"][["country", "m_c", "ratio_median", "ratio_lo", "ratio_hi"]].to_string(index=False))


if __name__ == "__main__":
    main()
