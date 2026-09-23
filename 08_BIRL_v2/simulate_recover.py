#!/usr/bin/env python3
"""
Simulation-recovery test for BIRL v2 (v2_country).  CLUSTER ONLY (loads the
222K-row data).

Two truth sets (country order = sorted names: Ethiopia, Malawi, Mali, Nigeria,
Tanzania, Uganda) simulate actions on the REAL q arrays and feasibility masks
with simulate_actions(); the model (flat priors by default, spec-critic P7) is
then fitted:

  --mode svi   smoke check: AutoMultivariateNormal guide, Adam 1e-2, 3000
               steps, POINT ESTIMATES only (guide median) + log-lik at truth
               vs at the estimate.  Reports the tolerances but is NOT the PASS
               verdict.
  --mode nuts  PASS verdict (spec-critic P9): 2 chains x 500 warmup + 500
               samples.  PASS iff, for every country in BOTH sets,
               |d rho| <= 0.25, |d s| <= 0.05, |d log beta| <= 0.2 (posterior
               median vs truth) AND >= 80% of the 36 truths (2 sets x 3 params
               x 6 countries) lie inside their 89% HPDI.  Also reports
               z-scores and log-lik(truth) vs log-lik(median).

Outputs: outputs/recovery/<mode>/{report.md, report.json, run.log,
         sim_actions_<set>.npy, nuts_posterior_<set>.npz | svi_estimates_<set>.json}

Usage (from 08_BIRL_v2/):
    python3 simulate_recover.py --mode svi
    python3 simulate_recover.py --mode nuts
    python3 simulate_recover.py --mode nuts --sets A --nuts-chains 2 --chain-method vectorized
    python3 simulate_recover.py --mode nuts --resume     # after a TIMEOUT: sets whose
        # nuts_posterior_<set>.npz exists are loaded, not re-fitted, and finished sets
        # already in report.json but not in --sets are kept, so A and B can be run as
        # separate jobs and still yield one PASS/FAIL verdict.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

parser = argparse.ArgumentParser(description="BIRL v2 simulation-recovery test")
parser.add_argument("--mode", required=True, choices=["svi", "nuts"])
parser.add_argument("--sets", nargs="+", default=["A", "B"], choices=["A", "B"])
parser.add_argument("--svi-steps", type=int, default=3000)
parser.add_argument("--svi-lr", type=float, default=1e-2)
parser.add_argument("--nuts-warmup", type=int, default=500)
parser.add_argument("--nuts-samples", type=int, default=500)
parser.add_argument("--nuts-chains", type=int, default=2)
parser.add_argument("--chain-method", choices=["parallel", "vectorized", "sequential"], default=None)
parser.add_argument("--hier-priors", action="store_true",
                    help="use the hierarchical (centred) priors instead of --flat-priors")
parser.add_argument("--noncentered", action="store_true", help="with --hier-priors only")
parser.add_argument("--s-max", type=float, default=0.6)
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--tol-rho", type=float, default=0.25)
parser.add_argument("--tol-s", type=float, default=0.05)
parser.add_argument("--tol-logbeta", type=float, default=0.2)
parser.add_argument("--hpdi-frac", type=float, default=0.8)
parser.add_argument("--tag", type=str, default=None, help="output subdirectory suffix")
parser.add_argument("--resume", action="store_true",
                    help="nuts: reuse nuts_posterior_<set>.npz where present and keep finished "
                         "sets from an existing report.json (per-set checkpointing)")
args = parser.parse_args()

import numpy as np

# src.config first: it sets numpyro's host device count BEFORE jax is imported
from src.config import (DATA_DIR, OUT_DIR, HPDI_PROB, DEVICE_INFO, PLATFORM, N_DEVICES,  # noqa: E402
                        log, add_file_log, peak_rss_gb)
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpyro  # noqa: E402
from numpyro.infer import SVI, Trace_ELBO  # noqa: E402
from numpyro.infer.autoguide import AutoMultivariateNormal  # noqa: E402
from numpyro.diagnostics import hpdi  # noqa: E402

from src.data_loader import load_data, model_kwargs
from src.models import v2_country, simulate_actions, derive_country_params, log_likelihood
from src.mcmc_runner import (run_mcmc, samples_by_chain, choose_chain_method, save_posterior,
                             load_posterior)

REC_DIR = OUT_DIR / "recovery" / (args.mode + (f"_{args.tag}" if args.tag else ""))
REC_DIR.mkdir(parents=True, exist_ok=True)
add_file_log(REC_DIR / "run.log")

TRUTH = {
    "A": {"rho": [1.5, 3.0, 2.0, 1.0, 2.5, 3.5],
          "s":   [0.20, 0.40, 0.30, 0.15, 0.50, 0.35],
          "beta": [3, 8, 5, 2, 6, 4]},
    "B": {"rho": [0.5, 4.5, 1.0, 3.0, 2.0, 1.5],
          "s":   [0.55, 0.10, 0.58, 0.05, 0.30, 0.45],
          "beta": [10, 1.5, 4, 6, 2, 8]},
}
CORE = ["rho_c", "s_c", "beta_c"]          # the 18 truths per set judged by the PASS rule
TOL = {"rho_c": args.tol_rho, "s_c": args.tol_s, "beta_c": args.tol_logbeta}


def err(site, est, truth):
    """|d rho|, |d s|, |d log beta| (gamma_c reported as |d gamma| in USD, not judged)."""
    if site == "beta_c":
        return abs(np.log(est) - np.log(truth))
    return abs(est - truth)


def summarize(est, truth, countries, draws=None):
    """est: site -> (6,) point estimates; draws: site -> (n, 6) or None (SVI)."""
    rows = []
    for site in CORE + ["gamma_c"]:
        for c, name in enumerate(countries):
            t = float(truth[site][c])
            e = float(est[site][c])
            row = {"param": site, "country": name, "truth": t, "estimate": e,
                   "abs_err": float(err(site, e, t)),
                   "tol": TOL.get(site), "judged": site in CORE}
            row["ok"] = bool(row["abs_err"] <= TOL[site]) if site in CORE else None
            if draws is not None:
                x = np.asarray(draws[site][:, c], np.float64)
                lo, hi = hpdi(x, prob=HPDI_PROB)
                sd = float(x.std(ddof=1))
                row.update({"sd": sd, "hpdi_lo": float(lo), "hpdi_hi": float(hi),
                            "in_hpdi": bool(lo <= t <= hi),
                            "z": float((e - t) / sd) if sd > 0 else float("nan")})
            rows.append(row)
    return rows


def model_kw(data_kwargs):
    return {"data": data_kwargs, "s_max": args.s_max,
            "flat_priors": not args.hier_priors, "noncentered": args.noncentered}


def fit_svi(data_kwargs, seed, m_c):
    guide = AutoMultivariateNormal(v2_country, init_loc_fn=numpyro.infer.init_to_median())
    svi = SVI(v2_country, guide, numpyro.optim.Adam(args.svi_lr), Trace_ELBO())
    t0 = time.time()
    res = svi.run(jax.random.PRNGKey(seed), args.svi_steps, progress_bar=False,
                  **model_kw(data_kwargs))
    losses = np.asarray(res.losses)
    log.info(f"    SVI {args.svi_steps} steps in {time.time()-t0:.0f}s; "
             f"loss first/last = {losses[0]:.1f} / {losses[-1]:.1f}")
    med = {k: np.asarray(v) for k, v in guide.median(res.params).items()}
    est = derive_country_params(med, m_c, s_max=args.s_max)
    info = {"elapsed_s": time.time() - t0, "loss_first": float(losses[0]),
            "loss_last": float(losses[-1]), "loss_min": float(np.nanmin(losses)),
            "finite_losses": bool(np.isfinite(losses).all()),
            "guide_median_latents": {k: v.tolist() for k, v in med.items()}}
    return est, info


def fit_nuts(data_kwargs, seed, method, npz_path=None):
    """Fit NUTS, or with --resume load a finished set's posterior from npz_path."""
    t0 = time.time()
    if args.resume and npz_path is not None and npz_path.exists():
        samples, div, meta = load_posterior(npz_path)
        elapsed = 0.0
        log.info(f"    NUTS: loaded existing {npz_path} (--resume; fitted {meta.get('date', '?')}), "
                 f"{next(iter(samples.values())).shape[:2]} draws, div={int(div.sum())}/{div.size}")
        resumed = True
    else:
        mcmc = run_mcmc(v2_country, model_kw(data_kwargs), args.nuts_warmup, args.nuts_samples,
                        args.nuts_chains, seed=seed, chain_method=method)
        samples, div, _ = samples_by_chain(mcmc)
        elapsed = time.time() - t0
        log.info(f"    NUTS {args.nuts_chains}x({args.nuts_warmup}+{args.nuts_samples}) in "
                 f"{elapsed/60:.1f} min, div={int(div.sum())}/{div.size}")
        resumed = False
    flat = {k: v.reshape((-1,) + v.shape[2:]) for k, v in samples.items()}
    est = {k: np.median(flat[k], axis=0) for k in CORE + ["gamma_c"]}
    return est, flat, samples, div, {"elapsed_s": elapsed, "div": int(div.sum()),
                                     "div_total": int(div.size), "loaded_from_npz": resumed}


def md_table(rows, with_post):
    if with_post:
        out = ["| param | country | truth | median | sd | z | 89% HPDI | in HPDI | abs err | tol | ok |",
               "|---|---|---|---|---|---|---|---|---|---|---|"]
    else:
        out = ["| param | country | truth | estimate | abs err | tol | ok |",
               "|---|---|---|---|---|---|---|"]
    for r in rows:
        ok = "-" if r["ok"] is None else ("ok" if r["ok"] else "FAIL")
        tol = "-" if r["tol"] is None else f"{r['tol']:.2f}"
        if with_post:
            out.append(f"| {r['param']} | {r['country']} | {r['truth']:.4g} | {r['estimate']:.4g} | "
                       f"{r['sd']:.3g} | {r['z']:+.2f} | [{r['hpdi_lo']:.4g}, {r['hpdi_hi']:.4g}] | "
                       f"{'yes' if r['in_hpdi'] else 'no'} | {r['abs_err']:.3g} | {tol} | {ok} |")
        else:
            out.append(f"| {r['param']} | {r['country']} | {r['truth']:.4g} | {r['estimate']:.4g} | "
                       f"{r['abs_err']:.3g} | {tol} | {ok} |")
    return "\n".join(out)


def verdict(report):
    """PASS rule on NUTS over both sets; SVI -> smoke summary; partial -> INCOMPLETE."""
    judged = [r for s in report["sets"].values() for r in s["rows"] if r["judged"]]
    n_ok = sum(r["ok"] for r in judged)
    tol_ok = bool(judged) and n_ok == len(judged)
    if args.mode == "svi":
        return ("SMOKE", f"SVI point estimates: {n_ok}/{len(judged)} tolerances met "
                f"(smoke check only; the PASS verdict is judged on --mode nuts)")
    n_in = sum(r["in_hpdi"] for r in judged)
    frac = n_in / len(judged) if judged else 0.0
    if set(report["sets"]) != {"A", "B"}:
        return ("INCOMPLETE", f"sets run: {sorted(report['sets'])}; PASS needs both A and B "
                f"({n_ok}/{len(judged)} tolerances, {n_in}/{len(judged)} in HPDI)")
    ok = tol_ok and frac >= args.hpdi_frac
    return ("PASS" if ok else "FAIL",
            f"{n_ok}/{len(judged)} tolerances met (need all), {n_in}/{len(judged)} = {frac:.0%} "
            f"truths inside 89% HPDI (need >= {args.hpdi_frac:.0%})")


def write_report(report):
    """report.json + report.md from report['sets'] (each set carries its own 'md' section,
    so sets kept from an earlier report.json are rendered too)."""
    md_sections = [line for name in sorted(report["sets"]) for line in report["sets"][name]["md"]]
    v, why = verdict(report)
    report["verdict"] = v
    report["verdict_reason"] = why
    report["peak_rss_gb"] = peak_rss_gb()
    (REC_DIR / "report.json").write_text(json.dumps(report, indent=2, default=str))
    head = [f"# BIRL v2 simulation-recovery report ({args.mode})", "",
            f"Date: {report['date']}  ", f"Platform: {PLATFORM} x {N_DEVICES}  ",
            f"Countries (order): {report['countries']}  ",
            f"m_c (USD): {', '.join(f'{c}={m:.2f}' for c, m in zip(report['countries'], report['m_c']))}  ",
            f"Priors: {'hierarchical' + (' non-centred' if args.noncentered else ' centred') if args.hier_priors else 'flat (independent wide per-country)'}, s_max={args.s_max}  ",
            (f"SVI: AutoMultivariateNormal, Adam lr={args.svi_lr}, {args.svi_steps} steps, point estimates (guide median)  "
             if args.mode == "svi" else
             f"NUTS: {args.nuts_chains} x ({args.nuts_warmup}+{args.nuts_samples}), chain_method={report.get('chain_method')}  "),
            f"Tolerances: |d rho| <= {args.tol_rho}, |d s| <= {args.tol_s}, |d log beta| <= {args.tol_logbeta}; "
            f"PASS (NUTS only) additionally needs >= {args.hpdi_frac:.0%} of the 36 truths inside their {HPDI_PROB:.0%} HPDI  ",
            "", f"**Verdict: {v}** - {why}", ""]
    (REC_DIR / "report.md").write_text("\n".join(head + md_sections))


def main():
    log.info("=" * 64)
    log.info(f"BIRL v2 simulation-recovery  mode={args.mode}  sets={args.sets}  "
             f"priors={'hier' if args.hier_priors else 'flat'}  s_max={args.s_max}")
    log.info(f"  Platform: {PLATFORM} x {N_DEVICES} {DEVICE_INFO['device_kinds']}")
    log.info(f"  Output: {REC_DIR}")
    log.info("=" * 64)
    data = load_data(DATA_DIR)
    countries = data["countries"]
    m_c = data["m_c_np"]
    base_kwargs = model_kwargs(data)
    method = choose_chain_method(args.nuts_chains, requested=args.chain_method,
                                 n_obs=data["N_obs"], n_actions=data["N_actions"])

    report = {"date": datetime.now().isoformat(), "mode": args.mode, "countries": countries,
              "m_c": [float(x) for x in m_c], "args": vars(args), "chain_method": method,
              "hpdi_prob": HPDI_PROB, "sets": {}, **DEVICE_INFO}
    prev_path = REC_DIR / "report.json"
    if args.resume and prev_path.exists():
        try:
            prev = json.loads(prev_path.read_text())
        except (OSError, ValueError) as e:
            log.warning(f"  --resume: could not read {prev_path} ({e}); starting a fresh report")
            prev = {}
        kept = {n: sr for n, sr in prev.get("sets", {}).items()
                if n not in args.sets and "rows" in sr and "md" in sr and prev.get("mode") == args.mode}
        if kept:
            report["sets"].update(kept)
            report["kept_from_previous_report"] = {"sets": sorted(kept), "date": prev.get("date")}
            log.info(f"  --resume: keeping finished set(s) {sorted(kept)} from {prev_path} "
                     f"(dated {prev.get('date')})")

    for si, set_name in enumerate(args.sets):
        tr = TRUTH[set_name]
        truth = {"rho_c": np.array(tr["rho"], float), "s_c": np.array(tr["s"], float),
                 "beta_c": np.array(tr["beta"], float)}
        truth["gamma_c"] = truth["s_c"] * m_c
        log.info(f"\n[SET {set_name}] truth rho={tr['rho']} s={tr['s']} beta={tr['beta']}")

        key = jax.random.PRNGKey(args.seed + 100 * si)
        sim = simulate_actions(key, truth["rho_c"], truth["s_c"], truth["beta_c"], data)
        data_kwargs = dict(base_kwargs, obs_action=jnp.asarray(sim, dtype=jnp.int32))
        sim_np = np.asarray(sim)
        agree = float((sim_np == np.asarray(data["obs_action"])).mean())
        log.info(f"  simulated {len(sim_np):,} actions; agreement with observed = {agree:.3f}")
        np.save(REC_DIR / f"sim_actions_{set_name}.npy", sim_np)
        ll_truth = float(log_likelihood(truth["rho_c"], truth["s_c"], truth["beta_c"], data_kwargs))
        log.info(f"  log-lik at truth = {ll_truth:.1f}")

        set_rep = {"truth": {k: v.tolist() for k, v in truth.items()},
                   "agreement_with_observed": agree, "loglik_truth": ll_truth}
        md = [f"## Truth set {set_name}", "",
              f"rho={tr['rho']}, s={tr['s']}, beta={tr['beta']}; simulated actions agree with "
              f"observed on {agree:.1%} of obs; log-lik(truth) = {ll_truth:.1f}", ""]

        if args.mode == "svi":
            log.info("  Fitting SVI (AutoMultivariateNormal)...")
            est, info = fit_svi(data_kwargs, args.seed + 10 + si, m_c)
            rows = summarize(est, truth, countries, draws=None)
            (REC_DIR / f"svi_estimates_{set_name}.json").write_text(
                json.dumps({k: v.tolist() for k, v in est.items()}, indent=2))
            label = "SVI point estimate"
        else:
            log.info(f"  Fitting NUTS ({method})...")
            npz_path = REC_DIR / f"nuts_posterior_{set_name}.npz"
            est, flat, by_chain, div, info = fit_nuts(data_kwargs, args.seed + 20 + si, method,
                                                      npz_path=npz_path)
            if not info["loaded_from_npz"]:
                save_posterior(by_chain, div, npz_path,
                               meta={"set": set_name, "truth": set_rep["truth"],
                                     "date": datetime.now().isoformat(), **DEVICE_INFO})
            rows = summarize(est, truth, countries, draws=flat)
            label = "NUTS posterior median"

        ll_est = float(log_likelihood(est["rho_c"], est["s_c"], est["beta_c"], data_kwargs))
        n_ok = sum(r["ok"] for r in rows if r["judged"])
        n_j = sum(r["judged"] for r in rows)
        log.info(f"  {label}: {n_ok}/{n_j} tolerances met; log-lik(estimate) = {ll_est:.1f} "
                 f"(truth {ll_truth:.1f}, diff {ll_est - ll_truth:+.1f})")
        set_rep.update({"fit_info": info, "estimate": {k: np.asarray(v).tolist() for k, v in est.items()},
                        "loglik_estimate": ll_est, "loglik_diff": ll_est - ll_truth,
                        "tolerances_met": n_ok, "tolerances_judged": n_j, "rows": rows})
        if args.mode == "nuts":
            set_rep["in_hpdi"] = sum(r["in_hpdi"] for r in rows if r["judged"])
        md += [f"### {label} ({n_ok}/{n_j} tolerances met; "
               + (f"{info['elapsed_s']:.0f}s, ELBO loss {info['loss_first']:.0f} -> {info['loss_last']:.0f}"
                  if args.mode == "svi" else
                  (f"loaded from {npz_path.name}" if info.get("loaded_from_npz") else f"{info['elapsed_s']/60:.1f} min")
                  + f", divergences {info['div']}/{info['div_total']}, "
                  f"{set_rep['in_hpdi']}/{n_j} truths in HPDI")
               + f"; log-lik truth {ll_truth:.1f} vs estimate {ll_est:.1f})", "",
               md_table(rows, with_post=(args.mode == "nuts")), ""]

        set_rep["md"] = md
        report["sets"][set_name] = set_rep
        write_report(report)            # after every set so partial results survive interruption

    v, why = verdict(report)
    log.info(f"\nRecovery ({args.mode}) verdict: {v} - {why}\n  report: {REC_DIR / 'report.md'}")
    print(f"RECOVERY mode={args.mode} verdict={v} {why}", flush=True)


if __name__ == "__main__":
    main()
