"""
Convergence diagnostics (R-hat, ESS, divergences, correlations) and
posterior predictive checks.
"""

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import jax
from numpyro.infer import Predictive

log = logging.getLogger("birl")


def run_diagnostics(mcmc, np_samples, name, out_dir, has_arviz=False):
    """Full convergence diagnostics (requires MCMC object). Saves report + CSVs."""
    report = []
    report.append(f"{'='*60}")
    report.append(f"Convergence Report: {name}")
    report.append(f"Date: {datetime.now().isoformat()}")
    report.append(f"{'='*60}")

    # --- Divergences ---
    try:
        div = np.array(mcmc.get_extra_fields()["diverging"])
        div_count = int(div.sum())
        div_rate = div_count / max(div.size, 1)
        ok = "PASS" if div_rate < 0.01 else "FAIL"
        report.append(f"\nDivergences: {div_count}/{div.size} ({div_rate:.4f})  [{ok}]")
    except Exception:
        report.append("\nDivergences: unavailable")

    # --- R-hat + ESS via ArviZ ---
    if has_arviz:
        try:
            import arviz as az
            idata = az.from_numpyro(mcmc)

            global_p = [p for p in ["mu_rho_G", "mu_alpha_G", "mu_lgamma_G",
                                    "sigma_rho_G", "sigma_alpha_G", "sigma_lgamma_G"]
                        if p in idata.posterior]
            country_p = [p for p in ["mu_rho_C", "mu_alpha_C", "mu_lgamma_C",
                                     "sigma_rho_C", "sigma_alpha_C", "sigma_lgamma_C"]
                         if p in idata.posterior]

            report.append("\nR-hat:")
            rhat_rows = []
            for p in global_p + country_p:
                rh = float(az.rhat(idata.posterior[p]).max())
                ok = "PASS" if rh < 1.01 else "FAIL"
                report.append(f"  {p:25s}: {rh:.4f}  [{ok}]")
                rhat_rows.append({"param": p, "rhat": rh})
            if rhat_rows:
                pd.DataFrame(rhat_rows).to_csv(out_dir / "rhat_summary.csv", index=False)

            report.append("\nESS:")
            ess_rows = []
            for p in global_p:
                eb = float(az.ess(idata.posterior[p], method="bulk").values)
                et = float(az.ess(idata.posterior[p], method="tail").values)
                ok = "PASS" if eb > 400 and et > 200 else "FAIL"
                report.append(f"  {p:25s}: bulk={eb:.0f}  tail={et:.0f}  [{ok}]")
                ess_rows.append({"param": p, "ess_bulk": eb, "ess_tail": et})
            if ess_rows:
                pd.DataFrame(ess_rows).to_csv(out_dir / "ess_summary.csv", index=False)
        except Exception as e:
            report.append(f"\nArviZ diagnostics error: {e}")

    # --- Posterior reasonableness + correlations ---
    _append_posterior_summary(report, np_samples, out_dir)

    report_text = "\n".join(report)
    with open(out_dir / "convergence_report.txt", "w") as f:
        f.write(report_text)
    log.info(f"\n{report_text}")
    return report_text


def quick_report(np_samples, name, out_dir=None):
    """Lightweight diagnostics from pickled samples (no MCMC object needed)."""
    report = []
    report.append(f"--- Quick Report: {name} ---")
    _append_posterior_summary(report, np_samples, out_dir=out_dir)
    report_text = "\n".join(report)
    log.info(report_text)

    if out_dir is not None:
        path = out_dir / f"quick_report_{name}.txt"
        with open(path, "w") as f:
            f.write(report_text)
    return report_text


def _append_posterior_summary(report, np_samples, out_dir=None):
    """Shared logic: posterior means, quantiles, correlations."""
    if "gamma_i" in np_samples:
        g = np_samples["gamma_i"].mean(0)
        report.append(f"\ngamma (USD): median={np.median(g):.1f}, "
                      f"P5={np.percentile(g,5):.1f}, P95={np.percentile(g,95):.1f}")

    if "rho_i" in np_samples:
        r = np_samples["rho_i"].mean(0)
        report.append(f"rho:   median={np.median(r):.2f}, "
                      f"P5={np.percentile(r,5):.2f}, P95={np.percentile(r,95):.2f}")

    if "alpha_i" in np_samples:
        a = np_samples["alpha_i"].mean(0)
        report.append(f"alpha: median={np.median(a):.3f}, "
                      f"P5={np.percentile(a,5):.3f}, P95={np.percentile(a,95):.3f}")

    # Parameter correlations (rho-gamma; alpha may not exist)
    corr_keys = [k for k in ["rho_i", "gamma_i"] if k in np_samples]
    if len(corr_keys) >= 2:
        n_sub = min(3000, np_samples[corr_keys[0]].shape[1])
        pairs = {"rho_gamma": ("rho_i", "gamma_i")}
        if "alpha_i" in np_samples:
            pairs["alpha_rho"] = ("alpha_i", "rho_i")
            pairs["alpha_gamma"] = ("alpha_i", "gamma_i")
        corr_data = {}
        report.append(f"\nPosterior correlations (subsample {n_sub} HH):")
        for pname, (k1, k2) in pairs.items():
            corrs = [np.corrcoef(np_samples[k1][:, i], np_samples[k2][:, i])[0, 1]
                     for i in range(n_sub)]
            mn = float(np.nanmean(corrs))
            ok = "PASS" if abs(mn) < 0.5 else "WARN"
            report.append(f"  {pname:15s}: {mn:.3f}  [{ok}]")
            corr_data[pname] = mn
        if out_dir is not None:
            pd.DataFrame([corr_data]).to_csv(out_dir / "correlation_matrix.csv", index=False)


def run_ppc(mcmc, model_fn, data, n_actions, action_labels, n_ppc=200, out_dir=None):
    """Run posterior predictive check from MCMC object (legacy)."""
    try:
        samples = mcmc.get_samples()
        _run_ppc_core(samples, model_fn, data, n_actions, action_labels, n_ppc, out_dir)
    except Exception as e:
        log.warning(f"  PPC failed: {e}")


def run_ppc_from_samples(np_samples, model_fn, data, n_actions, action_labels,
                         n_ppc=200, out_dir=None):
    """Run posterior predictive check from numpy samples dict (for chunked MCMC)."""
    import jax.numpy as jnp
    try:
        jax_samples = {k: jnp.array(v) for k, v in np_samples.items()}
        _run_ppc_core(jax_samples, model_fn, data, n_actions, action_labels, n_ppc, out_dir)
    except Exception as e:
        log.warning(f"  PPC failed: {e}")


def _run_ppc_core(samples, model_fn, data, n_actions, action_labels, n_ppc, out_dir):
    """Shared PPC logic."""
    n_total = next(iter(samples.values())).shape[0]
    thin = max(1, n_total // n_ppc)
    thinned = {k: v[::thin] for k, v in samples.items()}

    ppc_data = {k: v for k, v in data.items() if k != "obs_action"}
    ppc_data["obs_action"] = None

    predictive = Predictive(model_fn, posterior_samples=thinned)
    pred_out = predictive(jax.random.PRNGKey(99), **ppc_data)

    pred_actions = np.array(pred_out["obs_action"])
    obs_np = np.array(data["obs_action"])

    rows = []
    for a in range(n_actions):
        p_freq = float((pred_actions == a).mean())
        o_freq = float((obs_np == a).mean())
        rows.append({"action": a, "label": action_labels.get(str(a), f"a{a}"),
                     "predicted": p_freq, "observed": o_freq,
                     "diff": abs(p_freq - o_freq)})

    ppc_df = pd.DataFrame(rows)
    if out_dir is not None:
        ppc_df.to_csv(out_dir / "ppc_action_freq.csv", index=False)
    log.info(f"  PPC: max diff={ppc_df['diff'].max():.4f}")
