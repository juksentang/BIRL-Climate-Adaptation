"""
Posterior summaries, convergence diagnostics and posterior predictive checks
for BIRL v2 (adapted from 06's diagnostics).  write_all_diagnostics() writes,
for every run:

  summary.csv              median, mean, sd, 89% HPDI, rank-normalised r_hat,
                           bulk and tail ESS for rho_c, s_c, s_lat_c, gamma_c,
                           beta_c (+ rho_lat_c, lb_c) and the hyperparameters
  convergence.txt          human-readable report (all numbers below)
  ppc_action_freq.csv      PPC action-frequency table (action = crop x intensity),
                           overall + per country observed/predicted shares
  ppc_crop_x_intensity.csv PPC pivot crop x intensity (observed, predicted)
  model_diagnostics.json   the same numbers, machine-readable:
      divergences; r_hat / ESS extremes; per-country posterior correlations
      corr(rho_c, s_c), corr(log beta_c, rho_c), corr(log beta_c, s_c);
      P(s_c > 0.55); at the posterior median: share of obs whose chosen action
      has all five nodes inside the floor region (d < 0) and share of obs with
      p(chosen) < 1e-3 (spec-critic P12); per-country mean Spearman correlation
      between CE(rho=0.3) and CE(rho=4.5) across feasible actions (P5);
      float32 resolution guards at the posterior median: max reward (CE / m_c)
      and max |logit| over feasible cells, max q90 per country (USD);
      device / platform of the run.

r_hat and ESS follow Vehtari et al. (2021): split chains, rank-normalise
(z = Phi^-1((rank - 3/8) / (S + 1/4))), r_hat = max over (z, folded z),
bulk ESS = ESS of z, tail ESS = min ESS of the 5% / 95% quantile indicators;
the per-sequence ESS and Gelman-Rubin come from numpyro.diagnostics.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from scipy.special import ndtri
import jax
import jax.numpy as jnp
from numpyro.diagnostics import hpdi as _np_hpdi, gelman_rubin, effective_sample_size

from cropchoice.config import HPDI_PROB, EPS_FRAC, DEVICE_INFO
from cropchoice.models_v2 import (HYPER_SITES, COUNTRY_SITES, simulate_actions, compute_logits,
                        compute_ce, log_likelihood, q_from_log, five_points, obs_mask)

log = logging.getLogger("birl_v2")

S_HIGH = 0.55            # P(s_c > S_HIGH) per country (gate: all s_c in [0.05, 0.55])
P_CHOSEN_LOW = 1e-3
SPEARMAN_RHOS = (0.3, 4.5)


# =====================================================================
# Rank-normalised r_hat / ESS
# =====================================================================

def hpdi(x, prob=HPDI_PROB):
    """Highest-density interval (lo, hi) of a 1-D sample: the narrowest window that
    holds floor(prob * n) sorted values.  NaNs are dropped; (nan, nan) for an empty
    sample; a single value gives a degenerate interval.  Used by every summary in
    the package (fit_semipar, report, sobol) so interval conventions cannot drift."""
    x = np.sort(np.asarray(x, float).ravel()); x = x[~np.isnan(x)]; n = x.size
    if n == 0:
        return float("nan"), float("nan")
    if n == 1:
        return float(x[0]), float(x[0])
    k = min(max(int(np.floor(prob * n)), 1), n - 1)
    w = x[k:] - x[:n - k]; i = int(np.argmin(w))
    return float(x[i]), float(x[i + k])


def _split_chains(x):
    n_chains, n = x.shape
    h = n // 2
    return np.concatenate([x[:, :h], x[:, h:2 * h]], axis=0)


def _z_scale(x):
    r = rankdata(x, axis=None).reshape(x.shape)
    return ndtri((r - 0.375) / (x.size + 0.25))


def _safe(fn, x, positive=False):
    """Evaluate a numpyro diagnostic; non-finite (or, for ESS, non-positive,
    which numpyro's autocorrelation estimator can return on very short
    sequences) -> NaN."""
    try:
        with np.errstate(all="ignore"):
            v = float(fn(x))
        if not np.isfinite(v) or (positive and v <= 0):
            return np.nan
        return v
    except Exception:
        return np.nan


def rhat_rank(x):
    """Rank-normalised split-R-hat (max of bulk and folded). x: (n_chains, n_draws)."""
    if x.shape[1] < 4 or np.std(x) < 1e-12:
        return np.nan
    xs = _split_chains(x)
    z = _z_scale(xs)
    zf = _z_scale(np.abs(xs - np.median(xs)))
    return max(_safe(gelman_rubin, z), _safe(gelman_rubin, zf))


def ess_bulk(x):
    if x.shape[1] < 4 or np.std(x) < 1e-12:
        return np.nan
    return _safe(effective_sample_size, _z_scale(_split_chains(x)), positive=True)


def ess_tail(x):
    if x.shape[1] < 4 or np.std(x) < 1e-12:
        return np.nan
    xs = _split_chains(x)
    q05, q95 = np.quantile(xs, [0.05, 0.95])
    e = [_safe(effective_sample_size, _z_scale((xs <= q).astype(np.float64)), positive=True)
         for q in (q05, q95)]
    return np.nanmin(e) if not all(np.isnan(e)) else np.nan


def _site_stats(x, prob):
    """x: (n_chains, n_draws) for one scalar quantity."""
    x = np.asarray(x, np.float64)
    flat = x.reshape(-1)
    lo, hi = _np_hpdi(flat, prob=prob)
    return {"median": float(np.median(flat)), "mean": float(flat.mean()),
            "sd": float(flat.std(ddof=1)) if flat.size > 1 else 0.0,
            "hpdi_lo": float(lo), "hpdi_hi": float(hi),
            "r_hat": rhat_rank(x), "ess_bulk": ess_bulk(x), "ess_tail": ess_tail(x)}


def posterior_summary(samples, countries, prob=HPDI_PROB):
    """DataFrame: one row per hyperparameter and per (country site, country)."""
    rows = []
    for site in HYPER_SITES:
        if site in samples:
            rows.append({"param": site, "country": "", **_site_stats(samples[site], prob)})
    for site in COUNTRY_SITES:
        if site not in samples:
            continue
        for c, name in enumerate(countries):
            rows.append({"param": site, "country": name,
                         **_site_stats(samples[site][:, :, c], prob)})
    df = pd.DataFrame(rows)
    first = next(iter(samples.values()))
    df["n_chains"] = first.shape[0]
    df["n_draws"] = first.shape[1]
    return df


# =====================================================================
# Posterior correlations and tail probabilities
# =====================================================================

def _flat(samples, site):
    v = samples[site]
    return v.reshape(-1, v.shape[-1]).astype(np.float64)


def _corr(a, b):
    if a.std() < 1e-12 or b.std() < 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def posterior_correlations(samples, countries):
    """Per country: corr(rho_c, s_c), corr(log beta_c, rho_c), corr(log beta_c, s_c)."""
    out = {}
    r, s, lb = _flat(samples, "rho_c"), _flat(samples, "s_c"), np.log(_flat(samples, "beta_c"))
    for c, name in enumerate(countries):
        out[name] = {"rho_s": _corr(r[:, c], s[:, c]),
                     "logbeta_rho": _corr(lb[:, c], r[:, c]),
                     "logbeta_s": _corr(lb[:, c], s[:, c])}
    hyper = {}
    if "mu_rho" in samples and "mu_s" in samples:
        hyper["mu_rho_mu_s"] = _corr(samples["mu_rho"].reshape(-1), samples["mu_s"].reshape(-1))
    if "mu_rho" in samples and "mu_lb" in samples:
        hyper["mu_rho_mu_lb"] = _corr(samples["mu_rho"].reshape(-1), samples["mu_lb"].reshape(-1))
    return out, hyper


def prob_s_above(samples, countries, thr=S_HIGH):
    s = _flat(samples, "s_c")
    return {name: float((s[:, c] > thr).mean()) for c, name in enumerate(countries)}


def posterior_median_params(samples):
    return {k: np.median(_flat(samples, k), axis=0) for k in ("rho_c", "s_c", "beta_c")}


# =====================================================================
# Fit diagnostics at the posterior median (spec-critic P12, P5)
# =====================================================================

def floor_and_fit_shares(rho_med, s_med, beta_med, data):
    """At the posterior median: share of obs whose chosen action has all five
    nodes in the floor region (d = Y - gamma - eps < 0), share of obs with
    p(chosen) < 1e-3, summed log-likelihood; overall and per country."""
    m_c = np.asarray(data["m_c_np"], np.float64)
    ctry = np.asarray(data["obs_country_idx"])
    obs = np.asarray(data["obs_action"])
    rows = np.arange(len(obs))
    q = [np.exp(np.clip(np.asarray(data[k], np.float64)[rows, obs], -20.0, 20.0))
         for k in ("cf_log_q10", "cf_log_q50", "cf_log_q90")]
    nodes = np.stack(five_points(*q), axis=1)                       # (N, 5)
    d = nodes - (s_med[ctry] * m_c[ctry] + EPS_FRAC * m_c[ctry])[:, None]
    in_floor = np.all(d < 0, axis=1)

    logits = compute_logits(jnp.asarray(rho_med, jnp.float32), jnp.asarray(s_med, jnp.float32),
                            jnp.asarray(beta_med, jnp.float32), data)
    logp = np.asarray(jax.nn.log_softmax(logits, axis=-1))[rows, obs].astype(np.float64)
    low = logp < np.log(P_CHOSEN_LOW)
    # float32 resolution of the logits: |logit| ~ 1e6 resolves only ~0.1 nat
    mask = np.asarray(obs_mask(data))
    lg = np.asarray(logits, np.float64)
    max_abs_logit = float(np.abs(lg[mask]).max()) if mask.any() else float("nan")
    ce_med = np.asarray(compute_ce(jnp.asarray(rho_med, jnp.float32),
                                   jnp.asarray(s_med, jnp.float32), data), np.float64)
    reward = ce_med / m_c[ctry][:, None]
    max_reward = float(reward[mask].max()) if mask.any() else float("nan")
    ia = np.unravel_index(int(np.argmax(np.where(mask, np.abs(lg), -np.inf))), lg.shape)

    per_c = {}
    for c, name in enumerate(data["countries"]):
        sel = ctry == c
        per_c[name] = {"floor_share_chosen": float(in_floor[sel].mean()),
                       "p_chosen_lt_1e-3": float(low[sel].mean()),
                       "mean_logp_chosen": float(logp[sel].mean())}
    return {"floor_share_chosen": float(in_floor.mean()),
            "p_chosen_lt_1e-3": float(low.mean()),
            "loglik": float(logp.sum()),
            "mean_logp_chosen": float(logp.mean()),
            "max_reward": max_reward,
            "max_abs_logit": max_abs_logit,
            "max_abs_logit_cell": {"obs": int(ia[0]), "action": int(ia[1]),
                                   "country": data["countries"][int(ctry[ia[0]])]},
            "float32_logit_resolution_nat": float(max_abs_logit * 2 ** -23),
            "q90_max_usd": data.get("q90_max_usd"),
            "per_country": per_c}


def spearman_ce_by_country(s_med, data, rhos=SPEARMAN_RHOS):
    """Per-country mean over obs of the Spearman correlation, across feasible
    actions, between CE(rho=rhos[0]) and CE(rho=rhos[1]) at s_c = s_med.
    Close to 1 means rho mostly rescales the reward (acts as a temperature)."""
    n_country = data["N_country"]
    s = jnp.asarray(s_med, jnp.float32)
    ce = [np.asarray(compute_ce(jnp.full((n_country,), r, jnp.float32), s, data), np.float64)
          for r in rhos]
    mask = np.asarray(data["feasibility_mask"])
    cz = np.asarray(data["obs_cz_idx"])
    ctry = np.asarray(data["obs_country_idx"])
    rho_obs = np.full(len(cz), np.nan)
    for z in range(mask.shape[0]):
        rows = np.where(cz == z)[0]
        cols = np.where(mask[z])[0]
        if len(rows) == 0 or len(cols) < 3:
            continue
        a = rankdata(ce[0][np.ix_(rows, cols)], axis=1)
        b = rankdata(ce[1][np.ix_(rows, cols)], axis=1)
        a = a - a.mean(axis=1, keepdims=True)
        b = b - b.mean(axis=1, keepdims=True)
        denom = np.sqrt((a * a).sum(1) * (b * b).sum(1))
        with np.errstate(invalid="ignore", divide="ignore"):
            rho_obs[rows] = np.where(denom > 0, (a * b).sum(1) / denom, np.nan)
    out = {}
    for c, name in enumerate(data["countries"]):
        v = rho_obs[ctry == c]
        out[name] = float(np.nanmean(v)) if np.isfinite(v).any() else np.nan
    out["overall"] = float(np.nanmean(rho_obs)) if np.isfinite(rho_obs).any() else np.nan
    return out


# =====================================================================
# Posterior predictive check: action frequencies by crop x intensity
# =====================================================================

def _split_label(label):
    crop, intensity = label.rsplit("_", 1)
    return crop, intensity


def ppc_action_freq(samples, data, out_dir, n_ppc=100, seed=99):
    """Average simulated action frequencies over thinned posterior draws.

    Writes ppc_action_freq.csv (action, label, crop, intensity, observed,
    predicted, diff, obs_<country>, pred_<country>) and ppc_crop_x_intensity.csv
    (pivot).  Returns (DataFrame, info dict).
    """
    out_dir = Path(out_dir)
    flat = {k: _flat(samples, k) for k in ("rho_c", "s_c", "beta_c")}
    n_total = flat["rho_c"].shape[0]
    idx = np.linspace(0, n_total - 1, num=min(n_ppc, n_total)).astype(int)

    n_actions = data["N_actions"]
    n_country = data["N_country"]
    obs = np.asarray(data["obs_action"])
    ctry = np.asarray(data["obs_country_idx"])
    obs_cnt = np.zeros((n_country, n_actions))
    pred_cnt = np.zeros((n_country, n_actions))
    for c in range(n_country):
        obs_cnt[c] = np.bincount(obs[ctry == c], minlength=n_actions)

    key = jax.random.PRNGKey(seed)
    for i in idx:
        key, sub = jax.random.split(key)
        sim = np.asarray(simulate_actions(sub, flat["rho_c"][i], flat["s_c"][i],
                                          flat["beta_c"][i], data))
        for c in range(n_country):
            pred_cnt[c] += np.bincount(sim[ctry == c], minlength=n_actions)
    pred_cnt /= len(idx)

    labels = data["action_labels"]
    rows = []
    for a in range(n_actions):
        lab = labels.get(str(a), f"crop{a}_na") if isinstance(labels, dict) else labels[a]
        crop, inten = _split_label(lab)
        o = obs_cnt[:, a].sum() / obs_cnt.sum()
        p = pred_cnt[:, a].sum() / pred_cnt.sum()
        row = {"action": a, "label": lab, "crop": crop, "intensity": inten,
               "observed": o, "predicted": p, "diff": abs(o - p)}
        for c, name in enumerate(data["countries"]):
            row[f"obs_{name}"] = obs_cnt[c, a] / max(obs_cnt[c].sum(), 1)
            row[f"pred_{name}"] = pred_cnt[c, a] / max(pred_cnt[c].sum(), 1)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "ppc_action_freq.csv", index=False)
    pivot = df.pivot_table(index="crop", columns="intensity", values=["observed", "predicted"],
                           aggfunc="sum")
    pivot.to_csv(out_dir / "ppc_crop_x_intensity.csv")
    info = {"n_draws": int(len(idx)), "max_abs_diff": float(df["diff"].max()),
            "mean_abs_diff": float(df["diff"].mean()),
            "per_country_max_abs_diff": {
                name: float(np.abs(df[f"obs_{name}"] - df[f"pred_{name}"]).max())
                for name in data["countries"]}}
    log.info(f"  PPC ({len(idx)} draws): max |obs-pred| = {info['max_abs_diff']:.4f}, "
             f"mean = {info['mean_abs_diff']:.4f}")
    return df, info


# =====================================================================
# Reports
# =====================================================================

def _fmt(v, f="{:.3f}"):
    return "nan" if v is None or (isinstance(v, float) and np.isnan(v)) else f.format(v)


def convergence_text(diag, summary_df, countries, variant, extra_lines=None):
    lines = ["=" * 72, f"Convergence report: {variant}", f"Date: {diag['run']['date']}",
             f"Platform: {diag['run']['platform']} x {diag['run']['n_devices']} "
             f"{diag['run']['device_kinds']}  chain_method={diag['run'].get('chain_method')}",
             "=" * 72,
             f"Chains: {diag['run']['n_chains']}   Draws/chain: {diag['run']['n_draws']}",
             "m_c (USD): " + ", ".join(f"{c}={m:.2f}" for c, m in zip(countries, diag['run']['m_c']))]
    d = diag["divergences"]
    lines.append(f"\nDivergences: {d['count']}/{d['total']} ({d['rate']:.4f})  "
                 f"[{'PASS' if d['rate'] < 0.01 else 'FAIL'}]")
    c = diag["convergence"]
    lines.append(f"R-hat max:    {_fmt(c['r_hat_max'], '{:.4f}')}  "
                 f"[{'PASS' if (c['r_hat_max'] or 9) < 1.01 else 'FAIL'}]")
    lines.append(f"ESS bulk min: {_fmt(c['ess_bulk_min'], '{:.0f}')}  "
                 f"[{'PASS' if (c['ess_bulk_min'] or 0) > 400 else 'WARN'}]")
    lines.append(f"ESS tail min: {_fmt(c['ess_tail_min'], '{:.0f}')}  "
                 f"[{'PASS' if (c['ess_tail_min'] or 0) > 400 else 'WARN'}]")

    lines.append("\nParameter table (median [89% HPDI], r_hat, ess_bulk, ess_tail):")
    for _, r in summary_df.iterrows():
        tag = f"{r['param']}[{r['country']}]" if r["country"] else r["param"]
        lines.append(f"  {tag:22s} {r['median']:10.4f} [{r['hpdi_lo']:9.4f}, {r['hpdi_hi']:9.4f}]"
                     f"  r_hat={_fmt(r['r_hat'])}  ess_b={_fmt(r['ess_bulk'], '{:.0f}')}"
                     f"  ess_t={_fmt(r['ess_tail'], '{:.0f}')}")

    lines.append("\nPosterior correlations per country (rho_c,s_c | log beta_c,rho_c | log beta_c,s_c):")
    for name, v in diag["correlations"]["per_country"].items():
        flag = "  [WARN]" if any(abs(x) > 0.8 for x in v.values() if not np.isnan(x)) else ""
        lines.append(f"  {name:10s} {_fmt(v['rho_s']):>7s} | {_fmt(v['logbeta_rho']):>7s} | "
                     f"{_fmt(v['logbeta_s']):>7s}{flag}")
    for k, v in diag["correlations"]["hyper"].items():
        lines.append(f"  {k}: {_fmt(v)}")

    lines.append(f"\nP(s_c > {S_HIGH}) per country:")
    for name, v in diag["p_s_gt_055"].items():
        lines.append(f"  {name:10s} {v:.3f}{'  [WARN]' if v > 0.1 else ''}")

    f = diag["fit_at_median"]
    lines.append(f"\nAt the posterior median: loglik = {f['loglik']:.1f}, "
                 f"mean log p(chosen) = {f['mean_logp_chosen']:.3f}")
    lines.append(f"  share of obs with chosen action fully inside the floor (all 5 nodes d<0): "
                 f"{f['floor_share_chosen']:.4f}")
    lines.append(f"  share of obs with p(chosen) < 1e-3: {f['p_chosen_lt_1e-3']:.4f}")
    for name, v in f["per_country"].items():
        lines.append(f"    {name:10s} floor={v['floor_share_chosen']:.4f}  "
                     f"p<1e-3={v['p_chosen_lt_1e-3']:.4f}")
    lines.append(f"  float32 guards: max reward (CE/m_c) = {f['max_reward']:.4g}, "
                 f"max |logit| = {f['max_abs_logit']:.4g} "
                 f"(resolution {f['float32_logit_resolution_nat']:.2e} nat; "
                 f"cell obs={f['max_abs_logit_cell']['obs']} action={f['max_abs_logit_cell']['action']} "
                 f"{f['max_abs_logit_cell']['country']})"
                 + ("  [WARN: > 1e4]" if f['max_abs_logit'] > 1e4 else ""))
    if f.get("q90_max_usd"):
        lines.append("  max q90 per country (USD): "
                     + ", ".join(f"{n}={v:.4g}" for n, v in f["q90_max_usd"].items()))

    lines.append(f"\nSpearman corr(CE(rho={SPEARMAN_RHOS[0]}), CE(rho={SPEARMAN_RHOS[1]})) across "
                 f"feasible actions, mean per country (> 0.95: rho acts mostly as a temperature):")
    for name, v in diag["spearman_ce_rho"].items():
        lines.append(f"  {name:10s} {_fmt(v)}{'  [WARN]' if (not np.isnan(v) and v > 0.95) else ''}")

    p = diag["ppc"]
    lines.append(f"\nPPC ({p['n_draws']} draws): max |obs-pred| = {p['max_abs_diff']:.4f}, "
                 f"mean = {p['mean_abs_diff']:.4f}")
    if extra_lines:
        lines.append("")
        lines.extend(extra_lines)
    return "\n".join(lines)


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def write_all_diagnostics(samples, diverging, data, variant, run_dir, run_info=None,
                          n_ppc=100, seed=99, extra_lines=None):
    """Write summary.csv, convergence.txt, ppc_*.csv, model_diagnostics.json. Returns the dict."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    countries = data["countries"]
    first = next(iter(samples.values()))

    summary = posterior_summary(samples, countries)
    summary.to_csv(run_dir / "summary.csv", index=False)

    div = np.asarray(diverging) if diverging is not None else np.zeros(first.shape[:2], bool)
    corr_c, corr_h = posterior_correlations(samples, countries)
    med = posterior_median_params(samples)
    fit = floor_and_fit_shares(med["rho_c"], med["s_c"], med["beta_c"], data)
    spear = spearman_ce_by_country(med["s_c"], data)
    ppc_df, ppc_info = ppc_action_freq(samples, data, run_dir, n_ppc=n_ppc, seed=seed)

    params = {}
    for _, r in summary.iterrows():
        params.setdefault(r["param"], {})[r["country"] or "_"] = {
            k: r[k] for k in ("median", "mean", "sd", "hpdi_lo", "hpdi_hi",
                              "r_hat", "ess_bulk", "ess_tail")}
    diag = {
        "run": {"variant": variant, "date": datetime.now().isoformat(),
                "n_chains": int(first.shape[0]), "n_draws": int(first.shape[1]),
                "countries": countries, "m_c": [float(x) for x in data["m_c_np"]],
                **DEVICE_INFO, **(run_info or {})},
        "divergences": {"count": int(div.sum()), "total": int(div.size),
                        "rate": float(div.mean()) if div.size else 0.0},
        "convergence": {"r_hat_max": float(np.nanmax(summary["r_hat"])) if summary["r_hat"].notna().any() else np.nan,
                        "ess_bulk_min": float(np.nanmin(summary["ess_bulk"])) if summary["ess_bulk"].notna().any() else np.nan,
                        "ess_tail_min": float(np.nanmin(summary["ess_tail"])) if summary["ess_tail"].notna().any() else np.nan},
        "posterior_median": {k: {name: float(v[c]) for c, name in enumerate(countries)}
                             for k, v in med.items()},
        "params": params,
        "correlations": {"per_country": corr_c, "hyper": corr_h},
        "p_s_gt_055": prob_s_above(samples, countries),
        "fit_at_median": fit,
        "spearman_ce_rho": spear,
        "ppc": ppc_info,
    }
    text = convergence_text(diag, summary, countries, variant, extra_lines)
    (run_dir / "convergence.txt").write_text(text)
    (run_dir / "model_diagnostics.json").write_text(json.dumps(_jsonable(diag), indent=2))
    log.info("\n" + text)
    return diag
