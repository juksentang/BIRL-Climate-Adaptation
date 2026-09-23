"""NUTS estimation of the semi-parametric structural crop-choice model (country level).

  V_ia   = a_c * mu_ia + b_c * sigma_ia + c_c * sigma_ia^2
  logits = center_feasible(V) + ASC_{c,a}   (action 0 = reference),  obs_action ~ Categorical
  priors: a, b, c ~ N(0, 3);  ASC ~ N(0, 3).  No beta (scale lives in a, b, c).

Stages (one job): [1] timing 50+20 draws, 4 chains vectorized -> projected hours;
if > 6 h use dense_mass=True and 600+600.  [2] full run 4 x (1000+1000).
[3] ASC-only null log-lik (SVI) for the structural gain.  [4] outputs
(OUT_DIR/semipar/): summary.csv, derived.csv, ppc_action_freq.csv, run.log,
posterior.npz, results.json.

  python -m cropchoice fit-semipar [--warmup W --samples S --chains K --skip-timing --dense]
  EXP_TOY=1 python -m cropchoice fit-semipar        # synthetic data, local check
"""
import os, time, json, argparse


def toy_model_kwargs(seed=0, Nt=400, At=6, C=3):
    """Synthetic model_kwargs-like dict for local checks (no data files)."""
    import numpy as np, jax.numpy as jnp
    rng = np.random.default_rng(seed)
    _q50 = rng.lognormal(3, 0.7, (Nt, At)); _rs = rng.lognormal(0, 0.6, (Nt, At))
    _m = rng.random((Nt, At)) > 0.2; _m[:, 0] = True
    mk = {"q10": jnp.asarray(_q50 * np.maximum(1 - 0.5 * _rs, 0.05), jnp.float32), "q50": jnp.asarray(_q50, jnp.float32),
          "q90": jnp.asarray(_q50 * (1 + 0.8 * _rs), jnp.float32), "mask_obs": jnp.asarray(_m),
          "obs_action": jnp.asarray(np.array([rng.choice(np.flatnonzero(r)) for r in _m]), jnp.int32),
          "obs_country_idx": jnp.asarray(rng.integers(0, C, Nt), jnp.int32)}
    countries = [chr(65 + i) for i in range(C)]
    labels = [f"crop{i//3}_{['low','medium','high'][i%3]}" for i in range(At)]
    return mk, countries, labels


def main(argv=None):
    ap = argparse.ArgumentParser(description="NUTS fit of the semi-parametric choice model (country level)")
    ap.add_argument("--warmup", type=int, default=1000); ap.add_argument("--samples", type=int, default=1000)
    ap.add_argument("--chains", type=int, default=4); ap.add_argument("--skip-timing", action="store_true")
    ap.add_argument("--dense", action="store_true"); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None, help="run directory (default OUT_DIR/semipar)")
    args = ap.parse_args(argv)
    os.environ.setdefault("BIRL_HOST_DEVICES", "1")

    from cropchoice.config import log, DATA_DIR, OUT_DIR, add_file_log
    import jax, jax.numpy as jnp, numpy as np, pandas as pd
    import numpyro
    from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO
    from numpyro.infer.autoguide import AutoMultivariateNormal
    from numpyro.infer.util import log_density
    from cropchoice.data import load_data, model_kwargs
    from cropchoice.models import semipar_features, make_semipar_model, make_probs_fn, country_one_hot
    from cropchoice.diagnostics import rhat_rank, ess_bulk, ess_tail, hpdi

    RUN_DIR = (OUT_DIR / "semipar") if args.out is None else __import__("pathlib").Path(args.out)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    add_file_log(RUN_DIR / "run.log")
    say = log.info

    TOY = bool(os.environ.get("EXP_TOY"))
    if TOY:
        mk, countries, labels = toy_model_kwargs()
        C = len(countries); args.warmup, args.samples = 30, 30
    else:
        data = load_data(DATA_DIR); countries = list(data["countries"]); C = len(countries)
        mk = model_kwargs(data)
        cfg = json.load(open(DATA_DIR / "action_space_config.json"))
        al = cfg["action_labels"]; labels = [al[str(i)] if isinstance(al, dict) else al[i] for i in range(mk["q50"].shape[1])]

    feat = semipar_features(mk); MASK, ACT, CI = feat["MASK"], feat["ACT"], feat["CI"]
    N, A = feat["MU"].shape
    OH = country_one_hot(mk, C)
    say(f"semipar  N={N} A={A} C={C}  devices={jax.devices()}  toy={TOY}")
    sig_np = np.asarray(feat["SIG"]); mask_np = np.asarray(MASK)
    say(f"sigma over feasible cells: p5/p50/p95 = {np.nanpercentile(sig_np[mask_np], [5, 50, 95]).round(3).tolist()}; share > 1.2: {float(np.mean(sig_np[mask_np] > 1.2)):.3f}")

    model = make_semipar_model(feat, C, OH)

    def kernel(dense):
        return NUTS(model, target_accept_prob=0.8, max_tree_depth=10, dense_mass=dense,
                    init_strategy=numpyro.infer.init_to_median())

    # ── [1] timing ──
    dense = args.dense; warm, samp = args.warmup, args.samples
    if not args.skip_timing and not TOY:
        m = MCMC(kernel(dense), num_warmup=50, num_samples=20, num_chains=args.chains, chain_method="vectorized", progress_bar=False)
        t0 = time.time(); m.warmup(jax.random.PRNGKey(1), extra_fields=("num_steps",), collect_warmup=True); tw = time.time() - t0
        lw = float(np.mean(np.asarray(m.get_extra_fields()["num_steps"])))
        t0 = time.time(); m.run(jax.random.PRNGKey(2), extra_fields=("num_steps",)); ts = time.time() - t0
        ls = float(np.mean(np.asarray(m.get_extra_fields()["num_steps"])))
        s_lf = ts / max(ls * 20, 1)
        proj_h = (tw / 50 * warm + ts / 20 * samp) / 3600
        say(f"[TIMING] warmup {tw/50:.2f} s/draw (mean leapfrog {lw:.0f}, incl compile), sampling {ts/20:.2f} s/draw (leapfrog {ls:.0f}, {s_lf*1000:.1f} ms/leapfrog); projected full run {proj_h:.2f} h")
        if proj_h > 6:
            dense, warm, samp = True, 600, 600
            say("[TIMING] projected > 6 h: switching to dense_mass=True and 600+600")

    # ── [2] full run ──
    m = MCMC(kernel(dense), num_warmup=warm, num_samples=samp, num_chains=args.chains, chain_method="vectorized", progress_bar=False)
    t0 = time.time(); m.run(jax.random.PRNGKey(args.seed), extra_fields=("diverging", "num_steps"))
    say(f"[RUN] {args.chains} x ({warm}+{samp}) dense={dense}: {(time.time()-t0)/60:.1f} min")
    S = {k: np.asarray(v) for k, v in m.get_samples(group_by_chain=True).items()}
    ex = m.get_extra_fields(group_by_chain=True); div = np.asarray(ex["diverging"]); nsteps = np.asarray(ex["num_steps"])
    np.savez_compressed(RUN_DIR / "posterior.npz", **S, diverging=div, num_steps=nsteps)
    say(f"divergences {int(div.sum())}/{div.size} ({div.mean():.4f}); mean leapfrog/draw {nsteps.mean():.1f}")

    # ── [4a] summary ──
    rows = []
    for site, name in (("a_c", "a"), ("b_c", "b"), ("c_c", "c")):
        for i, cn in enumerate(countries):
            x = S[site][:, :, i]; lo, hi = hpdi(x)
            rows.append(dict(param=name, country=cn, median=float(np.median(x)), mean=float(x.mean()), sd=float(x.std()),
                             hpdi_lo=lo, hpdi_hi=hi, r_hat=float(rhat_rank(x)), ess_bulk=float(ess_bulk(x)), ess_tail=float(ess_tail(x))))
    asc_rhat = np.array([[rhat_rank(S["asc_c"][:, :, i, j]) for j in range(A - 1)] for i in range(C)])
    asc_ess = np.array([[ess_bulk(S["asc_c"][:, :, i, j]) for j in range(A - 1)] for i in range(C)])
    summ = pd.DataFrame(rows); summ.to_csv(RUN_DIR / "summary.csv", index=False)
    say("summary (median [89% HPDI] r_hat ess_bulk):")
    for r in rows:
        say(f"  {r['param']}[{r['country']:9s}] {r['median']:+8.3f} [{r['hpdi_lo']:+8.3f}, {r['hpdi_hi']:+8.3f}]  r_hat={r['r_hat']:.3f} ess={r['ess_bulk']:.0f}")
    say(f"ASC: r_hat max {np.nanmax(asc_rhat):.3f}, ess_bulk min {np.nanmin(asc_ess):.0f}")

    # ── [4b] derived quantities per draw ──
    a, b, c = S["a_c"].reshape(-1, C), S["b_c"].reshape(-1, C), S["c_c"].reshape(-1, C)
    drows = []
    for i, cn in enumerate(countries):
        d = {"country": cn}
        for s0 in (0.5, 1.0):
            q = -(b[:, i] + 2 * c[:, i] * s0) / a[:, i] * 0.1          # log income given up to cut sigma by 0.1
            lo, hi = hpdi(q); d[f"mda_s{s0}"] = float(np.median(q)); d[f"mda_s{s0}_lo"] = lo; d[f"mda_s{s0}_hi"] = hi
        sstar = -b[:, i] / (2 * c[:, i]); lo, hi = hpdi(sstar)
        d.update(sigma_star=float(np.median(sstar)), sigma_star_lo=lo, sigma_star_hi=hi, p_a_pos=float(np.mean(a[:, i] > 0)), p_c_pos=float(np.mean(c[:, i] > 0)))
        drows.append(d)
    pd.DataFrame(drows).to_csv(RUN_DIR / "derived.csv", index=False)
    say("derived: marginal dispersion aversion (log income given up per -0.1 sigma) at sigma=0.5 / 1.0; sigma* = -b/2c:")
    for d in drows:
        say(f"  {d['country']:9s} mda(0.5)={d['mda_s0.5']:+.3f} [{d['mda_s0.5_lo']:+.3f},{d['mda_s0.5_hi']:+.3f}]  mda(1.0)={d['mda_s1.0']:+.3f} [{d['mda_s1.0_lo']:+.3f},{d['mda_s1.0_hi']:+.3f}]  sigma*={d['sigma_star']:.2f} [{d['sigma_star_lo']:.2f},{d['sigma_star_hi']:.2f}]  P(a>0)={d['p_a_pos']:.2f}")

    # ── [4c] PPC ──
    probs_jit = make_probs_fn(feat, OH)
    flat = {k: v.reshape((-1,) + v.shape[2:]) for k, v in S.items()}
    n_ppc = min(100, flat["a_c"].shape[0]); pick = np.linspace(0, flat["a_c"].shape[0] - 1, n_ppc).astype(int)
    P = np.zeros((N, A), np.float64)
    for j in pick:
        P += np.asarray(probs_jit(jnp.asarray(flat["a_c"][j]), jnp.asarray(flat["b_c"][j]), jnp.asarray(flat["c_c"][j]), jnp.asarray(flat["asc_c"][j])))
    P /= n_ppc
    act_np = np.asarray(ACT); ci_np = np.asarray(CI)
    obs_f = np.bincount(act_np, minlength=A) / N; pred_f = P.mean(0)
    pp = pd.DataFrame({"action": range(A), "label": labels, "crop": [l.rsplit("_", 1)[0] for l in labels],
                       "intensity": [l.rsplit("_", 1)[1] for l in labels], "observed": obs_f, "predicted": pred_f, "diff": np.abs(obs_f - pred_f)})
    for i, cn in enumerate(countries):
        sel = ci_np == i
        pp[f"obs_{cn}"] = np.bincount(act_np[sel], minlength=A) / sel.sum(); pp[f"pred_{cn}"] = P[sel].mean(0)
    pp.to_csv(RUN_DIR / "ppc_action_freq.csv", index=False)
    ll_median = float(np.sum(np.log(np.maximum(P[np.arange(N), act_np], 1e-12))))
    old = OUT_DIR / "v2_country" / "ppc_action_freq.csv"
    old_diff = float(pd.read_csv(old)["diff"].mean()) if old.exists() else float("nan")
    say(f"PPC mean |obs-pred| by action: semipar {pp['diff'].mean():.4f} (max {pp['diff'].max():.4f}) vs v2_country {old_diff:.4f}")
    for _, r in pp.sort_values("observed", ascending=False).head(6).iterrows():
        say(f"  {r['label']:22s} obs {r['observed']:.3f} pred {r['predicted']:.3f}")

    # ── [3] ASC-only null log-lik ──
    null_model = lambda: model(null=True)
    guide = AutoMultivariateNormal(null_model, init_loc_fn=numpyro.infer.init_to_median())
    svi = SVI(null_model, guide, numpyro.optim.Adam(1e-2), Trace_ELBO())
    res = svi.run(jax.random.PRNGKey(0), 3000 if not TOY else 50, progress_bar=False)
    med = guide.median(res.params)
    _, tr = log_density(numpyro.handlers.substitute(null_model, data=med), (), {}, {})
    ll_null = float(tr["obs_action"]["fn"].log_prob(tr["obs_action"]["value"]).sum())
    say(f"log-lik: ASC-only null {ll_null:.1f}; semipar at PPC-averaged probs {ll_median:.1f}; structural gain {(ll_median-ll_null)/N:.4f} nat/obs")

    json.dump({"countries": countries, "N": N, "chains": args.chains, "warmup": warm, "samples": samp, "dense_mass": dense,
               "divergence_rate": float(div.mean()), "mean_leapfrog": float(nsteps.mean()),
               "r_hat_max_abc": float(summ["r_hat"].max()), "ess_min_abc": float(summ["ess_bulk"].min()),
               "asc_r_hat_max": float(np.nanmax(asc_rhat)), "asc_ess_min": float(np.nanmin(asc_ess)),
               "ll_null": ll_null, "ll_semipar": ll_median, "gain_per_obs": (ll_median - ll_null) / N,
               "ppc_mean_absdiff": float(pp["diff"].mean()), "ppc_mean_absdiff_v2": old_diff,
               "summary": rows, "derived": drows}, open(RUN_DIR / "results.json", "w"), indent=2)
    say(f"written {RUN_DIR}")
    return RUN_DIR
