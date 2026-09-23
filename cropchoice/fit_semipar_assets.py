"""NUTS estimation of the semi-parametric crop-choice model with household-asset heterogeneity.

  V_ia   = a_g * mu_ia + b_g * sigma_ia + c_g * sigma_ia^2,   g = country x within-country asset tercile
  logits = center_feasible(V) + ASC_{c,a}   (ASC by country x action, action 0 = reference; as fit_semipar)
  priors: a, b, c ~ N(0, 3) per group;  ASC ~ N(0, 3).  No beta.

Terciles are cut within country on hh_asset_index (33.3 / 66.7 percentiles of the
observations).  Observations with a missing asset index form a separate "na" group per
country when the country has >= --min-na such observations, otherwise they are dropped
(counts are reported).  Outputs (OUT_DIR/semipar_assets/): summary.csv, marginal_by_group.csv,
derived.csv, convergence.txt, results.json, posterior.npz, run.log.

  python -m cropchoice fit-semipar-assets [--warmup W --samples S --chains K --skip-timing --dense]
  EXP_TOY=1 python -m cropchoice fit-semipar-assets
"""
import os, time, json, argparse
from pathlib import Path

TERCILE_NAMES = ["T1_poor", "T2_mid", "T3_rich"]


def asset_groups(ci_np, asset, countries, min_na):
    """Group index per observation (-1 = dropped), group names [(country, tercile)]."""
    import numpy as np
    C = len(countries); N_all = len(ci_np)
    terc = np.full(N_all, -1, int)
    for i in range(C):
        sel = (ci_np == i) & np.isfinite(asset)
        if sel.sum() == 0:
            continue
        lo, hi = np.nanpercentile(asset[sel], [100 / 3, 200 / 3])
        terc[sel] = np.where(asset[sel] <= lo, 0, np.where(asset[sel] <= hi, 1, 2))
    group_names, gidx = [], np.full(N_all, -1, int)
    for i, cn in enumerate(countries):
        for t in range(3):
            sel = (ci_np == i) & (terc == t)
            gidx[sel] = len(group_names); group_names.append((cn, TERCILE_NAMES[t]))
        sel_na = (ci_np == i) & (terc == -1)
        if sel_na.sum() >= min_na:
            gidx[sel_na] = len(group_names); group_names.append((cn, "na"))
    return gidx, group_names


def main(argv=None):
    ap = argparse.ArgumentParser(description="NUTS fit of the semi-parametric choice model by country x asset tercile")
    ap.add_argument("--warmup", type=int, default=1000); ap.add_argument("--samples", type=int, default=1000)
    ap.add_argument("--chains", type=int, default=4); ap.add_argument("--skip-timing", action="store_true")
    ap.add_argument("--dense", action="store_true"); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-na", type=int, default=500, help="keep a per-country 'na' asset group only above this size")
    ap.add_argument("--out", default=None, help="run directory (default OUT_DIR/semipar_assets)")
    args = ap.parse_args(argv)
    os.environ.setdefault("BIRL_HOST_DEVICES", "1")

    from cropchoice.config import log, DATA_DIR, OUT_DIR, add_file_log
    import jax, jax.numpy as jnp, numpy as np, pandas as pd
    import numpyro
    from numpyro.infer import MCMC, NUTS
    from cropchoice.data import load_data, model_kwargs
    from cropchoice.broadcast import one_hot
    from cropchoice.models import semipar_features, make_semipar_model, compute_logits_semipar
    from cropchoice.diagnostics import rhat_rank, ess_bulk, ess_tail, hpdi
    from cropchoice.fit_semipar import toy_model_kwargs

    RUN_DIR = (OUT_DIR / "semipar_assets") if args.out is None else Path(args.out)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    add_file_log(RUN_DIR / "run.log")
    say = log.info

    TOY = bool(os.environ.get("EXP_TOY"))
    if TOY:
        mk, countries, _ = toy_model_kwargs(Nt=600)
        C = len(countries)
        rng = np.random.default_rng(1); asset = rng.normal(size=600); asset[rng.random(600) < 0.05] = np.nan
        args.warmup, args.samples = 30, 30
    else:
        data = load_data(DATA_DIR); countries = list(data["countries"]); C = len(countries)
        mk = model_kwargs(data)
        df = pd.read_parquet(DATA_DIR / "birl_sample.parquet", columns=["hh_asset_index"])
        asset = pd.to_numeric(df["hh_asset_index"], errors="coerce").values.astype(float)
        assert len(asset) == int(mk["obs_action"].shape[0])

    ci_np = np.asarray(mk["obs_country_idx"]); N_all = len(ci_np)
    gidx, group_names = asset_groups(ci_np, asset, countries, args.min_na)
    keep = gidx >= 0; n_drop = int((~keep).sum()); G = len(group_names)
    say(f"groups G={G}: " + ", ".join(f"{c}/{t}={int((gidx == g).sum())}" for g, (c, t) in enumerate(group_names)) + f"; dropped (asset missing, small) {n_drop}")
    keep_idx = jnp.asarray(np.flatnonzero(keep))

    def sub(x):
        return x[keep_idx] if hasattr(x, "shape") and x.shape[0] == N_all else x
    mk_k = {k: sub(v) for k, v in mk.items() if k in ("q10", "q50", "q90", "mask_obs", "obs_action", "obs_country_idx")}
    feat = semipar_features(mk_k); MASK, ACT, CI = feat["MASK"], feat["ACT"], feat["CI"]
    N, A = feat["MU"].shape
    GI = jnp.asarray(gidx[keep], jnp.int32)
    OH_C = one_hot(CI, C); OH_G = one_hot(GI, G)
    say(f"semipar_assets  N={N} (of {N_all}) A={A} C={C} G={G}  devices={jax.devices()}  toy={TOY}")

    model = make_semipar_model(feat, C, OH_C, oh_g=OH_G, G=G)

    def kernel(dense):
        return NUTS(model, target_accept_prob=0.8, max_tree_depth=10, dense_mass=dense,
                    init_strategy=numpyro.infer.init_to_median())

    dense = args.dense; warm, samp = args.warmup, args.samples
    if not args.skip_timing and not TOY:
        m = MCMC(kernel(dense), num_warmup=50, num_samples=20, num_chains=args.chains, chain_method="vectorized", progress_bar=False)
        t0 = time.time(); m.warmup(jax.random.PRNGKey(1), extra_fields=("num_steps",), collect_warmup=True); tw = time.time() - t0
        lw = float(np.mean(np.asarray(m.get_extra_fields()["num_steps"])))
        t0 = time.time(); m.run(jax.random.PRNGKey(2), extra_fields=("num_steps",)); ts = time.time() - t0
        ls = float(np.mean(np.asarray(m.get_extra_fields()["num_steps"])))
        proj_h = (tw / 50 * warm + ts / 20 * samp) / 3600
        say(f"[TIMING] warmup {tw/50:.2f} s/draw (leapfrog {lw:.0f}, incl compile), sampling {ts/20:.2f} s/draw (leapfrog {ls:.0f}); projected {proj_h:.2f} h")
        if proj_h > 2:
            dense, warm, samp = True, 600, 600
            say("[TIMING] projected > 2 h: switching to dense_mass=True and 600+600")

    m = MCMC(kernel(dense), num_warmup=warm, num_samples=samp, num_chains=args.chains, chain_method="vectorized", progress_bar=False)
    t0 = time.time(); m.run(jax.random.PRNGKey(args.seed), extra_fields=("diverging", "num_steps"))
    say(f"[RUN] {args.chains} x ({warm}+{samp}) dense={dense}: {(time.time()-t0)/60:.1f} min")
    S = {k: np.asarray(v) for k, v in m.get_samples(group_by_chain=True).items()}
    ex = m.get_extra_fields(group_by_chain=True); div = np.asarray(ex["diverging"]); nsteps = np.asarray(ex["num_steps"])
    np.savez_compressed(RUN_DIR / "posterior.npz", **S, diverging=div, num_steps=nsteps,
                        group_country=np.array([g[0] for g in group_names]), group_tercile=np.array([g[1] for g in group_names]))
    say(f"divergences {int(div.sum())}/{div.size} ({div.mean():.4f}); mean leapfrog/draw {nsteps.mean():.1f}")

    # ── summary per group ──
    n_g = np.bincount(gidx[keep], minlength=G)
    rows = []
    for site, name in (("a_g", "a"), ("b_g", "b"), ("c_g", "c")):
        for g, (cn, tn) in enumerate(group_names):
            x = S[site][:, :, g]; lo, hi = hpdi(x)
            rows.append(dict(param=name, country=cn, tercile=tn, n_obs=int(n_g[g]), median=float(np.median(x)), sd=float(x.std()),
                             hpdi_lo=lo, hpdi_hi=hi, r_hat=float(rhat_rank(x)), ess_bulk=float(ess_bulk(x)), ess_tail=float(ess_tail(x))))
    summ = pd.DataFrame(rows); summ.to_csv(RUN_DIR / "summary.csv", index=False)
    asc_rhat = np.array([[rhat_rank(S["asc_c"][:, :, i, j]) for j in range(A - 1)] for i in range(C)])
    asc_ess = np.array([[ess_bulk(S["asc_c"][:, :, i, j]) for j in range(A - 1)] for i in range(C)])

    # ── derived: poor vs rich contrasts, marginal sigma effect at 0.5 ──
    a, b, c = (S[k].reshape(-1, G) for k in ("a_g", "b_g", "c_g"))
    drows, mrows = [], []
    for g, (cn, tn) in enumerate(group_names):
        me = b[:, g] + 2 * c[:, g] * 0.5; lo, hi = hpdi(me)
        mrows.append(dict(country=cn, tercile=tn, n_obs=int(n_g[g]), marg_sigma_05=float(np.median(me)), lo=lo, hi=hi,
                          a=float(np.median(a[:, g])), b=float(np.median(b[:, g])), c=float(np.median(c[:, g]))))
    for cn in countries:
        idx = {tn: g for g, (c_, tn) in enumerate(group_names) if c_ == cn}
        if "T1_poor" not in idx or "T3_rich" not in idx:
            continue
        gp, gr = idx["T1_poor"], idx["T3_rich"]
        d = {"country": cn}
        for nm, arr in (("a", a), ("b", b), ("c", c)):
            diff = arr[:, gp] - arr[:, gr]; lo, hi = hpdi(diff)
            d[f"d{nm}_poor_minus_rich"] = float(np.median(diff)); d[f"d{nm}_lo"] = lo; d[f"d{nm}_hi"] = hi; d[f"p_d{nm}_gt0"] = float(np.mean(diff > 0))
        mp = b[:, gp] + c[:, gp]; mr = b[:, gr] + c[:, gr]; diff = mp - mr; lo, hi = hpdi(diff)
        d.update(dmarg05_poor_minus_rich=float(np.median(diff)), dmarg05_lo=lo, dmarg05_hi=hi, p_dmarg05_gt0=float(np.mean(diff > 0)))
        drows.append(d)
    pd.DataFrame(mrows).to_csv(RUN_DIR / "marginal_by_group.csv", index=False)
    pd.DataFrame(drows).to_csv(RUN_DIR / "derived.csv", index=False)

    # ── log-lik at the posterior median and structural gain vs the country-level run ──
    def loglik(a_, b_, c_, asc_):
        logp = jax.nn.log_softmax(compute_logits_semipar(a_, b_, c_, asc_, feat["MU"], feat["SIG"], MASK, OH_C, OH_G, center=True), axis=-1)
        return jnp.sum(jnp.take_along_axis(logp, ACT[:, None], axis=-1))

    med = {k: jnp.asarray(np.median(v.reshape((-1,) + v.shape[2:]), axis=0)) for k, v in S.items()}
    ll = float(jax.jit(loglik)(med["a_g"], med["b_g"], med["c_g"], med["asc_c"]))
    ref = OUT_DIR / "semipar" / "results.json"
    ref_ll = ref_null = float("nan")
    if ref.exists():
        rj = json.load(open(ref)); ref_ll, ref_null = rj.get("ll_semipar", float("nan")), rj.get("ll_null", float("nan"))
    say(f"log-lik at posterior median {ll:.1f} on N={N}; country-level semipar (N={N_all}) {ref_ll:.1f}; ASC null {ref_null:.1f}")

    with open(RUN_DIR / "convergence.txt", "w") as f:
        f.write(f"semipar_assets: {args.chains} x ({warm}+{samp}) dense={dense}; N={N} (dropped {n_drop}); G={G}\n")
        f.write(f"Divergences: {int(div.sum())}/{div.size} ({div.mean():.4f})  [{'PASS' if div.mean() < 0.01 else 'FAIL'}]\n")
        f.write(f"R-hat max (a,b,c): {summ['r_hat'].max():.4f}   ESS bulk min: {summ['ess_bulk'].min():.0f}   ASC r_hat max {np.nanmax(asc_rhat):.4f}, ess min {np.nanmin(asc_ess):.0f}\n\n")
        for r in rows:
            f.write(f"  {r['param']}[{r['country']:9s} {r['tercile']:8s} n={r['n_obs']:6d}] {r['median']:+8.3f} [{r['hpdi_lo']:+8.3f}, {r['hpdi_hi']:+8.3f}]  r_hat={r['r_hat']:.3f} ess={r['ess_bulk']:.0f}\n")
        f.write("\npoor - rich contrasts (median [89% HPDI], P(>0)):\n")
        for d in drows:
            f.write(f"  {d['country']:9s} da={d['da_poor_minus_rich']:+.3f} [{d['da_lo']:+.3f},{d['da_hi']:+.3f}] P={d['p_da_gt0']:.2f}  "
                    f"db={d['db_poor_minus_rich']:+.3f} P={d['p_db_gt0']:.2f}  dc={d['dc_poor_minus_rich']:+.3f} P={d['p_dc_gt0']:.2f}  "
                    f"dmarg(0.5)={d['dmarg05_poor_minus_rich']:+.3f} [{d['dmarg05_lo']:+.3f},{d['dmarg05_hi']:+.3f}] P={d['p_dmarg05_gt0']:.2f}\n")
    for line in open(RUN_DIR / "convergence.txt"):
        say(line.rstrip())

    json.dump({"countries": countries, "groups": group_names, "n_obs_group": n_g.tolist(), "N": N, "N_all": N_all, "dropped": n_drop,
               "chains": args.chains, "warmup": warm, "samples": samp, "dense_mass": dense,
               "divergence_rate": float(div.mean()), "mean_leapfrog": float(nsteps.mean()),
               "r_hat_max_abc": float(summ["r_hat"].max()), "ess_min_abc": float(summ["ess_bulk"].min()),
               "asc_r_hat_max": float(np.nanmax(asc_rhat)), "asc_ess_min": float(np.nanmin(asc_ess)),
               "ll_median_params": ll, "ll_semipar_country": ref_ll, "ll_null_country": ref_null,
               "gain_vs_country_semipar_per_obs": (ll - ref_ll) / N if np.isfinite(ref_ll) else None,
               "summary": rows, "derived": drows, "marginal_by_group": mrows}, open(RUN_DIR / "results.json", "w"), indent=2)
    say(f"written {RUN_DIR}")
    return RUN_DIR
