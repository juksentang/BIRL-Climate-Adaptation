#!/usr/bin/env python3
"""
BIRL v2 runner (country-level Stone-Geary CRRA with smooth floor,
certainty-equivalent reward).  ALL real-data runs belong on the cluster.

Usage (from 08_BIRL_v2/):
    python3 run_v2.py --variant v2_country --timing            # 50+20, 1 chain: s/step, RSS, platform
    python3 run_v2.py --variant v2_country                     # 4 chains x (1000 + 1000)
    python3 run_v2.py --variant v2_country_gfix --s-fixed 0.3
    python3 run_v2.py --variant v2_country --s-max 0.8         # robustness (auto tag _smax08)
    python3 run_v2.py --variant v2_country --flat-priors --noncentered
    python3 run_v2.py --variant v2_country --chains 2 --chain-method vectorized --resume

Outputs: outputs/<variant>[_<tag>]/{posterior.npz, summary.csv, convergence.txt,
         ppc_action_freq.csv, ppc_crop_x_intensity.csv, model_diagnostics.json,
         run.log, run_info.json, timing.json (timing runs)}
The tag is --tag, or built from the non-default flags (smax08, flat, nc, sfix<val>,
timing) so runs never overwrite each other.  Sampling is chunked (--chunk, default
250 draws) with a checkpoint after each chunk; --resume continues from it.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path


parser = argparse.ArgumentParser(description="BIRL v2 MCMC runner")
parser.add_argument("--variant", required=True, choices=["v2_country", "v2_country_gfix"])
parser.add_argument("--s-fixed", type=float, default=0.3, help="s_c for v2_country_gfix")
parser.add_argument("--s-max", type=float, default=0.6, help="upper bound of s_c (default 0.6)")
parser.add_argument("--flat-priors", action="store_true",
                    help="independent wide per-country priors, no hyperparameters")
parser.add_argument("--noncentered", action="store_true",
                    help="non-centred hierarchical parameterisation (default: centred)")
parser.add_argument("--timing", action="store_true",
                    help="50 warmup + 20 (+20 repeated) samples, 1 chain: per-phase s/draw and "
                         "s/leapfrog (compile separated), peak device memory, projections")
parser.add_argument("--chains", type=int, default=None, help="chain count (default 4; 1 for --timing)")
parser.add_argument("--chain-method", choices=["parallel", "vectorized", "sequential"], default=None,
                    help="override the automatic choice (accelerator: parallel if devices >= "
                         "chains else vectorized; CPU: sequential -- pass 'parallel' to pmap "
                         "the chains over BIRL_HOST_DEVICES host devices, ~1.5 GB RAM each)")
parser.add_argument("--warmup", type=int, default=1000)
parser.add_argument("--samples", type=int, default=1000)
parser.add_argument("--chunk", type=int, default=250, help="draws per checkpoint chunk")
parser.add_argument("--resume", action="store_true",
                    help="continue from a checkpoint (mcmc_state.pkl / warmup_state.pkl) or reuse "
                         "an existing posterior.npz")
parser.add_argument("--stop-after-chunk", type=int, default=None,
                    help="TESTING: exit right after this many checkpoints of this process are "
                         "written (run stays resumable with --resume); used by the smoke job")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--n-ppc", type=int, default=100)
parser.add_argument("--tag", type=str, default=None, help="output subdirectory suffix")
args = parser.parse_args()

import numpy as np

from cropchoice.config import (DATA_DIR, OUT_DIR, DEVICE_INFO, N_DEVICES, PLATFORM, S_MAX,
                        JAX_CACHE_DIR, log, add_file_log, peak_rss_gb, device_memory_gb)
from cropchoice.data import load_data, model_kwargs
from cropchoice.models_v2 import MODELS
from cropchoice.inference import (run_mcmc_timed, run_mcmc_chunked, samples_by_chain, save_posterior,
                             load_posterior, choose_chain_method)
from cropchoice.diagnostics import write_all_diagnostics


def auto_tag():
    if args.tag is not None:
        return args.tag
    parts = []
    if abs(args.s_max - S_MAX) > 1e-12:
        parts.append("smax" + f"{args.s_max:g}".replace(".", ""))
    if args.variant == "v2_country_gfix" and abs(args.s_fixed - 0.3) > 1e-12:
        parts.append("sfix" + f"{args.s_fixed:g}".replace(".", ""))
    if args.flat_priors:
        parts.append("flat")
    if args.noncentered:
        parts.append("nc")
    if args.timing:
        parts.append("timing")
    return "_".join(parts)


TAG = auto_tag()
RUN_DIR = OUT_DIR / (args.variant + (f"_{TAG}" if TAG else ""))
RUN_DIR.mkdir(parents=True, exist_ok=True)
add_file_log(RUN_DIR / "run.log")


def atomic_json(path, obj):
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, str(path))


def build_model_kwargs(data):
    mk = {"data": model_kwargs(data), "s_max": args.s_max,
          "flat_priors": args.flat_priors, "noncentered": args.noncentered}
    if args.variant == "v2_country_gfix":
        mk["s_fixed"] = args.s_fixed
    return mk


def finish(samples, diverging, data, run_info, extra_lines=None):
    """Save posterior + all diagnostics."""
    save_posterior(samples, diverging, RUN_DIR / "posterior.npz",
                   meta={"variant": args.variant, "countries": data["countries"],
                         "m_c": [float(x) for x in data["m_c_np"]],
                         "s_max": args.s_max, "flat_priors": args.flat_priors,
                         "noncentered": args.noncentered,
                         "s_fixed": args.s_fixed if args.variant == "v2_country_gfix" else None,
                         "date": datetime.now().isoformat(), **DEVICE_INFO})
    write_all_diagnostics(samples, diverging, data, args.variant, RUN_DIR, run_info=run_info,
                          n_ppc=args.n_ppc, seed=args.seed + 7, extra_lines=extra_lines)
    atomic_json(RUN_DIR / "run_info.json", {**run_info, **DEVICE_INFO, "args": vars(args)})


def main():
    t_total = time.time()
    log.info("=" * 64)
    log.info(f"BIRL v2  variant={args.variant}  tag={TAG or '-'}  timing={args.timing}  "
             f"seed={args.seed}")
    log.info(f"  s_max={args.s_max}  flat_priors={args.flat_priors}  noncentered={args.noncentered}"
             + (f"  s_fixed={args.s_fixed}" if args.variant == "v2_country_gfix" else ""))
    log.info(f"  Platform: {PLATFORM} x {N_DEVICES} {DEVICE_INFO['device_kinds']}  "
             f"(JAX {DEVICE_INFO['jax_version']}, numpyro {DEVICE_INFO['numpyro_version']})")
    log.info(f"  Data: {DATA_DIR}")
    log.info(f"  Output: {RUN_DIR}")
    log.info(f"  JAX compilation cache: {JAX_CACHE_DIR}")
    log.info("=" * 64)

    data = load_data(DATA_DIR)
    mk = build_model_kwargs(data)
    model_fn = MODELS[args.variant]
    rss_after_load = peak_rss_gb()
    log.info(f"  Peak RSS after data load: {rss_after_load:.2f} GB")
    shape_kw = {"n_obs": data["N_obs"], "n_actions": data["N_actions"]}

    # ── Timing mode ──
    if args.timing:
        n_warm, n_samp = 50, 20
        n_chains = args.chains or 1
        method = choose_chain_method(n_chains, requested=args.chain_method, **shape_kw)
        log.info(f"[TIMING] {n_warm}+{n_samp}(+{n_samp} steady-state pass), {n_chains} chain(s), {method}")
        t0 = time.time()
        mcmc, ph = run_mcmc_timed(model_fn, mk, n_warm, n_samp, n_chains, seed=args.seed,
                                  chain_method=method, extra_fields=("num_steps", "accept_prob"))
        elapsed = time.time() - t0
        samples, div, extra = samples_by_chain(mcmc)          # the steady-state pass
        per_step = elapsed / (n_warm + 2 * n_samp)
        rss = peak_rss_gb()
        dev_mem = device_memory_gb()            # None on CPU; XLA preallocation makes RSS/nvidia-smi useless
        peak_dev = dev_mem.get("peak_gb") if dev_mem else None
        n_steps = extra.get("num_steps")
        mean_leapfrog = float(np.mean(n_steps)) if n_steps is not None else float("nan")
        w, s1, s2 = ph["warmup"], ph["sampling_first"], ph["sampling_steady"]
        s_per_leapfrog = s2["s_per_leapfrog"]                 # compile-free, same GPU + chain layout
        compile_sampling_s = max(s1["elapsed_s"] - s2["elapsed_s"], 0.0)
        # Projection for a (--warmup + --samples) run on THIS device with THIS chain
        # layout: leapfrogs x s/leapfrog.  The warmup mean leapfrog count comes from
        # only n_warm early (deep-tree) draws, so the warmup half is an upper bound.
        proj_h = (s_per_leapfrog * (w["mean_leapfrog_per_draw"] * args.warmup
                                    + s2["mean_leapfrog_per_draw"] * args.samples)) / 3600.0
        proj_naive_h = per_step * (args.warmup + args.samples) / 3600.0
        info = {"variant": args.variant, "tag": TAG, "n_warmup": n_warm, "n_samples": n_samp,
                "n_chains": n_chains, "chain_method": method,
                "elapsed_s": elapsed, "per_step_s_incl_compile": per_step,
                "phases": ph, "compile_sampling_s_est": compile_sampling_s,
                "mean_leapfrog_per_draw": mean_leapfrog,               # steady-state sampling pass
                "mean_leapfrog_per_draw_warmup": w["mean_leapfrog_per_draw"],
                "s_per_leapfrog_steady": s_per_leapfrog,
                "s_per_draw_steady": s2["s_per_draw"],
                "mean_accept_prob": float(np.mean(extra["accept_prob"])) if "accept_prob" in extra else None,
                "div_count": int(div.sum()), "div_total": int(div.size),
                "peak_rss_gb": rss, "rss_after_load_gb": rss_after_load,
                "peak_device_gb": peak_dev,
                "peak_device_gb_per_chain": peak_dev / n_chains if peak_dev else None,
                "device_memory": dev_mem,
                "projected_full_run_h": proj_h,
                "projected_full_run_h_naive_incl_compile": proj_naive_h,
                "projection_note": (f"projected_full_run_h = s_per_leapfrog_steady x "
                                    f"(warmup leapfrog/draw x {args.warmup} + steady leapfrog/draw x "
                                    f"{args.samples}) / 3600, for {n_chains} chain(s) {method} on "
                                    f"{DEVICE_INFO.get('device_kinds')}. For another GPU / chain layout "
                                    f"rescale s_per_leapfrog_steady: MIG 1g.10gb -> full H100 is roughly "
                                    f"/7 (SMs+bandwidth); 1 -> 4 vectorized lock-stepped chains is "
                                    f"roughly x3-4 on a memory-bound kernel, plus lock-step waits on the "
                                    f"deepest tree. The warmup term is an upper bound (mean depth over "
                                    f"{n_warm} early draws)."),
                "date": datetime.now().isoformat(), **DEVICE_INFO}
        atomic_json(RUN_DIR / "timing.json", info)
        line = (f"TIMING variant={args.variant} platform={PLATFORM} devices={N_DEVICES} "
                f"chains={n_chains} {method} elapsed_s={elapsed:.1f} "
                f"warmup_s_per_draw={w['s_per_draw']:.2f} steady_s_per_draw={s2['s_per_draw']:.2f} "
                f"s_per_leapfrog={s_per_leapfrog:.4f} leapfrog_per_draw_warmup={w['mean_leapfrog_per_draw']:.1f} "
                f"leapfrog_per_draw_steady={s2['mean_leapfrog_per_draw']:.1f} "
                f"compile_s~{compile_sampling_s:.0f} peak_rss_gb={rss:.2f} "
                + (f"peak_device_gb={peak_dev:.2f} (per chain {peak_dev / n_chains:.2f}, "
                   f"limit {dev_mem.get('limit_gb', float('nan')):.1f}) " if peak_dev else "")
                + f"div={int(div.sum())}/{div.size} "
                f"projected_{args.warmup}+{args.samples}_same_layout_h={proj_h:.2f} "
                f"(naive incl. compile {proj_naive_h:.2f})")
        log.info(line)
        print(line, flush=True)
        finish(samples, div, data, {"chain_method": method, "mode": "timing", **info},
               extra_lines=[f"[timing run: {n_warm}+{n_samp}+{n_samp}, steady {s2['s_per_draw']:.2f} s/draw, "
                            f"{s_per_leapfrog:.4f} s/leapfrog, peak RSS {rss:.2f} GB"
                            + (f", peak device {peak_dev:.2f} GB" if peak_dev else "")
                            + f", {PLATFORM}]"])
        log.info(f"Timing run complete in {time.time()-t_total:.1f}s")
        return

    # ── Full run ──
    n_chains = args.chains or 4
    method = choose_chain_method(n_chains, requested=args.chain_method, **shape_kw)
    log.info(f"[RUN] {args.warmup}+{args.samples}, {n_chains} chains, chain_method={method}, "
             f"chunk={args.chunk}, resume={args.resume}")

    post_path = RUN_DIR / "posterior.npz"
    mcmc_min = None
    if post_path.exists() and args.resume:
        samples, div, _ = load_posterior(post_path)
        log.info(f"[SKIP] MCMC: loaded existing {post_path} (--resume); recomputing diagnostics")
    else:
        t1 = time.time()
        res = run_mcmc_chunked(model_fn, mk, args.warmup, args.samples, n_chains,
                               seed=args.seed, chunk_size=args.chunk,
                               out_dir=RUN_DIR, chain_method=method,
                               resume=args.resume, stop_after_chunk=args.stop_after_chunk)
        if res is None:
            log.info(f"[STOP] --stop-after-chunk {args.stop_after_chunk}: exiting after "
                     f"{(time.time() - t1) / 60:.1f} min with the checkpoint kept in {RUN_DIR}")
            return
        samples, div = res
        mcmc_min = (time.time() - t1) / 60
        log.info(f"[DONE] MCMC in {mcmc_min:.1f} min, peak RSS {peak_rss_gb():.2f} GB")

    dev_mem = device_memory_gb()
    run_info = {"mode": "full", "chain_method": method, "n_chains_requested": n_chains,
                "warmup": args.warmup, "samples": args.samples, "chunk": args.chunk,
                "mcmc_minutes": mcmc_min, "peak_rss_gb": peak_rss_gb(),
                "peak_device_gb": dev_mem.get("peak_gb") if dev_mem else None,
                "device_memory": dev_mem, "tag": TAG}
    finish(samples, div, data, run_info,
           extra_lines=[f"[full run: {n_chains} x ({args.warmup}+{args.samples}), {method}, "
                        f"{PLATFORM} x {N_DEVICES}, peak RSS {peak_rss_gb():.2f} GB]"])
    log.info(f"BIRL v2 {args.variant}{'_' + TAG if TAG else ''} complete. "
             f"Total {(time.time()-t_total)/60:.1f} min")


if __name__ == "__main__":
    main()
