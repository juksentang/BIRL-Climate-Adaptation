#!/usr/bin/env python3
"""
BIRL MCMC runner. Two variants only:
  - hier_noalpha: Main model (hierarchical ρ+γ, no α, learnable β)
  - R3:           Robustness (fixed ρ=1.5, hierarchical γ, learnable β)

Usage:
    python3 run_birl.py --variant hier_noalpha
    python3 run_birl.py --variant R3
    python3 run_birl.py --variant hier_noalpha --dry-run
"""

import argparse
import json
import os
import time
from datetime import datetime

import numpy as np

# ── Variant registry ──
VARIANTS = {
    "hier_noalpha": (2000, 3000, 42,  "Main: hierarchical rho+gamma, no alpha"),
    "R3":           (1000, 3000, 103, "Robustness: fixed rho=1.5"),
}

parser = argparse.ArgumentParser(description="BIRL MCMC runner")
parser.add_argument("--variant", required=True, choices=VARIANTS.keys())
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--dry-run", action="store_true")
args = parser.parse_args()

warmup, samples, default_seed, desc = VARIANTS[args.variant]
seed = args.seed if args.seed is not None else default_seed

from src.config import BASE_DIR, DATA_DIR, OUT_BASE, N_DEVICES, GCS_BUCKET, log
from src.data_loader import load_all
from src.models import birl_hier_noalpha, birl_r3
from src.mcmc_runner import (
    run_mcmc, run_mcmc_chunked, save_posterior,
    load_checkpoint_full, get_divergence_info,
    sync_to_gcs, sync_dir_to_gcs,
)
from src.diagnostics import quick_report, run_ppc_from_samples
from src.posterior import extract_and_save_posterior

n_chains = min(4, N_DEVICES)
OUT_DIR = OUT_BASE / args.variant
OUT_DIR.mkdir(parents=True, exist_ok=True)
GCS_DIR = f"{GCS_BUCKET}/{args.variant}"
(BASE_DIR / "logs").mkdir(exist_ok=True)


def atomic_json_write(path, data):
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.rename(tmp, str(path))


def get_variant_config(variant, data):
    if variant == "hier_noalpha":
        return birl_hier_noalpha, data.main_data
    elif variant == "R3":
        return birl_r3, data.main_data


# ── SVI guide → MCMC init values ──
SVI_GUIDES = {
    "hier_noalpha": "svi_hier_noalpha_guide.npz",
    "R3":           "svi_r3_guide.npz",
}

def load_svi_init(variant):
    """Load SVI posterior means as MCMC initialization values.

    SVI AutoNormal stores '{param}_auto_loc' for each latent site.
    Strip the '_auto_loc' suffix to get the NumPyro site name.
    """
    guide_file = DATA_DIR / SVI_GUIDES.get(variant, "")
    if not guide_file.exists():
        log.info(f"  SVI guide not found: {guide_file}, using default init")
        return None

    import jax.numpy as jnp
    guide = np.load(str(guide_file))
    init_values = {}
    for k in guide.keys():
        if k.endswith("_auto_loc"):
            site_name = k.replace("_auto_loc", "")
            init_values[site_name] = jnp.array(guide[k])

    # AutoNormal stores HalfNormal sites in unconstrained space (can be negative).
    # init_to_value expects constrained values (>0), so transform via exp().
    HALFNORMAL_SITES = {"sigma_rho_G", "sigma_rho_W", "sigma_gamma_G", "sigma_gamma_W"}
    for site in HALFNORMAL_SITES:
        if site in init_values:
            init_values[site] = jnp.exp(init_values[site])

    log.info(f"  SVI init: {len(init_values)} sites from {guide_file.name}")
    for k, v in sorted(init_values.items()):
        if v.ndim == 0:
            log.info(f"    {k}: {float(v):.4f}")
        else:
            log.info(f"    {k}: shape={v.shape}, mean={float(v.mean()):.4f}")
    return init_values


def main():
    log.info("=" * 60)
    log.info(f"BIRL MCMC: variant={args.variant}  seed={seed}  "
             f"warmup={warmup}  samples={samples}  chains={n_chains}")
    log.info(f"Description: {desc}")
    log.info(f"Output: {OUT_DIR}")
    log.info(f"GCS:    {GCS_DIR}")
    log.info(f"Time: {datetime.now().isoformat()}")
    log.info("=" * 60)
    t_total = time.time()

    # ── Step 0: Load Data ──
    log.info("\n[RUN] Loading data...")
    data = load_all(DATA_DIR)
    model_fn, data_kwargs = get_variant_config(args.variant, data)
    log.info(f"  Model: {model_fn.__name__}")

    if args.dry_run:
        log.info("\n[DRY-RUN] Data loaded. Exiting.")
        return

    # ── Load SVI posterior as init ──
    log.info("\n[RUN] Loading SVI initialization...")
    init_values = load_svi_init(args.variant)

    # init_values keep SVI's original shapes — NumPyro's parallel chain
    # machinery (vmap) handles replication internally.

    # ── Step 1: Timing Test ──
    N_WARMUP_TIMING = 200
    N_SAMPLE_TIMING = 50
    ABORT_THRESHOLD = 0.30

    timing_path = OUT_DIR / "timing.json"
    if timing_path.exists():
        with open(timing_path) as f:
            cfg = json.load(f)
        log.info(f"\n[SKIP] Timing: cached {cfg['per_step']:.3f}s/step")
        if cfg.get("sample_div_rate", 0) > ABORT_THRESHOLD:
            log.error(f"ABORT: Cached div_rate={cfg['sample_div_rate']:.0%}")
            return
    else:
        log.info(f"\n[RUN] Timing test ({N_WARMUP_TIMING}+{N_SAMPLE_TIMING}, 1 chain)...")
        t0 = time.time()
        mcmc_t = run_mcmc(model_fn, data_kwargs,
                          N_WARMUP_TIMING, N_SAMPLE_TIMING, 1, seed=0,
                          init_values=init_values)
        elapsed = time.time() - t0
        per_step = elapsed / (N_WARMUP_TIMING + N_SAMPLE_TIMING)
        div_info = get_divergence_info(mcmc_t)
        cfg = {"variant": args.variant, "per_step": per_step,
               "elapsed_s": elapsed,
               "sample_div_count": div_info["div_count"],
               "sample_div_total": div_info["div_total"],
               "sample_div_rate": div_info["div_rate"],
               "n_warmup_timing": N_WARMUP_TIMING,
               "n_sample_timing": N_SAMPLE_TIMING}
        atomic_json_write(timing_path, cfg)
        sync_to_gcs(timing_path, GCS_DIR)
        log.info(f"[DONE] Timing: {per_step:.3f}s/step, "
                 f"div={div_info['div_count']}/{div_info['div_total']} "
                 f"({div_info['div_rate']:.0%})")
        del mcmc_t

        if cfg["sample_div_rate"] > ABORT_THRESHOLD:
            log.error(f"ABORT: div_rate={cfg['sample_div_rate']:.0%} > {ABORT_THRESHOLD:.0%}")
            abort_report = {"variant": args.variant, "status": "ABORTED",
                            "reason": f"div_{cfg['sample_div_rate']:.2f}", **cfg}
            atomic_json_write(OUT_DIR / "abort_report.json", abort_report)
            sync_to_gcs(OUT_DIR / "abort_report.json", GCS_DIR)
            return

    # ── Step 2: Main MCMC ──
    posterior_path = OUT_DIR / "posterior.pkl"
    all_diverging = None

    if posterior_path.exists():
        np_samples, all_diverging = load_checkpoint_full(posterior_path)
        log.info(f"\n[SKIP] MCMC: loaded from checkpoint")
    else:
        log.info(f"\n[RUN] MCMC: {warmup}+{samples}, {n_chains} chains...")
        t1 = time.time()
        np_samples, all_diverging = run_mcmc_chunked(
            model_fn, data_kwargs, warmup, samples, n_chains,
            seed=seed, chunk_size=500,
            out_dir=OUT_DIR, gcs_dir=GCS_DIR,
        )
        save_posterior(np_samples, all_diverging, "posterior", OUT_DIR, gcs_dir=GCS_DIR)
        log.info(f"[DONE] MCMC: {time.time()-t1:.1f}s")

    # ── Step 3: Diagnostics ──
    diag_path = OUT_DIR / "convergence_report.txt"
    if not diag_path.exists():
        log.info("\n[RUN] Diagnostics...")
        quick_report(np_samples, args.variant, OUT_DIR)
        div_total = all_diverging.size if all_diverging is not None else 0
        div_count = int(all_diverging.sum()) if all_diverging is not None else 0
        div_rate = div_count / max(div_total, 1)
        lines = [
            f"{'='*60}", f"Convergence: {args.variant}", f"{'='*60}", "",
            f"Divergences: {div_count}/{div_total} ({div_rate:.4f})  "
            f"[{'PASS' if div_rate < 0.01 else 'FAIL'}]",
        ]
        qr = OUT_DIR / f"quick_report_{args.variant}.txt"
        if qr.exists():
            lines.append(""); lines.append(qr.read_text())
        with open(diag_path, "w") as f:
            f.write("\n".join(lines))
        sync_to_gcs(diag_path, GCS_DIR)

    # ── Step 4: Posterior Extraction ──
    hh_path = OUT_DIR / "main_hh_params.parquet"
    if not hh_path.exists():
        log.info("\n[RUN] Posterior extraction...")
        extract_and_save_posterior(
            np_samples, data.hh_list, data.hh_countries, data.countries, OUT_DIR)
        for f in OUT_DIR.glob("main_*.csv"):
            sync_to_gcs(f, GCS_DIR)
        sync_to_gcs(hh_path, GCS_DIR)

    # ── Step 5: PPC ──
    ppc_path = OUT_DIR / "ppc_action_freq.csv"
    if not ppc_path.exists():
        log.info("\n[RUN] PPC...")
        run_ppc_from_samples(
            np_samples, model_fn, data_kwargs,
            data.N_ACTIONS, data.action_cfg["action_labels"],
            n_ppc=200, out_dir=OUT_DIR)
        if ppc_path.exists():
            sync_to_gcs(ppc_path, GCS_DIR)

    # ── Final sync ──
    sync_dir_to_gcs(OUT_DIR, GCS_DIR)

    total_time = time.time() - t_total
    log.info(f"\n{'='*60}")
    log.info(f"BIRL {args.variant} complete.  Total: {total_time/60:.1f} min")
    log.info(f"{'='*60}")


if __name__ == "__main__":
    main()
