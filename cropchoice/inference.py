"""
NUTS execution with chunked checkpoint/resume (adapted from 06's mcmc_runner;
GCS sync dropped).  Posterior samples are saved as posterior.npz (one numpy
array per site, shape (n_chains, n_draws, ...)), never as pickle.  Only the
resume state (HMC state after warmup / after the last chunk) between chunks is
pickled, in mcmc_state.pkl, and it is deleted once the run completes.

chain_method selection (spec): on an accelerator, 'parallel' when
jax.device_count() >= chains, else 'vectorized'; on CPU 'sequential' unless
--chain-method parallel is requested explicitly (pmap over BIRL_HOST_DEVICES
host devices runs one full-data gradient per chain concurrently, ~1.5 GB each
under XLA-CPU, which is OOM territory on a small machine).  The decision and
the estimated per-chain footprint are logged.
"""

import os
import json
import time
import pickle
import logging
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import numpyro
from numpyro.infer import MCMC, NUTS

from cropchoice.config import N_DEVICES, PLATFORM, host_ram_gb, device_memory_gb

log = logging.getLogger("birl_v2")

DEFAULT_CHUNK = 250
TARGET_ACCEPT = 0.8
MAX_TREE_DEPTH = 10
STATE_FILE = "mcmc_state.pkl"
PARTIAL_FILE = "samples_partial.npz"
WARMUP_FILE = "warmup_state.pkl"
CHAIN_METHODS = ("parallel", "vectorized", "sequential")


def make_kernel(model_fn):
    return NUTS(model_fn, target_accept_prob=TARGET_ACCEPT,
                max_tree_depth=MAX_TREE_DEPTH,
                init_strategy=numpyro.infer.init_to_median())


# rough per-chain working set of one full-data gradient evaluation: ~30 (N, A)
# float32 residual / temporary arrays (five-node stacks, exp, where, softmax, ...)
ARRAYS_PER_CHAIN = 30
ACCELERATOR_PLATFORMS = ("gpu", "tpu", "cuda", "rocm")


def chain_footprint_gb(n_obs, n_actions, n_arrays=ARRAYS_PER_CHAIN):
    """Estimated device/host memory of one chain's gradient evaluation in GB."""
    return n_obs * n_actions * 4 * n_arrays / 1e9


def choose_chain_method(n_chains, requested=None, n_devices=None, platform=None,
                        n_obs=None, n_actions=None):
    """Accelerator: 'parallel' if devices >= chains, else 'vectorized'.
    CPU: 'sequential' (memory: n_chains concurrent full-data gradients) unless
    `requested` says otherwise.  Logs the decision and the per-chain footprint."""
    n_devices = N_DEVICES if n_devices is None else n_devices
    platform = PLATFORM if platform is None else platform
    fp = chain_footprint_gb(n_obs, n_actions) if (n_obs and n_actions) else None
    fp_txt = f", est. {fp:.2f} GB per chain" if fp is not None else ""
    if requested:
        if requested not in CHAIN_METHODS:
            raise ValueError(f"chain_method must be one of {CHAIN_METHODS}")
        method, why = requested, "requested"
    elif n_chains <= 1:
        method, why = "sequential", "single chain"
    elif platform in ACCELERATOR_PLATFORMS:
        if n_devices >= n_chains:
            method, why = "parallel", f"{n_devices} {platform} devices >= {n_chains} chains"
        else:
            method, why = "vectorized", f"{n_devices} {platform} device(s) < {n_chains} chains"
    else:
        ram = host_ram_gb()
        ram_txt = f"{ram:.1f} GB host RAM" if ram else "host RAM unknown"
        method = "sequential"
        why = (f"cpu: {n_chains} concurrent full-data gradients would need "
               f"~{n_chains} x {fp:.1f} GB ({ram_txt}); use --chain-method parallel to opt in"
               if fp is not None else
               f"cpu ({ram_txt}); use --chain-method parallel to opt in")
    log.info(f"  chain_method={method} ({why}{fp_txt})")
    if method == "parallel" and n_devices < n_chains:
        log.warning(f"  chain_method=parallel with {n_devices} device(s) < {n_chains} chains: "
                    f"numpyro will run the chains sequentially")
    return method


def default_chains(requested=None, default=4):
    return int(requested) if requested else default


# =====================================================================
# One-shot MCMC (timing test / recovery runs)
# =====================================================================

def run_mcmc(model_fn, model_kwargs, n_warmup, n_samples, n_chains, seed=42,
             chain_method="parallel", progress_bar=False, extra_fields=("num_steps",)):
    """Run NUTS in one shot. model_kwargs are passed as **kwargs to the model
    (e.g. {"data": ..., "s_max": ..., "flat_priors": ...}). Returns the MCMC object."""
    mcmc = MCMC(make_kernel(model_fn), num_warmup=n_warmup, num_samples=n_samples,
                num_chains=n_chains, chain_method=chain_method, progress_bar=progress_bar)
    mcmc.run(jax.random.PRNGKey(seed), extra_fields=tuple(extra_fields), **model_kwargs)
    return mcmc


def run_mcmc_timed(model_fn, model_kwargs, n_warmup, n_samples, n_chains, seed=42,
                   chain_method="sequential", extra_fields=("num_steps", "accept_prob")):
    """Timing run in three timed phases so compile time and the warmup phase can be
    separated from the steady-state cost of one leapfrog step:
      1. warmup  (collect_warmup=True -> num_steps of every warmup draw are kept)
      2. sampling pass 1 (includes the compile of the sampling loop)
      3. sampling pass 2 from the same post-warmup state (compile cached -> the
         steady-state s/draw and s/leapfrog)
    Returns (mcmc, phases) where phases[name] = {"elapsed_s", "n_draws",
    "leapfrog_total", "mean_leapfrog_per_draw", "s_per_draw", "s_per_leapfrog"}."""
    ef = tuple(extra_fields)
    mcmc = MCMC(make_kernel(model_fn), num_warmup=n_warmup, num_samples=n_samples,
                num_chains=n_chains, chain_method=chain_method, progress_bar=False)
    phases = {}

    def _phase(name, t0):
        extra = mcmc.get_extra_fields(group_by_chain=True)
        steps = np.asarray(extra["num_steps"]) if "num_steps" in extra else None
        dt = time.time() - t0
        n_draws = int(steps.size) if steps is not None else n_chains * (n_warmup if name == "warmup" else n_samples)
        tot = float(steps.sum()) if steps is not None else float("nan")
        phases[name] = {"elapsed_s": dt, "n_draws": n_draws, "leapfrog_total": tot,
                        "mean_leapfrog_per_draw": tot / max(n_draws, 1),
                        "s_per_draw": dt / max(n_draws / max(n_chains, 1), 1),
                        "s_per_leapfrog": dt / max(tot / max(n_chains, 1), 1.0)}
        log.info(f"  [TIMING] {name}: {dt:.1f}s, {n_draws} draws, mean leapfrog/draw "
                 f"{phases[name]['mean_leapfrog_per_draw']:.1f}, "
                 f"{phases[name]['s_per_draw']:.2f} s/draw, {phases[name]['s_per_leapfrog']:.3f} s/leapfrog")

    t0 = time.time()
    mcmc.warmup(jax.random.PRNGKey(seed), extra_fields=ef, collect_warmup=True, **model_kwargs)
    _phase("warmup", t0)
    t0 = time.time()
    mcmc.run(jax.random.PRNGKey(seed + 1), extra_fields=ef, **model_kwargs)
    _phase("sampling_first", t0)                     # includes the sampling-loop compile
    mcmc.post_warmup_state = mcmc.last_state
    t0 = time.time()
    mcmc.run(jax.random.PRNGKey(seed + 2), extra_fields=ef, **model_kwargs)
    _phase("sampling_steady", t0)                    # compile cached: steady-state cost
    return mcmc, phases


def samples_by_chain(mcmc):
    """(samples dict, diverging, extra) with a leading chain axis."""
    samples = {k: np.asarray(v) for k, v in mcmc.get_samples(group_by_chain=True).items()}
    extra = {k: np.asarray(v) for k, v in mcmc.get_extra_fields(group_by_chain=True).items()}
    div = extra.pop("diverging", None)
    if div is None:
        first = next(iter(samples.values()))
        div = np.zeros(first.shape[:2], dtype=bool)
    return samples, div, extra


# =====================================================================
# Chunked MCMC with checkpoint/resume
# =====================================================================

def run_mcmc_chunked(model_fn, model_kwargs, n_warmup, n_samples, n_chains, seed=42,
                     chunk_size=DEFAULT_CHUNK, out_dir=None, chain_method="parallel",
                     progress_bar=False, resume=False, stop_after_chunk=None):
    """NUTS with periodic checkpoints (to survive SLURM time limits).

    1. Warmup runs in one shot (adaptation needs continuity); the post-warmup
       HMC state is pickled to out_dir/warmup_state.pkl so a time-out during
       the first sampling chunk does not repeat the (long) warmup.
    2. Sampling is split into chunks of `chunk_size` draws per chain.
    3. After each chunk the accumulated samples (npz) and the HMC state
       (pickle) are written to out_dir.  With resume=True and a matching
       checkpoint present, warmup is skipped and sampling continues from it
       (or, with only warmup_state.pkl present, sampling starts from it);
       without resume a stale checkpoint is discarded with a warning.
       The stored chain_method must match: re-submit with the same --gres.

    stop_after_chunk=N (testing only): return None right after the N-th
    checkpoint of this process is written, leaving the run resumable.

    Returns (samples dict (n_chains, n_draws, ...), diverging (n_chains, n_draws)),
    or None when stopped early by stop_after_chunk.
    """
    out_dir = Path(out_dir) if out_dir else None
    state_path = out_dir / STATE_FILE if out_dir else None
    partial_path = out_dir / PARTIAL_FILE if out_dir else None
    warm_path = out_dir / WARMUP_FILE if out_dir else None
    if stop_after_chunk is not None and not out_dir:
        raise ValueError("stop_after_chunk needs out_dir (nothing to resume from otherwise)")
    kernel = make_kernel(model_fn)

    resumed = False
    done = 0
    all_samples, all_div = {}, None
    have_ckpt = bool(state_path and state_path.exists() and partial_path.exists())
    if have_ckpt and resume:
        ckpt = _load_state(state_path)
        want = {"n_chains": n_chains, "n_samples_target": n_samples, "chain_method": chain_method,
                "seed": seed, "n_warmup": n_warmup, "chunk_size": chunk_size}
        mismatch = {k: (ckpt.get(k), v) for k, v in want.items() if ckpt.get(k, v) != v}
        if not mismatch:
            all_samples, all_div, meta = load_posterior(partial_path)
            post_warmup_state = ckpt["post_warmup_state"]
            done = int(ckpt["samples_collected"])
            # The checkpoint is two files written one after the other; a kill between
            # them leaves the npz one chunk ahead of the pkl.  The pkl (HMC state +
            # count) is authoritative: truncate the draws to its count so the chunk
            # after it is not sampled twice.
            n_have = int(next(iter(all_samples.values())).shape[1]) if all_samples else 0
            if n_have != done or meta.get("samples_collected", done) != done:
                log.warning(f"  Checkpoint files disagree: {PARTIAL_FILE} holds {n_have} draws "
                            f"(meta {meta.get('samples_collected')}), {STATE_FILE} says {done}; "
                            f"truncating the draws to {done}")
                if n_have < done:
                    raise RuntimeError(f"{PARTIAL_FILE} has fewer draws ({n_have}) than "
                                       f"{STATE_FILE} records ({done}); delete both and restart")
                all_samples = {k: v[:, :done] for k, v in all_samples.items()}
                all_div = all_div[:, :done]
            resumed = True
            log.info(f"  Resumed from checkpoint: {done}/{n_samples} draws per chain")
        else:
            log.warning(f"  Checkpoint config mismatch {mismatch} (stored, requested); restarting")
    elif have_ckpt:
        log.warning("  Checkpoint present but --resume not given: discarding it and restarting")
        for p in (state_path, partial_path):
            p.unlink()

    if not resumed:
        post_warmup_state = None
        warm_cfg = {"n_chains": n_chains, "chain_method": chain_method, "seed": seed,
                    "n_warmup": n_warmup}
        if warm_path and warm_path.exists():
            if resume:
                w = _load_state(warm_path)
                mismatch = {k: (w.get(k), v) for k, v in warm_cfg.items() if w.get(k, v) != v}
                if not mismatch:
                    post_warmup_state = w["post_warmup_state"]
                    log.info(f"  Resumed post-warmup state from {WARMUP_FILE}: warmup skipped, "
                             f"sampling starts at 0/{n_samples}")
                else:
                    log.warning(f"  {WARMUP_FILE} config mismatch {mismatch} (stored, requested); "
                                f"re-running warmup")
            else:
                log.warning(f"  {WARMUP_FILE} present but --resume not given: discarding it")
                warm_path.unlink()
        if post_warmup_state is None:
            log.info(f"  Warmup: {n_warmup} steps x {n_chains} chains ({chain_method})...")
            t0 = time.time()
            mcmc_w = MCMC(kernel, num_warmup=n_warmup, num_samples=1, num_chains=n_chains,
                          chain_method=chain_method, progress_bar=progress_bar)
            mcmc_w.run(jax.random.PRNGKey(seed), **model_kwargs)
            post_warmup_state = mcmc_w.last_state
            log.info(f"  Warmup done: {time.time()-t0:.1f}s")
            del mcmc_w
            if warm_path:
                _save_state(warm_path, {"post_warmup_state": post_warmup_state, **warm_cfg})
                log.info(f"  Saved: {warm_path} (post-warmup HMC state; --resume skips warmup)")

    remaining = n_samples - done
    chunk_idx = done // chunk_size
    chunks_this_process = 0
    mem_logged = False
    while remaining > 0:
        this_chunk = min(chunk_size, remaining)
        chunk_idx += 1
        t1 = time.time()
        mcmc_c = MCMC(kernel, num_warmup=0, num_samples=this_chunk, num_chains=n_chains,
                      chain_method=chain_method, progress_bar=progress_bar)
        mcmc_c.post_warmup_state = post_warmup_state
        mcmc_c.run(jax.random.PRNGKey(seed + 1000 * chunk_idx), extra_fields=("num_steps",),
                   **model_kwargs)

        s, d, _ = samples_by_chain(mcmc_c)
        if not all_samples:
            all_samples, all_div = s, d
        else:
            all_samples = {k: np.concatenate([all_samples[k], s[k]], axis=1) for k in all_samples}
            all_div = np.concatenate([all_div, d], axis=1)

        post_warmup_state = mcmc_c.last_state
        done += this_chunk
        remaining -= this_chunk
        dt = time.time() - t1
        log.info(f"  Chunk {chunk_idx}: {this_chunk} draws in {dt:.1f}s "
                 f"({dt/this_chunk:.2f} s/draw), div={int(d.sum())}/{d.size}, "
                 f"total {done}/{n_samples}")

        if out_dir:
            save_posterior(all_samples, all_div, partial_path,
                           meta={"n_chains": n_chains, "samples_collected": done})
            _save_state(state_path, {
                "post_warmup_state": post_warmup_state, "samples_collected": done,
                "n_samples_target": n_samples, "n_chains": n_chains,
                "chain_method": chain_method, "seed": seed, "n_warmup": n_warmup,
                "chunk_size": chunk_size})
        if not mem_logged:                       # after the first chunk of this process
            mem_logged = _log_device_memory(n_chains) is not None or PLATFORM == "cpu"
        del mcmc_c
        chunks_this_process += 1
        if stop_after_chunk is not None and chunks_this_process >= stop_after_chunk and remaining > 0:
            log.info(f"  [STOP] stop_after_chunk={stop_after_chunk}: checkpoint at {done}/{n_samples} "
                     f"draws kept in {out_dir}; re-run with --resume to continue")
            return None

    log.info(f"  Sampling complete: {n_chains} x {done} draws, "
             f"div={int(all_div.sum())}/{all_div.size} ({all_div.mean():.2%})")
    if out_dir:
        for p in (state_path, partial_path, warm_path):
            if p.exists():
                p.unlink()
    return all_samples, all_div


def _log_device_memory(n_chains):
    """Log accelerator memory (peak / in use / limit) and the per-chain share."""
    mem = device_memory_gb()
    if mem and "peak_gb" in mem:
        log.info(f"  Device memory {mem.get('device')}: peak {mem['peak_gb']:.2f} GB, "
                 f"in use {mem.get('in_use_gb', float('nan')):.2f} GB, "
                 f"limit {mem.get('limit_gb', float('nan')):.2f} GB "
                 f"(~{mem['peak_gb'] / max(n_chains, 1):.2f} GB per chain)")
    return mem


# =====================================================================
# Posterior I/O (npz, no pickle)
# =====================================================================

def save_posterior(samples, diverging, path, meta=None):
    """Write samples (dict of arrays) + diverging + JSON meta to an npz file atomically."""
    path = Path(path)
    tmp = str(path) + ".tmp.npz"
    payload = {k: np.asarray(v) for k, v in samples.items()}
    payload["__diverging__"] = np.asarray(diverging)
    payload["__meta__"] = np.array(json.dumps(meta or {}))
    np.savez(tmp, **payload)
    os.replace(tmp, str(path))
    log.info(f"  Saved: {path} ({path.stat().st_size/1e6:.1f} MB)")


def load_posterior(path):
    """Returns (samples dict, diverging, meta dict)."""
    z = np.load(str(path))
    samples = {k: z[k] for k in z.files if not k.startswith("__")}
    div = z["__diverging__"] if "__diverging__" in z.files else None
    meta = json.loads(str(z["__meta__"])) if "__meta__" in z.files else {}
    return samples, div, meta


def flatten_chains(samples):
    """(n_chains, n_draws, ...) -> (n_chains*n_draws, ...)."""
    return {k: v.reshape((-1,) + v.shape[2:]) for k, v in samples.items()}


# =====================================================================
# Resume state (pickle of the HMC state only)
# =====================================================================

def _to_numpy(d):
    if isinstance(d, dict):
        return {k: _to_numpy(v) for k, v in d.items()}
    if hasattr(d, "_fields"):
        return type(d)(*[_to_numpy(getattr(d, f)) for f in d._fields])
    if isinstance(d, (list, tuple)):
        return type(d)([_to_numpy(v) for v in d])
    if isinstance(d, jax.Array):
        return np.asarray(d)
    return d


def _to_jax(d):
    if isinstance(d, dict):
        return {k: _to_jax(v) for k, v in d.items()}
    if hasattr(d, "_fields"):
        return type(d)(*[_to_jax(getattr(d, f)) for f in d._fields])
    if isinstance(d, (list, tuple)):
        return type(d)([_to_jax(v) for v in d])
    if isinstance(d, np.ndarray):
        return jnp.asarray(d)
    return d


def _save_state(path, state):
    tmp = str(path) + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(_to_numpy(state), f)
    os.replace(tmp, str(path))


def _load_state(path):
    with open(path, "rb") as f:
        state = pickle.load(f)
    state["post_warmup_state"] = _to_jax(state["post_warmup_state"])
    return state
