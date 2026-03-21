"""
MCMC execution with chunked checkpoint/resume, and GCS sync for preemptible VMs.

Sampling is split into chunks (default 500 samples). After each chunk the
posterior + MCMC state are saved to disk and synced to GCS, so a preempted
TPU spot VM can resume from the last chunk.

Round 1 change: added run_mcmc_chunked / save_mcmc_state / load_mcmc_state.
"""

import os
import pickle
import logging
import subprocess
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import numpyro
from numpyro.infer import MCMC, NUTS

log = logging.getLogger("birl")

# ── Default chunk size for checkpoint ──
DEFAULT_CHUNK = 500


# =====================================================================
# GCS sync helpers
# =====================================================================

def sync_to_gcs(local_path, gcs_dir=None):
    """Best-effort upload a file to GCS. Failures logged but never block."""
    from src.config import GCS_BUCKET
    if gcs_dir is None:
        gcs_dir = GCS_BUCKET
    gcs_path = f"{gcs_dir}/{Path(local_path).name}"
    try:
        subprocess.run(
            ["gsutil", "-q", "cp", str(local_path), gcs_path],
            timeout=300, check=True, capture_output=True,
        )
        log.info(f"  GCS: {gcs_path}")
    except FileNotFoundError:
        log.debug("  GCS: gsutil not found (local dev)")
    except Exception as e:
        log.warning(f"  GCS upload failed (non-fatal): {e}")


def sync_dir_to_gcs(local_dir, gcs_dir=None):
    """Best-effort rsync an entire directory to GCS."""
    from src.config import GCS_BUCKET
    if gcs_dir is None:
        gcs_dir = GCS_BUCKET
    try:
        subprocess.run(
            ["gsutil", "-m", "-q", "rsync", "-r", str(local_dir), gcs_dir],
            timeout=600, check=True, capture_output=True,
        )
        log.info(f"  GCS rsync: {local_dir} → {gcs_dir}")
    except FileNotFoundError:
        log.debug("  GCS: gsutil not found (local dev)")
    except Exception as e:
        log.warning(f"  GCS rsync failed (non-fatal): {e}")


# =====================================================================
# Simple (non-chunked) MCMC — used for timing tests
# =====================================================================

def run_mcmc(model_fn, data_kwargs, n_warmup, n_samples, n_chains, seed=42,
             init_values=None):
    """Run NUTS MCMC in one shot. Returns the MCMC object."""
    if init_values is not None:
        init_strategy = numpyro.infer.init_to_value(values=init_values)
    else:
        init_strategy = numpyro.infer.init_to_median()
    kernel = NUTS(
        model_fn,
        target_accept_prob=0.8,
        max_tree_depth=10,
        init_strategy=init_strategy,
    )
    mcmc = MCMC(
        kernel,
        num_warmup=n_warmup,
        num_samples=n_samples,
        num_chains=n_chains,
        chain_method="parallel",
        progress_bar=True,
    )
    mcmc.run(jax.random.PRNGKey(seed), **data_kwargs)
    return mcmc


# =====================================================================
# Chunked MCMC with checkpoint/resume
# =====================================================================

def run_mcmc_chunked(model_fn, data_kwargs, n_warmup, n_samples, n_chains,
                     seed=42, chunk_size=DEFAULT_CHUNK,
                     out_dir=None, gcs_dir=None):
    """Run NUTS MCMC with periodic checkpoints for preemption resilience.

    Strategy:
      1. Warmup runs in one shot (not chunked — adaptation needs continuity).
      2. Sampling is split into chunks of `chunk_size`.
      3. After each chunk, posterior samples + MCMC state are saved.
      4. On resume (state file exists), warmup is skipped and sampling
         continues from the last saved state.

    Note: Multi-chain uses init_to_median (not SVI init) because
    init_to_value + chain_method='parallel' has shape incompatibilities.
    SVI init is used for timing test (1 chain) only.

    Returns (np_samples_dict, diverging_array).
    """
    import time

    state_path = out_dir / "mcmc_state.pkl" if out_dir else None

    kernel = NUTS(
        model_fn,
        target_accept_prob=0.8,
        max_tree_depth=10,
        init_strategy=numpyro.infer.init_to_median(),
    )

    # ── Check for existing state (resume after preemption) ──
    resumed = False
    samples_so_far = 0
    all_samples = {}
    all_diverging = []

    if state_path and state_path.exists():
        log.info(f"  Resuming from checkpoint: {state_path}")
        ckpt = _load_mcmc_state(state_path)
        samples_so_far = ckpt["samples_collected"]
        all_samples = ckpt["samples"]
        all_diverging = ckpt["diverging"]
        post_warmup_state = ckpt["post_warmup_state"]
        resumed = True
        log.info(f"  Resumed: {samples_so_far}/{n_samples} samples already collected")

    # ── Warmup (only if not resumed) ──
    if not resumed:
        log.info(f"  Warmup: {n_warmup} steps, {n_chains} chains...")
        t0 = time.time()
        mcmc_warmup = MCMC(
            kernel,
            num_warmup=n_warmup,
            num_samples=1,  # need >=1 for parallel chains; sample is discarded
            num_chains=n_chains,
            chain_method="parallel",
            progress_bar=True,
        )
        mcmc_warmup.run(jax.random.PRNGKey(seed), **data_kwargs)
        post_warmup_state = mcmc_warmup.last_state
        log.info(f"  Warmup done: {time.time()-t0:.1f}s")
        del mcmc_warmup

    # ── Sampling in chunks ──
    remaining = n_samples - samples_so_far
    chunk_idx = samples_so_far // chunk_size

    while remaining > 0:
        this_chunk = min(chunk_size, remaining)
        chunk_idx += 1
        log.info(f"  Sampling chunk {chunk_idx}: {this_chunk} samples "
                 f"({samples_so_far}/{n_samples} done)...")
        t1 = time.time()

        mcmc_chunk = MCMC(
            kernel,
            num_warmup=0,
            num_samples=this_chunk,
            num_chains=n_chains,
            chain_method="parallel",
            progress_bar=True,
        )
        mcmc_chunk.post_warmup_state = post_warmup_state
        mcmc_chunk.run(
            jax.random.PRNGKey(seed + chunk_idx),
            **data_kwargs,
        )

        # Collect samples from this chunk
        chunk_samples = {k: np.array(v) for k, v in mcmc_chunk.get_samples().items()}
        try:
            chunk_div = np.array(mcmc_chunk.get_extra_fields()["diverging"])
        except Exception:
            chunk_div = np.zeros(this_chunk * n_chains, dtype=bool)

        # Accumulate
        if not all_samples:
            all_samples = chunk_samples
        else:
            all_samples = {k: np.concatenate([all_samples[k], chunk_samples[k]], axis=0)
                           for k in all_samples}
        all_diverging.append(chunk_div)

        # Update state for next chunk
        post_warmup_state = mcmc_chunk.last_state
        samples_so_far += this_chunk
        remaining -= this_chunk

        elapsed_chunk = time.time() - t1
        div_rate = float(chunk_div.sum() / max(chunk_div.size, 1))
        log.info(f"  Chunk {chunk_idx} done: {elapsed_chunk:.1f}s, "
                 f"div={chunk_div.sum()}/{chunk_div.size} ({div_rate:.0%}), "
                 f"total={samples_so_far}/{n_samples}")

        # Save checkpoint
        if state_path:
            _save_mcmc_state(state_path, {
                "samples": all_samples,
                "diverging": all_diverging,
                "post_warmup_state": post_warmup_state,
                "samples_collected": samples_so_far,
                "n_samples_target": n_samples,
                "n_chains": n_chains,
                "chunk_size": chunk_size,
            })
            if gcs_dir:
                sync_to_gcs(state_path, gcs_dir)
            log.info(f"  Checkpoint saved: {samples_so_far}/{n_samples}")

        del mcmc_chunk

    # ── Merge divergences ──
    all_div = np.concatenate(all_diverging) if all_diverging else np.array([], dtype=bool)

    log.info(f"  Sampling complete: {samples_so_far} samples, "
             f"div={all_div.sum()}/{all_div.size} "
             f"({all_div.sum()/max(all_div.size,1):.1%})")

    return all_samples, all_div


def _save_mcmc_state(path, state_dict):
    """Atomic save of MCMC state to pickle."""
    tmp = str(path) + ".tmp"
    # Convert JAX arrays in post_warmup_state to numpy for pickling
    safe_state = _jax_state_to_numpy(state_dict)
    with open(tmp, "wb") as f:
        pickle.dump(safe_state, f)
    os.rename(tmp, str(path))


def _load_mcmc_state(path):
    """Load MCMC state from pickle, converting numpy back to JAX."""
    with open(path, "rb") as f:
        state = pickle.load(f)
    state["post_warmup_state"] = _numpy_state_to_jax(state["post_warmup_state"])
    return state


def _jax_state_to_numpy(d):
    """Recursively convert JAX arrays to numpy for pickling."""
    if isinstance(d, dict):
        return {k: _jax_state_to_numpy(v) for k, v in d.items()}
    elif hasattr(d, '_fields'):  # namedtuple — must check BEFORE tuple
        return type(d)(*[_jax_state_to_numpy(getattr(d, f)) for f in d._fields])
    elif isinstance(d, (list, tuple)):
        return type(d)([_jax_state_to_numpy(v) for v in d])
    elif isinstance(d, jnp.ndarray):
        return np.array(d)
    return d


def _numpy_state_to_jax(d):
    """Recursively convert numpy arrays back to JAX for MCMC resume."""
    if isinstance(d, dict):
        return {k: _numpy_state_to_jax(v) for k, v in d.items()}
    elif hasattr(d, '_fields'):  # namedtuple — must check BEFORE tuple
        return type(d)(*[_numpy_state_to_jax(getattr(d, f)) for f in d._fields])
    elif isinstance(d, (list, tuple)):
        return type(d)([_numpy_state_to_jax(v) for v in d])
    elif isinstance(d, np.ndarray):
        return jnp.array(d)
    return d


# =====================================================================
# Checkpoint I/O (final posterior — unchanged from original)
# =====================================================================

def save_posterior(np_samples, diverging, name, out_dir, gcs_dir=None):
    """Save final posterior + divergences as pickle with atomic write + GCS sync."""
    final_path = out_dir / f"{name}.pkl"
    tmp_path = out_dir / f"{name}.pkl.tmp"

    extra = {}
    if diverging is not None and len(diverging) > 0:
        extra["diverging"] = np.array(diverging)

    with open(tmp_path, "wb") as f:
        pickle.dump({"posterior": np_samples, "extra": extra}, f)
    os.rename(str(tmp_path), str(final_path))

    size_mb = final_path.stat().st_size / 1e6
    log.info(f"  Saved: {final_path} ({size_mb:.1f} MB)")
    sync_to_gcs(final_path, gcs_dir)
    return np_samples


def save_checkpoint(mcmc, name, out_dir, gcs_dir=None):
    """Legacy: save posterior from MCMC object (used by timing test)."""
    samples = mcmc.get_samples()
    np_samples = {k: np.array(v) for k, v in samples.items()}

    extra = {}
    try:
        ef = mcmc.get_extra_fields()
        if "diverging" in ef:
            extra["diverging"] = np.array(ef["diverging"])
    except Exception:
        pass

    final_path = out_dir / f"{name}.pkl"
    tmp_path = out_dir / f"{name}.pkl.tmp"

    with open(tmp_path, "wb") as f:
        pickle.dump({"posterior": np_samples, "extra": extra}, f)
    os.rename(str(tmp_path), str(final_path))

    size_mb = final_path.stat().st_size / 1e6
    log.info(f"  Saved: {final_path} ({size_mb:.1f} MB)")
    sync_to_gcs(final_path, gcs_dir)
    return np_samples


def load_checkpoint(path):
    """Load posterior samples from a checkpoint pickle."""
    with open(path, "rb") as f:
        return pickle.load(f)["posterior"]


def load_checkpoint_full(path):
    """Load posterior samples AND divergence array from checkpoint pickle."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    np_samples = data["posterior"]
    diverging = data.get("extra", {}).get("diverging", None)
    return np_samples, diverging


def get_divergence_info(mcmc):
    """Extract divergence count and rate from MCMC object."""
    try:
        div = np.array(mcmc.get_extra_fields()["diverging"])
        return {"div_count": int(div.sum()), "div_total": int(div.size),
                "div_rate": float(div.sum() / max(div.size, 1))}
    except Exception:
        return {"div_count": -1, "div_total": -1, "div_rate": -1.0}
