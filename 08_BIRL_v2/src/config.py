"""
Global configuration for BIRL v2: device setup, paths, model constants, logging.

MUST be imported before any other src/ module (and before jax anywhere in the
process): it reads BIRL_HOST_DEVICES and calls numpyro.set_host_device_count()
BEFORE JAX is initialised, so that chain_method='parallel' can use several CPU
"devices" when no accelerator is present.  On a GPU node set
BIRL_HOST_DEVICES=1 (the sbatch scripts do); the GPU count is unaffected.

Environment overrides (all optional):
  BIRL_HOST_DEVICES   host (CPU) device count for numpyro (default 4)
  BIRL_DATA_DIR       directory holding 06's birl_sample.parquet,
                      env_model_output.npz, action_space_config.json
                      (default ../06_BIRL_MCMC/data relative to this package)
  BIRL_OUT_DIR        output root (default <package>/outputs)
  BIRL_JAX_CACHE      JAX persistent compilation cache directory (default
                      <BIRL_OUT_DIR>/.jax_cache; set to "0" to disable).  Every
                      chunk / --resume / variant re-jits the 222K-row NUTS
                      kernel (numpyro cannot cache it: the nested data dict is
                      unhashable), so the compiled kernel is reused across
                      processes from this cache instead.
"""

import os
import logging
from pathlib import Path

# ── Device setup (before JAX init) ──
BIRL_HOST_DEVICES = int(os.environ.get("BIRL_HOST_DEVICES", "4"))

import numpyro  # noqa: E402
numpyro.set_host_device_count(BIRL_HOST_DEVICES)

import jax  # noqa: E402  — must come after set_host_device_count

# ── Paths (relative to the package so the same code runs on the cluster) ──
BASE_DIR = Path(__file__).resolve().parent.parent            # 08_BIRL_v2/
DATA_DIR = Path(os.environ.get(
    "BIRL_DATA_DIR", str((BASE_DIR.parent / "06_BIRL_MCMC" / "data")))).resolve()
OUT_DIR = Path(os.environ.get("BIRL_OUT_DIR", str(BASE_DIR / "outputs"))).resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Persistent compilation cache (before the first compile) ──
JAX_CACHE_DIR = os.environ.get("BIRL_JAX_CACHE", str(OUT_DIR / ".jax_cache"))
if JAX_CACHE_DIR not in ("", "0", "none", "off"):
    try:
        Path(JAX_CACHE_DIR).mkdir(parents=True, exist_ok=True)
        jax.config.update("jax_compilation_cache_dir", JAX_CACHE_DIR)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.5)
    except Exception as _e:  # never fatal: the cache is an optimisation only
        JAX_CACHE_DIR = f"disabled ({_e})"
else:
    JAX_CACHE_DIR = "disabled"

# ── Model constants (data-scale, not parameters) ──
S_MAX = 0.6                 # default upper bound of s_c (gamma_c = s_c * m_c); --s-max overrides
EPS_FRAC = 0.02             # eps_c = EPS_FRAC * m_c: utility floor on surplus and smoothing scale k
RHO_LO, RHO_HI = 0.1, 5.0   # rho_c = RHO_LO + (RHO_HI - RHO_LO) * sigmoid(rho_lat_c)
LOG_BETA_LO, LOG_BETA_HI = -4.0, 6.0   # beta_c = exp(clip(lb_c, LO, HI))
P_TAYLOR = 1e-3             # |1 - rho| below this -> Taylor branch of the power mean
                            # (the centred direct branch is float32-accurate down to here;
                            #  the cumulant expansion is ~1e-8 rel at |p| = 1e-3)
QUAD_W = (0.10, 0.20, 0.40, 0.20, 0.10)   # weights on (q10, (q10+q50)/2, q50, (q50+q90)/2, q90)
LOG_CLIP = 20.0             # clip on log q before exp (as in 06)
HPDI_PROB = 0.89

# ── Logging ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger("birl_v2")


def add_file_log(path):
    """Attach a file handler (e.g. outputs/<variant>/run.log) to the root logger."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(path)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logging.getLogger().addHandler(fh)
    return fh


# ── Device detection ──
def device_info():
    """Platform / device description written to run.log, timing.json and model_diagnostics.json."""
    devs = jax.devices()
    return {
        "jax_version": jax.__version__,
        "numpyro_version": numpyro.__version__,
        "platform": jax.default_backend(),
        "n_devices": jax.device_count(),
        "n_local_devices": jax.local_device_count(),
        "device_kinds": sorted({getattr(d, "device_kind", d.platform) for d in devs}),
        "host_devices_requested": BIRL_HOST_DEVICES,
    }


DEVICE_INFO = device_info()
N_DEVICES = DEVICE_INFO["n_devices"]
PLATFORM = DEVICE_INFO["platform"]
log.info(f"JAX {DEVICE_INFO['jax_version']} / numpyro {DEVICE_INFO['numpyro_version']}: "
         f"{N_DEVICES} x {PLATFORM} device(s) {DEVICE_INFO['device_kinds']} "
         f"(BIRL_HOST_DEVICES={BIRL_HOST_DEVICES}); compilation cache: {JAX_CACHE_DIR}")


def host_ram_gb():
    """Physical host RAM in GB (None if it cannot be read)."""
    try:
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1e9
    except (ValueError, OSError, AttributeError):
        return None


def device_memory_gb(device=None):
    """Accelerator memory stats of one device in GB: {bytes_in_use, peak_bytes_in_use,
    bytes_limit} -> {'in_use_gb', 'peak_gb', 'limit_gb'}; None on CPU or when the
    backend does not report them.  XLA preallocates most of a GPU, so this (not
    nvidia-smi / RSS) is the model's real device footprint."""
    if PLATFORM == "cpu":
        return None
    try:
        dev = device if device is not None else jax.local_devices()[0]
        st = dev.memory_stats()
        if not st:
            return None
        out = {"device": str(dev)}
        for src, dst in (("bytes_in_use", "in_use_gb"), ("peak_bytes_in_use", "peak_gb"),
                         ("bytes_limit", "limit_gb")):
            if src in st:
                out[dst] = st[src] / 1e9
        return out
    except Exception:
        return None


def peak_rss_gb():
    """Peak resident set size of this process in GB (Linux ru_maxrss is KB)."""
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
