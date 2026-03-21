"""
Global configuration: paths, device detection, logging.
MUST be imported before any other src/ module (sets device count before JAX init).
"""

import os
import logging
from pathlib import Path

import numpyro

# Set host device count for CPU multi-chain parallelism BEFORE JAX init.
# On TPU/GPU this flag is harmless (real devices override it).
# Set before JAX init. On CPU this creates virtual devices; on TPU/GPU harmless.
_N_HOST = min(8, os.cpu_count() or 4)
numpyro.set_host_device_count(_N_HOST)

import jax  # noqa: E402  — must come after set_host_device_count

# ── Paths (all relative to 05_BIRL/) ──
BASE_DIR = Path(__file__).resolve().parent.parent  # 05_BIRL/
DATA_DIR = BASE_DIR / "data"

OUT_BASE = BASE_DIR / "outputs"
OUT_BASE.mkdir(parents=True, exist_ok=True)

# ── Logging ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(BASE_DIR / "birl.log"),
    ],
)
log = logging.getLogger("birl")

# ── Device detection ──
devices = jax.devices()
N_LOCAL_DEVICES = jax.local_device_count()
PLATFORM = devices[0].platform  # 'tpu', 'gpu', or 'cpu'
log.info(f"JAX: {N_LOCAL_DEVICES} x {PLATFORM}")

# Available chains = device count. Callers choose min(desired, N_DEVICES).
N_DEVICES = N_LOCAL_DEVICES  # 64 on single-machine TPU, 4 on 4-chip VM, etc.
log.info(f"Available devices: {N_DEVICES}")

# ── GCS ──
GCS_BUCKET = "gs://subsahra/birl"

# ── Optional dependencies ──
try:
    import arviz as az  # noqa: F401
    HAS_ARVIZ = True
except ImportError:
    HAS_ARVIZ = False

try:
    from scipy.stats import spearmanr  # noqa: F401
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
