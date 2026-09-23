"""
pytest configuration for cropchoice (toy and real-data tests).

Markers:
  toy       pure-math tests on a synthetic <= 50-row x 27-action dataset
            (no 06 data needed):    python3 -m pytest -m toy tests/
  realdata  tests that load the 222K-row 06 data (cluster only):
                                    python3 -m pytest -m realdata tests/
The `toy` fixture builds the synthetic dataset with the six real m_c values
so floor / scale behaviour is representative; `real_data` loads 06's files
once per session (skipped when they are absent).  The real-data tests run
over ALL observations and actions: the cells that stress the Taylor switch
and the float32 path are rare, so a random subset would miss them.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# cropchoice.config must be imported before jax anywhere (sets the host device count)
from cropchoice.config import DATA_DIR  # noqa: E402
import jax.numpy as jnp  # noqa: E402

COUNTRIES = ["Ethiopia", "Malawi", "Mali", "Nigeria", "Tanzania", "Uganda"]
M_C_REF = np.array([19.86, 19.12, 129.96, 102.74, 26.03, 25.46])   # USD, from the spec
INTENSITIES = ("low", "medium", "high")


def pytest_configure(config):
    config.addinivalue_line("markers", "toy: pure-math tests on synthetic data (no 06 data)")
    config.addinivalue_line("markers", "realdata: tests that load the 222K-row 06 data (cluster)")


def make_toy_data(n_obs=40, n_actions=27, n_country=6, n_cz=9, seed=0):
    """Synthetic loader-like dict: q arrays around the real m_c with wide spread
    (q10 often below the floor), random country x zone feasibility mask with
    >= 3 feasible actions per zone, and a feasible chosen action per obs."""
    rng = np.random.default_rng(seed)
    m_c = M_C_REF[:n_country].astype(np.float32)
    ctry = rng.integers(0, n_country, n_obs).astype(np.int32)
    cz = rng.integers(0, n_cz, n_obs).astype(np.int32)
    q50 = (m_c[ctry][:, None] * rng.lognormal(0.0, 1.0, size=(n_obs, n_actions))).astype(np.float32)
    q10 = (q50 * rng.uniform(0.05, 0.8, size=q50.shape)).astype(np.float32)
    q90 = (q50 * rng.uniform(1.2, 5.0, size=q50.shape)).astype(np.float32)
    mask = rng.random((n_cz, n_actions)) < 0.7
    for z in range(n_cz):
        mask[z, rng.choice(n_actions, 3, replace=False)] = True
    obs_action = np.array([rng.choice(np.where(mask[z])[0]) for z in cz], dtype=np.int32)
    labels = {str(a): f"crop{a // 3}_{INTENSITIES[a % 3]}" for a in range(n_actions)}
    return dict(
        obs_action=jnp.array(obs_action), obs_country_idx=jnp.array(ctry), obs_cz_idx=jnp.array(cz),
        feasibility_mask=jnp.array(mask),
        cf_log_q10=jnp.log(jnp.array(q10)), cf_log_q50=jnp.log(jnp.array(q50)),
        cf_log_q90=jnp.log(jnp.array(q90)),
        m_c=jnp.array(m_c), N_country=n_country,
        countries=COUNTRIES[:n_country], m_c_np=m_c.astype(np.float64),
        N_obs=n_obs, N_actions=n_actions, N_cz=n_cz, action_labels=labels,
        mask_np=mask, ctry_np=ctry, cz_np=cz,
    )


@pytest.fixture(scope="session")
def toy():
    return make_toy_data()


@pytest.fixture(scope="session")
def real_data():
    for f in ("birl_sample.parquet", "env_model_output.npz", "action_space_config.json"):
        if not (DATA_DIR / f).exists():
            pytest.skip(f"06 data not available: {DATA_DIR / f}")
    from cropchoice.data import load_data
    return load_data(DATA_DIR)

