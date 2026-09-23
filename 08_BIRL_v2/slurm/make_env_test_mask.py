"""Reproduce the Step 04 environment-model household split as a boolean mask.

Step 04 (04_env_model.ipynb, cell 5) shuffles the unique hh_id_merge values
with numpy's default_rng(42) and takes the first 15% as the test households.
The env-model predictions for the observed cells are in-sample for the other
85%; `exp_leak.py` uses this mask to compare the two groups.  Row order is
that of birl_sample.parquet (identical to the loader's).

Writes slurm/env_test_mask.npy (bool, one entry per observation).
Run from 08_BIRL_v2/:  python3 slurm/make_env_test_mask.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from src.config import DATA_DIR

df = pd.read_parquet(DATA_DIR / "birl_sample.parquet", columns=["hh_id_merge"])
rng = np.random.default_rng(42)
uh = df["hh_id_merge"].unique()
rng.shuffle(uh)
test_hh = set(uh[: int(len(uh) * 0.15)])
mask = df["hh_id_merge"].isin(test_hh).values
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "env_test_mask.npy")
np.save(out, mask)
print(f"{out}: {mask.sum()} test rows of {len(mask)} (expected 32530 of 222023)")
