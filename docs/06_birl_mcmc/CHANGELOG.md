# CHANGELOG - 05_BIRL_MCMC

## Design (2026-03-19)

Clean rewrite based on SVI experiments (05_BIRL_SVI, 8 variants, ~25 min).

### SVI Findings That Shaped This Pipeline

1. **α not identifiable**: Individual α posteriors uniform [-1.5, +1.5]; country α 5/6 hit bounds; α-γ compensate (Nigeria α flips sign when γ changes). **Dropped α entirely.**

2. **β must be learnable**: SVI found β≈0.14. Fixed β=5 was 36x too high, collapsing MCMC step sizes to 1e-9. β gets LogNormal prior centered at ~e^1≈2.7.

3. **Reward centering must be mean-only**: Full z-score (dividing by std) erases the natural utility scale needed for ρ/γ identification. Center-only (mean-subtract) preserves obs-level variation.

4. **Parameter bounds required**: ρ∈[0.1, 5.0] and γ∈[0.1, 30.0] via sigmoid. Without bounds, both diverge to infinity. γ upper bound from consumption median $33.8.

5. **hier_noalpha is the best model**: ELBO -677,586, PPC 0.071. CE ranking Spearman=0.886 (p=0.019) vs R3.

### Model Specifications

**hier_noalpha** (Main):
- β: learnable, LogNormal prior
- ρ: Global → Country → HH, sigmoid bounded [0.1, 5.0]
- γ: Global → Country → HH, sigmoid bounded [0.1, 30.0]
- α: none
- Reward: center-only (mean-subtract) before softmax

**R3** (Robustness):
- Same as hier_noalpha but ρ fixed at 1.5 for all HH

### Files

| File | Status |
|------|--------|
| `src/models.py` | New: `_center_reward`, `birl_hier_noalpha`, `birl_r3` |
| `src/posterior.py` | Updated: country unbounded→bounded transform, beta in globals |
| `run_birl.py` | New: two variants only |
| `src/config.py` | Copied from 05_Archived_2, unchanged |
| `src/data_loader.py` | Copied from 05_Archived_2, unchanged |
| `src/mcmc_runner.py` | Copied from 05_Archived_2, unchanged |
| `src/diagnostics.py` | Copied from 05_Archived_2, unchanged |
| `src/robustness.py` | Copied from 05_Archived_2, unchanged |

### Lineage

- `05_Archived/` — Original pre-Round-1 code
- `05_Archived_2/` — Round 1 fixes (P0-P9, chunked MCMC, SVI runner)
- `05_BIRL_SVI/` — SVI experiments and analysis
- **`05_BIRL_MCMC/`** — This pipeline (clean rewrite for MCMC)
