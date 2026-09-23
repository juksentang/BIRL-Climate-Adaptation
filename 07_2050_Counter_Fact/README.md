# Step 07 — 2050 climate counterfactual and policy scenarios

Stages (run with `scripts/run_pipeline.py --only N`, or the equivalent `python -m cropchoice` command):

| Stage | Script | Input | Output |
|---|---|---|---|
| 0 | `00_export_gps_points.py`, `00_extract_cmip6_*` | LSMS GPS points | GEE exports in `data/cmip6_raw/` |
| 1 | `01_process_climate.py` | CMIP6 exports | ensemble deltas at each observation |
| 2 | `02_generate_cf_matrices.py` | Step 04 models + deltas | `data/ssp245_cf.npz`, `data/ssp585_cf.npz` (q10/q50/q90 per observation × 27 actions) |
| 3 | `03_choice_counterfactual.py` | CF matrices + `08_BIRL_v2/outputs/semipar/posterior.npz` | `results/choice_cf/{metrics,crop_shares}.parquet` |
| 4 | `04_report_choice_cf.py` | stage 3 | `results/choice_cf/tables/*.csv`, `figures/*.pdf` |
| 5 | `05_sobol_policy_sweep.py` | stage 2 + posterior | `results/choice_cf_sobol/` (Saltelli sweep, optional) |

**Stage 3 was rewritten on 2026-09-23** against the Step 08 semi-parametric choice model. The design is in `CHOICE_CF_SPEC.md` (policies as monotone income transforms on the quantiles, exact downside metrics, logsum compensating variation, cost accounting). `results/choice_cf_sens/<setting>/` holds ten policy-intensity sensitivity runs.

The original Stone-Geary certainty-equivalent stage (`scripts/03_compute_welfare_v1.py`, `src/welfare_engine.py`, `src/reporting.py`) is kept only to reproduce paper v1.1; it needs the Step 06 posterior. Its products are in `results/v1_stone_geary/`. Do not use them for new work.

Superseded documents in `docs/07_counterfactual/` describe the v1 stage and carry a banner saying so.
