# Structural crop-choice model for climate adaptation in Sub-Saharan Africa

A Bayesian structural discrete-choice model of smallholder crop and input choices, estimated on 222,023 plot-season observations from 15,644 households in six LSMS-ISA countries (Ethiopia, Malawi, Mali, Nigeria, Tanzania, Uganda; 2008–2023), and used to simulate crop choices, incomes and downside risk under 2050 climate (CMIP6, SSP2-4.5 / SSP5-8.5) and under transfer, floor and index-insurance policies.

一句话：六国农户的作物选择结构模型，加 2050 气候与政策反事实。**三种跑法，按你有多少数据和算力选。**

---

## Three ways to run this

| Tier | What you need | Command | Time | What you get |
|---|---|---|---|---|
| **0 — figures from tracked results** | Python, `pip install -e .`; no data | `jupyter notebook notebooks/00_paper_figures.ipynb` | ~1 min | Every figure and table in the paper, regenerated from the CSV/JSON summaries tracked in this repo |
| **1 — re-estimate the model and the base counterfactual** | Two derived data files (see *Data*), a free Colab GPU (CPU works, slower) | open `notebooks/01_reproduce_colab.ipynb` in Colab, or `python -m cropchoice fit-semipar && python -m cropchoice counterfactual` | ~20 min on a T4/L4, ~1 h on 4 CPU cores | The country-level posterior (SVI, optional NUTS), the ssp585 counterfactual and the headline figure |
| **2 — full pipeline** (optional) | Raw LSMS-ISA access, Google Earth Engine, a GPU cluster with SLURM | see *Step map* and `08_BIRL_v2/slurm/README.md` | hours to days | Environment model, all MCMC variants, simulation-recovery test, Sobol sweep, robustness experiments |

Tier 0 is what a reader or reviewer needs. Tier 1 is what a co-author needs to check the numbers. Tier 2 is for extending the work.

### Install

```bash
git clone https://github.com/juksentang/BIRL-Climate-Adaptation.git
cd BIRL-Climate-Adaptation
pip install -e .            # installs the `cropchoice` package and pinned dependencies (requirements.txt)
python -m pytest tests -q -m toy   # optional: math checks on synthetic data, no real data needed
```

Tier 1 on a GPU additionally needs a CUDA build of JAX (`pip install "jax[cuda12]"`); the Colab notebook does this for you.

### Data

This repository contains code, tracked result summaries and the country-level posterior (`08_BIRL_v2/outputs/semipar/posterior.npz`, six countries × (a, b, c) + crop fixed effects, no household data). Tier 1 needs two derived files that we cannot redistribute because they are built from restricted LSMS-ISA microdata:

- `06_BIRL_MCMC/data/birl_sample.parquet` (222,023 × 245, the analysis sample)
- `06_BIRL_MCMC/data/env_model_output.npz` (predicted income quantiles q10/q50/q90 for every observation × 27 actions)

`DATA_ACCESS.md` explains how to obtain the LSMS-ISA surveys and rebuild these files (Steps 01–04), and what can be shared with collaborators directly. `python -m cropchoice` reads them from `06_BIRL_MCMC/data/` by default; override with `BIRL_DATA_DIR=/path`.

---

## Step map

```
LSMS-ISA + geospatial panel (514K × 211, built in ../Nigeria/)
    │  Steps 01–03  screening, action space (9 crops × 3 input intensities = 27), FDH frontier      local, ~30 s
    ▼
birl_sample.parquet (222K × 245)
    │  Step 04      LightGBM μ / σ environment model → q10/q50/q90 for 222K × 27                  Colab, ~2.5 h
    ▼
env_model_output.npz
    │  Step 08      semi-parametric choice model  V = a·μ + b·σ + c·σ² + crop FE,  NUTS             one H100, ~6 min
    ▼
08_BIRL_v2/outputs/semipar/posterior.npz  (tracked)
    │  Step 07      CMIP6 2050 deltas → counterfactual quantiles → choice probabilities, incomes,     stage 3: ~1 min GPU
    ▼              downside risk, policy scenarios, Sobol sensitivity
07_2050_Counter_Fact/results/choice_cf*/   (tables, figures, indices tracked)
```

| Step | Directory | Status | Runner | Where |
|---|---|---|---|---|
| 01 | `01_Data_Screening/` | live | `screen_and_clean.py` | local |
| 02 | `02_Action_Space/` | live | `build_action_space.py` | local |
| 03 | `03_FDH/` | live | `run_fdh.py` | local |
| 04 | `04_Env_Model/` | live | `04_env_model.ipynb` | Colab |
| 05 | `05_BIRL_SVI/` | **superseded** (paper v1.1 only) | `05_BIRL_SVI_Colab.ipynb` | Colab |
| 06 | `06_BIRL_MCMC/` | **superseded** (paper v1.1 only) | `run_birl.py` | GCP TPU |
| 07 | `07_2050_Counter_Fact/` | live; stage 3 rewritten 2026-09 | `scripts/run_pipeline.py`, `python -m cropchoice counterfactual` | local / cluster |
| 08 | `08_BIRL_v2/` | live | `python -m cropchoice fit-semipar`, `slurm/*.sbatch` | cluster |

Steps 05 and 06 are kept unchanged so that paper v1.1 can be reproduced; see the `SUPERSEDED.md` in each. Their results were superseded because the CRRA / Stone-Geary parameters they estimate turned out not to be identified from crop choice (`docs/08_birl_v2/STATUS_2026-09-20.md`).

The shared code lives in the `cropchoice` package (data loading, the choice models, NUTS/SVI runners, diagnostics, policy transforms, the counterfactual engine, reporting). The scripts under `07_2050_Counter_Fact/scripts/` and `08_BIRL_v2/slurm/` are thin wrappers around it. The Step 07 design is specified in `07_2050_Counter_Fact/CHOICE_CF_SPEC.md`.

### Step 07 stages

```bash
cd 07_2050_Counter_Fact
python scripts/run_pipeline.py --only 1     # CMIP6 deltas at the LSMS points (needs GEE exports in data/cmip6_raw)
python scripts/run_pipeline.py --only 2     # counterfactual quantile matrices ssp245_cf.npz, ssp585_cf.npz
python scripts/run_pipeline.py --only 3     # choice-model counterfactual  (= python -m cropchoice counterfactual)
python scripts/run_pipeline.py --only 4     # tables and figures           (= python -m cropchoice report)
python scripts/run_pipeline.py --only 5     # Sobol sweep of the policy parameters (optional, GPU)
```

Stages 3–5 only need `posterior.npz` (tracked), the two CF matrices and the two derived data files.

### Step 08 on a cluster (optional)

`08_BIRL_v2/slurm/README.md` documents the Rorqual (Alliance Canada) package: copy `slurm/env.example.sh` to `slurm/env.sh` and fill in your user and allocation; `sync_up.sh`, `setup_venv.sh`, `submit_chain.sh`, `sb.sh` (adds `--account`/`--output` at submission), `sync_down.sh`. Any SLURM cluster with an NVIDIA GPU works with the same scripts after editing `env.sh`.

---

## Main results (September 2026)

Full account with tables: `docs/08_birl_v2/STATUS_2026-09-20.md`. In short:

- Within the CRRA / Stone-Geary family, risk aversion ρ and the subsistence threshold γ are **not identified** from crop choice: ρ sits at its bounds in every country, γ at its bound in three. No classical utility family (CRRA, rank-dependent, expected shortfall, safety-first) fits the choices.
- A semi-parametric choice model V = a·μ + b·σ + c·σ² with crop fixed effects is identified (0 divergences, r̂ ≤ 1.003). Dispersion aversion is present in all six countries; the **level coefficient a falls with poverty** (Nigeria 1.71, Mali 1.34, Uganda 0.91, Ethiopia 0.41, Tanzania ≈ 0, Malawi < 0), and within every country the poorest asset tercile has a significantly lower a.
- Hence in the four low-income countries a variance cut moves crop choice about twice as much as an income transfer, in Mali one third as much (holds on 75% of the policy-parameter box; the level of the ratio depends on the transfer size).
- Floors and index insurance shift choices toward safer, lower-mean crops. Safety net vs insurance per resource dollar has no robust cross-country ranking.
- 2050 (SSP5-8.5): expected income −27% in Mali, −5% in Nigeria, slightly up elsewhere.

Key numbers: 6 countries, 222,023 observations, 15,644 households, 27 actions, environment-model out-of-sample R² = 0.596, 5 CMIP6 GCMs, 2 SSPs, 6 policy scenarios, K = 200 posterior draws through every counterfactual.

---

## A note on names

Directory names and environment variables keep the prefix `BIRL` (Bayesian inverse reinforcement learning), the name under which the project started. Methodologically, single-step maximum-entropy IRL with known dynamics reduces to the structural discrete-choice model estimated here, so the current papers describe the method as a Bayesian structural (revealed-preference) crop-choice model. The prefix is kept only to avoid breaking paths on the cluster and in older documents.

## Data sources

- **LSMS-ISA**: World Bank Living Standards Measurement Study — Integrated Surveys on Agriculture
- **CHIRPS**: Climate Hazards Group InfraRed Precipitation with Station data (1997–2023, via GEE)
- **MODIS**: MOD13A1 NDVI 16-day composite (2007–2023, via GEE)
- **ERA5**: ECMWF ERA5-Land reanalysis temperature (2000–2020)
- **ISRIC**: SoilGrids v2.0 (7 properties × 5 depths, via GEE)
- **NASADEM**: NASA DEM elevation/slope/ruggedness (via GEE)
- **ACLED**: Armed Conflict Location & Event Data (1997–2025)
- **Nelson**: Accessibility to cities travel time (2015, via GEE)
- **CMIP6**: NASA/GDDP-CMIP6 downscaled projections (0.25°, 2040–2060, via GEE)

## Changelog

### 2026-09-23 — Step 07 rewrite, refactor for onboarding

- Step 07's welfare stage now runs on the Step 08 choice model: policies act on the income distribution (transfer, variance cut, floor, index insurance with basis risk and loading), outcomes are crop shares, expected income, exact downside probability and expected shortfall, a logsum compensating variation where the level coefficient is positive, and calibrated-ρ certainty equivalents as sensitivity; a Sobol sweep covers the policy parameters. The Stone-Geary stage is kept as `scripts/03_compute_welfare_v1.py`; its products moved to `results/v1_stone_geary/`.
- Step 08 gained the asset-tercile version of the model (`slurm/run_semipar_assets.py`).
- Shared code consolidated into the `cropchoice` package with `pyproject.toml`, tests moved to `tests/`, two notebooks added (`notebooks/`), `posterior.npz` tracked, superseded documents marked, this README rewritten around the three tiers.

### 2026-09 — Step 08: re-estimation and repositioning

- **Why.** Diagnostics on the paper v1.1 results (Step 06, `hier_noalpha`) showed that the country-level γ estimates sat at an upper bound derived from the pooled per-plot income median, that the un-scaled reward let ρ absorb choice noise (β = 0.14), and that the 31K household-level parameters were not identified. The policy ranking (safety nets vs. insurance) inherited these artefacts.
- **What was done.** New Step 08 (`08_BIRL_v2/`): country-level model, share-parameterised γ, CE-scaled reward, one-hot parameter broadcasting (a gather-transpose scatter had made gradients 300x slower), chunked NUTS with checkpoint/resume, simulation-recovery test, 96 unit tests, and a Rorqual (H100) SLURM package. Runs take minutes instead of hours.
- **What was found.** Within the CRRA / Stone-Geary family, ρ and γ are not identified from crop choice (ρ at its bounds in every country). The environment model shows no leakage on held-out households, and its σ is not a familiarity proxy. Choices load on the predicted 10th percentile 2 to 14x more than on the 90th; no classical utility family fits. A semi-parametric choice model V = a·μ + b·σ + c·σ² with crop fixed effects is identified: dispersion aversion is uniform across countries, level sensitivity rising with income.
- **Consequences.** Step 06 is kept only to reproduce paper v1.1. Step 07's welfare stage was rewritten against the Step 08 model (2026-09-23, above). Full account for collaborators: `docs/08_birl_v2/STATUS_2026-09-20.md`.
- **Repository.** Only small summaries under `08_BIRL_v2/outputs/` are tracked; cluster user and allocation live in an untracked `slurm/env.sh` (template `env.example.sh`), and `slurm/sb.sh` supplies `--account`/`--output` at submission.

## Acknowledgments

Cloud computing resources were provided by the Google Cloud TPU Research Cloud (TRC) program (Steps 05-06). The Step 08 re-estimation was enabled in part by support provided by Calcul Québec (calculquebec.ca) and the Digital Research Alliance of Canada (alliancecan.ca), on the Rorqual cluster. Geospatial data extraction was supported by the Google Earth Engine (GEE) academic research quota.

## Contributors
