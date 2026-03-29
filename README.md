## Acknowledgments

Cloud computing resources were provided by the Google Cloud TPU Research Cloud (TRC) program. Geospatial data extraction was supported by the Google Earth Engine (GEE) academic research quota.

# BIRL Formal Analysis Pipeline

Bayesian Inverse Reinforcement Learning for Smallholder Agricultural Decision-Making.
6 Sub-Saharan African countries (Ethiopia, Malawi, Mali, Nigeria, Tanzania, Uganda), 2008–2023.

## Directory Structure

```

Formal Analysis/
├── README.md
│
├── data/                              ← Shared data
│   ├── all_countries_panel_birl.parquet   ← Full panel (514K × 211)
│   └── birl_sample.parquet                ← Analysis sample (222K × 244)
│
├── 01_Data_Screening/                 ← Sample selection & cleaning
├── 02_Action_Space/                   ← 27 actions (9 crops × 3 intensity)
├── 03_FDH/                            ← Order-m FDH frontier estimation
├── 04_Env_Model/                      ← LightGBM environment model (Colab)
├── 05_BIRL_SVI/                       ← Variational Inference prototype (Colab)
├── 06_BIRL_MCMC/                      ← MCMC posterior inference (GCP)
├── 07_2050_Counter_Fact/              ← 2050 climate counterfactual & policy welfare
│
└── docs/                              ← Build guides, reports, and analysis notes
```

## Pipeline Overview

```
all_countries_panel_birl.parquet (514K × 211)
    │
    ▼  Steps 01-03 (local, ~30s)
birl_sample.parquet (222K × 244, with actions + FDH efficiency)
    │
    ▼  Step 04 (Colab, ~2.5h)
env_model_output.npz + model_mu.txt + model_sigma.txt
    │
    ▼  Step 05→06 (Colab/GCP, hours, Complete analysis requires 8Chips TPU V4)
posterior.pkl (MCMC: ρ_c, γ_c per country, 12K samples)
    │
    ▼  Step 07 (local, ~30min)
CE tables, climate loss, policy value, synergy
```

## Pipeline Steps

| Step | Directory | Runner | Environment | Time |
|------|-----------|--------|-------------|-----:|
| 01 | `01_Data_Screening/` | `screen_and_clean.py` | Local | ~6s |
| 02 | `02_Action_Space/` | `build_action_space.py` | Local | ~5s |
| 03 | `03_FDH/` | `run_fdh.py` | Local | ~19s |
| 04 | `04_Env_Model/` | `04_env_model.ipynb` | Colab | ~2.5h |
| 05 | `05_BIRL_SVI/` | `05_BIRL_SVI_Colab.ipynb` | Colab | ~hours |
| 06 | `06_BIRL_MCMC/` | `run_birl.py` | GCP VM | ~hours |
| 07 | `07_2050_Counter_Fact/` | `scripts/run_pipeline.py` | Local | ~30min |

### Step Descriptions

- **01 Data Screening**: Filter panel to 222K obs / 15.6K HH (≥2 waves); winsorize; handle missing values; derive variables
- **02 Action Space**: Construct 9 crops × 3 input intensity = 27 discrete actions; build AEZ feasibility masks
- **03 FDH**: Order-m Free Disposal Hull frontier (m=25, B=200); efficiency scores; survival thresholds
- **04 Env Model**: LightGBM μ+σ models (Optuna tuned); predict income distribution (q10/q50/q90) for 222K×27; OOS R²=0.596
- **05 BIRL SVI**: Variational inference prototype — 7 model variants explored; used for initialization of Step 06
- **06 BIRL MCMC**: Hierarchical Bayesian IRL via NumPyro NUTS; 12K posterior samples, 0% divergence; final model: `hier_noalpha`
- **07 2050 Counterfactual**: CMIP6 5-GCM ensemble → delta method → CF income matrices → Stone-Geary CE → policy welfare (6 scenarios × 6 countries × 3K posterior samples)

## Execution

### Steps 01-03 (local, ~10mins)

```bash
cd "Formal Analysis"
python3 01_Data_Screening/screen_and_clean.py
python3 02_Action_Space/build_action_space.py
python3 03_FDH/run_fdh.py
```

### Step 04 (Google Colab)

Upload `04_Env_Model/04_env_model.ipynb` to Colab with GPU runtime.

### Steps 05-06 (remote)

- **05**: Upload `05_BIRL_SVI/05_BIRL_SVI_Colab.ipynb` to Colab (CPU sufficient)
- **06**: Deploy `06_BIRL_MCMC/` to GCP VM. See `06_BIRL_MCMC/Analysis.md` for details.

### Step 07 (local, ~1hour)

```bash
cd "07_2050_Counter_Fact"
python3 scripts/run_pipeline.py          # Stages 1-3
python3 scripts/run_pipeline.py --from 2 # Resume from stage 2
python3 scripts/run_pipeline.py --only 3 # Welfare computation only
```

Prerequisites: Step 04 models + Step 06 posterior + CMIP6 data (via GEE).

## Key Numbers

| Metric | Value |
|--------|-------|
| Countries | 6 (Ethiopia, Malawi, Mali, Nigeria, Tanzania, Uganda) |
| Observations (screened) | 222,023 |
| Households | 15,644 |
| Actions | 27 (9 crops × 3 intensity) |
| AEZ zones | 6 |
| FDH efficiency (mean) | 0.289 |
| Variables (final sample) | 244 |
| Env Model R² (OOS) | 0.596 |
| MCMC samples | 12,000 (4 chains × 3,000) |
| MCMC divergence | 0% |
| GCMs (2050 ensemble) | 5 (ACCESS-CM2, MIROC6, MRI-ESM2-0, INM-CM5-0, IPSL-CM6A-LR) |
| SSP scenarios | 2 (SSP2-4.5, SSP5-8.5) |
| Policy scenarios | 6 (baseline, 2 climate, insurance, safety net, combined) |

## Data Sources

- **LSMS-ISA**: World Bank Living Standards Measurement Study — Integrated Surveys on Agriculture
- **CHIRPS**: Climate Hazards Group InfraRed Precipitation with Station data (1997–2023, via GEE)
- **MODIS**: MOD13A1 NDVI 16-day composite (2007–2023, via GEE)
- **ERA5**: ECMWF ERA5-Land reanalysis temperature (2000–2020)
- **ISRIC**: SoilGrids v2.0 (7 properties × 5 depths, via GEE)
- **NASADEM**: NASA DEM elevation/slope/ruggedness (via GEE)
- **ACLED**: Armed Conflict Location & Event Data (1997–2025)
- **Nelson**: Accessibility to cities travel time (2015, via GEE)
- **CMIP6**: NASA/GDDP-CMIP6 downscaled projections (0.25°, 2040–2060, via GEE)

## Contributors

