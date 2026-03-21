# Data Access & Licensing

This repository contains **code only**. All data files (`.parquet`, `.npz`, `.pkl`, `.csv` with GPS coordinates) are excluded from version control via `.gitignore`.

To reproduce results, you must obtain the following datasets under their respective licenses.

---

## Primary Survey Data (Restricted)

All household-level panel data derive from World Bank LSMS-ISA surveys. These are **free to access** but require registration and agreement to terms of use.

| File in Repo | Source Dataset | Access URL |
|-------------|---------------|------------|
| `data/all_countries_panel_birl.parquet` | LSMS-ISA (6 countries, see below) | See per-country links |
| `data/birl_sample.parquet` | Derived from above (Steps 01-03) | Reproduce from code |
| `04_Env_Model/env_model_output.npz` | Model predictions on above | Reproduce from code |
| `04_Env_Model/*_predictions.parquet` | Model predictions on above | Reproduce from code |
| `07_2050_Counter_Fact/data/ssp*_cf.npz` | Counterfactual predictions on above | Reproduce from code |
| `07_2050_Counter_Fact/data/gps_points_for_gee.csv` | GPS coordinates from LSMS | See below |
| `07_2050_Counter_Fact/data/cmip6_processed/obs_climate_ssp*.parquet` | CMIP6 deltas at LSMS GPS points | Reproduce from code |

### Per-Country LSMS-ISA Access

| Country | Survey | Waves | Access |
|---------|--------|-------|--------|
| Ethiopia | Ethiopia Socioeconomic Survey (ESS) | 1-4 (2011-2019) | https://microdata.worldbank.org/index.php/catalog/lsms/?page=1&country%5B%5D=74 |
| Malawi | Integrated Household Survey (IHS) / IHPS | 3-5 (2010-2019) | https://microdata.worldbank.org/index.php/catalog/lsms/?page=1&country%5B%5D=151 |
| Mali | Enquete Agricole de Conjoncture Integree (EACI) | 1-2 (2014-2017) | https://microdata.worldbank.org/index.php/catalog/lsms/?page=1&country%5B%5D=155 |
| Nigeria | General Household Survey-Panel (GHS-Panel) | 1-4 (2010-2018) | https://microdata.worldbank.org/index.php/catalog/lsms/?page=1&country%5B%5D=178 |
| Tanzania | National Panel Survey (NPS) | 1-5 (2008-2020) | https://microdata.worldbank.org/index.php/catalog/lsms/?page=1&country%5B%5D=216 |
| Uganda | National Panel Survey (UNPS) | 1-8 (2009-2019) | https://microdata.worldbank.org/index.php/catalog/lsms/?page=1&country%5B%5D=227 |

**Registration**: Create a free account at https://microdata.worldbank.org, agree to the Terms of Use for each dataset, then download. GPS data requires the "Geospatial" supplement for each wave.

**Note on GPS coordinates**: LSMS applies random displacement (0-2 km urban, 0-5 km rural) to household GPS locations before public release. Even so, we do not distribute GPS files; they must be obtained directly from the World Bank.

---

## MCMC Posterior & Household-Level Parameters (Derived, Restricted)

These files contain estimated parameters at the household level, linked to LSMS household IDs.

| File | Content | How to Reproduce |
|------|---------|-----------------|
| `06_BIRL_MCMC/outputs/*/posterior.pkl` | Full MCMC posterior (12K samples, includes per-HH params) | Run Step 06 on GCP |
| `06_BIRL_MCMC/outputs/*/mcmc_state.pkl` | MCMC sampler state | Run Step 06 |
| `06_BIRL_MCMC/outputs/*/main_hh_params.parquet` | Per-HH parameter summaries with `hh_id_merge` | Run Step 06 |
| `05_BIRL_SVI/results/*_hh_params.parquet` | SVI per-HH parameter summaries | Run Step 05 |

---

## Geospatial Data (Open Access)

These datasets are publicly available and were extracted via Google Earth Engine (GEE).

| Source | Variables | Resolution | License | Access |
|--------|-----------|-----------|---------|--------|
| CHIRPS v2.0 | Precipitation | 0.05deg | Open | https://www.chc.ucsb.edu/data/chirps |
| MODIS MOD13A1 | NDVI | 500m | Open | https://lpdaac.usgs.gov/products/mod13a1v061/ |
| ERA5-Land | Temperature | 0.1deg | Copernicus | https://cds.climate.copernicus.eu |
| ISRIC SoilGrids v2.0 | 7 soil properties | 250m | CC-BY 4.0 | https://soilgrids.org |
| NASADEM | Elevation/Slope/TRI | 30m | Open | https://lpdaac.usgs.gov/products/nasadem_hgtv001/ |
| ACLED | Conflict events | Point | Terms apply | https://acleddata.com (free for research) |
| Nelson (2019) | Travel time to cities | 1km | CC-BY 4.0 | https://malariaatlas.org/research-project/accessibility-to-cities/ |
| NASA/GDDP-CMIP6 | Climate projections | 0.25deg | Open | `ee.ImageCollection("NASA/GDDP-CMIP6")` in GEE |

The `07_2050_Counter_Fact/data/cmip6_raw/` CSV files are direct GEE exports from NASA/GDDP-CMIP6 and can be freely redistributed.

---

## Files Safe to Distribute

The following files contain no individual-level data and can be shared openly:

- All `.py`, `.js` source code
- All `.md` documentation
- `02_Action_Space/action_space_config.json`
- `04_Env_Model/model_mu.txt`, `model_sigma.txt` (LightGBM tree structures)
- `04_Env_Model/env_model_metrics.json`, `importance_*.csv`
- `04_Env_Model/study_*.pkl` (Optuna hyperparameter search)
- `07_2050_Counter_Fact/data/posterior_country_params.npz` (6 country-level params only)
- `07_2050_Counter_Fact/data/ndvi_model_*.joblib` (regression weights)
- `07_2050_Counter_Fact/data/cmip6_raw/*.csv` (public CMIP6 data)
- `07_2050_Counter_Fact/data/cmip6_processed/ensemble_deltas.parquet` (climate deltas, no HH data)
- `07_2050_Counter_Fact/results/*.csv` and `ce_posterior_samples.npz` (country-level aggregates)
- `05_BIRL_SVI/results/*_guide.npz`, `*_elbo.csv`, `*_ppc.csv`, `*_svi_checkpoint.pkl` (population-level)
- `06_BIRL_MCMC/outputs/*/main_country_params.csv`, `main_global_params.csv`, `ppc_action_freq.csv`, `correlation_matrix.csv`
- `.ipynb` notebooks (after clearing outputs with `jupyter nbconvert --clear-output`)

---

## Reproduction Steps

1. Register at https://microdata.worldbank.org and download all 6 LSMS-ISA country datasets
2. Run the upstream data pipeline to produce `all_countries_panel_birl.parquet` (see Nigeria/ directory)
3. Run Formal Analysis Steps 01-03 locally (~30s)
4. Run Step 04 on Google Colab (~2.5h)
5. Run Steps 05-06 on GCP with TPU v4 (see `docs/06_birl_mcmc/`)
6. Run Step 07 locally (~30min)
