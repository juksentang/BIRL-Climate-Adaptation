# Notebooks — two ways to reproduce the results

| notebook | needs | runtime | what you get |
|---|---|---|---|
| `00_paper_figures.ipynb` | Python with `pandas`, `numpy`, `matplotlib`; **no data, no `cropchoice` install** | about 1 minute on a laptop | the three main figures and the main SI tables, rebuilt from the CSV/JSON summaries tracked in this repository (`08_BIRL_v2/outputs/`, `07_2050_Counter_Fact/results/`); PDFs land in `notebooks/figures_tier0/` (ignored by git) |
| `01_reproduce_colab.ipynb` (script form: `01_reproduce.py`) | `pip install -e .` in the repository root; the two derived data files plus `action_space_config.json` (restricted LSMS-ISA derivatives, see `DATA_ACCESS.md`; set `BIRL_DATA_DIR` if they are not in `06_BIRL_MCMC/data/`) and `07_2050_Counter_Fact/data/ssp585_cf.npz`; a GPU is convenient, not required | see below | the semi-parametric choice model refitted by SVI (optional NUTS), a table against the tracked NUTS posterior, the climate × policy counterfactual with your own posterior draws, and the headline figure against the tracked result |

## Tier 1 runtime (recorded check)

`01_reproduce.py` is the same notebook as a plain script (`# %%` cells) and is what
we run on the cluster to check the notebook end to end. Recorded on one
NVIDIA H100 MIG slice (1g.10gb, 1/7 of a GPU), job of 2026-09-23:

| step | time |
|---|---|
| data load (222,023 × 27) | TIER1_LOAD |
| SVI, 6000 steps | TIER1_SVI |
| counterfactual, 2 climates × 6 policies × 50 draws | TIER1_CF |
| whole script | TIER1_WALL |

A Colab T4 is comparable to the slice; on 4 CPU cores expect roughly 15–30 min for the SVI and skip NUTS.

## What to compare

* Section 2 of `01`: SVI point estimates of `a, b, c` next to the medians of the
  tracked NUTS run (`08_BIRL_v2/outputs/semipar/summary.csv`). They agree to about
  two decimals (recorded values below).
* Section 4: the switching ratio "variance cut / income transfer" per country next
  to `07_2050_Counter_Fact/results/choice_cf/tables/headline_ratio.csv`.

TIER1_TABLE

## Colab

Uncomment the first two lines of the install cell (`git clone … && pip install -e .`,
`pip install "jax[cuda12]"`), upload or mount the data files and point
`BIRL_DATA_DIR` at them. Everything else is unchanged.
