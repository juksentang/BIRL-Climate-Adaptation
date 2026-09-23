# Refactor plan: make the pipeline runnable by a notebook-level user (2026-09-23)

Goal: three entry tiers. (0) regenerate every paper figure/table from tracked summaries with no data, in one notebook; (1) re-estimate the choice model and the base counterfactual from the two derived data files in one Colab notebook (free GPU, ~20 min; CPU fallback ~1 h); (2) the full pipeline (env model, MCMC variants, Sobol) on a cluster, clearly marked optional.

Branch: `refactor-onboarding`. Numerical results must not change: the tracked tables under `07_2050_Counter_Fact/results/choice_cf/tables/` and `08_BIRL_v2/outputs/semipar/results.json` (log-lik at the posterior median) are the regression targets.

## Package layout (new, top level)

```
cropchoice/                     importable package, `pip install -e .` (pyproject.toml at repo root)
  __init__.py
  config.py        constants (QUAD_W, Z=1.2816, LOG_CLIP, EPS_FRAC, S_MAX, INFEASIBLE_LOGIT), repo-relative default
                   paths, env overrides (BIRL_DATA_DIR, BIRL_OUT_DIR, BIRL_HOST_DEVICES kept for cluster compatibility)
  data.py          load_sample(data_dir) -> dict; model_kwargs(); add_precomputed(); observed_shares(); select_obs()
                   (from 08_BIRL_v2/src/data_loader.py)
  quantiles.py     mu_sigma(q10,q50,q90); five_nodes(); e5(); log-clipped exp (from 08 models + 07 choice_engine;
                   ONE definition)
  broadcast.py     one_hot(ci, C); per_obs(oh, v) with Precision.HIGHEST (one definition)
  models.py        semi-parametric models: country-level (run_semipar), group-level (asset terciles), null;
                   center_reward; a `compute_logits_semipar(a,b,c,asc,data)` used by fitting AND counterfactual
  models_v2.py     the Stone-Geary/CRRA variants moved verbatim from 08_BIRL_v2/src/models.py (kept for the record)
  inference.py     NUTS runner with chunked checkpoint/resume, timing run, SVI helper (from 08 mcmc_runner +
                   the SVI pattern in the exp_* scripts)
  diagnostics.py   rank-normalised r_hat, bulk/tail ESS, hpdi (single implementation, replaces the 4 copies),
                   summary.csv writer, convergence report, PPC tables (from 08 diagnostics)
  policies.py      apply_policy() and cost accounting (from 07 choice_engine), DEFAULT_PARAMS
  counterfactual.py load_posterior_thinned(), scenario_metrics(), run_scenarios() -> long DataFrames
                   (from 07 choice_engine + 03 script body)
  report.py        table/figure helpers shared by 07/scripts/04_report_choice_cf.py and the notebooks
  cli.py           `python -m cropchoice <cmd>`: fit-semipar, fit-semipar-assets, counterfactual, report, sobol,
                   with the same flags the scripts have today
tests/             moved from 08_BIRL_v2/tests + NEW tests for policies (monotone inverse ystar vs apply_policy,
                   insurance basis=0 boundary, cost accounting) and for compute_logits_semipar equality with the
                   old run_semipar definition on a toy
notebooks/
  00_paper_figures.ipynb   no data: reads tracked csv/json, regenerates every figure and table
  01_reproduce_colab.ipynb data: two derived files -> SVI fit (+ optional NUTS) -> base counterfactual -> headline
requirements.txt / pyproject.toml   pinned (jax, numpyro, numpy, pandas, pyarrow, scipy, matplotlib, pytest)
```

Existing step directories stay where they are. Scripts in `07_2050_Counter_Fact/scripts/` and `08_BIRL_v2/slurm/` become thin wrappers that import `cropchoice` (no `sys.path` hacks, no `spec_from_file_location`); the sbatch files keep working (`pip install -e .` in setup_venv.sh). `08_BIRL_v2/src/` is removed once everything imports from `cropchoice` (git history keeps it).

## Known subtlety to document (not to change silently)

`run_semipar.py` centres V over the feasible set before adding ASCs (`center_reward`); the counterfactual does not. Choice probabilities are identical; the logsum level is not. `cropchoice.models.compute_logits_semipar(..., center=True)` keeps the estimation behaviour; the counterfactual calls it with `center=False` and the spec/README states that the logsum-based CV is relative to a per-observation constant fixed by convention.

## Documentation

- Root `README.md` rewritten around the three tiers: what each needs, exact commands, expected runtime, where results land; then the step map; then data access; then the changelog (existing) and acknowledgments.
- `07_2050_Counter_Fact/README.md` (new), `05_BIRL_SVI/SUPERSEDED.md`, `06_BIRL_MCMC/SUPERSEDED.md`.
- Superseded banners at the top of `docs/06_birl_mcmc/Analysis.md`, `docs/06_birl_mcmc/CHANGELOG.md` (fix title), `docs/07_counterfactual/*.md` (fix `plan.md` title).
- v1 Stone-Geary products moved to `07_2050_Counter_Fact/results/v1_stone_geary/`.
- `.gitignore`: whitelist `08_BIRL_v2/outputs/semipar/posterior.npz` (2.5 MB, no personal data; the counterfactual's only posterior input).
- Wording: "BIRL" stays in directory names and env-variable prefixes; human-readable text says "structural crop-choice model (Bayesian discrete choice)"; the methods note that single-step MaxEnt IRL reduces to it.
