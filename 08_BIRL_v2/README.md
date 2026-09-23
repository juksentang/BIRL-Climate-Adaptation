# 08_BIRL_v2 — Country-level BIRL with certainty-equivalent reward (Phase 1)

> **Status (2026-09-20).** The Stone-Geary/CRRA variants below (`v2_country`, `v2_country_gfix`) ran cleanly but showed that ρ and γ (or the share s) are not identified from crop choice: ρ sits at its bounds in every country. The model that is identified is the semi-parametric choice model in `slurm/run_semipar.py` (V = a·μ + b·σ + c·σ² + crop fixed effects). Full account, tables and next steps: `../docs/08_birl_v2/STATUS_2026-09-20.md`. The experiment scripts `slurm/exp_*.py` are the evidence chain (reward specs, risk-signal existence, utility families, leakage, familiarity).

Bayesian inverse reinforcement learning of farmers' crop x input-intensity
choices, re-specified to fix three defects of the 06_BIRL_MCMC model
(`birl_hier_noalpha`). Nothing under `06_BIRL_MCMC/`, `07_2050_Counter_Fact/`,
`04_Env_Model/` or `paper/` is modified; 06's data files are read in place
(`../06_BIRL_MCMC/data/{birl_sample.parquet, env_model_output.npz,
action_space_config.json}`, override with `BIRL_DATA_DIR`).

**Execution policy.** Nothing that loads the 222K-row dataset or runs JAX on
real arrays is executed on the laptop. Locally allowed: `python3 -m py_compile`
and importing the modules on tiny synthetic arrays (`pytest -m toy`).
pytest, `--timing`, the recovery test and all MCMC run on Rorqual (Alliance
Canada, H100) through the `slurm/` package (see "Running on Rorqual" below).

## What changed vs 06 and why

| Defect in 06 | v2 fix |
|---|---|
| Subsistence floor gamma bounded in absolute USD [0.1, 30] hit the bound in 4/6 countries (per-country medians of the chosen action's q50: Ethiopia 19.86, Malawi 19.12, Mali 129.96, Nigeria 102.74, Tanzania 26.03, Uganda 25.46 USD). | `gamma_c = s_c * m_c`, `s_c in (0, S_MAX=0.6)` (`--s-max 0.8` for robustness), `m_c` = per-country median of the chosen action's q50, a data constant from the loader. |
| Reward = mean-centred expected utility: the utility scale depends on rho, one global beta cannot serve all rho, rho acts as a per-household temperature confounded with choice noise. | Reward = certainty equivalent CE (USD) / `m_c`, computed directly as a weighted power mean of the surplus nodes in log space (never EU then inverse). CE is on the income scale for every rho, so `beta_c` is a genuine choice-noise parameter. |
| 31K household-level latents, not identified; NUTS 6.9 s/step. | Country-level parameters only: 6 x {rho, s, beta} + 6 hyperparameters, centred parameterisation (country parameters are data-dominated; `--noncentered` available). |
| Hard floor `max(Y - gamma, eps)` has a kink. | Smooth floor `surplus = eps + 0.5 (d + sqrt(d^2 + k^2))`, `d = Y - gamma - eps`, `k = eps`: C1, >= eps, equals `max(Y - gamma, eps)` away from the kink. |

### Model

```
nodes x_k, k=1..5 on (q10, (q10+q50)/2, q50, (q50+q90)/2, q90),  w = (.1,.2,.4,.2,.1)
surplus_k = eps + 0.5 (d_k + sqrt(d_k^2 + eps^2)),   d_k = x_k - gamma - eps
gamma_c = s_c m_c,  eps_c = 0.02 m_c
p = 1 - rho
m = sum_k w_k log surplus_k,  c_k = log surplus_k - m
log CE_surplus = m + log1p( sum_k w_k expm1(p c_k) ) / p                       |p| >= 1e-3
log CE_surplus = m + (p/2) Var_w(log surplus) + (p^2/6) kappa3_w(log surplus)  |p| <  1e-3
CE = gamma + exp(log CE_surplus)
reward  = CE / m_c[country],  mean-centred over feasible actions (country x zone empirical mask)
logits  = beta_c * reward_c,  -1e10 where infeasible;   obs_action ~ Categorical(logits)

mu_rho ~ N(0,1),  sigma_rho ~ HalfN(1),  rho_lat_c ~ N(mu_rho, sigma_rho),  rho_c  = 0.1 + 4.9 sigmoid(rho_lat_c)
mu_s   ~ N(-1,1), sigma_s   ~ HalfN(1),  s_lat_c   ~ N(mu_s, sigma_s),      s_c    = S_MAX sigmoid(s_lat_c)
mu_lb  ~ N(1,1),  sigma_lb  ~ HalfN(1),  lb_c      ~ N(mu_lb, sigma_lb),    beta_c = exp(clip(lb_c, -4, 6))
--flat-priors: rho_lat_c ~ N(0,2), s_lat_c ~ N(-1,2), lb_c ~ N(1,2), no hyperparameters (recovery test)
--noncentered: *_raw ~ N(0,1), lat = mu + sigma * raw (same site names exposed as deterministics)
```

The direct branch is algebraically the spec's `(1/p) logsumexp_k(p log
surplus_k + log w_k)`, written in the centred, cancellation-free form (the sum
inside `log1p` is O(p^2 Var), so dividing by p loses no float32 digits); it is
exact for every p != 0 and accurate to ~3e-7 relative down to |p| = 1e-3, which
lets the switch to the Taylor branch sit at `P_TAYLOR = 1e-3` (config.py)
instead of 0.02.  The Taylor branch is the cumulant expansion around p = 0 and
is only the p -> 0 limit; its third-cumulant term is an addition to the spec's
second-order form (deliberate: it costs nothing and tightens the switch). At
|p| = 1e-3 the two branches differ by ~1e-8 relative, i.e. the summed
log-likelihood is continuous in rho_c to ~1e-3 nat (tests T2), so NUTS
trajectories that cross rho = 1 +- 1e-3 see no potential-energy step.

One function, `compute_logits(rho_c, s_c, beta_c, data)` in `src/models.py`,
is used by the numpyro models, `simulate_actions` and `log_likelihood`.

Variants: `v2_country` (rho, s, beta sampled) and `v2_country_gfix`
(`s_c` fixed, default 0.3 via `--s-fixed`; only rho and beta sampled).

## Layout

```
08_BIRL_v2/
  README.md
  run_v2.py              MCMC driver (timing test / full run, chunked checkpoints, --resume)
  simulate_recover.py    simulation-recovery test (--mode svi smoke | --mode nuts PASS verdict)
  src/config.py          BIRL_HOST_DEVICES -> numpyro.set_host_device_count BEFORE jax import;
                         JAX persistent compilation cache (BIRL_JAX_CACHE, default outputs/.jax_cache);
                         paths (BIRL_DATA_DIR, BIRL_OUT_DIR overrides), constants, device_info(),
                         device_memory_gb(), host_ram_gb()
  src/data_loader.py     load_data(): obs arrays, country x zone feasibility mask, log-q arrays, m_c,
                         precomputed q10/q50/q90 (USD) + mask_obs, max-q90 check; model_kwargs(), select_obs()
  src/models.py          smooth_surplus, log_power_mean, certainty_equivalent, compute_logits,
                         log_likelihood, v2_country, v2_country_gfix, simulate_actions,
                         derive_country_params
  src/mcmc_runner.py     NUTS (target_accept 0.8, max_tree_depth 10, init_to_median),
                         choose_chain_method, chunked sampling with checkpoint/resume, npz I/O
  src/diagnostics.py     summary.csv, convergence.txt, PPC tables, model_diagnostics.json
  tests/conftest.py      `toy` fixture (synthetic 40 x 27), `real_data` (all 222K obs); markers
  tests/test_models.py   T1-T7 (+ guards);  -m toy runs without data, -m realdata on the cluster
  slurm/                 Rorqual package: env.sh, setup_venv.sh, sync_up/down.sh, 00-04 *.sbatch,
                         submit_chain.sh, status.sh, README.md
  outputs/<variant>[_<tag>]/   posterior.npz, summary.csv, convergence.txt, ppc_action_freq.csv,
                         ppc_crop_x_intensity.csv, model_diagnostics.json, run.log, run_info.json,
                         timing.json (timing runs)
  outputs/recovery/<mode>/     report.md, report.json, sim_actions_<set>.npy, nuts_posterior_<set>.npz
```

Data (read-only, from 06): `birl_sample.parquet` (222,023 obs), `env_model_output.npz`
(q10/q50/q90, (222023, 27) float32), `action_space_config.json` (27 actions = crop x intensity).
Country order everywhere = sorted names: Ethiopia, Malawi, Mali, Nigeria, Tanzania, Uganda.
m_c (USD) = 19.86, 19.12, 129.96, 102.74, 26.03, 25.46 (checked by test T6).

## How to run (cluster; from this directory)

```bash
python3 -m pytest -m toy tests/                              # pure-math tests, no data (also OK locally)
python3 -m pytest tests/                                     # all tests incl. real data
python3 run_v2.py --variant v2_country --timing              # 50+20, 1 chain: s/step, peak RSS, platform
python3 simulate_recover.py --mode svi                       # smoke: AutoMVN, Adam 1e-2, 3000 steps
python3 simulate_recover.py --mode nuts                      # PASS verdict: flat priors, 2 x (500+500)
python3 run_v2.py --variant v2_country                       # 4 chains x (1000 + 1000)
python3 run_v2.py --variant v2_country_gfix --s-fixed 0.3
python3 run_v2.py --variant v2_country --s-max 0.8           # -> outputs/v2_country_smax08/
python3 run_v2.py --variant v2_country --resume              # continue after a SLURM time-out
```

`run_v2.py` flags: `--variant {v2_country,v2_country_gfix} --s-fixed --s-max
--flat-priors --noncentered --timing --chains --chain-method
{parallel,vectorized,sequential} --warmup --samples --chunk --resume --seed
--n-ppc --tag`. Output directory tag = `--tag` or auto (`smax08`, `sfix<v>`,
`flat`, `nc`, `timing`).

Devices: `config.py` reads `BIRL_HOST_DEVICES` (default 4; set 1 on GPU nodes)
before importing JAX. `chain_method`: on an accelerator `parallel` when
`jax.device_count() >= chains`, else `vectorized`; on CPU `sequential` (each
chain's gradient evaluation holds ~30 (222K, 27) float32 arrays, ~0.7 GB, and
XLA-CPU fuses little, so pmapping 4 chains over host devices needs 4-8 GB);
`--chain-method parallel` opts into the pmap path. The decision, the
estimated per-chain footprint and host RAM are logged. `--chains` /
`--chain-method` override. Every run writes the platform / device list into
`run.log`, `run_info.json`, `timing.json` and `model_diagnostics.json`; on a
GPU the timing run and the first chunk also log `peak_device_gb` from
`jax.local_devices()[0].memory_stats()` (XLA preallocates the card, so RSS /
nvidia-smi say nothing about the footprint) and the per-chain share.

Compilation cache: `config.py` enables JAX's persistent compilation cache
(`BIRL_JAX_CACHE`, default `outputs/.jax_cache`, `0` disables) so chunks,
`--resume` and the gfix / smax variants reuse the compiled NUTS kernel across
processes (numpyro's own cache cannot: the nested `data` dict is unhashable).

Checkpointing: warmup runs in one shot, sampling in chunks of `--chunk` (250)
draws with `samples_partial.npz` + `mcmc_state.pkl` written after each chunk
(deleted on completion). The pkl (HMC state + draw count + chains / samples /
chain_method / seed / warmup / chunk) is authoritative: on `--resume` the npz
is truncated to its count if a kill landed between the two writes, and a
checkpoint whose settings differ from the command line is discarded with a
warning. Re-running the same command with `--resume` continues
from the checkpoint (or just recomputes diagnostics from an existing
`posterior.npz`); without `--resume` a stale checkpoint is discarded.

`posterior.npz` holds one numpy array per site with shape `(n_chains, n_draws,
...)` plus `__diverging__` and a JSON `__meta__`. Load with
`src.mcmc_runner.load_posterior`.

## Diagnostics (every run)

* `summary.csv`: median, mean, sd, 89% HPDI, rank-normalised split r_hat,
  bulk and tail ESS (Vehtari et al. 2021, computed in numpy on top of
  numpyro.diagnostics) for `rho_c, s_c, s_lat_c, gamma_c, beta_c` (+ `rho_lat_c,
  lb_c`) and the hyperparameters.
* `convergence.txt` / `model_diagnostics.json`: divergences; r_hat / ESS extremes;
  per-country posterior correlations corr(rho_c, s_c), corr(log beta_c, rho_c),
  corr(log beta_c, s_c); P(s_c > 0.55); at the posterior median the share of obs
  whose chosen action has all five nodes inside the floor region (d < 0) and the
  share with p(chosen) < 1e-3; per-country mean Spearman correlation between
  CE(rho=0.3) and CE(rho=4.5) across feasible actions (> 0.95 means rho acts
  mostly as a temperature); PPC summary; platform.
* `ppc_action_freq.csv` (action = crop x intensity, overall + per-country observed
  / predicted shares averaged over 100 thinned draws) and `ppc_crop_x_intensity.csv`.

## Simulation recovery (`simulate_recover.py`)

Truth sets on the REAL q arrays and masks (country order = sorted names):
`A` interior rho=[1.5,3.0,2.0,1.0,2.5,3.5], s=[.20,.40,.30,.15,.50,.35],
beta=[3,8,5,2,6,4]; `B` near-bound rho=[0.5,4.5,1.0,3.0,2.0,1.5],
s=[.55,.10,.58,.05,.30,.45], beta=[10,1.5,4,6,2,8].

* `--mode svi`: AutoMultivariateNormal, Adam 1e-2, 3000 steps, point estimates
  (guide median), log-lik at truth vs estimate. Reports the tolerances as a smoke
  check; NOT the PASS verdict.
* `--mode nuts`: `--flat-priors` model, 2 chains x 500 warmup + 500 samples.
  PASS iff for every country in both sets |d rho| <= 0.25, |d s| <= 0.05,
  |d log beta| <= 0.2 (posterior median vs truth) AND >= 80% of the 36 truths lie
  inside their 89% HPDI; z-scores and log-lik(truth) vs log-lik(median) reported.
  Running a single set gives verdict INCOMPLETE.

Outputs `outputs/recovery/<mode>/report.md` + `report.json`.

## Tests (`tests/test_models.py`)

| | what | marker |
|---|---|---|
| T1 | q10=q50=q90=Y0 => CE == Y0 (1e-3 rel; 2e-3 at Y0=0.5 m_c, s=0.3 where the smooth floor's own bias is 1.1e-3) for rho in {0.3, 0.985, 0.995, 1, 1.005, 1.015, 1.03, 2.5, 4.5}, Y0 in {2, 0.5} m_c, s in {.1,.2,.3}; and CE == gamma + smooth_surplus(Y0) at 2e-4 | toy |
| T2 | trend-corrected jump at the Taylor switch (rho = 1 -+ 1e-3, grid spacing 2e-4) < 1e-4 CE on every cell (worst cell reported); both branch formulas agree at |p| = 1e-3 (2e-5); per-country summed log-lik (beta = 5, s = 0.3, float64 evaluation of the model) jumps < 1e-2 nat across each switch | toy, realdata (all 222K x 27 cells) |
| T3 | float64 numpy power-mean reference vs JAX, rho in {0.3, 0.99, 1, 2.5, 4.5, 1 -+ 1.5e-3}, 1e-4 rel on every cell (worst cell reported) | toy, realdata (all cells) |
| T4 | finite jax.grad of the summed log-lik w.r.t. (mu_rho, mu_s, mu_lb) and w.r.t. (rho_c, s_c, log beta_c) at rho in {0.1, 0.99, 1, 1.01, 4.99}, never all-zero; at lb = 7 (past the clip) finite with non-zero d/d rho, d/d s | toy, realdata |
| T5 | logits finite, infeasible == -1e10, centred, simulate_actions feasible; precomputed q/mask arrays give identical logits | toy, realdata |
| T6 | loader m_c == independent pandas computation == the six spec values | realdata |
| T7 | CE increasing in each node income (JAX non-decreasing + float64 strict) | toy |
| guards | smooth-floor properties, transforms / derive_country_params, model traces (centred, non-centred, flat, s_max, gfix) | toy |

## Phase 1 gates (judged from the cluster outputs)

All `s_c` posterior medians in [0.05, 0.55] with P(s_c > 0.55) small, and the
rho ranking across countries stable across `v2_country`, `v2_country_gfix` and
06; recovery `--mode nuts` verdict PASS; r_hat < 1.01, no divergences.

## Running on Rorqual

All computation runs on Rorqual (Alliance Canada; account `<account>`,
project root `$SCRATCH/birl_v2`) through the `slurm/` package; see
`slurm/README.md` for the details, the expected times, the meaning of every
output and the Phase 1 gates. From this directory, on the laptop:

```bash
bash slurm/sync_up.sh          # code + the three 06 data files -> $SCRATCH/birl_v2
bash slurm/setup_venv.sh       # once, on the login node: modules + venv + pip install --no-index
bash slurm/submit_chain.sh     # 00_smoke + 00b_timing (gates) -> 01_recover_nuts, 02_main, 03_gfix, 04_smax08 (held)
bash slurm/status.sh           # squeue + tails of the logs
bash slurm/sync_down.sh        # outputs/ back to the laptop
```

Stack on the cluster (verified 2026-09-19/20): `StdEnv/2023 python/3.11 cuda/12.9
cudnn/9.13.1.26 arrow/25.0.0` (cuda/12.9 rather than the default 12.6: XLA warns
that ptxas <= 12.6.2 miscompiles clamping edge cases, and the gradient path clips); wheelhouse `jax = jaxlib = jax_cuda12_plugin =
jax_cuda12_pjrt = 0.10.2`, `numpyro 0.21.0`, `numpy 2.4.2`, `pandas 2.3.3`,
`scipy 1.17.1`, `pytest 9.1.1`, pyarrow from the arrow module. GPUs: `00_smoke`
on a MIG `h100_1g.10gb` slice, `01_recover_nuts` on `h100_2g.20gb`, the three
main runs on one full H100 (`--chains 4`, `chain_method=vectorized`, 24 h,
`--resume` for re-submission after a time-out: warmup state and every 250-draw
chunk are checkpointed). `01_recover_nuts` has 12 h and checkpoints per truth
set. `submit_chain.sh` submits 01-04 held; release them after reading
`00b_timing`'s TIMING line (4 chains vectorized on one H100, i.e. the main
jobs' own layout). Every job sets
`BIRL_HOST_DEVICES=1` and asserts that JAX sees a GPU before starting.

## Status

Implementation complete; verified locally only by `py_compile` and the toy
import check (20 synthetic rows). Unit tests, `--timing`, recovery and MCMC
have not been run: they run on Rorqual via `slurm/`.
