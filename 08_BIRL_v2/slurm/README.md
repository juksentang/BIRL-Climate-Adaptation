# slurm/ — running BIRL v2 on Rorqual

Target: Rorqual (Calcul Québec / Alliance Canada), account `def-zhiming_gpu`,
user `jsentang`, project root `/scratch/jsentang/birl_v2`. Every value below was
verified live on the login node `rorqual3` on 2026-09-19 (not taken from the web
docs). All scripts source `env.sh`; the laptop-side scripts re-exec themselves
over `ssh rorqual` when they are not on the cluster, so every command here can be
typed on the laptop from `08_BIRL_v2/`.

## Files

| file | runs on | what |
|---|---|---|
| `env.sh` | both | user, account, paths, module list, pinned wheel versions, gres names |
| `setup_venv.sh` | login node, once | `module load …`, `python -m venv $REMOTE_ROOT/venv`, `pip install --no-index` the pinned set, prints versions + `jax.devices()`, runs the toy pytest (`--force` rebuilds) |
| `sync_up.sh` | laptop | rsync the package (no `outputs/`, `__pycache__`, caches) to `$REMOTE_ROOT/08_BIRL_v2` and the three 06 data files to `$REMOTE_ROOT/06_BIRL_MCMC/data` (`-n` dry run) |
| `00_smoke.sbatch` | MIG `h100_1g.10gb`, 3 h | `pytest -q tests/` (toy + realdata) ; checkpoint/resume round-trip (`run_v2.py --chains 2 --warmup 5 --samples 4 --chunk 2 --tag ckpt`: stop after chunk 1 → `--resume` → `--resume` again) ; `simulate_recover.py --mode svi` — the gate |
| `00b_timing.sbatch` | 1 × H100, 2 h | `run_v2.py --variant v2_country --timing --chains 4` (vectorized on one device = the main jobs' layout; 50 warmup + 20 + 20 draws, phases timed separately) → `outputs/v2_country_timing/timing.json` — the walltime gate; runs concurrently with 00 |
| `01_recover_nuts.sbatch` | MIG `h100_2g.20gb`, 12 h | `simulate_recover.py --mode nuts --resume` (flat priors, sets A+B, 2 × (500+500); per-set checkpoint, re-submit after a time-out) — the PASS verdict |
| `02_main.sbatch` | 1 × H100, 24 h | `run_v2.py --variant v2_country --chains 4 --resume` |
| `03_gfix.sbatch` | 1 × H100, 24 h | `run_v2.py --variant v2_country_gfix --s-fixed 0.3 --chains 4 --resume` |
| `04_smax08.sbatch` | 1 × H100, 24 h | `run_v2.py --variant v2_country --s-max 0.8 --chains 4 --resume` → `outputs/v2_country_smax08/` |
| `submit_chain.sh` | either | submits 00 and 00b, then 01–04 **held** with `--dependency=afterok:<00>:<00b>`; prints the `scontrol release` command and records all job ids; `--release` releases them |
| `status.sh` | either | `squeue` + `sacct` + result one-liners + tail of every log |
| `sync_down.sh` | laptop | rsync `outputs/` back (skips `.jax_cache`, in-flight checkpoints) |

Every sbatch: `--account=def-zhiming_gpu`, one `--gres` as above, `--nodes=1`,
`--cpus-per-task=4` (8 for the main runs), `--mem=32G` (64G main),
`--output=/scratch/jsentang/birl_v2/08_BIRL_v2/outputs/slurm_logs/%x-%j.out` (a literal path: Slurm opens it before the
script runs, so the directory must exist — `sync_up.sh` / `submit_chain.sh` create it; a manual `sbatch` on a
fresh clone fails at start with no log), `set -euo pipefail`, `module load
StdEnv/2023 python/3.11 cuda/12.9 cudnn/9.13.1.26 arrow/25.0.0`, `source
$VENV/bin/activate`, `export BIRL_HOST_DEVICES=1`, `cd $REMOTE_ROOT/08_BIRL_v2`,
then `python3 -c "import jax; print(jax.devices())"` with an assertion that the
first device is a GPU (a silent CPU fallback would burn the allocation), then the
real command.

## Order of submission

```bash
cd "…/Formal Analysis/08_BIRL_v2"
bash slurm/sync_up.sh              # 1. code + data -> /scratch/jsentang/birl_v2   (~2 min, 103 MB data)
bash slurm/setup_venv.sh           # 2. once; ~3 min; ends with versions, jax.devices(), toy pytest
bash slurm/submit_chain.sh         # 3. 00 + 00b -> {01,02,03,04} HELD; prints job ids + the release command
bash slurm/status.sh               # 4. repeat; `bash slurm/status.sh 40` for longer tails
bash slurm/submit_chain.sh --release   # 5. after 00 and 00b passed and the TIMING line looks sane
bash slurm/sync_down.sh            # 6. when 02-04 are done -> outputs/ locally
```

`submit_chain.sh --smoke-only`, `--no-timing`, `--skip-smoke`, `--after ID[,ID]`, `--only
02_main,03_gfix`, `--no-hold` cover the partial cases. Rorqual's scheduler runs
`kill_invalid_depend`, so if `00_smoke` exits non-zero the four dependants are
cancelled, not left pending. A main run that hits the 24 h wall is re-submitted
with the same script and the **same `--gres`** (`sbatch slurm/02_main.sbatch`
from `$REMOTE_PKG`, or `submit_chain.sh --skip-smoke --only 02_main`): `--resume`
continues from the 250-draw checkpoint, or from `warmup_state.pkl` if the
time-out hit during the first chunk (the 1000-step warmup is never repeated).
The checkpoint stores `chain_method`; a re-submission on a different device
count (e.g. four MIG slices → `parallel`) mismatches it and restarts from
scratch. `01_recover_nuts` likewise: re-submit the same script, `--resume`
loads the finished set's `nuts_posterior_<set>.npz` and keeps it in
`report.json`, so the PASS verdict can be assembled from two jobs.

**Do not run `sync_up.sh` while jobs are queued or running** — it refuses
(`--force` overrides): the pending sbatch scripts import `run_v2.py` / `src`
when they start, so an rsync `--delete` of the code tree changes what they run.

## Expected times (to be corrected from the first `TIMING` line)

| job | GPU | walltime requested | expectation |
|---|---|---|---|
| 00_smoke | 1g.10gb | 3 h | pytest over all 222K × 27 cells: 1 min 43 s observed (96 passed); ckpt round-trip (2 chains vectorized, 9 early deep-tree draws, three processes) ~20–30 min; SVI 2 × 3000 steps ~ minutes. The first submission's 1-chain 50+20 timing step was cancelled after 70 min without finishing on this slice (≥ 60 s/draw: deep warmup trees at `max_tree_depth=10`) — hence 00b |
| 00b_timing | h100 | 2 h | 4 chains vectorized × (50 + 20 + 20) draws on the main jobs' GPU; expected well under 1 h, and its `projected_full_run_h` is the direct estimate for 02/03/04 |
| 01_recover_nuts | 2g.20gb | 12 h | 2 sets × 2 chains × 1000 draws, vectorized; plausibly many hours — the per-set checkpoint + `--resume` makes a second submission finish set B without repeating A |
| 02/03/04 | h100 | 24 h | 4 × (1000+1000) vectorized. Gate on 00b's `projected_full_run_h` (same GPU, same 4-chain vectorized layout; the warmup term uses the mean depth of 50 early draws and is an upper bound). If it exceeds ~20 h, the warmup + chunk checkpoints make a second 24 h submission finish it (release anyway, re-submit with `--resume` after the TIMEOUT) |

Partition tiers are picked automatically from `--time`: ≤ 3 h → `gpubase_bygpu_b1`,
≤ 12 h → `b2`, ≤ 24 h → `b3` (36 full-H100 nodes), up to 7 days (`b5`); only queue
priority changes with the tier, charging is on use. Billing weights per GPU-hour:
1g.10gb 1743, 2g.20gb 3486, 3g.40gb 5229, full h100 12200, so the whole chain costs
≈ 3 × 1743 + 2 × 12200 + 12 × 3486 + 3 × 24 × 12200 ≈ 9.5e5 billing units at most (less: charged on use).

`timing.json` fields (`run_v2.py --timing`): the run is three timed phases — warmup
(50 draws, `collect_warmup`), a first 20-draw sampling pass (includes the sampling-loop
compile) and a second 20-draw pass from the same post-warmup state (compile cached).
`phases.*.s_per_draw`, `phases.*.mean_leapfrog_per_draw`, `s_per_leapfrog_steady`,
`compile_sampling_s_est`, `projected_full_run_h` (= `s_per_leapfrog_steady` × (warmup
leapfrogs/draw × 1000 + steady leapfrogs/draw × 1000) / 3600, same GPU + layout) and
`projected_full_run_h_naive_incl_compile` (the old elapsed/draw × 2000). Run with
`--chains 1` on a MIG slice the number must be rescaled (1g.10gb → full H100 ≈ ÷7,
1 → 4 vectorized lock-stepped chains ≈ ×3–4); 00b avoids that by measuring in place.

## What each output means

* `outputs/slurm_logs/<name>-<jobid>.out` — the job log: node, gres, `jax.devices()`, then the program's stdout/stderr.
* `outputs/v2_country_timing/timing.json` (from 00b) — per-phase `s_per_draw` / `mean_leapfrog_per_draw`, `s_per_leapfrog_steady`, `compile_sampling_s_est`, `peak_device_gb` (XLA's real footprint; nvidia-smi is meaningless because XLA preallocates), `projected_full_run_h` (4 chains vectorized, 1000+1000, one full H100 = the main jobs' layout). Also the `TIMING …` line in `run.log`.
* `outputs/v2_country_ckpt/run.log` — the smoke's checkpoint round-trip: must contain `Saved: … warmup_state.pkl`, `[STOP] stop_after_chunk=1`, `Resumed from checkpoint: 2/4`, `Chunk 2:` and `[SKIP] MCMC: loaded existing`.
* `sstat`/`sacct` show `gres/gpuutil=0` for MIG slices (NVML has no per-instance utilization) while `gres/gpumem` is allocated and the CPU sits at 100 % (CUDA spin-wait): that is NOT a CPU fallback; the `jax.devices()` assertion at the top of each log is the check. `status.sh` prints `gres/gpumem` for running jobs.
* `outputs/recovery/svi/report.md` — SVI smoke: point estimates vs truth for sets A and B, verdict `SMOKE` (informative only).
* `outputs/recovery/nuts/report.md|json` — the recovery verdict: `PASS` iff for every country in both sets |Δρ| ≤ 0.25, |Δs| ≤ 0.05, |Δ log β| ≤ 0.2 (posterior median vs truth) and ≥ 80 % of the 36 truths lie inside their 89 % HPDI; z-scores and log-lik(truth) vs log-lik(median) per set.
* `outputs/<variant>/summary.csv` — median, mean, sd, 89 % HPDI, rank-normalised split r̂, bulk/tail ESS for `rho_c, s_c, s_lat_c, gamma_c, beta_c` and the hyperparameters.
* `outputs/<variant>/convergence.txt`, `model_diagnostics.json` — divergences, r̂/ESS extremes, per-country corr(ρ, s), corr(log β, ρ), corr(log β, s), P(s_c > 0.55), share of obs with the chosen action inside the floor, share with p(chosen) < 1e-3, Spearman CE(ρ=0.3) vs CE(ρ=4.5) per country (> 0.95 ⇒ ρ acts mostly as a temperature), PPC summary, platform.
* `outputs/<variant>/ppc_action_freq.csv`, `ppc_crop_x_intensity.csv` — observed vs posterior-predictive action shares, overall and per country.
* `outputs/<variant>/posterior.npz` — one array per site, shape `(n_chains, n_draws, …)` + `__diverging__` + `__meta__` (load with `src.mcmc_runner.load_posterior`).

## Phase 1 gates (judged from these outputs)

1. `00_smoke` exit 0: all tests pass on the real data, timing finite, SVI smoke ran.
2. `outputs/recovery/nuts/report.md` verdict **PASS**.
3. `outputs/v2_country/summary.csv`: every `s_c` posterior median in **[0.05, 0.55]** and `P(s_c > 0.55)` small (the floor is identified, not pinned at a bound as in 06); r̂ < 1.01, no divergences.
4. The ranking of `rho_c` across the six countries is **stable across `v2_country`, `v2_country_gfix` and 06** (`06_BIRL_MCMC` outputs); `v2_country_smax08` should reproduce `v2_country` (the 0.6 cap is not binding).

## How the versions were chosen (verified 2026-09-19 on rorqual3)

* `avail_wheels --all-versions jax jaxlib jax_cuda12_plugin jax_cuda12_pjrt numpyro`: the newest jax in the wheelhouse is 0.11.1, but its jaxlib 0.11.1 exists only for cp312–cp314 and there is **no** `jax_cuda12_plugin` 0.11.1 at all (plugins stop at 0.10.2). The newest *consistent* set for python 3.11 is therefore **jax 0.10.2 = jaxlib 0.10.2 (cp311, x86-64-v3) = jax_cuda12_plugin 0.10.2 (cp311) = jax_cuda12_pjrt 0.10.2**; `jax-0.10.2` METADATA requires `jaxlib>=0.10.1,<=0.10.2` and `jax-cuda12-plugin==0.10.2` for the `cuda12` extra, so the set is self-consistent. No PyPI fallback is needed.
* **numpyro 0.21.0** (wheelhouse; requires `jax>=0.7.0`). The code was written against numpyro 0.19 / JAX 0.8.1 locally; the toy pytest at the end of `setup_venv.sh` and the smoke job check the combination.
* `jax_cuda12_plugin` METADATA wants CUDA 12.1+ / cuDNN ≥ 9.8 (`with-cuda` extra); the Alliance build dlopens them from the loaded modules. First choice was `cuda/12.6` (the default) + `cudnn/9.10.0.56`; the first smoke log then showed XLA's `*** WARNING *** Invoking ptxas with version 12.6.77 … CUDA versions up to 12.6.2 miscompile certain edge cases around clamping. Please upgrade to CUDA 12.6.3 or newer.` Since the gradient path clips (`models.py`: `jnp.clip(…, -T_CLIP, T_CLIP)`, `.clip(min=1)`, the log-beta clip), the stack was moved to **`cuda/12.9` (cudacore 12.9.1, ptxas 12.9) + `cudnn/9.13.1.26`** (verified 2026-09-20 on rorqual3: loads together; `cudnn/9.10.0.56` does not load with cuda/12.9). No venv rebuild: the wheel dlopens whichever CUDA/cuDNN the modules provide. The smoke's last summary line counts the ptxas-12.6 warnings in its own log (must be 0).
* `pip install --no-index --dry-run` of the pinned set in a throwaway venv on the login node resolved every dependency from the wheelhouse (`numpy 2.4.2, scipy 1.17.1, ml_dtypes 0.5.4, opt_einsum 3.4.0, multipledispatch 1.0.0, tqdm 4.70.1, pandas 2.3.3, pytest 9.1.1`). `pandas` is pinned < 3 because the loader was written against pandas 2.x. `pyarrow` comes from `module load arrow/25.0.0` (the wheelhouse only has a dummy `pyarrow-9999` stub); it imports inside the venv.
* GRES names: `sacctmgr show tres` lists `gpu:h100`, `gpu:h100_1g.10gb`, `gpu:h100_2g.20gb`, `gpu:h100_3g.40gb`, `gpu:h100_4g.40gb`; `sbatch --test-only` accepted `--gres=gpu:h100_1g.10gb:1`, `gpu:h100_2g.20gb:1` and `gpu:h100:1` with the exact `--time/--mem/--cpus` used here (placed in `gpubase_bygpu_b1` / `_b2`). `module load …` works under `set -euo pipefail`.
* Login-node quirk: any XLA CPU compile on the login node must run under a small CPU affinity (`taskset -c 0-3`, `XLA_FLAGS=--xla_cpu_multi_thread_eigen=false`): XLA sizes its Eigen pool from the 192 visible cores and `pthread_create` fails with EAGAIN under the login-node process cap (`Fatal Python error: Aborted` in `backend_compile_and_load`). `setup_venv.sh` does this; compute nodes are cgroup-confined to `--cpus-per-task` and unaffected. With that guard all 80 toy tests passed on the login node with jax 0.10.2 / numpyro 0.21.0 (8.4 s).
* Accounts: `def-zhiming_gpu`, `def-zhiming_cpu` (QOS normal, MaxSubmit 1000). `$SCRATCH` = `/scratch/jsentang` (19 T free). `DependencyParameters=kill_invalid_depend`.
