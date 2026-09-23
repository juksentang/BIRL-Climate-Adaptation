# Step 07 stage 3 (rewrite): choice-model counterfactual and welfare — spec

Date 2026-09-23. Replaces the Stone-Geary CE welfare stage (`03_compute_welfare.py`, kept as `03_compute_welfare_v1.py`) with a stage built on the Step 08 semi-parametric choice model. Stages 1-2 (climate → CF quantile matrices) are unchanged.

## Inputs

| What | Where | Notes |
|---|---|---|
| Baseline quantiles | `04_Env_Model/env_model_output.npz` (= `06_BIRL_MCMC/data/env_model_output.npz`) | q10/q50/q90 (N=222,023 × A=27) float32; `action_ids` must equal arange(27) (assert) |
| 2050 quantiles | `07_2050_Counter_Fact/data/ssp245_cf.npz`, `ssp585_cf.npz` | same shape/order; produced by stage 2 with aid = crop_idx*3 + intensity |
| Choice-model posterior | `08_BIRL_v2/outputs/semipar/posterior.npz` | `a_c,b_c,c_c` (4,1000,6); `asc_c` (4,1000,6,26). Flatten chains, thin to K=200 draws (every 20th) |
| Feasibility, countries, m_c, observed actions, country index | `08_BIRL_v2/src/data_loader.load_data(DATA_DIR)` + `model_kwargs` | `mask_obs` (N,A), `obs_country_idx`, `obs_action`, `m_c` (6,), `countries` sorted |
| Household / zone ids for aggregation | `06_BIRL_MCMC/data/birl_sample.parquet` cols `hh_id_merge`, `action_zone`, `wave` | row order identical to the loader |

## Choice model (exactly as estimated)

μ = log1p(max(q50,0)); σ = clip((log1p(max(q90,0)) − log1p(max(q10,0))) / (2·1.2816), 0.01, 5)
V_ia = a_c μ_ia + b_c σ_ia + c_c σ²_ia
U_ia = V_ia + ASC_{c,a} (ASC of action 0 is 0); infeasible actions get −1e10
P(a|i) = softmax over feasible actions of U_ia
Do NOT centre V here (centring cancels in P but not in the logsum).
Note (2026-09-23): the estimation script (`run_semipar.py`) centres V over the feasible set before adding the ASCs (`center_reward`). Choice probabilities are identical either way; the logsum level is not pinned by the likelihood, so the logsum-based compensating variation is defined relative to this per-observation convention (differences between scenarios are what is reported).

## Scenarios

climate ∈ {baseline, ssp245, ssp585} × policy ∈ {none, transfer, contraction, safety_net, insurance, both}

Policies are monotone transforms of income y, so they act on the quantiles pointwise: q_k' = g(q_k) for k ∈ {10,50,90}. m_c is the country median of the chosen-action q50 (baseline, from the loader) and is the money unit throughout.

- **transfer** (stylised, level only): g(y) = y + t·m_c, t = 0.10. Cost per plot-season = t·m_c.
- **contraction** (stylised, variance only): mean-preserving contraction of the three quantiles toward q50: q_k' = q50 + λ(q_k − q50), λ = 0.5. No cost (a pure "what if σ fell" probe). Note: this leaves q50 fixed and shrinks σ; μ is unchanged.
- **safety_net** (income floor): g(y) = max(y, F_c), F_c = f·m_c, f = 0.5 (sensitivity 0.3, 0.7). Cost per plot-season = E[max(F_c − y, 0)] under the 5-node rule (weights .1,.2,.4,.2,.1 on q10, (q10+q50)/2, q50, (q50+q90)/2, q90) of the **chosen** action's baseline distribution, averaged over the country (a transfer cost, not behaviour-adjusted; report also the behaviour-adjusted version = Σ_a P(a) E[max(F − y_a,0)]).
- **insurance** (index-type indemnity with basis risk and loading): payout(y) = (1 − β_basis)·max(T_c − y, 0), T_c = 0.5·m_c, β_basis = 0.3; premium π_c = (1 + loading)·E[payout] with loading = 0.2, E[payout] computed like the safety-net cost (chosen-action baseline distribution, country average); g(y) = y + payout(y) − π_c. Monotone non-decreasing (slope β_basis below T, 1 above), so quantiles map pointwise. Cost to the public = loading share only if the premium is subsidised; report E[payout] and π_c separately.
- **both** = safety_net then insurance.

After the transform recompute μ', σ' from q'. Flags: --f, --t, --lam, --basis, --loading, --trigger (all as shares of m_c where relevant).

## Outputs per (climate, policy, country, draw k)

1. **Crop shares**: mean over the country's observations of P(a|i), for the 27 actions, plus aggregated by crop (9) and by intensity (3).
2. **Expected income** E[y] = Σ_a P(a|i)·E5[y_a'] (5-node mean), country mean; and expected median Σ_a P(a|i) q50'.
3. **Downside probability** P(y < θ_c), θ_c = 0.3·m_c (sensitivity 0.5): Σ_a P(a|i)·Φ((log1p θ_c − μ'_ia)/σ'_ia), country mean.
4. **Logsum welfare** W_i = log Σ_{a feasible} exp(U_ia'); country mean W̄. Money metric (only where a_c > 0): CV_log = ΔW̄ / a_c relative to (baseline, none), and CV_pct = exp(CV_log) − 1 (per cent of income). Report per draw; for countries with a_c ≤ 0 in a draw mark as NaN and report the share of draws with a_c > 0 (expected: Tanzania, Malawi mostly NaN).
5. **Calibrated-ρ certainty equivalent** (sensitivity only): CE_ρ(a) = exp(μ' + (1−ρ)σ'²/2) − 1 for ρ ∈ {1.5, 2.5, 3.5}; behavioural CE = Σ_a P(a|i) CE_ρ(a), country mean.
6. **Policy cost** as defined above, and effect per dollar: ΔE[y] / cost, ΔP(y<θ) / cost, CV_pct / cost.
7. **Headline test**: for each country and climate, the change in crop shares (L1 distance Σ_a |ΔP(a)| / 2, "share of plots that switch") under **transfer** vs under **contraction**; the ratio contraction/transfer is the level-vs-variance responsiveness. Expected: ratio rises as income falls (Tanzania, Malawi ≫ Nigeria, Mali).

Uncertainty: 89% HPDI over the K draws for every quantity. Also report the (baseline, none) predicted crop shares against observed shares as a PPC.

## Files

- `07_2050_Counter_Fact/src/choice_engine.py`: pure functions: `load_posterior_thinned(K)`, `mu_sigma(q10,q50,q90)`, `apply_policy(q10,q50,q90, policy, params, m_c_obs, cost_info)`, `choice_probs(mu,sig,draw_params,mask,country_idx)`, `scenario_metrics(...)` returning a dict of arrays; jax if available else numpy (must run on the cluster GPU; numpy fallback for toy tests).
- `07_2050_Counter_Fact/scripts/03_choice_counterfactual.py`: loops climates × policies × draws (batch draws on the GPU, loop scenarios), writes `results/choice_cf/metrics.parquet` (long format: climate, policy, country, draw, metric, value) and `results/choice_cf/crop_shares.parquet` (climate, policy, country, draw, action, share) and `results/choice_cf/run_info.json` (parameters, K, git commit, timing). CLI flags for policy parameters and --toy (synthetic data, no real files) for local checks.
- `07_2050_Counter_Fact/scripts/03_choice_counterfactual.sbatch`: Rorqual, 1 × h100_1g.10gb, 2 h, sources `${REMOTE_ROOT}/08_BIRL_v2/slurm/env.sh` via the same ENV_FILE pattern as 08's sbatch files, submitted with `08_BIRL_v2/slurm/sb.sh`. `08_BIRL_v2/slurm/sync_up.sh` must also push `07_2050_Counter_Fact/{src,scripts}` and `07_2050_Counter_Fact/data/ssp245_cf.npz, ssp585_cf.npz` to `${REMOTE_ROOT}/07_2050_Counter_Fact/`; `sync_down.sh` must pull `07_2050_Counter_Fact/results/choice_cf/`.
- `07_2050_Counter_Fact/scripts/04_report_choice_cf.py`: reads the two parquet files and writes `results/choice_cf/tables/*.csv` (one per output item above, median [89% HPDI]) and `results/choice_cf/figures/*.pdf` (crop-share shifts by country × climate; headline ratio by country; downside probability by scenario; CV where defined).
- Old stage kept: rename `03_compute_welfare.py` → `03_compute_welfare_v1.py`; `run_pipeline.py` stage 3 points to the new script.

## Local rule

No real-data computation on the laptop: `--toy` only (synthetic N=500, A=27, C=6). Real runs on Rorqual.
