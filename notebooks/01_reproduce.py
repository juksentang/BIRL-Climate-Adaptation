# %% [markdown]
# # Tier 1 — re-estimate the choice model and the base counterfactual
#
# This notebook refits the semi-parametric crop-choice model
# $V_{ia} = a_c\,\mu_{ia} + b_c\,\sigma_{ia} + c_c\,\sigma_{ia}^2$ (+ crop fixed effects)
# on the two derived data files, then runs the climate and policy counterfactual
# with the posterior you just estimated (or, optionally, with the tracked posterior).
#
# **What you need**
#
# | item | where |
# |---|---|
# | this repository, installed with `pip install -e .` | `git clone …` (placeholder below) |
# | `birl_sample.parquet`, `env_model_output.npz`, `action_space_config.json` | restricted LSMS-ISA derivatives, see `DATA_ACCESS.md`; put them in `06_BIRL_MCMC/data/` or point `BIRL_DATA_DIR` at them |
# | `07_2050_Counter_Fact/data/ssp585_cf.npz` (2050 quantiles) | produced by Step 07 stages 1–2; shipped with the derived data |
# | a GPU (Colab T4/L4 is enough) | CPU works: SVI ≈ 15–30 min, skip NUTS |
#
# **Expected runtimes** (from the run recorded in `notebooks/README.md`): data load
# < 10 s; SVI fit (6000 steps) ≈ 1 min on a GPU slice; optional NUTS (4 chains,
# 400+400) ≈ 5 min on a full GPU; counterfactual (K = 50 draws, baseline + SSP5-8.5,
# six policies) ≈ 1 min.

# %% [markdown]
# ## 0. Install (Colab) and locate the data
#
# On Colab, uncomment the first two lines. Locally, run `pip install -e .` once in
# the repository root and skip them.

# %%
# !git clone https://github.com/<org>/<repo>.git && cd <repo> && pip install -e . -q
# !pip install -q "jax[cuda12]"       # GPU build of JAX (Colab); the CPU build is installed by pip install -e .
import os, time, json
from pathlib import Path

# repository root = the first parent that contains cropchoice/
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "cropchoice").is_dir())
os.chdir(ROOT)
os.environ.setdefault("BIRL_HOST_DEVICES", "1")                 # one device; chains run vectorised
# os.environ["BIRL_DATA_DIR"] = "/content/drive/MyDrive/birl_data"  # Colab: where the three derived files are
import numpy as np, pandas as pd, jax, jax.numpy as jnp
try:
    display  # IPython
except NameError:  # plain python (e.g. the cluster check of this file)
    display = print
print("repository:", ROOT); print("JAX devices:", jax.devices())

# %% [markdown]
# ## 1. Load the derived data
#
# `load_data` builds the observation arrays (222,023 plot-seasons × 27 actions),
# the country × zone feasibility mask and $m_c$ (the country median of the chosen
# action's predicted median income, the money unit of the model).

# %%
from cropchoice.config import DATA_DIR
from cropchoice.data import load_data, model_kwargs
from cropchoice.models import semipar_features, make_semipar_model, country_one_hot

t0 = time.time()
data = load_data(DATA_DIR)
mk = model_kwargs(data)
countries = list(data["countries"]); C = len(countries)
feat = semipar_features(mk); OH = country_one_hot(mk, C)
N, A = feat["MU"].shape
print(f"N = {N:,} plot-seasons, A = {A} actions, C = {C} countries  ({time.time() - t0:.1f} s)")
print("m_c (USD):", dict(zip(countries, np.round(np.asarray(data['m_c']), 2))))

# %% [markdown]
# ## 2. Fit the model by SVI (≈ 1 min on a GPU)
#
# A multivariate-normal variational approximation of the posterior over the
# 18 country coefficients and 156 crop fixed effects. Point estimates below should
# match Table 1 of the paper to about two decimals; the NUTS run in section 3 gives
# the exact posterior.

# %%
import numpyro
from numpyro.infer import SVI, Trace_ELBO, MCMC, NUTS
from numpyro.infer.autoguide import AutoMultivariateNormal

model = make_semipar_model(feat, C, OH)                      # centred V + ASCs, exactly as estimated
guide = AutoMultivariateNormal(model, init_loc_fn=numpyro.infer.init_to_median())
svi = SVI(model, guide, numpyro.optim.Adam(1e-2), Trace_ELBO())
t0 = time.time()
res = svi.run(jax.random.PRNGKey(0), 6000, progress_bar=False)
med = {k: np.asarray(v) for k, v in guide.median(res.params).items()}
print(f"SVI: 6000 steps in {time.time() - t0:.0f} s; ELBO loss {float(res.losses[0]):.0f} -> {float(res.losses[-1]):.0f}")
svi_table = pd.DataFrame({"a": med["a_c"], "b": med["b_c"], "c": med["c_c"]}, index=countries).round(2)
display(svi_table)

# %% [markdown]
# ### Compare with the tracked NUTS posterior
#
# `08_BIRL_v2/outputs/semipar/summary.csv` holds the medians and 89% HPDIs of the
# full NUTS run that the paper reports.

# %%
ref = pd.read_csv(ROOT / "08_BIRL_v2" / "outputs" / "semipar" / "summary.csv")
ref = ref[ref.param.isin(["a", "b", "c"])].pivot(index="country", columns="param", values="median").reindex(countries).round(2)
cmp = pd.concat({"SVI (this run)": svi_table, "NUTS (tracked)": ref}, axis=1)
display(cmp)

# %% [markdown]
# ## 3. Optional: NUTS (≈ 5 min on a full GPU; skip on CPU)
#
# Set `RUN_NUTS = True` to draw from the exact posterior. 400 warm-up + 400 draws
# per chain is enough to reproduce the HPDIs to the reported precision.

# %%
RUN_NUTS = False
K_DRAWS = 50
if RUN_NUTS:
    t0 = time.time()
    mcmc = MCMC(NUTS(model, target_accept_prob=0.8, max_tree_depth=10, init_strategy=numpyro.infer.init_to_median()),
                num_warmup=400, num_samples=400, num_chains=4, chain_method="vectorized", progress_bar=False)
    mcmc.run(jax.random.PRNGKey(1))
    post = {k: np.asarray(v) for k, v in mcmc.get_samples().items()}
    print(f"NUTS: 4 x (400+400) in {(time.time() - t0) / 60:.1f} min; divergences {int(np.asarray(mcmc.get_extra_fields()['diverging']).sum())}")
    idx = np.linspace(0, post["a_c"].shape[0] - 1, K_DRAWS).astype(int)
    draws = {"a": post["a_c"][idx], "b": post["b_c"][idx], "c": post["c_c"][idx],
             "asc": np.concatenate([np.zeros((K_DRAWS, C, 1), np.float32), post["asc_c"][idx]], axis=-1)}
else:
    # K draws from the SVI guide
    samp = guide.sample_posterior(jax.random.PRNGKey(2), res.params, sample_shape=(K_DRAWS,))
    draws = {"a": np.asarray(samp["a_c"]), "b": np.asarray(samp["b_c"]), "c": np.asarray(samp["c_c"]),
             "asc": np.concatenate([np.zeros((K_DRAWS, C, 1), np.float32), np.asarray(samp["asc_c"])], axis=-1)}
print({k: v.shape for k, v in draws.items()})

# %% [markdown]
# ## 4. Counterfactual: current climate and SSP5-8.5, six policies (≈ 1 min)
#
# Policies are monotone transforms of the predicted income distribution
# (`cropchoice.policies`): income transfer, variance cut, income floor, index
# insurance, floor + insurance. The model gives the choice probabilities under each;
# the metrics are crop shares, expected income, downside probability, expected
# shortfall and switching shares (see `07_2050_Counter_Fact/CHOICE_CF_SPEC.md`).
#
# To use the tracked posterior instead of your own draws, set
# `USE_TRACKED_POSTERIOR = True`.

# %%
from cropchoice.counterfactual import real_inputs, run_scenarios, long_tables
from cropchoice.policies import DEFAULT_PARAMS

USE_TRACKED_POSTERIOR = False
CLIMATES = ["baseline", "ssp585"]
POLICIES = ["none", "transfer", "contraction", "safety_net", "insurance", "both"]
t0 = time.time()
inp = real_inputs(CLIMATES, K_DRAWS, posterior=ROOT / "08_BIRL_v2" / "outputs" / "semipar" / "posterior.npz")
if not USE_TRACKED_POSTERIOR:
    inp["draws"] = draws
resu, info = run_scenarios(inp, CLIMATES, POLICIES, DEFAULT_PARAMS, batch=10, say=lambda s: None)
metrics, shares = long_tables(resu, inp["countries"])
print(f"counterfactual: {len(CLIMATES)} climates x {len(POLICIES)} policies x {K_DRAWS} draws in {time.time() - t0:.0f} s")

# %% [markdown]
# ### The headline: variance cut versus income transfer
#
# For each country, the share of plots that switch crop or intensity under a pure
# variance cut divided by the share that switch under an income transfer
# (SSP5-8.5). Compare with `07_2050_Counter_Fact/results/choice_cf/tables/headline_ratio.csv`.

# %%
sw = metrics[(metrics.metric == "switch_share") & (metrics.climate == "ssp585")]
piv = sw.pivot_table(index=["country", "draw"], columns="policy", values="value")
ratio = (piv["contraction"] / piv["transfer"]).groupby("country")
order = list(pd.Series(inp["m_c"], index=inp["countries"]).sort_values().index)
head = pd.DataFrame({"ratio_median": ratio.median(), "q05": ratio.quantile(0.055), "q95": ratio.quantile(0.945)}).reindex(order).round(2)
tracked = pd.read_csv(ROOT / "07_2050_Counter_Fact" / "results" / "choice_cf" / "tables" / "headline_ratio.csv")
head["tracked (K=200 NUTS)"] = tracked[tracked.climate == "ssp585"].set_index("country")["ratio_median"].reindex(order).round(2)
display(head)

# %%
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None; print("matplotlib not installed: skipping the plot")
if plt:
    COLOR = dict(zip(order, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]))
    m_c_s = pd.Series(inp["m_c"], index=inp["countries"]).reindex(order)
    fig, ax = plt.subplots(figsize=(5, 3), dpi=120)
    for i, cn in enumerate(order):
        ax.errorbar(i, head.loc[cn, "ratio_median"],
                    yerr=[[head.loc[cn, "ratio_median"] - head.loc[cn, "q05"]], [head.loc[cn, "q95"] - head.loc[cn, "ratio_median"]]],
                    fmt="o", color=COLOR[cn], capsize=2)
        ax.plot(i, head.loc[cn, "tracked (K=200 NUTS)"], "x", color="#444444")
    ax.axhline(1, color="#999999", lw=0.6, ls="--"); ax.set_yscale("log")
    ax.set_xticks(range(len(order))); ax.set_xticklabels([f"{c}\n{m:.0f} USD" for c, m in zip(order, m_c_s)], fontsize=7)
    ax.set_ylabel("switching: variance cut / income transfer")
    ax.set_title("Your run (dots, 89% interval) vs the tracked result (x)", fontsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    plt.tight_layout(); plt.show()

# %% [markdown]
# ## 5. What to look at next
#
# * `metrics` and `shares` are the long tables that `python -m cropchoice report`
#   turns into the paper's tables and figures (`--results` pointing at a directory
#   holding `metrics.parquet`, `crop_shares.parquet`, `run_info.json`).
# * `python -m cropchoice sobol` sweeps the policy sizes; `python -m cropchoice fit-semipar-assets`
#   estimates the asset-tercile version.
# * The full pipeline (environment model, MCMC variants, Sobol) is documented in the
#   root `README.md` (Tier 2).
