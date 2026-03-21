# Step 7: 2050 Climate Counterfactual Analysis — Build Documentation

**方法**: Delta Method + LightGBM环境模型 + Stone-Geary CRRA效用 + 后验不确定性传播
**运行环境**: Local (Python 3.12, NumPy/LightGBM/scikit-learn)
**总计算时间**: ~23分钟 (Stage 1: 30s, Stage 2: 2min, Stage 3: 20min, Stage 4: 30s)
**前置条件**: Step 04 (Environment Model), Step 06 (BIRL MCMC Posterior), CMIP6 GEE Extraction
**日期**: 2026-03-21

---

## 1. 架构

将BIRL后验的描述性发现（"各国农户风险偏好不同"）转化为处方性政策建议（"在2050年气候变化下，哪个国家需要什么类型的干预"）。

核心指标: **确定性等价 (Certainty Equivalent, CE)** — 使农户效用等于期望效用的确定性收入水平。

### 1.1 分析链

```
Step 04: LightGBM环境模型         Step 06: BIRL MCMC后验
  model_mu.txt + model_sigma.txt     posterior.pkl (rho_c, gamma_c)
           │                                │
           ▼                                ▼
┌─────────────────────────────────────────────────────────┐
│ Stage 0: 导出GPS点 (4,436个唯一点)                       │
│   birl_sample.parquet → gps_points_for_gee.csv          │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│ GEE提取: CMIP6 NASA/GDDP-CMIP6                          │
│   5 GCMs × 2 SSPs × 2 periods = 20 CSVs                │
│   Per-point growing-season statistics                   │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│ Stage 1: 气候处理                                        │
│   Ensemble delta factors (5-GCM平均)                    │
│   降雨: 乘性 × (future/baseline)                         │
│   温度: 加性 + (future - baseline)                       │
│   NDVI: 经验模型 f(rain, temp, country) R²=0.60/0.66    │
│   → 替换222,023 obs的7个气候特征                         │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│ Stage 2: 反事实矩阵生成                                  │
│   LightGBM(model_mu) → μ̂ = E[log(y)|s_2050, a]         │
│   LightGBM(model_sigma) → σ̂ = Std[log(y)|s_2050, a]    │
│   LogNormal → q10/q50/q90 (USD空间)                     │
│   → (222,023 × 27) × 4 matrices per SSP                │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│ Stage 3: 福利计算                                        │
│   Load posterior (rho_c, gamma_c) → 3,000 samples       │
│   6 scenarios × 6 countries × 3,000 posterior samples   │
│   Stone-Geary CRRA + 5点求积 → CE                       │
│   max(CE) over feasible actions → optimal CE per obs    │
│   → climate loss, policy value, synergy                 │
└─────────────────────────────────────────────────────────┘
```

### 1.2 关键假设

1. **风险偏好不变**: 2050年的rho和gamma与当前相同（保守——贫困加剧可能推高gamma）
2. **非气候特征不变**: 土壤、地形、家庭特征、市场接入保持当前水平（隔离纯气候效应）
3. **作物技术不变**: 环境模型学到的气候-产量关系在2050年仍成立（无品种改良）
4. **基于CE而非选择概率**: 不预测"农户会改种什么"，只预测"福利变多少"——不依赖beta

---

## 2. 文件结构

```
07_2050_Counter_Fact/
├── counterfactual_build.md              ← 本文件
├── 2050_analysis_report.md              # 完整分析报告（结果数据+论文叙事）
├── plan.md                              # 原始设计文档
│
├── src/                                 # 核心Python模块
│   ├── __init__.py
│   ├── config.py                        # 所有常量、路径、场景、特征列表
│   ├── climate_processing.py            # Delta method + NDVI经验模型
│   ├── counterfactual_engine.py         # 反事实矩阵生成（LightGBM预测）
│   ├── welfare_engine.py                # CE计算、安全网模型、场景运行
│   └── reporting.py                     # 控制台结果汇报
│
├── scripts/                             # Pipeline脚本
│   ├── 00_export_gps_points.py          # Stage 0: 导出GPS点
│   ├── 00_extract_cmip6_gee.js          # GEE JavaScript提取（Code Editor版）
│   ├── 00_extract_cmip6_python.py       # GEE Python API提取
│   ├── 01_process_climate.py            # Stage 1: 气候处理
│   ├── 02_generate_cf_matrices.py       # Stage 2: 反事实矩阵
│   ├── 03_compute_welfare.py            # Stage 3: CE计算
│   └── run_pipeline.py                  # 端到端orchestrator
│
├── data/
│   ├── cmip6_raw/                       # 20 CSVs (5 GCMs × 2 SSPs × 2 periods)
│   │   └── cmip6_{GCM}_{SSP}_{period}.csv
│   ├── cmip6_processed/
│   │   ├── ensemble_deltas.parquet      # 4,436 points × delta factors
│   │   ├── obs_climate_ssp245.parquet   # 222,023 obs × 7 adjusted features
│   │   └── obs_climate_ssp585.parquet   # 222,023 obs × 7 adjusted features
│   ├── ssp245_cf.npz                    # (222,023 × 27) × 4 matrices
│   ├── ssp585_cf.npz                    # (222,023 × 27) × 4 matrices
│   ├── ndvi_model_growing.joblib        # GBR for NDVI growing mean
│   ├── ndvi_model_preseason.joblib      # GBR for NDVI preseason mean
│   ├── posterior_country_params.npz     # Lightweight posterior (512 KB)
│   └── gps_points_for_gee.csv          # 4,436 unique GPS points
│
├── results/
│   ├── ce_by_scenario_country.csv       # 核心: 6 scenarios × 6 countries
│   ├── ce_posterior_samples.npz         # 3,000-sample arrays for CI
│   ├── climate_loss.csv                 # CE(S0) - CE(S2) per country
│   ├── policy_value.csv                 # CE(policy) - CE(S2) per country
│   └── synergy.csv                      # Combined - (Insurance + Safety net)
│
└── _archived_plots/                     # Archived plotting code
    ├── 04_make_figures.py
    ├── visualization.py
    └── fig*.pdf
```

---

## 3. 运行方法

### 3.1 前置条件

- Step 04完成: `04_Env_Model/model_mu.txt`, `model_sigma.txt`, `env_model_output.npz`
- Step 06完成: `06_BIRL_MCMC/outputs/hier_noalpha/posterior.pkl`（或lightweight `data/posterior_country_params.npz`）
- CMIP6 CSVs已从GEE提取到 `data/cmip6_raw/`
- Python依赖: numpy, pandas, lightgbm, scikit-learn, joblib, pyarrow

### 3.2 端到端运行

```bash
cd "Formal Analysis/07_2050_Counter_Fact"

# Stage 0 (仅首次): 导出GPS点给GEE
python scripts/00_export_gps_points.py

# GEE提取 (二选一):
#   方案A: GEE Code Editor — 上传CSV到Asset, 运行 00_extract_cmip6_gee.js
#   方案B: Python API — python scripts/00_extract_cmip6_python.py
# → 20个CSV存入 data/cmip6_raw/

# Stages 1-3: 完整pipeline
python scripts/run_pipeline.py --from 1

# 或单独运行:
python scripts/run_pipeline.py --only 1   # Stage 1: 气候处理
python scripts/run_pipeline.py --only 2   # Stage 2: 反事实矩阵
python scripts/run_pipeline.py --only 3   # Stage 3: CE + 福利计算
```

### 3.3 单独阶段

```bash
python scripts/01_process_climate.py       # Stage 1 (~30s)
python scripts/02_generate_cf_matrices.py  # Stage 2 (~2min)
python scripts/03_compute_welfare.py       # Stage 3 (~20min)
```

---

## 4. 输入数据

### 4.1 来自Step 04 (Environment Model)

| 文件 | 内容 | 用途 |
|------|------|------|
| `model_mu.txt` | LightGBM均值模型 (1,334 trees) | 预测E[log(y)\|s,a] |
| `model_sigma.txt` | LightGBM波动性模型 | 预测Std[log(y)\|s,a] |
| `env_model_output.npz` | 基线反事实矩阵 (222,023 × 27 × 4) | S0 baseline CE |
| `env_model_metrics.json` | sigma_scale=1.626, 特征列表 | 校准参数 |

### 4.2 来自Step 06 (BIRL MCMC)

| 文件 | 内容 | 用途 |
|------|------|------|
| `posterior.pkl` (2.8 GB) | 12,000 MCMC样本 (rho_c, gamma_c 等) | 后验不确定性传播 |
| `posterior_country_params.npz` (512 KB) | rho_c/gamma_c 轻量版 | 替代pkl（优先使用） |
| `birl_sample.parquet` | 222,023 obs, 36特征, action_id | 观测数据+GPS坐标 |
| `action_space_config.json` | 27 actions, feasibility mask | 可行动作掩码 |

### 4.3 CMIP6数据 (GEE提取)

| 参数 | 设置 |
|------|------|
| 数据集 | NASA/GDDP-CMIP6 (0.25°, 日数据) |
| GCM Ensemble | ACCESS-CM2, MIROC6, MRI-ESM2-0, INM-CM5-0, IPSL-CM6A-LR |
| 基线时段 | 2005-2014 (historical), 或 2005-2023 (historical + SSP) |
| 未来时段 | 2045-2055 (以2050为中心的10年气候态) |
| SSP情景 | SSP2-4.5 (中等排放) + SSP5-8.5 (高排放) |
| 提取粒度 | Per GPS point (4,436个唯一点) |
| 聚合 | 国家特定生长季 (月份) |
| 变量 | rainfall_gs_sum, rainfall_gs_cv, tmean_gs, tmax_gs |

---

## 5. 处理Pipeline详解

### Stage 0: GPS点导出

从 `birl_sample.parquet` 提取 4,436 个唯一 (lat, lon) 点，附带country标签，导出CSV供GEE上传。

```
输入: birl_sample.parquet (222,023 obs)
输出: data/gps_points_for_gee.csv (4,436 rows)
```

### Stage 1: 气候处理 (`01_process_climate.py`)

**Step 1a — Ensemble Delta Factors**:
- 读入20个CMIP6 CSV (5 GCMs × 2 SSPs × {baseline, future})
- 按GPS点取5-GCM平均 → ensemble mean delta factors
- 降雨delta: **乘性** `delta_rain = future_mean / baseline_mean`（clip baseline >= 1.0 mm）
- 降雨CV delta: **乘性** `delta_cv = future_cv / baseline_cv`（clip baseline >= 0.01）
- 温度delta: **加性** `delta_temp = future_mean - baseline_mean`
- 输出: `cmip6_processed/ensemble_deltas.parquet` (4,436 points × 8 delta columns)

**Step 1b — NDVI经验模型**:
- CMIP6不包含NDVI → 从当前数据拟合经验关系
- 模型: GradientBoostingRegressor (200 trees, depth=4, subsample=0.8, lr=0.05)
- 特征: `[rainfall_growing_sum_final, era5_tmean_growing_final, country_enc]`
- 训练集: ~202,847 obs（排除NaN）
- NDVI growing mean: **R² = 0.602**
- NDVI preseason mean: **R² = 0.657**
- 输出: `data/ndvi_model_growing.joblib`, `data/ndvi_model_preseason.joblib`

**Step 1c — 应用Delta到观测**:
- 对每个SSP，将222,023 obs通过GPS坐标匹配到delta factors
- 替换7个气候特征:

| 特征 | Delta方法 | 公式 |
|------|----------|------|
| `rainfall_growing_sum_final` | 乘性 | current × delta_rain |
| `rainfall_10yr_mean_final` | 乘性 | current × delta_rain |
| `rainfall_10yr_cv_final` | 乘性 | current × delta_rain_cv |
| `era5_tmean_growing_final` | 加性 | current + delta_tmean |
| `era5_tmax_growing_final` | 加性 | current + delta_tmax |
| `ndvi_growing_mean_final` | 经验预测 | GBR(adjusted_rain, adjusted_temp, country) |
| `ndvi_preseason_mean_final` | 经验预测 | GBR(adjusted_rain, adjusted_temp, country) |

- GPS匹配率: 194,825 / 222,023 = **87.8%**（未匹配obs保留原始气候值）
- 输出: `cmip6_processed/obs_climate_{ssp}.parquet`

### Stage 2: 反事实矩阵生成 (`02_generate_cf_matrices.py`)

- 读入Step 04训练好的LightGBM双模型
- 对2050 climate-adjusted DataFrame，遍历27个action:
  - 替换 action_crop + input_intensity，保持34个state特征不变
  - 编码categorical特征（与训练时完全一致）
  - model_mu预测 → mu = E[log(y)]
  - model_sigma预测 → sig_raw → clip(0.1) × kappa(1.626) → sig
  - LogNormal变换:
    ```
    q10 = exp(mu - 1.2816 × sig) - 1
    q50 = exp(mu) - 1
    q90 = exp(mu + 1.2816 × sig) - 1
    sigma_usd = (q90 - q10) / 2.56
    ```
- 输出: `data/{ssp}_cf.npz`，每个包含 (q10, q50, q90, sigma)，shape=(222,023, 27)，float32
- 附带诊断: 按国家比较baseline vs 2050 q50中位数

### Stage 3: 福利计算 (`03_compute_welfare.py`)

**Step 3a — 后验加载**:
- 优先读取lightweight `posterior_country_params.npz` (512 KB)，否则读 `posterior.pkl` (2.8 GB)
- Sigmoid变换: `rho_c = 0.1 + 4.9 × sigmoid(rho_c_unbounded)` (bounded [0.1, 5.0])
- Gamma: `gamma_c = 0.1 + 29.9 × sigmoid(gamma_c_unbounded)` (bounded [0.1, 30.0])
- Thinning: 12,000 → **3,000样本** (每4个取1个)

**Step 3b — 可行性掩码**:
- 从 `birl_sample.parquet` + `action_space_config.json` 重建
- Country × Zone (CZ) 组合索引
- Empirical feasibility: (cz, action_id) 组合在数据中出现 → feasible

**Step 3c — CE计算 (per scenario × per country × per posterior sample)**:
- Stone-Geary CRRA效用（完全匹配models.py:40-52的NumPy移植版）:
  ```
  U(Y) = [(Y - gamma)^(1-rho) - 1] / (1-rho)
  ```
  rho接近1时用Taylor展开: `U ≈ log(surplus) × (1 + 0.5×(1-rho)×log(surplus))`

- 5点求积:
  ```
  y1=q10, y2=(q10+q50)/2, y3=q50, y4=(q50+q90)/2, y5=q90
  EU = 0.10×U(y1) + 0.20×U(y2) + 0.40×U(y3) + 0.20×U(y4) + 0.10×U(y5)
  ```

- CE反解:
  ```
  CE = gamma + ((1-rho)×EU + 1)^(1/(1-rho))
  ```

- 保险干预: `rho_effective = rho × rho_factor` (e.g. 0.5)
- 安全网干预: **income floor model** (非gamma scaling):
  ```
  floor = gamma + (country_median_income - gamma) × coverage
  Y_effective = max(Y, floor)
  ```
  gamma本身不变（偏好参数），floor保证正surplus

- 大国子采样: `max_obs = 10,000`（固定seed=42）
- Optimal CE: `max(CE_mat) over feasible actions per obs`
- 取国家中位数作为汇总指标

**Step 3d — 派生指标**:
- Climate loss: `CE(S0) - CE(S2)` (per posterior sample → median + 95% CI)
- Policy value: `CE(Sx) - CE(S2)` for S3/S4/S5
- Synergy: `CE(S5) - CE(S2) - [(CE(S3) - CE(S2)) + (CE(S4) - CE(S2))]`

---

## 6. 输出文件

### 6.1 data/ 目录

| 文件 | 大小 | 内容 |
|------|------|------|
| `cmip6_raw/*.csv` (20个) | ~15KB each | 5 GCMs × 2 SSPs × 2 periods, per-point气候统计 |
| `cmip6_processed/ensemble_deltas.parquet` | ~200KB | 4,436 points × 8 delta factors (两个SSP) |
| `cmip6_processed/obs_climate_ssp245.parquet` | ~6MB | 222,023 obs × 10 cols (GPS + 7 climate) |
| `cmip6_processed/obs_climate_ssp585.parquet` | ~6MB | 同上 |
| `ssp245_cf.npz` | ~56MB | (222,023 × 27) × 4 反事实矩阵 (float32) |
| `ssp585_cf.npz` | ~56MB | 同上 |
| `ndvi_model_growing.joblib` | ~2MB | GBR模型 (200 trees, NDVI growing) |
| `ndvi_model_preseason.joblib` | ~2MB | GBR模型 (200 trees, NDVI preseason) |
| `posterior_country_params.npz` | 512KB | rho_c/gamma_c unbounded (12,000 × 6) |
| `gps_points_for_gee.csv` | ~200KB | 4,436 unique GPS points |

### 6.2 results/ 目录

| 文件 | 内容 |
|------|------|
| `ce_by_scenario_country.csv` | **核心**: 6 scenarios × 6 countries, CE median + 95% CI |
| `ce_posterior_samples.npz` | 36 arrays (6×6), 每个3,000个CE values |
| `climate_loss.csv` | CE(S0) - CE(S2), absolute + percentage |
| `policy_value.csv` | Insurance / Safety net / Combined values + ratio |
| `synergy.csv` | Combined - (Insurance + Safety net), absolute + percentage |

---

## 7. 源代码模块

### 7.1 `src/config.py` — 配置中心

所有常量的单一来源:
- **路径**: STEP04_DIR, STEP06_DIR, DATA_DIR, RESULTS_DIR 等
- **环境模型常量**: SIGMA_SCALE=1.626, Z10=-1.2816, Z90=1.2816
- **BIRL参数边界**: RHO_LO=0.1, RHO_HI=5.0; GAMMA_LO=0.1, GAMMA_HI=30.0
- **36个特征列**: FEATURE_COLS (精确顺序匹配 env_model_metrics.json)
- **7个气候特征**: CLIMATE_FEATURES → 分MULTIPLICATIVE (3) / ADDITIVE (2) / NDVI (2)
- **动作空间**: 9 CROP_ORDER × 3 INTENSITY_ORDER = 27 N_ACTIONS
- **GCM/SSP/时段**: 5 GCMS, 2 SSPS, baseline/future periods
- **生长季**: GROWING_SEASONS per country (months + cross_year flag)
- **6个政策情景**: SCENARIOS dict (climate, rho_factor, floor_coverage)
- **后验稀疏化**: THIN_FACTOR = 4

### 7.2 `src/climate_processing.py` — 气候处理

4个核心函数:
- `compute_ensemble_deltas()`: 读20个CMIP6 CSV → per-point ensemble-mean deltas
- `train_ndvi_models()`: 拟合NDVI = f(rain, temp, country) 经验模型
- `apply_deltas_to_observations()`: 替换222,023 obs的7个气候特征
- `print_signal_diagnostics()`: 按国家打印气候变化信号幅度

### 7.3 `src/counterfactual_engine.py` — 反事实矩阵

3个核心函数:
- `load_env_models()`: 加载LightGBM model_mu + model_sigma
- `generate_counterfactual_matrix()`: 遍历27 actions, 预测→LogNormal变换→USD矩阵
- `validate_cf_matrices()`: 按国家比较baseline vs 2050 q50

关键实现: categorical编码严格匹配训练时的类别顺序 (ACTION_CROP_CATS, COUNTRY_CATS, SEASON_CATS)

### 7.4 `src/welfare_engine.py` — 福利引擎

核心函数:
- `load_country_posteriors()`: 读MCMC后验 → sigmoid变换 → thinning
- `build_feasibility_data()`: Country×Zone可行性掩码
- `stone_geary_numpy()`: Stone-Geary CRRA效用 (NumPy向量化, 精确移植自JAX models.py)
- `compute_ce_matrix()`: CE = U⁻¹(EU), 5点求积
- `compute_country_ce_batched()`: per-country batch CE, 包含安全网floor和子采样
- `run_all_scenarios()`: 按climate key分组加载，6 scenarios × 6 countries × 3,000 samples
- `compute_derived_metrics()`: climate loss, policy value, synergy

内存优化: 按climate key分组加载矩阵（baseline/ssp245/ssp585各加载一次后释放）

### 7.5 `src/reporting.py` — 汇报

- `print_summary_table()`: 控制台打印CE/loss/policy/synergy表格

### 7.6 `scripts/run_pipeline.py` — Pipeline Orchestrator

- 支持 `--from N` 和 `--only N` 参数
- Stages 0-3 顺序执行，每阶段独立subprocess
- 记录各阶段耗时

---

## 8. 设计决策

### 8.1 Delta Method (非直接替换)

**选择**: 用CMIP6 baseline/future的**比值（降雨）或差值（温度）**修正当前观测值，而非直接使用CMIP6绝对值。

**理由**: GCM的绝对值有系统偏差（bias），但"变化量"相对可靠。Delta method保留了当前观测的空间精细结构，只叠加GCM预测的变化信号。这是downscaling的标准做法。

**具体**:
- 降雨: 乘性delta，因为降雨变化是比例性的（干旱区减少10% vs 湿润区减少10%，含义不同）
- 温度: 加性delta，因为温度变化近似均匀（+1.5°C对所有点几乎相同）
- 降雨CV: 乘性delta，因为CV是无量纲比率

### 8.2 NDVI经验模型 (非CMIP6直接输出)

**选择**: 用GBR经验模型从adjusted rain + temp预测2050 NDVI，而非使用GCM植被输出。

**理由**: CMIP6 GDDP数据集不包含NDVI/LAI等植被变量。MODIS NDVI是遥感产品，不在气候模型输出中。经验模型R² = 0.60/0.66，说明降雨+温度+国家解释了大部分NDVI变异。

**局限**: 经验关系假设CO2施肥效应、土地利用变化等不改变rain-NDVI关系。这在2050时间尺度上可能偏保守。

### 8.3 安全网: Income Floor Model (非Gamma Scaling)

**选择**: `Y_effective = max(Y, floor)`, 其中 `floor = gamma + (median_income - gamma) × coverage`。gamma本身**不改变**。

**之前的错误方法**: 早期版本通过 `gamma_effective = gamma × factor` (factor < 1) 模拟安全网。这导致某些国家安全网效果为负——因为降低gamma改变了效用函数的凹度，对高rho国家产生反直觉的效用下降。

**修正理由**:
- gamma是偏好参数（生存底线），不是政策杠杆。真实安全网（如Ethiopia PSNP, Malawi SCTP）通过转移支付提升低收入端，不改变偏好
- Income floor模型直接截断收入分布下尾，保证surplus = Y - gamma > 0
- coverage = 0.5 意味着floor在gamma和中位收入的中点，确保合理的保底水平
- 修正后**所有国家安全网效果均为正值**

### 8.4 后验稀疏化 (Thinning by 4)

**选择**: 12,000 MCMC samples → thin by 4 → 3,000 samples。

**理由**: CE计算是6 scenarios × 6 countries × K samples的循环，K从12,000降到3,000可将Stage 3从~80min降到~20min。MCMC chain的自相关使相邻样本信息冗余，thin by 4有效消除自相关。95% CI估计在3,000 samples下已非常稳定。

### 8.5 大国子采样 (max_obs = 10,000)

**选择**: 对每个国家，若obs > 10,000则随机子采样（固定seed=42）。

**理由**: Nigeria有~76,000 obs，对每个posterior sample做full computation太慢。由于CE取的是median（稳健统计量），10,000个obs足以给出稳定的中位数估计。子采样仅影响CE的采样噪声，不影响posterior不确定性传播。

### 8.6 5-GCM Ensemble Mean

**选择**: 5个GCM的delta取简单平均，不做加权。

**理由**: GCM weighting（基于历史表现）在区域尺度上争议较大，简单平均是IPCC AR6推荐的default approach。5个GCM选择覆盖了气候敏感性的不同区间（INM-CM5-0偏冷, IPSL-CM6A-LR偏热）。

### 8.7 Lightweight Posterior (.npz vs .pkl)

**选择**: 从2.8 GB的完整posterior.pkl中预提取rho_c/gamma_c到512 KB的.npz文件。

**理由**: 完整posterior包含所有household-level参数（alpha, gamma_raw等），但Step 07只需要country-level的rho_c和gamma_c。预提取避免每次运行加载2.8 GB。如果.npz不存在则fallback到完整pkl。

---

## 9. 政策情景

### 9.1 情景定义

| ID | 名称 | 气候 | rho_factor | floor_coverage | 经济学含义 |
|----|------|------|:----------:|:--------------:|-----------|
| **S0** | 当前基线 | 当前 | 1.0 | 0.0 | 无气候变化，无干预 |
| **S1** | SSP2-4.5 | 2050中等 | 1.0 | 0.0 | 中等排放路径，无干预 |
| **S2** | SSP5-8.5 | 2050高排放 | 1.0 | 0.0 | 高排放路径，无干预 (主要对照) |
| **S3** | 保险 | SSP5-8.5 | **0.5** | 0.0 | 天气指数保险降低有效风险暴露 50% |
| **S4** | 安全网 | SSP5-8.5 | 1.0 | **0.5** | 社会转移保障收入不低于gamma和中位数中点 |
| **S5** | 组合 | SSP5-8.5 | **0.5** | **0.5** | 保险 + 安全网同时实施 |

### 9.2 情景逻辑

**气候损失** = S0 - S2: 气候变化的纯福利冲击

**保险价值** = S3 - S2: 在最差气候下，保险带来的CE改善
- 机制: rho × 0.5 → 降低凹度 → 更高CE（即使收入分布不变）
- 适用性: 对高rho国家（Mali 3.02, Malawi 3.30）更有效

**安全网价值** = S4 - S2: 在最差气候下，保底收入带来的CE改善
- 机制: 截断下尾 → min(Y) ≥ floor → 大幅提升EU → 更高CE
- 适用性: 对高gamma国家（Ethiopia 29.82, Mali 29.12）更有效

**组合价值** = S5 - S2: 两工具同时实施的总效果

**协同效应** = (S5-S2) - [(S3-S2) + (S4-S2)]: 正=超线性(互补), 负=次线性(替代)

---

## 10. 核心数据

### 10.1 BIRL后验参数 (来自Step 06)

| 国家 | rho (风险厌恶) | 95% CI | gamma (生存底线, $) | 95% CI |
|------|:-------------:|--------|:------------------:|--------|
| Nigeria | **1.35** | [1.25, 1.45] | 17.49 | [14.69, 20.66] |
| Uganda | **1.62** | [1.47, 1.78] | **1.80** | [0.52, 3.67] |
| Ethiopia | 2.39 | [2.23, 2.56] | **29.82** | [29.38, 29.99] |
| Tanzania | 2.96 | [2.08, 3.90] | 25.42 | [5.75, 29.93] |
| Mali | **3.02** | [2.77, 3.27] | 29.12 | [26.58, 29.96] |
| Malawi | **3.30** | [3.04, 3.57] | 29.63 | [28.53, 29.98] |

### 10.2 气候变化信号 (SSP5-8.5)

| 国家 | 降雨 Delta | 温度 Delta | 降雨CV Delta | NDVI Delta |
|------|:---------:|:---------:|:-----------:|:---------:|
| Ethiopia | **+22.6%** | +1.5°C | -11.4% | +5.8% |
| Nigeria | +12.4% | +1.5°C | +7.7% | -1.6% |
| Uganda | +5.8% | +1.4°C | +11.7% | -4.5% |
| Tanzania | +6.4% | +1.5°C | +2.8% | -1.7% |
| Malawi | -0.4% | **+1.8°C** | **+24.9%** | +2.1% |
| Mali | +0.1% | **+1.9°C** | **+28.8%** | **-19.5%** |

### 10.3 反事实收入变化 (环境模型预测)

| 国家 | 基线q50中位数 | SSP5-8.5 q50 | Delta% |
|------|:-----------:|:-----------:|:------:|
| Mali | $127.0 | $87.7 | **-30.9%** |
| Nigeria | $104.2 | $87.8 | **-15.8%** |
| Uganda | $27.4 | $29.1 | +5.9% |
| Tanzania | $30.4 | $31.2 | +2.7% |
| Malawi | $20.8 | $20.9 | +0.7% |
| Ethiopia | $19.9 | $19.9 | +0.3% |

### 10.4 CE结果 (6 scenarios × 6 countries)

| 情景 | Ethiopia | Malawi | Mali | Nigeria | Tanzania | Uganda |
|------|:--------:|:------:|:----:|:-------:|:--------:|:------:|
| **S0 当前** | $34.43 | $31.38 | $181.63 | $133.25 | $30.39 | $33.30 |
| S1 SSP2-4.5 | $34.31 | $31.36 | $119.20 | $118.82 | $30.41 | $36.66 |
| **S2 SSP5-8.5** | $34.19 | $31.37 | $122.14 | $112.31 | $30.41 | $36.69 |
| S3 保险 | $46.18 | $32.57 | $173.35 | $163.01 | $35.31 | $51.63 |
| S4 安全网 | $51.68 | $32.09 | $169.80 | $147.03 | $44.81 | $46.06 |
| **S5 组合** | $60.93 | $33.39 | $200.14 | $180.06 | $53.57 | $56.07 |

### 10.5 气候损失: CE(S0) - CE(S2)

| 国家 | 气候损失 ($) | 95% CI | 损失比例 |
|------|:-----------:|--------|:-------:|
| **Mali** | **$59.34** | [$57.15, $61.14] | **-32.7%** |
| **Nigeria** | **$21.01** | [$19.91, $23.78] | **-15.7%** |
| Ethiopia | $0.25 | [$0.17, $0.34] | -0.7% |
| Malawi | $0.00 | [$0.00, $0.01] | ~0% |
| Tanzania | ~$0 | [-$0.53, $0.00] | ~0% |
| Uganda | **-$3.44** | [-$3.81, -$3.11] | **+10.3%** |

### 10.6 政策价值: CE(policy) - CE(S2)

| 国家 | 保险 ($) | 安全网 ($) | 组合 ($) | 保险/安全网比率 |
|------|:-------:|:---------:|:-------:|:-------------:|
| Mali | +$51.06 | +$47.62 | +$77.96 | 1.07 |
| Nigeria | +$50.57 | +$34.78 | +$67.66 | 1.45 |
| Ethiopia | +$12.00 | +$17.49 | +$26.77 | 0.69 |
| Tanzania | +$6.08 | +$15.06 | +$24.19 | 0.40 |
| Uganda | +$14.93 | +$9.33 | +$19.35 | 1.60 |
| Malawi | +$1.25 | +$0.73 | +$2.03 | 1.72 |

### 10.7 协同效应

| 国家 | 协同 ($) | 协同比例 | 类型 |
|------|:-------:|:-------:|:----:|
| Mali | -$20.66 | -26.7% | 次线性 (替代) |
| Nigeria | -$17.64 | -26.1% | 次线性 (替代) |
| Uganda | -$4.89 | -25.1% | 次线性 (替代) |
| Ethiopia | -$2.75 | -10.3% | 弱次线性 |
| Tanzania | **+$2.92** | **+12.1%** | **超线性 (互补)** |
| Malawi | +$0.06 | +2.7% | ~线性 |

---

## 11. 核心发现摘要

1. **气候损失集中在West Africa**: Mali (-33%) 和 Nigeria (-16%) 承受 >90% 的总福利损失，主要由温度上升(+1.9°C)和NDVI恶化(-20%)驱动
2. **East Africa可能受益**: Uganda (+10%) 由于降雨增加(+6%)和低rho(1.62)，福利反而改善
3. **保险对所有国家有效**: Mali +$51, Nigeria +$51, 但Malawi仅+$1.25（受限于低收入基数$20）
4. **安全网对所有国家有效**: 修正后(income floor model) 无一国出现负效果。Ethiopia(+51%)和Tanzania(+49%)尤其受益
5. **政策排序因国而异**: Ethiopia/Tanzania → 安全网优先; Mali → 两者并重; Nigeria/Uganda → 保险略优
6. **组合干预在大多数国家呈次线性**: 保险和安全网有部分替代性（-20~-26%），唯Tanzania呈超线性(+12%)
7. **一刀切政策的浪费**: 组合干预效果差异38倍 (Mali $78 vs Malawi $2)
8. **风险偏好是政策效果的关键调节变量**: 相同气候冲击 × 不同rho/gamma → 截然不同的政策响应
