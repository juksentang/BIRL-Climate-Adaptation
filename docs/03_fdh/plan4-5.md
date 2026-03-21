# Step 4–5: Environment Model & BIRL — Final Implementation
## 基于全部诊断结果的确定性方案

**状态**: 所有方法论选择已确定，无待定项
**输入**: `birl_sample.parquet` (222,023 obs × 244 cols, 6国, 15,644 HH)
**决策单位**: plot-crop observation (222,023 个)
**行为参数单位**: household (15,644 个, 每户共享 α_i, γ_i, λ_i)

---

## 关键诊断结论（决策依据）

| 诊断 | 结果 | 决策影响 |
|------|------|---------|
| NUTS per-step | **0.9s** (1000HH pilot) | Full NUTS可行，不需要SVI |
| SVI | **ELBO→NaN，彻底失败** | 排除SVI选项 |
| α-γ相关 | **-0.021** | 近乎完美识别，三参数可行 |
| R̂ | 全部 **< 1.005** | NUTS收敛良好 |
| Divergence | **0** | 后验几何健康 |
| Funnel check | **无漏斗** | Non-centered参数化正确工作 |
| ICC | **0.240** | 有效维度~12K，层级模型正确 |
| LogNormal | **skew=-0.03, kurt=0.14** | 参数化分布假设合理 |

---

## 目录

### Step 4: 环境模型
- [4.1 架构总览](#41)
- [4.2 产出变量](#42)
- [4.3 特征工程](#43)
- [4.4 数据拆分](#44)
- [4.5 Model A: 均值模型 (μ)](#45)
- [4.6 Model B: 波动性模型 (σ)](#46)
- [4.7 Optuna调参](#47)
- [4.8 分位数提取与验证](#48)
- [4.9 反事实预测矩阵](#49)
- [4.10 DEA上界约束](#410)
- [4.11 可解释性分析](#411)
- [4.12 鲁棒性: QRF](#412)
- [4.13 模型选择决策](#413)
- [4.14 输出清单](#414)

### Step 5: BIRL
- [5.1 架构总览](#51)
- [5.2 效用函数完整规格](#52)
- [5.3 选择模型 (Softmax)](#53)
- [5.4 三层层级贝叶斯结构](#54)
- [5.5 先验设定与文献依据](#55)
- [5.6 协变量扩展模型](#56)
- [5.7 NumPyro完整实现](#57)
- [5.8 数据准备 (JAX Arrays)](#58)
- [5.9 MCMC配置](#59)
- [5.10 全量运行策略](#510)
- [5.11 收敛诊断](#511)
- [5.12 失败案例决策树](#512)
- [5.13 后验提取与报告](#513)
- [5.14 鲁棒性检验矩阵](#514)
- [5.15 输出清单](#515)

### 附录
- [A. 计算资源汇总](#appendix-a)
- [B. 论文写法指南](#appendix-b)
- [C. 文件结构总览](#appendix-c)

---

<a id="41"></a>
# Step 4: 环境模型

## 4.1 架构总览

### 环境模型的角色

环境模型回答：**"如果农户 i 在状态 s 下选择动作 a，其收入分布是什么？"**

它不关心农户为什么选a——那是BIRL的工作。

```
                    ┌─────────────────────────┐
  状态 s_it ───────▶│  Model A (LightGBM)     │──▶ μ̂ = E[log(y)|s,a]
  动作 a_it ───────▶│  → 预测条件均值          │
                    └─────────────────────────┘
                    ┌─────────────────────────┐
  状态 s_it ───────▶│  Model B (LightGBM)     │──▶ σ̂ = Std[log(y)|s,a]
  动作 a_it ───────▶│  → 预测条件波动性        │
                    └─────────────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │ LogNormal分位数    │
                    │ q10 = exp(μ̂-1.28σ̂)-1 │
                    │ q50 = exp(μ̂)-1       │──▶ BIRL效用函数
                    │ q90 = exp(μ̂+1.28σ̂)-1 │
                    │ 单调性: 数学保证     │
                    └───────────────────┘
```

### 为什么选择LightGBM + LogNormal参数化

| 候选方案 | 分位数单调 | 缺失值 | 速度 | 分布假设 | 选择 |
|---------|:---------:|:------:|:----:|:-------:|:----:|
| **LightGBM LogNormal** | **数学保证** | **原生** | **极快** | LogNormal (已验证) | **主模型** |
| QRF | 数学保证 | 需预处理 | 慢10-60× | 无 | 鲁棒性 |
| LightGBM 3×Quantile | ❌ 需post-hoc修正 | 原生 | 快 | 无 | 排除 |
| NGBoost | 数学保证 | 需预处理 | 中等 | 参数族 | 备选 |

LogNormal假设已通过诊断5验证: 整体skew=-0.03, kurt=0.14, 所有层|skew|<1.3, |kurt|<3.5。

---

<a id="42"></a>
## 4.2 产出变量

```python
y = np.log(df['harvest_value_USD_w'].clip(lower=0.01) + 1)
```

| 决策 | 选择 | 理由 |
|------|------|------|
| 变量 | harvest_value_USD_w | 间作多作物kg不可加总；与BIRL效用函数一致 |
| 变换 | log(y + 1) | 高度右偏→近似正态；LogNormal参数化的基础 |
| 零值 | clip(0.01) → log(0.01+1)=0.01 | 保留零产出obs供模型学习crop failure概率 |
| Winsorization | 已在Step 1完成 | max=$50,000 (winsorized) |

---

<a id="43"></a>
## 4.3 特征工程

### 完整特征列表 (~38维)

```
═══ 动作变量 (反事实预测时替换这些) ═══

  action_crop           分类, 8类     LightGBM原生categorical
  input_intensity       有序, 0/1/2   低/中/高

═══ Plot-level状态 (每个obs独有) ═══

  plot_area_GPS         连续, ha      地块面积
  intercropped          0/1           是否间作 (状态, 非动作)
  plot_owned            0/1           自有地
  irrigated             0/1           灌溉

═══ 气候 (GEE _final, 修正窗口) ═══

  rainfall_growing_sum_final    连续, mm     生长季累计降雨
  rainfall_10yr_mean_final      连续, mm     10年均月降雨
  rainfall_10yr_cv_final        连续         降雨变异系数
  ndvi_preseason_mean_final     连续         季前NDVI
  ndvi_growing_mean_final       连续         生长季NDVI

═══ 温度 (ERA5, 5.5%缺失→LightGBM原生处理) ═══

  era5_tmean_growing_final      连续, °C     生长季日均温
  era5_tmax_growing_final       连续, °C     生长季日最高温

═══ 土壤化学 (ISRIC 0-5cm层, ~6.5%缺失→LightGBM原生) ═══

  clay_0-5cm            连续, %       粘土含量
  sand_0-5cm            连续, %       沙含量
  soc_0-5cm             连续, g/kg    有机碳
  nitrogen_0-5cm        连续, g/kg    土壤总氮 (注意区分化肥nitrogen_kg)
  phh2o_0-5cm           连续          pH值

═══ 地形 (NASADEM) ═══

  elevation_m           连续, m       海拔
  slope_deg             连续, °       坡度

═══ 家庭特征 (household级, 同一HH所有obs共享) ═══

  hh_size               连续          家庭规模
  hh_asset_index        连续          资产指数
  hh_dependency_ratio   连续          抚养比
  age_manager           连续          管理者年龄
  female_manager        0/1           女性管理者
  formal_education_manager  0/1       正规教育
  livestock             0/1           有牲畜
  nonfarm_enterprise    0/1           非农经营
  hh_electricity_access 0/1           通电

═══ 市场与基础设施 ═══

  travel_time_city_min  连续, 分钟    到最近城市旅行时间
  urban                 0/1           城乡

═══ 冲突 (ACLED) ═══

  conflict_events_25km_12m    连续    25km内12月冲突事件数
  conflict_nearest_event_km   连续    最近冲突距离

═══ DEA (来自Step 3) ═══

  dea_efficiency        连续          技术效率 (Order-m FDH)

═══ 时空 ═══

  country               分类, 6类     LightGBM原生categorical
  year                  连续          年份
  season                分类, 2类     季节
```

### LightGBM分类特征声明

```python
categorical_features = ['action_crop', 'country', 'season']
# LightGBM原生处理: 不需要one-hot或target encoding
# 传入字符串列, 在lgb.Dataset中指定categorical_feature参数
```

### 缺失值处理

```
LightGBM原生处理NaN——不需要预处理。
模型自动学习"缺失值该走左子树还是右子树"。

缺失情况:
  era5温度:        5.5% NaN  → LightGBM原生处理
  soil (ISRIC):    6.5% NaN  → LightGBM原生处理
  dea_efficiency:  4.5% NaN  → LightGBM原生处理
  其他特征:        < 1% NaN  → LightGBM原生处理

不需要任何imputer。这是选LightGBM而非QRF的实用优势之一。
```

---

<a id="44"></a>
## 4.4 数据拆分

```python
from sklearn.model_selection import GroupKFold

# ── Household-grouped split ──
# 同一农户的所有obs在同一fold (避免信息泄漏)

unique_hh = df['hh_id_merge'].unique()
rng = np.random.default_rng(42)
rng.shuffle(unique_hh)

n_test = int(len(unique_hh) * 0.15)   # 15% HH for holdout test
test_hh = set(unique_hh[:n_test])
trainval_hh = set(unique_hh[n_test:])

test_mask = df['hh_id_merge'].isin(test_hh)
trainval_mask = ~test_mask

X_trainval, y_trainval = X[trainval_mask], y[trainval_mask]
X_test, y_test = X[test_mask], y[test_mask]
groups_trainval = df.loc[trainval_mask, 'hh_id_merge']

# ── CV folds for Optuna ──
gkf = GroupKFold(n_splits=3)
folds = list(gkf.split(X_trainval, y_trainval, groups=groups_trainval))
```

### 预期拆分规模

```
Train+Val:  ~188,700 obs, ~13,300 HH (85%)
Test:       ~33,300 obs,  ~2,340 HH (15%)
每个CV fold: ~125,800 train / ~62,900 val
```

---

<a id="45"></a>
## 4.5 Model A: 均值模型 (μ)

### 目标

预测 E[log(harvest_value_USD_w + 1) | state, action]

### 实现

```python
import lightgbm as lgb

# 构建Dataset
dtrain = lgb.Dataset(
    X_trainval, y_trainval,
    categorical_feature=categorical_features,
    free_raw_data=False,
)

# 默认参数 (Optuna会覆盖)
base_params_mu = {
    'objective': 'regression',
    'metric': 'rmse',
    'verbose': -1,
    'n_jobs': -1,
    'seed': 42,
}
```

---

<a id="46"></a>
## 4.6 Model B: 波动性模型 (σ)

### 目标

预测 |residual| = |log(y) - μ̂(s,a)|，即条件波动性。

### 实现

```python
# Step 1: 用Model A的最优模型计算残差
mu_pred_trainval = model_mu.predict(X_trainval)
residuals = y_trainval - mu_pred_trainval

# Step 2: 目标变量 = 绝对残差
y_sigma = np.abs(residuals)

# Step 3: 训练sigma模型
dtrain_sigma = lgb.Dataset(
    X_trainval, y_sigma,
    categorical_feature=categorical_features,
    free_raw_data=False,
)

base_params_sigma = {
    'objective': 'regression',
    'metric': 'mae',      # MAE更适合预测绝对残差
    'verbose': -1,
    'n_jobs': -1,
    'seed': 42,
}
```

### σ的下界约束

```python
# 预测后clip: 避免sigma=0导致q10=q50=q90
# 最小sigma=0.1 (在log空间, 对应USD空间约10%的波动)
sigma_pred = model_sigma.predict(X).clip(lower=0.1)
```

### 为什么预测|residual|而非residual²

```
|residual|对异常值更鲁棒:
  - residual² 对大残差赋予极大权重
  - 农业数据有很多异常值（即使winsorize后）
  - MAE loss + |residual|目标 → 对异常值自然鲁棒

替代方案: log(residual²) → 但需要额外处理residual=0的情况
推荐: |residual| + MAE, 简单且有效
```

---

<a id="47"></a>
## 4.7 Optuna调参

### Model A 调参

```python
import optuna

def objective_mu(trial):
    params = base_params_mu.copy()
    params.update({
        'num_leaves':       trial.suggest_int('num_leaves', 63, 255),
        'learning_rate':    trial.suggest_float('lr', 0.01, 0.1, log=True),
        'min_child_samples': trial.suggest_int('min_child', 20, 100, log=True),
        'subsample':        trial.suggest_float('subsample', 0.6, 0.9),
        'colsample_bytree': trial.suggest_float('colsample', 0.5, 0.9),
        'reg_alpha':        trial.suggest_float('reg_alpha', 1e-3, 10, log=True),
        'reg_lambda':       trial.suggest_float('reg_lambda', 1e-3, 10, log=True),
    })

    cv_result = lgb.cv(
        params, dtrain,
        num_boost_round=2000,
        folds=folds,
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
    )
    return cv_result['valid rmse-mean'][-1]

study_mu = optuna.create_study(direction='minimize',
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5))
study_mu.optimize(objective_mu, n_trials=40, show_progress_bar=True)
```

### Model B 调参

```python
def objective_sigma(trial):
    params = base_params_sigma.copy()
    params.update({
        'num_leaves':       trial.suggest_int('num_leaves', 31, 127),
        'learning_rate':    trial.suggest_float('lr', 0.01, 0.1, log=True),
        'min_child_samples': trial.suggest_int('min_child', 30, 200, log=True),
        'subsample':        trial.suggest_float('subsample', 0.6, 0.9),
        'colsample_bytree': trial.suggest_float('colsample', 0.5, 0.9),
        'reg_alpha':        trial.suggest_float('reg_alpha', 1e-3, 10, log=True),
        'reg_lambda':       trial.suggest_float('reg_lambda', 1e-3, 10, log=True),
    })

    cv_result = lgb.cv(
        params, dtrain_sigma,
        num_boost_round=1500,
        folds=folds,
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
    )
    return cv_result['valid mae-mean'][-1]

study_sigma = optuna.create_study(direction='minimize',
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5))
study_sigma.optimize(objective_sigma, n_trials=30, show_progress_bar=True)
```

### 最终模型训练

```python
# Model A
best_params_mu = study_mu.best_params | base_params_mu
best_round_mu = len(lgb.cv(best_params_mu, dtrain, num_boost_round=2000,
                           folds=folds,
                           callbacks=[lgb.early_stopping(50)])['valid rmse-mean'])
model_mu = lgb.train(best_params_mu, dtrain, num_boost_round=best_round_mu)

# Residuals for Model B
mu_pred_all = model_mu.predict(X_trainval)
y_sigma_all = np.abs(y_trainval - mu_pred_all)
dtrain_sigma = lgb.Dataset(X_trainval, y_sigma_all,
                            categorical_feature=categorical_features)

# Model B
best_params_sigma = study_sigma.best_params | base_params_sigma
best_round_sigma = ...  # 同上
model_sigma = lgb.train(best_params_sigma, dtrain_sigma,
                         num_boost_round=best_round_sigma)
```

### 计时预期

```
Model A Optuna: 40 trials × ~2min = ~80min
Model B Optuna: 30 trials × ~2min = ~60min
最终训练:       ~5min
总计:           ~2.5小时 (CPU多核)
```

---

<a id="48"></a>
## 4.8 分位数提取与验证

### 分位数公式

```python
from scipy.stats import norm

z10 = norm.ppf(0.10)   # -1.2816
z50 = 0.0              # median = mean for Normal
z90 = norm.ppf(0.90)   # +1.2816

def extract_quantiles(model_mu, model_sigma, X):
    """从LogNormal参数提取分位数 (在USD空间)。"""
    mu = model_mu.predict(X)
    sigma = model_sigma.predict(X).clip(lower=0.1)

    q10_log = mu + z10 * sigma
    q50_log = mu + z50 * sigma   # = mu
    q90_log = mu + z90 * sigma

    q10_usd = np.exp(q10_log) - 1
    q50_usd = np.exp(q50_log) - 1
    q90_usd = np.exp(q90_log) - 1

    # σ in USD space (正态近似)
    sigma_usd = (q90_usd - q10_usd) / 2.56

    return q10_usd, q50_usd, q90_usd, sigma_usd

q10_test, q50_test, q90_test, sigma_test = extract_quantiles(
    model_mu, model_sigma, X_test)
```

### 单调性保证

```
由数学结构保证:
  sigma > 0 (clip下限0.1)
  z10 < z50 < z90 (-1.28 < 0 < +1.28)
  → q10_log < q50_log < q90_log
  → exp()单调递增
  → q10_usd < q50_usd < q90_usd   ∀ obs

无需任何post-hoc修正。交叉率 = 0%，永远。
```

### 验证指标

```python
from sklearn.metrics import r2_score, mean_absolute_error

# 1. q50 (均值模型) 的OOS R²
r2 = r2_score(y_test, model_mu.predict(X_test))
print(f"Model A q50 OOS R²: {r2:.4f}")
# 预期: 0.30-0.50

# 2. Model B (σ模型) 的OOS MAE
sigma_pred_test = model_sigma.predict(X_test).clip(lower=0.1)
resid_test = np.abs(y_test - model_mu.predict(X_test))
mae_sigma = mean_absolute_error(resid_test, sigma_pred_test)
print(f"Model B σ OOS MAE: {mae_sigma:.4f}")

# 3. Calibration: 实际落在q10以下的比例应≈10%
y_test_usd = np.exp(y_test) - 1
cal_10 = (y_test_usd < q10_test).mean()
cal_50 = (y_test_usd < q50_test).mean()
cal_90 = (y_test_usd < q90_test).mean()
print(f"Calibration: q10={cal_10:.3f}(→0.10), q50={cal_50:.3f}(→0.50), "
      f"q90={cal_90:.3f}(→0.90)")
# 可接受偏差: ±0.03

# 4. Pinball loss
def pinball(y_true, y_pred, tau):
    r = y_true - y_pred
    return np.mean(np.where(r >= 0, tau * r, (tau - 1) * r))

for tau, q in [(0.10, q10_test), (0.50, q50_test), (0.90, q90_test)]:
    pl = pinball(y_test_usd, q, tau)
    print(f"Pinball(τ={tau}): {pl:.4f}")

# 5. 分位数单调性 (应恒为100%)
mono = ((q10_test <= q50_test) & (q50_test <= q90_test)).mean()
print(f"Monotonicity: {mono:.4f}")  # 应为1.0000
```

---

<a id="49"></a>
## 4.9 反事实预测矩阵

### 目标

对每个obs，生成所有feasible动作的收入分布预测。BIRL softmax需要比较不同动作的效用。

### 向量化实现

```python
def generate_counterfactual_matrix(model_mu, model_sigma, df,
                                    feature_cols, action_features,
                                    feasibility_mask, N_actions=24):
    """
    生成 (N_obs × N_actions) 的反事实预测矩阵。
    向量化: 按action循环(24次), 每次预测全部obs。
    """
    N_obs = len(df)

    # 输出矩阵
    q10_matrix = np.full((N_obs, N_actions), np.nan, dtype=np.float32)
    q50_matrix = np.full((N_obs, N_actions), np.nan, dtype=np.float32)
    q90_matrix = np.full((N_obs, N_actions), np.nan, dtype=np.float32)
    sigma_matrix = np.full((N_obs, N_actions), np.nan, dtype=np.float32)

    # 状态特征 (不含动作特征)
    state_cols = [c for c in feature_cols if c not in action_features]

    for action_id in range(N_actions):
        crop_idx = action_id // 3
        intensity = action_id % 3
        crop_name = idx_to_crop[crop_idx]

        # 构造反事实特征: 替换动作, 保持状态
        X_cf = df[state_cols].copy()
        X_cf['action_crop'] = crop_name
        X_cf['input_intensity'] = intensity

        # 预测
        mu_cf = model_mu.predict(X_cf)
        sigma_cf = model_sigma.predict(X_cf).clip(lower=0.1)

        q10_cf = np.exp(mu_cf + z10 * sigma_cf) - 1
        q50_cf = np.exp(mu_cf) - 1
        q90_cf = np.exp(mu_cf + z90 * sigma_cf) - 1
        sigma_usd_cf = (q90_cf - q10_cf) / 2.56

        q10_matrix[:, action_id] = q10_cf
        q50_matrix[:, action_id] = q50_cf
        q90_matrix[:, action_id] = q90_cf
        sigma_matrix[:, action_id] = sigma_usd_cf

    # 应用feasibility mask: infeasible → NaN
    for zone_id in range(feasibility_mask.shape[0]):
        zone_obs = (df['zone_idx'] == zone_id).values
        for a in range(N_actions):
            if not feasibility_mask[zone_id, a]:
                q10_matrix[zone_obs, a] = np.nan
                q50_matrix[zone_obs, a] = np.nan
                q90_matrix[zone_obs, a] = np.nan
                sigma_matrix[zone_obs, a] = np.nan

    return q10_matrix, q50_matrix, q90_matrix, sigma_matrix

# 执行
q10_mat, q50_mat, q90_mat, sigma_mat = generate_counterfactual_matrix(
    model_mu, model_sigma, df, feature_cols,
    action_features=['action_crop', 'input_intensity'],
    feasibility_mask=mask_array,
)
```

### 性能

```
24次LightGBM predict × 222K obs = 5.3M predictions
LightGBM predict速度: ~0.1s per 222K → 总计 ~3秒
加上特征构造开销: ~30秒

对比QRF: 同样操作需要10-30分钟
```

---

<a id="410"></a>
## 4.10 DEA上界约束

```python
# q90不应超过DEA前沿值 × 1.1 (10%容差)
frontier = df['dea_frontier_value_USD'].values[:, None]  # (N_obs, 1)
frontier_cap = frontier * 1.1

# 对每个action的q90做soft clip
q90_mat_clipped = np.minimum(q90_mat, np.broadcast_to(frontier_cap, q90_mat.shape))

# 同步更新sigma
sigma_mat = np.where(
    np.isnan(q90_mat_clipped), np.nan,
    (q90_mat_clipped - q10_mat).clip(min=0) / 2.56
)

# DEA缺失的obs (4.5%) 不做clip
q90_mat = np.where(np.isnan(frontier_cap), q90_mat, q90_mat_clipped)
```

---

<a id="411"></a>
## 4.11 可解释性分析

### SHAP

```python
import shap

explainer = shap.TreeExplainer(model_mu)
shap_sample = X_test.sample(n=5000, random_state=42)
shap_values = explainer.shap_values(shap_sample)

shap.summary_plot(shap_values, shap_sample, feature_names=feature_cols,
                  max_display=15, show=False)
plt.tight_layout()
plt.savefig('figures/shap_summary_mu.png', dpi=150)
```

### 偏依赖图 (关键关系)

```
必须检查的关系 (应与农学知识一致):
  1. rainfall_growing_sum → yield: 先升后平或倒U型
  2. era5_tmean_growing → yield: 倒U型 (高温胁迫)
  3. input_intensity → yield: 单调递增 (但边际递减)
  4. plot_area_GPS → yield per ha: 反向 (inverse farm size)
  5. conflict_events_25km → yield: 负向
  6. elevation_m → yield: 因作物而异

如果某个关系方向不符合农学知识 → 检查数据质量或特征工程
```

### Calibration plot

```python
quantiles_check = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
for tau in quantiles_check:
    z = norm.ppf(tau)
    q_log = mu_test + z * sigma_test_pred
    q_usd = np.exp(q_log) - 1
    actual = (y_test_usd < q_usd).mean()
    print(f"τ={tau:.2f}: expected={tau:.2f}, actual={actual:.3f}")

# 绘制calibration图
# x轴: 名义分位数, y轴: 实际覆盖率
# 理想: 45°线
```

---

<a id="412"></a>
## 4.12 鲁棒性: QRF

```python
from quantile_forest import RandomForestQuantileRegressor
from sklearn.impute import SimpleImputer

# QRF不支持NaN → 中位数填补 + 缺失指示变量
imputer = SimpleImputer(strategy='median')
X_trainval_imp = imputer.fit_transform(X_trainval_numeric)
X_test_imp = imputer.transform(X_test_numeric)

# QRF分类特征需要编码
from sklearn.preprocessing import OrdinalEncoder
oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X_trainval_cat = oe.fit_transform(X_trainval[categorical_features])

# 合并
X_trainval_qrf = np.hstack([X_trainval_imp, X_trainval_cat, X_trainval_missing_flags])
X_test_qrf = np.hstack([X_test_imp, X_test_cat, X_test_missing_flags])

# Optuna调参 (简化, 重点参数)
def objective_qrf(trial):
    model = RandomForestQuantileRegressor(
        n_estimators=trial.suggest_int('n_est', 300, 800, step=100),
        max_depth=trial.suggest_int('max_depth', 12, 25),
        min_samples_leaf=trial.suggest_int('min_leaf', 20, 80, log=True),
        max_features=trial.suggest_float('max_feat', 0.3, 0.7),
        n_jobs=-1, random_state=42,
    )
    model.fit(X_tr_qrf, y_tr)
    preds = model.predict(X_va_qrf, quantiles=[0.10, 0.50, 0.90])

    loss_10 = pinball(y_va, np.exp(preds[:,0])-1, 0.10)
    loss_50 = pinball(y_va, np.exp(preds[:,1])-1, 0.50)
    loss_90 = pinball(y_va, np.exp(preds[:,2])-1, 0.90)
    return 2.0*loss_10 + 1.0*loss_50 + 0.5*loss_90

study_qrf = optuna.create_study(direction='minimize')
study_qrf.optimize(objective_qrf, n_trials=40)  # ~7-8小时

# 最终QRF训练 + 反事实预测 (慢, ~10-30min)
# 对比LightGBM结果
```

---

<a id="413"></a>
## 4.13 模型选择决策

```python
# 关键对比
print("=== Model Comparison ===")
print(f"q50 R²:  LGBM={r2_lgbm:.4f}, QRF={r2_qrf:.4f}")
print(f"q10 cal: LGBM={cal10_lgbm:.3f}, QRF={cal10_qrf:.3f}")
print(f"q50 cal: LGBM={cal50_lgbm:.3f}, QRF={cal50_qrf:.3f}")

# Counterfactual matrix一致性
from scipy.stats import spearmanr
r_q50, _ = spearmanr(q50_mat_lgbm.flatten(), q50_mat_qrf.flatten())
r_q10, _ = spearmanr(q10_mat_lgbm.flatten(), q10_mat_qrf.flatten())
print(f"CF matrix Spearman: q50={r_q50:.4f}, q10={r_q10:.4f}")
```

### 决策规则

```
如果 |R²_LGBM - R²_QRF| < 0.02 且 CF Spearman > 0.95:
  → 选LGBM (速度优势, 无缺失值预处理)
  → 报告QRF一致性确认分布假设不驱动结果

如果 R²_QRF > R²_LGBM + 0.05:
  → 选QRF (精度差距大到不能忽略)
  → LogNormal假设可能在某些层不成立

通常: LGBM和QRF差距极小, 选LGBM
```

---

<a id="414"></a>
## 4.14 输出清单

### 文件

```
Formal Analysis/04_Environment_Model/
├── train_env_model.py              # 主训练脚本
├── generate_counterfactuals.py     # 反事实预测
├── train_qrf_robustness.py         # QRF鲁棒性
├── env_model_report.md             # 完整报告
├── models/
│   ├── lgbm_mu.txt                 # Model A (booster)
│   ├── lgbm_sigma.txt              # Model B (booster)
│   ├── optuna_mu.pkl               # Optuna study A
│   ├── optuna_sigma.pkl            # Optuna study B
│   └── qrf_robustness.joblib       # QRF模型 (鲁棒性)
├── predictions/
│   ├── cf_q10_matrix.npy           # (222023, 24) float32
│   ├── cf_q50_matrix.npy           # (222023, 24) float32
│   ├── cf_q90_matrix.npy           # (222023, 24) float32
│   ├── cf_sigma_matrix.npy         # (222023, 24) float32
│   └── test_predictions.parquet    # Test set预测
└── figures/
    ├── shap_summary_mu.png
    ├── shap_summary_sigma.png
    ├── calibration_plot.png
    ├── pdp_rainfall.png
    ├── pdp_temperature.png
    └── lgbm_vs_qrf_scatter.png
```

### 验证清单

```
□ Model A q50 OOS R² > 0.25
□ Calibration偏差 < 0.03 (所有τ)
□ 分位数单调率 = 100%
□ SHAP前5特征包含rainfall和plot_area
□ 偏依赖图方向符合农学知识
□ DEA clip影响 < 5%的obs
□ 反事实矩阵无NaN (除infeasible actions外)
□ q50, q10, q90 > 0 (除零产出obs外)
□ LGBM vs QRF: CF Spearman > 0.90
□ 反事实矩阵文件 < 500MB (npy格式)
```

---

<a id="51"></a>
# Step 5: BIRL层级贝叶斯推断

## 5.1 架构总览

### 确定方案

```
推断方法:    Full NUTS (SVI已排除——ELBO→NaN)
层级结构:    三层 (全局 → 国家 → 农户)
参数化:      Non-centered (pilot确认有效)
参数:        3个/户 (α, γ, λ)
总参数:      ~52K名义, ~12K有效 (ICC=0.24)
硬件:        TPU V5e
预计时间:    12-20小时 (全量)
```

### 决策依据

```
Per-step = 0.9s (pilot 1000 HH) → Full NUTS可行
SVI ELBO→NaN → 排除SVI
α-γ相关 = -0.021 → 三参数识别良好
R̂ < 1.005, 0 divergences → 后验几何健康
ICC = 0.24 → 有效维度12K, 层级模型必要
```

---

<a id="52"></a>
## 5.2 效用函数完整规格

### 总效用

```
r_i(s, a; θ_i) = u(μ_subj) - γ_i · DownsideRisk_i(a) - λ_i · LiquidityStress_i(a)
```

### CRRA效用

```
u(y) = y^(1-ρ) / (1-ρ)

ρ = 1.5 (固定)
来源: Rosenzweig & Binswanger (1993)
鲁棒性: ρ ∈ {1.0, 1.5, 2.0}

为什么固定ρ:
  ρ和γ同时自由→识别混淆（两者都刻画风险态度的不同方面）
  固定ρ→γ专门刻画对下尾的额外关注
```

### 主观收入预期

```
μ_subj(s, a; α_i) = q50(s, a) - κ_μ · α_i · σ(s, a)

q50, σ: 来自Step 4环境模型的反事实预测
α_i:    信念偏差 (>0=悲观, <0=乐观, =0无偏)
κ_μ:    1.0 (固定, 标准化尺度)

机制: 悲观农户(α>0)系统性低估预期收入
     → 主观μ比客观q50低
     → 波动性σ越大, 偏差越大
```

### 下尾风险 (DownsideRisk)

```
DownsideRisk_i(a) = max{0, ȳ_i - q10_subj(s, a)}

q10_subj = q10(s,a) - κ_q · α_i · σ(s,a)
κ_q = 1.5 (固定, 下尾对偏差更敏感)

ȳ_i = 同(country, action_crop)层中harvest_value_USD_w的P25
      (Safety-First生存阈值, 参照Fafchamps 1992, Dercon 1996)
      敏感性: P10, P25, P50

触发条件: 当q10_subj < ȳ_i时, DownsideRisk > 0, γ_i可识别
触发率: ~25% obs (方案B, P25阈值)
```

### 流动性压力 (LiquidityStress)

```
LiquidityStress_i(a) = max{0, MinCons_i - μ_subj(s, a)}

MinCons_i = totcons_USD的P10 by (country, cons_quint) × plot_share
            缺失时: country × urban中位数 × plot_share

触发条件: 当μ_subj < MinCons_i时, 压力 > 0, λ_i可识别
```

### 固定参数汇总

| 参数 | 值 | 来源 | 鲁棒性 |
|------|---:|------|--------|
| ρ (CRRA) | 1.5 | Rosenzweig & Binswanger 1993 | {1.0, 1.5, 2.0} |
| β (温度) | 5.0 | IRL文献标准 | {2.0, 5.0, 10.0} |
| κ_μ | 1.0 | 标准化 | — |
| κ_q | 1.5 | 下尾更敏感 | {1.0, 1.5, 2.0} |

---

<a id="53"></a>
## 5.3 选择模型 (Softmax with Action Mask)

```
π(a | s; θ_i) = exp{β · r_i(s, a; θ_i)} · I_feas(a, zone_i)
                ─────────────────────────────────────────────
                Σ_{a'} exp{β · r_i(s, a'; θ_i)} · I_feas(a', zone_i)

β = 5.0: 效用差1单位 → 选择概率比 e^5 ≈ 148:1
I_feas: 动作掩码, infeasible动作logit = -∞

似然 (plot-crop level):
  L_i = Π_t Π_s Π_p π(a_{itsp} | s_{itsp}; θ_i)

  t: wave, s: season, p: plot-crop
  θ_i 在同一农户所有决策点间共享
```

---

<a id="54"></a>
## 5.4 三层层级贝叶斯结构

### 为什么三层

```
两层问题: Mali(2轮)和Uganda(7轮)共享一个全局先验→不合适
三层方案: 全局 → 国家(6个) → 农户(15,644个)

信息流:
  Uganda丰富数据 → 稳定全局层
  全局层 → 约束Mali的国家层 (数据少, 需要借力)
  Mali国家层 → 约束Mali农户的参数 (强shrinkage)
  Nigeria国家层 → 较弱约束 (ICC=0.40, 个体参数有信息)
```

### 完整模型

```
══ 全局层 (6个参数) ══

μ_γ^G  ~ Normal(0.5, 0.5)      σ_γ^G  ~ HalfNormal(0.3)
μ_α^G  ~ Normal(0.0, 0.2)      σ_α^G  ~ HalfNormal(0.15)
μ_λ^G  ~ Normal(0.3, 0.3)      σ_λ^G  ~ HalfNormal(0.2)

══ 国家层 (36个参数: 6国 × 6) ══ [Non-centered]

μ_γ^c_raw ~ Normal(0,1)  →  μ_γ^c = μ_γ^G + σ_γ^G · μ_γ^c_raw
σ_γ^c     ~ HalfNormal(0.2)
(同理 α, λ)

══ 农户层 (46,932个参数: 15,644 × 3) ══ [Non-centered]

γ_raw_i ~ Normal(0,1)  →  γ_i = μ_γ^{c(i)} + σ_γ^{c(i)} · γ_raw_i
(同理 α, λ)

══ 似然 ══

a_{obs} ~ Categorical(logits = β · reward_matrix · mask)
```

---

<a id="55"></a>
## 5.5 先验设定与文献依据

| 参数 | 先验 | 经济含义 | 文献 |
|------|------|---------|------|
| μ_γ^G | N(0.5, 0.5) | SSA小农普遍风险厌恶 | Binswanger 1980; Yesuf & Bluffstone 2009 |
| σ_γ^G | HalfN(0.3) | 国家间差异允许适度变化 | — |
| μ_α^G | N(0, 0.2) | 不预设偏差方向 | 弱信息先验 |
| σ_α^G | HalfN(0.15) | 国家间信念差异较小 | — |
| μ_λ^G | N(0.3, 0.3) | SSA信贷约束普遍 | Fafchamps 2003; Karlan et al. 2014 |
| σ_λ^G | HalfN(0.2) | 国家间流动性差异 | — |

先验均为弱信息: 2σ范围覆盖合理参数空间全域。

---

<a id="56"></a>
## 5.6 协变量扩展模型

### 基础模型 (主结果)

θ_i的国家层均值不依赖观测特征。

### 扩展模型 (机制分析)

```
μ_γ^c(i) = β_γ0^c + β_γ1·female + β_γ2·asset + β_γ3·conflict + β_γ4·age
μ_α^c(i) = β_α0^c + β_α1·education + β_α2·rainfall_cv + β_α3·electricity
μ_λ^c(i) = β_λ0^c + β_λ1·asset + β_λ2·nonfarm + β_λ3·travel_time

β系数先验: Normal(0, 0.5) (弱信息)
仅在基础模型收敛后运行。
```

---

<a id="57"></a>
## 5.7 NumPyro完整实现

```python
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro import plate, sample, deterministic

def birl_model(
    obs_action,            # (N_obs,) int32
    obs_hh_idx,            # (N_obs,) int32 → [0, N_hh)
    obs_country_idx,       # (N_obs,) int32 → [0, 6)
    hh_country_idx,        # (N_hh,) int32 → [0, 6)
    obs_zone,              # (N_obs,) int32 → [0, N_zones)
    feasibility_mask,      # (N_zones, N_actions) bool
    cf_q10,                # (N_obs, N_actions) float32
    cf_q50,                # (N_obs, N_actions) float32
    cf_sigma,              # (N_obs, N_actions) float32
    survival_threshold,    # (N_obs,) float32
    min_consumption,       # (N_obs,) float32
    N_hh=15644,
    N_country=6,
):
    N_obs, N_actions = cf_q50.shape

    # Fixed params
    rho = 1.5
    beta = 5.0
    kappa_mu = 1.0
    kappa_q = 1.5

    # ═══ Global layer ═══
    mu_gamma_G = sample("mu_gamma_G", dist.Normal(0.5, 0.5))
    sigma_gamma_G = sample("sigma_gamma_G", dist.HalfNormal(0.3))
    mu_alpha_G = sample("mu_alpha_G", dist.Normal(0.0, 0.2))
    sigma_alpha_G = sample("sigma_alpha_G", dist.HalfNormal(0.15))
    mu_lambda_G = sample("mu_lambda_G", dist.Normal(0.3, 0.3))
    sigma_lambda_G = sample("sigma_lambda_G", dist.HalfNormal(0.2))

    # ═══ Country layer (non-centered) ═══
    with plate("countries", N_country):
        mu_gamma_C_raw = sample("mu_gamma_C_raw", dist.Normal(0, 1))
        sigma_gamma_C = sample("sigma_gamma_C", dist.HalfNormal(0.2))
        mu_alpha_C_raw = sample("mu_alpha_C_raw", dist.Normal(0, 1))
        sigma_alpha_C = sample("sigma_alpha_C", dist.HalfNormal(0.1))
        mu_lambda_C_raw = sample("mu_lambda_C_raw", dist.Normal(0, 1))
        sigma_lambda_C = sample("sigma_lambda_C", dist.HalfNormal(0.15))

    mu_gamma_C = deterministic("mu_gamma_C",
                               mu_gamma_G + sigma_gamma_G * mu_gamma_C_raw)
    mu_alpha_C = deterministic("mu_alpha_C",
                               mu_alpha_G + sigma_alpha_G * mu_alpha_C_raw)
    mu_lambda_C = deterministic("mu_lambda_C",
                                mu_lambda_G + sigma_lambda_G * mu_lambda_C_raw)

    # ═══ Household layer (non-centered) ═══
    with plate("households", N_hh):
        gamma_raw = sample("gamma_raw", dist.Normal(0, 1))
        alpha_raw = sample("alpha_raw", dist.Normal(0, 1))
        lambda_raw = sample("lambda_raw", dist.Normal(0, 1))

    gamma_i = deterministic("gamma_i",
        mu_gamma_C[hh_country_idx] + sigma_gamma_C[hh_country_idx] * gamma_raw)
    alpha_i = deterministic("alpha_i",
        mu_alpha_C[hh_country_idx] + sigma_alpha_C[hh_country_idx] * alpha_raw)
    lambda_i = deterministic("lambda_i",
        mu_lambda_C[hh_country_idx] + sigma_lambda_C[hh_country_idx] * lambda_raw)

    # ═══ Map to obs level ═══
    gamma_obs = gamma_i[obs_hh_idx][:, None]     # (N_obs, 1)
    alpha_obs = alpha_i[obs_hh_idx][:, None]
    lambda_obs = lambda_i[obs_hh_idx][:, None]

    # ═══ Subjective income moments ═══
    mu_subj = cf_q50 - kappa_mu * alpha_obs * cf_sigma
    q10_subj = cf_q10 - kappa_q * alpha_obs * cf_sigma

    # ═══ Utility ═══
    mu_subj_safe = jnp.clip(mu_subj, 1e-6, None)
    u_income = mu_subj_safe ** (1 - rho) / (1 - rho)

    y_bar = survival_threshold[:, None]
    downside = jnp.maximum(0.0, y_bar - q10_subj)

    mincons = min_consumption[:, None]
    liq_stress = jnp.maximum(0.0, mincons - mu_subj)

    reward = u_income - gamma_obs * downside - lambda_obs * liq_stress

    # ═══ Action mask + softmax ═══
    obs_mask = feasibility_mask[obs_zone]
    logits = jnp.where(obs_mask, beta * reward, -1e10)

    # ═══ Likelihood ═══
    with plate("observations", N_obs):
        sample("obs_action", dist.Categorical(logits=logits), obs=obs_action)
```

---

<a id="58"></a>
## 5.8 数据准备 (JAX Arrays)

```python
import jax.numpy as jnp

# ── Index mappings ──
hh_list = df['hh_id_merge'].unique()
hh_to_idx = {h: i for i, h in enumerate(hh_list)}
df['hh_idx'] = df['hh_id_merge'].map(hh_to_idx).astype(int)

countries = sorted(df['country'].unique())
country_to_idx = {c: i for i, c in enumerate(countries)}
df['country_idx'] = df['country'].map(country_to_idx).astype(int)

hh_info = df.drop_duplicates('hh_id_merge').sort_values('hh_idx')
hh_country_idx = hh_info['country_idx'].values

# ── JAX arrays ──
obs_action = jnp.array(df['action_id'].values, dtype=jnp.int32)
obs_hh_idx = jnp.array(df['hh_idx'].values, dtype=jnp.int32)
obs_country_idx = jnp.array(df['country_idx'].values, dtype=jnp.int32)
obs_zone = jnp.array(df['zone_idx'].values, dtype=jnp.int32)
hh_country_idx_jax = jnp.array(hh_country_idx, dtype=jnp.int32)

# ── Counterfactual matrices (from Step 4) ──
cf_q10 = jnp.array(np.nan_to_num(q10_mat, nan=0.0), dtype=jnp.float32)
cf_q50 = jnp.array(np.nan_to_num(q50_mat, nan=0.0), dtype=jnp.float32)
cf_sigma = jnp.array(np.nan_to_num(sigma_mat, nan=1.0), dtype=jnp.float32)
# NaN→0 for q10/q50 (infeasible, masked out by -1e10 logit)
# NaN→1 for sigma (avoid 0×alpha=0 masking alpha's effect on feasible actions)

# ── Feasibility mask ──
feasibility_mask = jnp.array(mask_array, dtype=jnp.bool_)

# ── Thresholds ──
survival_threshold = jnp.array(
    df['survival_threshold_B'].fillna(0).values, dtype=jnp.float32)
min_consumption = jnp.array(
    df['min_consumption'].fillna(0).values, dtype=jnp.float32)

# ── Dimensions ──
N_hh = len(hh_list)
N_country = len(countries)
N_obs = len(df)
N_actions = mask_array.shape[1]

print(f"N_obs={N_obs:,}, N_hh={N_hh:,}, N_country={N_country}")
print(f"N_actions={N_actions}, N_zones={mask_array.shape[0]}")
```

### 内存估算

```
cf_q10 + cf_q50 + cf_sigma: 3 × 222K × 24 × 4B = 64 MB
Index arrays: ~5 MB
Total data: ~70 MB
NUTS states: ~200-500 MB
Posterior storage: 52K params × 20K samples × 4B = 4.2 GB
Total: ~5 GB → TPU V5e (16GB HBM) OK
```

---

<a id="59"></a>
## 5.9 MCMC配置

```python
kernel = NUTS(
    birl_model,
    target_accept_prob=0.8,
    max_tree_depth=10,
    init_strategy=numpyro.infer.init_to_median(),
)

mcmc = MCMC(
    kernel,
    num_warmup=2000,
    num_samples=5000,
    num_chains=4,
    chain_method='parallel',
    progress_bar=True,
)

mcmc.run(
    jax.random.PRNGKey(42),
    obs_action=obs_action,
    obs_hh_idx=obs_hh_idx,
    obs_country_idx=obs_country_idx,
    hh_country_idx=hh_country_idx_jax,
    obs_zone=obs_zone,
    feasibility_mask=feasibility_mask,
    cf_q10=cf_q10,
    cf_q50=cf_q50,
    cf_sigma=cf_sigma,
    survival_threshold=survival_threshold,
    min_consumption=min_consumption,
    N_hh=N_hh,
    N_country=N_country,
)
```

---

<a id="510"></a>
## 5.10 全量运行策略

### 时间估算

```
Pilot (1000 HH, 14K obs): 0.9s/step
全量 (15,644 HH, 222K obs): 预估 3-8s/step

保守估计 (5s/step):
  4 chains × 7000 steps × 5s = 140,000s = 38.9h → 可能太长

乐观估计 (3s/step):
  4 chains × 7000 steps × 3s = 84,000s = 23.3h → 可接受

紧凑配置 (如果太慢):
  warmup=1500, samples=3000, 4 chains
  4 × 4500 × 5s = 90,000s = 25h
```

### 分阶段执行

```
Phase 1: Pilot全量计时 (30分钟)
  100 warmup + 100 samples, 1 chain, 全量数据
  → 确认真实per-step时间
  → 决定warmup和samples数量

Phase 2: 正式运行
  基于Phase 1的计时选择配置:
    < 3s: 2000 warmup + 5000 samples
    3-5s: 1500 warmup + 3000 samples
    5-8s: 1000 warmup + 2000 samples
    > 8s: 降维 (decision-point level或子采样)

Phase 3: 检查点
  每500 samples保存checkpoint
  如果Colab断开, 可从checkpoint恢复
```

### 降维Fallback (如果>8s/step)

```
Fallback A: 子采样obs (推荐)
  每户保留最多8个obs (随机)
  222K → ~100K obs
  per-step约减半

Fallback B: Decision-point level
  每个(hh, wave, season)只保留主作物
  222K → ~42K obs
  per-step约减5倍

Fallback C: 子采样HH
  只用10K HH (随机)
  per-step按比例减少
  鲁棒性中用不同10K子集交叉验证
```

---

<a id="511"></a>
## 5.11 收敛诊断

```python
import arviz as az

idata = az.from_numpyro(mcmc)

# ── R̂ ──
rhat = az.rhat(idata)
global_params = ['mu_gamma_G','mu_alpha_G','mu_lambda_G',
                 'sigma_gamma_G','sigma_alpha_G','sigma_lambda_G']
for p in global_params:
    print(f"R̂({p}): {float(rhat[p]):.4f}")
# 目标: 全部 < 1.01

# ── ESS ──
ess_bulk = az.ess(idata, method='bulk')
ess_tail = az.ess(idata, method='tail')
for p in global_params:
    print(f"ESS_bulk({p}): {float(ess_bulk[p]):.0f}, "
          f"ESS_tail: {float(ess_tail[p]):.0f}")
# 目标: bulk > 400, tail > 200

# ── Divergence ──
div = mcmc.get_extra_fields()['diverging']
div_rate = div.sum() / div.size
print(f"Divergent: {div.sum()}/{div.size} ({div_rate:.4f})")
# 目标: < 1%

# ── Trace plots ──
az.plot_trace(idata, var_names=global_params)
plt.savefig('diagnostics/trace_global.png', dpi=150, bbox_inches='tight')

# ── α-γ posterior correlation ──
posterior = mcmc.get_samples()
alpha_all = np.array(posterior['alpha_i'])    # (N_samples, N_hh)
gamma_all = np.array(posterior['gamma_i'])
corrs = [np.corrcoef(alpha_all[:,i], gamma_all[:,i])[0,1]
         for i in range(min(N_hh, 2000))]
print(f"α-γ corr: mean={np.mean(corrs):.3f}, std={np.std(corrs):.3f}")
# 目标: |mean| < 0.3 (pilot已确认: -0.021)

# ── Posterior Predictive Check ──
from numpyro.infer import Predictive
pred = Predictive(birl_model, posterior_samples=mcmc.get_samples(),
                  num_samples=1000)
pred_actions = np.array(pred(jax.random.PRNGKey(99), ...)['obs_action'])

for a in range(N_actions):
    p = (pred_actions == a).mean()
    q = (np.array(obs_action) == a).mean()
    if q > 0.01:
        print(f"Action {a}: pred={p:.3f}, actual={q:.3f}")

# KL divergence
from scipy.stats import entropy
pred_dist = np.bincount(pred_actions.flatten(), minlength=N_actions)
pred_dist = pred_dist / pred_dist.sum()
actual_dist = np.bincount(np.array(obs_action), minlength=N_actions)
actual_dist = actual_dist / actual_dist.sum()
kl = entropy(actual_dist + 1e-10, pred_dist + 1e-10)
print(f"KL(actual || predicted): {kl:.4f}")
# 目标: < 0.1
```

---

<a id="512"></a>
## 5.12 失败案例决策树

```
R̂ > 1.05?
  ├─ 是 → 哪些参数?
  │   ├─ α和γ → 识别混淆 → 固定α=0, 只估(γ,λ) → 重跑
  │   ├─ σ系列 → Non-centered不够 → 增加warmup到3000
  │   └─ 全部 → 收紧先验 (σ_α: 0.15→0.05)
  └─ 否 → OK

ESS < 50?
  ├─ 是 → 检查non-centered是否生效
  │   ├─ 是 → 减少N_obs (Fallback A/B)
  │   └─ 否 → 修复参数化
  └─ 否 → OK

Divergence > 10%?
  ├─ 是 → target_accept_prob 0.8→0.95
  │   ├─ 仍>10% → 检查reward数值 → clip到[-100,100]
  │   └─ <5% → 继续
  └─ 否 → OK

Per-step > 8s (全量)?
  ├─ 是 → Fallback A: 子采样到~100K obs → 重测per-step
  │   ├─ 仍>8s → Fallback B: decision-point level (~42K obs)
  │   └─ <5s → 用子采样版本
  └─ 否 → 直接跑
```

---

<a id="513"></a>
## 5.13 后验提取与报告

### 农户级参数

```python
posterior = mcmc.get_samples()
gamma_samples = np.array(posterior['gamma_i'])  # (20000, 15644)
alpha_samples = np.array(posterior['alpha_i'])
lambda_samples = np.array(posterior['lambda_i'])

hh_params = pd.DataFrame({
    'hh_id_merge':       hh_list,
    'country':           hh_countries,

    'gamma_mean':        gamma_samples.mean(0),
    'gamma_std':         gamma_samples.std(0),
    'gamma_q025':        np.quantile(gamma_samples, 0.025, axis=0),
    'gamma_q975':        np.quantile(gamma_samples, 0.975, axis=0),

    'alpha_mean':        alpha_samples.mean(0),
    'alpha_std':         alpha_samples.std(0),
    'alpha_q025':        np.quantile(alpha_samples, 0.025, axis=0),
    'alpha_q975':        np.quantile(alpha_samples, 0.975, axis=0),
    'alpha_positive_prob': (alpha_samples > 0).mean(0),

    'lambda_mean':       lambda_samples.mean(0),
    'lambda_std':        lambda_samples.std(0),
    'lambda_q025':       np.quantile(lambda_samples, 0.025, axis=0),
    'lambda_q975':       np.quantile(lambda_samples, 0.975, axis=0),
})
```

### 国家级参数

```python
country_params = pd.DataFrame({
    'country':       countries,
    'mu_gamma':      posterior['mu_gamma_C'].mean(0),
    'mu_gamma_q025': np.quantile(posterior['mu_gamma_C'], 0.025, axis=0),
    'mu_gamma_q975': np.quantile(posterior['mu_gamma_C'], 0.975, axis=0),
    'mu_alpha':      posterior['mu_alpha_C'].mean(0),
    'mu_alpha_q025': np.quantile(posterior['mu_alpha_C'], 0.025, axis=0),
    'mu_alpha_q975': np.quantile(posterior['mu_alpha_C'], 0.975, axis=0),
    'sigma_gamma':   posterior['sigma_gamma_C'].mean(0),
    'sigma_alpha':   posterior['sigma_alpha_C'].mean(0),
})
```

### 关键可视化

```
1. 后验密度: α, γ, λ 全体分布 + 按国家
2. 空间地图: α_mean, γ_mean 在GPS坐标上 (geopandas)
3. 异质性: 按gender, wealth, education, conflict分组箱线图
4. α-γ联合后验: 散点图 (识别性)
5. Shrinkage: 按obs/HH分组, 后验std vs group size
6. 国家对比: 6国的(μ_α, μ_γ)后验及95%CI
```

---

<a id="514"></a>
## 5.14 鲁棒性检验矩阵

| # | 变体 | 改动 | 目的 |
|---|------|------|------|
| R1 | 两参数 | 只估(α,γ), λ=国家级常数 | 检验三参数是否过参数化 |
| R2 | β=2.0 | 更随机的选择模型 | β敏感性 |
| R3 | β=10.0 | 更理性的选择模型 | β敏感性 |
| R4 | ρ=1.0 | 更低的曲率 | CRRA敏感性 |
| R5 | ρ=2.0 | 更高的曲率 | CRRA敏感性 |
| R6 | 阈值=P10 | 更极端的生存线 | 阈值敏感性 |
| R7 | 阈值=P50 | 更宽松的生存线 | 阈值敏感性 |
| R8 | Decision-point level | 每(hh,wave,season)取主作物 | 决策单位敏感性 |
| R9 | QRF反事实 | 用QRF替代LGBM | 环境模型敏感性 |

每个变体报告: α,γ后验均值与主结果的Spearman相关。
目标: ≥6/9个变体相关 > 0.85。

---

<a id="515"></a>
## 5.15 输出清单

### 文件

```
Formal Analysis/05_BIRL/
├── prepare_birl_data.py           # 数据准备
├── run_birl.py                    # 主MCMC (Colab/TPU)
├── run_birl_robustness.py         # 鲁棒性变体
├── extract_posterior.py           # 后验提取
├── birl_diagnostics.py            # 诊断
├── birl_report.md                 # 完整报告
├── data/
│   └── birl_jax_inputs.npz        # JAX arrays
├── posterior/
│   ├── birl_posterior.nc           # ArviZ NetCDF
│   ├── birl_hh_params.parquet     # 农户级汇总
│   ├── birl_country_params.csv    # 国家级汇总
│   └── birl_global_params.csv     # 全局参数
├── diagnostics/
│   ├── trace_global.png
│   ├── trace_country.png
│   ├── rhat_summary.csv
│   ├── ppc_action_freq.csv
│   ├── alpha_gamma_joint.png
│   └── convergence_report.md
└── figures/
    ├── posterior_density_by_country.png
    ├── spatial_map_alpha.png
    ├── spatial_map_gamma.png
    ├── heterogeneity_boxplots.png
    └── shrinkage_vs_nobs.png
```

### 最终验证清单

```
□ 所有全局+国家参数 R̂ < 1.01
□ 所有全局+国家参数 bulk_ESS > 400
□ Divergent transitions < 1%
□ α-γ后验相关 |r| < 0.3
□ γ全局均值: P(μ_γ^G > 0) > 95%
□ Posterior Predictive KL < 0.1
□ 按wave数分组: 2-wave HH后验std > 5-wave HH
□ 鲁棒性: ≥6/9变体相关 > 0.85
□ 计算成本 < $100
```

---

<a id="appendix-a"></a>
# 附录A: 计算资源汇总

| Step | 方法 | 硬件 | 时间 | 成本 |
|------|------|------|-----:|-----:|
| 4 Optuna(μ) | LightGBM CV×40 | CPU 16核 | ~80min | 免费 |
| 4 Optuna(σ) | LightGBM CV×30 | CPU 16核 | ~60min | 免费 |
| 4 反事实 | LightGBM predict | CPU | ~30s | 免费 |
| 4 QRF (鲁棒) | QRF Optuna×40 | CPU 16核 | ~7hr | 免费 |
| 5 BIRL主结果 | NUTS 4ch×7000 | TPU V5e | 12-24hr | ~$30-60 |
| 5 鲁棒性×9 | NUTS×9 | TPU V5e | 9×8hr | ~$150-200 |
| **总计** | | | | **~$200-300** |

<a id="appendix-b"></a>
# 附录B: 论文写法指南

### 环境模型段落

> We model the conditional distribution of log harvest value as Normal with heteroskedastic variance: log(y + 1) | s, a ~ N(μ(s,a), σ²(s,a)), where both μ(·) and σ(·) are estimated by gradient-boosted decision trees (LightGBM; Ke et al. 2017) with hyperparameters selected via Bayesian optimization (Optuna; Akiba et al. 2019) using 3-fold household-grouped cross-validation. This parameterization guarantees monotonicity of conditional quantiles by construction. As robustness, we re-estimate using Quantile Random Forests (Meinshausen 2006), which impose no distributional assumptions (Appendix Table X).

### BIRL段落

> We estimate farmer-level behavioral parameters using a three-level hierarchical Bayesian model with No-U-Turn Sampling (NUTS; Hoffman & Gelman 2014) implemented in NumPyro (Phan et al. 2019). The hierarchy (global → country → household) enables information pooling across farmers within countries and across countries, particularly important for data-sparse countries with only two survey waves. All household-level parameters use non-centered parameterizations to ensure efficient sampling. Pre-estimation diagnostics on a 1,000-household pilot confirmed: per-step NUTS time of 0.9 seconds, zero divergent transitions, R̂ < 1.005, and near-zero α–γ posterior correlation (r = −0.02), indicating clean identification.

<a id="appendix-c"></a>
# 附录C: 关键引用

```
环境模型:
  Ke et al. (2017) — LightGBM
  Akiba et al. (2019) — Optuna
  Meinshausen (2006) — Quantile Random Forest

BIRL:
  Hoffman & Gelman (2014) — NUTS
  Phan, Pradhan & Jankowiak (2019) — NumPyro
  Ramachandran & Amir (2007) — Bayesian IRL
  Ziebart et al. (2008) — Maximum Entropy IRL

经济学:
  Roy (1952) — Safety-First
  Fafchamps (1992) — 下尾风险与作物选择
  Dercon (1996) — 风险与贫困
  Rosenzweig & Binswanger (1993) — CRRA参数
  Binswanger (1980) — 风险态度实验
  Karlan et al. (2014) — 信贷约束
  Fafchamps (2003) — 农村市场与信贷

统计:
  Cazals, Florens & Simar (2002) — Order-m FDH
```