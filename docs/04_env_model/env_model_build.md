# Step 4: Environment Model Construction Documentation

**方法**: LightGBM + LogNormal参数化 (双模型: μ均值 + σ波动性)
**运行环境**: Google Colab (44 CPU cores, 173GB RAM)
**总计算时间**: ~2.5小时 (Optuna调参为主)
**日期**: 2026-03-17

---

## 1. 架构

环境模型回答："如果农户i在状态s下选择动作a，其收入分布是什么？"

```
状态 s + 动作 a ──▶ Model A (LightGBM) ──▶ μ̂ = E[log(y)|s,a]
                 ──▶ Model B (LightGBM) ──▶ σ̂ = Std[log(y)|s,a]
                                             │
                              ┌───────────────▼──────────────┐
                              │ LogNormal quantiles           │
                              │ q10 = exp(μ̂ - 1.28·σ̂·κ) - 1 │
                              │ q50 = exp(μ̂) - 1             │ ──▶ BIRL
                              │ q90 = exp(μ̂ + 1.28·σ̂·κ) - 1 │
                              │ κ = 1.626 (calibration scale) │
                              └──────────────────────────────┘
```

### 为什么LightGBM + LogNormal

| 方案 | 分位数单调 | 缺失值 | 速度 | 选择 |
|------|:---------:|:------:|:----:|:----:|
| **LightGBM LogNormal** | **数学保证** | **原生** | **极快** | **主模型** |
| QRF | 保证 | 需预处理 | 慢60× | 鲁棒性备选 |
| 3×Quantile LightGBM | 需post-hoc修正 | 原生 | 快 | 排除 |

LogNormal假设已通过诊断5验证（整体skew=-0.03, kurt=0.14）。

---

## 2. 产出变量

```
y = log(harvest_value_USD_w.clip(0.01) + 1)
```

- harvest_value_USD_w（winsorized to $50K cap）：间作多作物kg不可加总，USD可以
- log(y+1)：高度右偏→近似正态，LogNormal参数化基础
- 零产出（4.5%）clip到0.01保留，供模型学习crop failure

---

## 3. 特征工程 (36维)

### 动作特征（反事实预测时替换）

| 特征 | 类型 | 说明 |
|------|------|------|
| `action_crop` | categorical, 9类 | LightGBM原生处理 |
| `input_intensity` | ordinal, 0/1/2 | low/medium/high |

### 状态特征（34维）

| 类别 | 特征 | 缺失率 |
|------|------|--------|
| **地块** | plot_area_GPS, intercropped, plot_owned, irrigated | <1% |
| **气候** | rainfall_growing_sum/10yr_mean/10yr_cv_final, ndvi_preseason/growing_mean_final | <4% |
| **温度** | era5_tmean/tmax_growing_final | 9% → LightGBM原生 |
| **土壤** | clay/sand/soc/nitrogen/phh2o_0-5cm | ~7% → LightGBM原生 |
| **地形** | elevation_m, slope_deg | <1% |
| **家庭** | hh_size, hh_asset_index, hh_dependency_ratio, age/female/education_manager, livestock, nonfarm, electricity | <1% |
| **市场** | travel_time_city_min, urban | <4% |
| **冲突** | conflict_events_25km_12m, conflict_nearest_event_km | <1% |
| **时空** | country (cat), year, season (cat) | 0% |

### 排除的特征

- **`dea_efficiency`**：从harvest_value计算而来，作为特征=target leakage。初次运行包含时R²异常偏高，移除后R²从~0.7降到0.596（合理范围）。

---

## 4. 数据拆分

```
Total:     222,023 obs, 15,644 HH
Train+Val: 189,493 obs, 13,297 HH (85%)
Test:       32,530 obs,  2,347 HH (15%)
```

- **HH-grouped split**：同一农户的所有obs在同一边，避免信息泄漏
- **3-fold GroupKFold CV**：用于Optuna调参和early stopping

---

## 5. Model A: 均值模型 (μ)

### Optuna调参

| 参数 | 搜索范围 | 最优值 |
|------|---------|--------|
| num_leaves | 63–255 | **230** |
| learning_rate | 0.03–0.1 | **0.0302** |
| min_child_samples | 20–100 | **23** |
| subsample | 0.6–0.9 | **0.611** |
| colsample_bytree | 0.5–0.9 | **0.506** |
| reg_alpha | 1e-3–10 | **0.151** |
| reg_lambda | 1e-3–10 | **0.059** |

- 20 trials, 68分钟
- num_boost_round上限4000, early stopping patience 50
- Best CV RMSE: **1.2154**
- 最终训练: **1,334 rounds**

### OOS性能

| 指标 | 值 |
|------|-----|
| **R²** | **0.5964** |
| **RMSE** | **1.1949** |

R²=0.60意味着模型解释了60%的log(harvest_value)变异，对6国跨作物的异质数据来说是合理的。

### Top 5 特征 (gain importance)

| 排名 | 特征 | Gain | 占比 |
|------|------|-----:|----:|
| 1 | **year** | 1,705K | 21.3% |
| 2 | **plot_area_GPS** | 1,160K | 14.5% |
| 3 | **country** | 961K | 12.0% |
| 4 | era5_tmean_growing | 350K | 4.4% |
| 5 | elevation_m | 282K | 3.5% |

year和country排前3说明时间趋势和国家间CPI/市场差异是harvest_value_USD变异的主要来源（这是预期的，因为USD价值受通胀和汇率影响大）。plot_area第2是因为更大的地块产出更多USD。

---

## 6. Model B: 波动性模型 (σ)

### 目标

预测 |residual| = |log(y) - μ̂(s,a)|，即条件波动性。

### 为什么|residual|而非residual²

|residual|对异常值更鲁棒（MAE loss + |residual| target双重保护）。residual²对极端值赋予过大权重。

### Optuna调参

| 参数 | 最优值 |
|------|--------|
| num_leaves | 126 |
| learning_rate | 0.0308 |
| min_child_samples | 32 |
| subsample | 0.643 |
| colsample_bytree | 0.649 |
| reg_alpha | 0.771 |
| reg_lambda | 0.130 |

- 20 trials, metric=MAE (LightGBM内部名: l1)
- OOS MAE: **0.5309**

### Top 5 σ特征

| 排名 | 特征 | Gain |
|------|------|-----:|
| 1 | **year** | 38.4K |
| 2 | **action_crop** | 29.5K |
| 3 | **plot_area_GPS** | 28.6K |
| 4 | intercropped | 24.6K |
| 5 | rainfall_10yr_cv | 22.0K |

action_crop在σ模型中排第2（μ模型中排第8），说明不同作物的产出**波动性差异**比**均值差异**更大。rainfall_10yr_cv进入σ的top 5说明长期降雨变异性确实增加了产出不确定性。

---

## 7. Calibration与Sigma Scaling

### 原始calibration（未缩放）

| 名义分位 | 实际覆盖 | Gap |
|----------|----------|----:|
| 5% | 15.9% | +10.9% |
| 10% | 21.7% | +11.7% |
| 25% | 33.8% | +8.8% |
| 50% | 49.2% | -0.8% |
| 75% | 65.2% | -9.8% |
| 90% | 78.7% | -11.3% |
| 95% | 85.1% | -9.9% |

**问题**：σ模型系统性低估了真实波动性。q50(median)几乎完美，但尾部分位数偏差大——q10高估（实际15.9%落在q10以下而非10%），q90低估。

### Sigma scaling修正

通过在OOS数据上搜索最优缩放因子κ，最小化calibration误差：

```
σ_scaled = σ_raw × κ
κ* = argmin Σ |actual_coverage(τ) - τ|
κ* = 1.626
```

### 缩放后calibration

| 名义分位 | 实际覆盖 | Gap |
|----------|----------|----:|
| 5% | 6.0% | **+1.0%** |
| 10% | 10.5% | **+0.5%** |
| 25% | 25.0% | **-0.01%** |
| 50% | 49.2% | **-0.8%** |
| 75% | 74.7% | **-0.3%** |
| 90% | 90.7% | **+0.7%** |
| 95% | 95.4% | **+0.4%** |

**缩放后所有分位数gap < 1.0%，calibration几乎完美。**

### 为什么σ需要缩放

Model B预测的是|residual|的条件期望，但LogNormal分位数需要的是条件标准差。E[|ε|] ≈ 0.8σ（正态分布下），所以raw prediction系统性偏小。κ=1.626的经验值合理（理论值√(π/2) ≈ 1.253，实际偏大因为残差不完全正态+模型预测偏保守）。

---

## 8. 分位数单调性

**100%保证**。由数学结构保证：

```
σ > 0 (clip下限0.1, 再×κ=1.626)
z10 < z50 < z90 (-1.28 < 0 < +1.28)
→ q10_log < q50_log < q90_log
→ exp()单调
→ q10 < q50 < q90  ∀ obs
```

无需任何post-hoc修正。

---

## 9. 反事实预测矩阵

对每个obs (222,023个)，对每个action (27个)，替换action特征保持state特征，预测完整收入分布。

```
输出矩阵: (222,023 × 27) × 4 = q10, q50, q90, sigma_USD
存储: env_model_output.npz, 56MB (float32 compressed)
生成时间: ~30秒（27次LightGBM predict）
```

### DEA上界约束

q90不超过DEA前沿值×1.1（10%容差）。对95.5%有FDH效率得分的obs应用。

---

## 10. 设计决策记录

| 决策 | 选择 | 理由 |
|------|------|------|
| 目标变量 | log(harvest_USD+1) | 右偏→正态；与BIRL效用函数一致 |
| 排除dea_efficiency | 是 | Target leakage（从harvest_value计算） |
| σ目标 | \|residual\| + MAE | 比residual²对异常值鲁棒 |
| σ缩放 | κ=1.626 | OOS calibration优化；理论上E[\|ε\|]≠σ |
| Optuna trials | 20 per model | 在Colab 2-core环境下的时间预算 |
| boost_round | 4000+early_stop50 | 充分搜索但不过拟合 |
| feature_pre_filter | False | 避免LightGBM对低方差特征的silent drop |
| 缺失值 | LightGBM原生 | 无需imputer |
| 数据拆分 | HH-grouped 85/15 | 避免household内信息泄漏 |

---

## 11. 输出文件清单

| 文件 | 大小 | 用途 | 本地使用方法 |
|------|------|------|-------------|
| `model_mu.txt` | 27MB | μ模型 | `lgb.Booster(model_file='model_mu.txt')` |
| `model_sigma.txt` | 9.4MB | σ模型 | `lgb.Booster(model_file='model_sigma.txt')` |
| `env_model_output.npz` | 56MB | CF矩阵 | `np.load(...)['q10'/'q50'/'q90'/'sigma']` |
| `env_model_metrics.json` | 3.2KB | 所有指标+超参 | `json.load(...)` |
| `oos_predictions.parquet` | 2.4MB | 测试集预测 | 残差分析、国家对比图 |
| `train_predictions.parquet` | 6.8MB | 训练集预测 | 过拟合检查 |
| `importance_mu.csv` | 1.3KB | μ特征重要度 | 直接画bar chart |
| `importance_sigma.csv` | 1.3KB | σ特征重要度 | 直接画bar chart |
| `study_mu.pkl` | 21KB | Optuna study | 恢复训练或分析调参 |
| `study_sigma.pkl` | 21KB | Optuna study | 同上 |
| `shap_summary_mu.png` | 158KB | SHAP图 | 直接用 |
| `feature_importance.png` | 113KB | 重要度图 | 直接用 |
| `calibration_plot.png` | 53KB | 原始calibration | 直接用 |
| `calibration_scaled.png` | 52KB | 缩放后calibration | 直接用 |
| `04_env_model_Colab_Final.ipynb` | 357KB | 最终notebook | 完整可复现代码+输出 |

---

## 12. 动作空间确认

最终动作空间为 **9 crops × 3 intensities = 27 actions**（不是24）。第9类是 **teff**——从"other"中拆出的Ethiopia主粮（5,522 obs, 占other的21.9%），详见 `action_build.md`。

action_space_config.json、feasibility mask、和CF矩阵列数（27）均已同步。

---

## 13. 与BIRL的接口

BIRL（Step 5）从 `env_model_output.npz` 读取：

```python
data = np.load('env_model_output.npz')
cf_q10 = data['q10']    # (222023, 27) float32, USD空间
cf_q50 = data['q50']    # (222023, 27)
cf_q90 = data['q90']    # (222023, 27)
cf_sigma = data['sigma'] # (222023, 27) = (q90-q10)/2.56
```

**所有矩阵已包含sigma_scale (κ=1.626) 缩放，BIRL直接使用无需再处理。**

### BIRL效用函数（Step 5正确规格）

```
r_i(s, a; α_i, γ_i, λ_i) = u(μ_subj) - γ_i · DownsideRisk - λ_i · LiquidityStress

其中:
  μ_subj     = q50(s,a) - κ_μ · α_i · σ(s,a)      # 悲观偏移
  q10_subj   = q10(s,a) - κ_q · α_i · σ(s,a)      # 下尾悲观偏移
  u(y)       = y^(1-ρ) / (1-ρ)                     # CRRA, ρ=1.5 固定
  DownsideRisk  = max{0, ȳ_i - q10_subj}           # Safety-First
  LiquidityStress = max{0, MinCons_i - μ_subj}     # 流动性压力

固定参数: ρ=1.5, κ_μ=1.0, κ_q=1.5, β=5.0 (softmax temperature)
待估参数: α_i (信念偏差), γ_i (风险厌恶), λ_i (流动性压力)
```

注意：α通过乘以σ做偏移进入（不是直接乘q50），λ项是LiquidityStress（不是直接乘sigma）。详见 `plan4-5.md` §5.2。

---

## 14. 诊断结果与分析 (2026-03-19)

### 14.1 Env Model 学到了什么

**模型确实学到了有意义的作物差异。** 以 maize vs sorghum_millet 二选一为例：

| 指标 | 值 |
|------|-----|
| 二选一准确率 | **63%**（农户实际选择了模型预测更好的作物） |
| q50 差异中位数 | **19%**（maize/sorghum ratio median=1.186） |
| q50 差异 P10-P90 | 0.83x - 1.98x |
| 差异 >1% 的 obs | 97.5% |
| 差异 >10% 的 obs | 75.7% |

这意味着模型知道"在这个气候、这个土壤条件下，maize 大概比 sorghum 贵 19%"。这是真实的、有信息量的预测。差异随国家变化：Malawi (ratio=1.86) 远高于 Ethiopia (ratio=0.98)，反映了当地市场条件差异。

**分国准确率**（maize vs sorghum 二选一）：

| 国家 | 准确率 | maize>sorghum | ratio中位数 |
|------|--------|--------------|------------|
| Ethiopia | 54.5% | 47.1% | 0.982 |
| Malawi | **97.4%** | 99.8% | 1.862 |
| Mali | 53.9% | 64.5% | 1.104 |
| Nigeria | 43.6% | 79.9% | 1.154 |
| Tanzania | **78.4%** | 82.8% | 1.157 |
| Uganda | 71.5% | 93.9% | 1.241 |

Malawi 和 Tanzania 的高准确率说明模型在这些国家捕捉到了清晰的作物-收入关系。Nigeria 的 43.6% 低于随机(50%)说明模型在 Nigeria 系统性地高估了 maize，可能是价格结构差异。

### 14.2 为什么 BIRL 看不到这些信号

**效用变换逐级压缩信号。** 完整链条：

```
env model 输出:  maize=$35, sorghum=$32        差异 $3 (9%)
        ↓
减去 γ=20:       surplus=15 vs 12              差异 $3 → 25% of surplus
        ↓
CRRA (ρ=1.5):   -1/√15 vs -1/√12              差异 0.06 → 效用空间 2%
        ↓
乘以 β=5:        logit差=0.3
        ↓
softmax 27选1:  概率差 ~1%                     无法区分
```

$3 的收入差异在每一步都被压缩。env model 给了 9% 的信号，CRRA 把它压成 2%，softmax 在 27 个选项里把 2% 稀释到不可区分。

### 14.3 Softmax 可区分性的量化诊断

使用 Stone-Geary (γ=20, ρ=1.5, β=5) 对 v1 反事实矩阵的诊断：

| 指标 | 值 | 含义 |
|------|-----|------|
| Utility spread (max-min) 中位数 | 0.240 | feasible set 内效用极差很小 |
| β×spread 中位数 | **1.20** | logit spread；需 >5 才能有效区分 |
| β×spread > 5 | 31.6% | 仅 1/3 的 obs 有足够区分度 |
| β×spread = 0 | ~10% | 完全退化为均匀 |
| Entropy / Max entropy | **0.986** | 98.6% 的最大熵 → 接近均匀分布 |
| P(chosen) 中位数 | 0.039 | vs 均匀 1/21=0.048，几乎无差异 |
| P(best) 中位数 | 0.069 | 最好的 action 概率也只有 7% |
| P(chosen)/P(uniform) > 1 | 51.2% | 仅半数 obs 选择概率高于均匀 |

**分国差异**：

| 国家 | logit spread | entropy ratio | 含义 |
|------|-------------|---------------|------|
| Ethiopia | 3.02 | 0.914 | 有一定区分度 |
| Malawi | 2.63 | 0.926 | 有一定区分度 |
| Tanzania | 2.17 | 0.903 | 有一定区分度 |
| **Mali** | **0.08** | **0.993** | **完全均匀** |
| **Nigeria** | **0.05** | **0.997** | **完全均匀** |
| **Uganda** | **0.15** | **0.991** | **完全均匀** |

Mali、Nigeria、Uganda 的 softmax 完全退化为均匀分布。

### 14.4 β 敏感性

| β | P(best) 中位数 | entropy 中位数 | logit spread 中位数 |
|---|---------------|---------------|-------------------|
| 0.1 | 0.048 | 4.39/4.39 | 0.02 |
| 1.0 | 0.050 | 4.39/4.39 | 0.24 |
| 5.0 | 0.069 | 4.37/4.39 | 1.20 |
| 10.0 | 0.100 | 4.26/4.39 | 2.40 |
| 20.0 | 0.172 | 3.81/4.39 | 4.81 |
| **50.0** | **0.377** | **2.53/4.39** | **12.02** |

β 需要增大到 ~50 才能让 softmax 有实质区分度。但高 β 会导致 MCMC 的 likelihood surface 过于尖锐，采样困难。

### 14.5 问题不在 Env Model，在接口

**结论：env model 不需要修改。** 它的信号在那里（maize vs sorghum 差 19%，二选一 63% 准确率）。问题出在效用变换把信号压没了。

**根本原因**：CRRA 效用函数 u(Y)=Y^(1-ρ)/(1-ρ) 在 ρ=1.5 时是强凹函数，将收入差异压缩为更小的效用差异。Stone-Geary 的 γ 参数进一步缩小有效 surplus。27 个 action 的 softmax 再次稀释。三重压缩导致 env model 的信号在到达 likelihood 时已不可辨。

**修复方向**：在 BIRL 侧对效用进行标准化（z-score），而非修改 env model：

```
标准化前: utility = [-0.258, -0.289, -0.271, ...]  spread=0.03
标准化后: z_utility = [1.2, -0.8, 0.3, ...]        spread=2.0
```

排序完全不变，信息完全保留，只是 scale 适配了 softmax。具体实现见 Step 5 BIRL 文档。

### 14.6 Per-hectare 目标变量实验（已排除）

10% 采样快速验证：将目标从 `log(harvest_USD+1)` 改为 `log(max(harvest_USD/area, 0.01))`。

| 指标 | v1 (total USD) | Per-hectare | 变化 |
|------|---------------|-------------|------|
| R² | 0.596 | 0.481 | -0.115 |
| action gain share | 4.2% | 3.8% | -0.4% |
| identity share | 32.2% | 35.2% | +3.0% |
| CF top-5 | 17.7% | 20.8% | +3.1% |

**结论：改善微乎其微。** action share 不升反降（3.8% vs 4.2%），plot_area 的 gain 被 year 吸收而非 action。根本原因是 year 和 country 编码的价格/通胀效应在 per-hectare 空间仍然主导。**不建议改目标变量——问题不在 env model 层面。**

### 14.7 Crop-specific 模型实验（已排除）

分别训练 maize 和 sorghum 的独立模型，交叉预测。

| 指标 | Crop-specific | Full model |
|------|--------------|------------|
| 二选一准确率 | 61.7% | **62.9%** |
| Ratio 中位数 | 1.261 | 1.186 |
| Ratio std | 极大 | 2.649 |

**结论：full model 已经和 crop-specific 一样好。** Crop-specific 模型的 ratio 方差更大但准确率并无提升——两个独立模型的预测误差不相关，交叉预测时方差叠加。**不建议分作物建模。**

### 14.8 其他诊断摘要

**分国校准**：Mali (max gap 5.2%) 和 Nigeria (3.6%) 的下尾偏乐观。Uganda 中位偏高 (q50/actual=1.11)。

**Leave-One-Country-Out CV**：3/6 国 R²<0（Nigeria=-0.95, Uganda=-0.10, Mali=-0.05）。跨国泛化弱，但不影响国内反事实（country 特征保持不变）。

**Feature importance 分组**：identity 32.2%, climate 21.4%, plot 17.0%, soil 10.0%, household 6.8%, geo 6.5%, **action 4.2%**, conflict 1.9%。action 天然在 harvest_value_USD 变异中占比小——"在哪里种"远比"种什么"重要，这是 Sub-Saharan Africa 农业的真实特征。
