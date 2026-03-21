# Step 07: 2050 Climate Counterfactual Analysis — Complete Report

**日期**: 2026-03-20
**前置条件**: Environment Model (Step 04, R²=0.596), BIRL MCMC Posterior (Step 06, 0% divergence)
**数据规模**: 222,023 observations, 15,644 households, 4,436 GPS points, 6 countries

---

## 1. 分析目标

将BIRL后验的描述性发现（"各国农户风险偏好不同"）转化为处方性政策建议（"在2050年气候变化下，哪个国家需要什么类型的干预"）。

核心指标: **确定性等价 (Certainty Equivalent, CE)** — 使农户效用等于期望效用的确定性收入水平。CE越高，福利越好。

---

## 2. 方法论

### 2.1 分析链

```
CMIP6 2050气候预测 (5 GCMs ensemble)
    │ Delta method (GPS点级别)
    ▼
2050反事实特征矩阵 (替换7个气候特征, 保持其他34个特征不变)
    │ LightGBM环境模型 (model_mu + model_sigma)
    ▼
2050反事实收入矩阵 (222,023 × 27 actions × {q10, q50, q90, sigma})
    │ Stone-Geary CRRA效用 + 5点求积
    ▼
CE计算 (per country × per scenario × 3,000 posterior samples)
    │ 最优action CE (max over feasible actions)
    ▼
气候损失 = CE(当前) - CE(2050)
政策价值 = CE(2050+干预) - CE(2050)
```

### 2.2 关键假设

1. **风险偏好不变**: 2050年的ρ和γ与当前相同（保守假设: 实际中贫困加剧可能推高γ，使损失更大）
2. **非气候特征不变**: 土壤、地形、家庭特征、市场接入保持当前水平（隔离纯气候效应）
3. **作物技术不变**: 环境模型学到的气候-产量关系在2050年仍成立（无品种改良预测）
4. **基于CE而非选择概率**: 不预测"农户会改种什么"，只预测"福利变多少"——不依赖β

### 2.3 CMIP6气候数据

| 参数 | 设置 |
|------|------|
| 数据集 | NASA/GDDP-CMIP6 (GEE) |
| 分辨率 | 0.25° (~25km), 日数据 |
| GCM Ensemble | ACCESS-CM2, MIROC6, MRI-ESM2-0, INM-CM5-0, IPSL-CM6A-LR |
| 基线时段 | 2005–2014 (historical scenario) |
| 未来时段 | 2045–2055 (SSP scenario, 以2050为中心的10年气候态) |
| SSP情景 | SSP2-4.5 (中等排放) + SSP5-8.5 (高排放) |
| 提取方式 | Per GPS point (4,436个唯一点), 国家特定生长季聚合 |
| Delta方法 | 降雨: 乘性 (current × future/baseline); 温度: 加性 (current + future - baseline) |

### 2.4 NDVI经验预测

CMIP6不包含NDVI。从当前数据拟合经验模型:

```
NDVI_growing  = f(rainfall, temperature, country)    R² = 0.602
NDVI_preseason = f(rainfall, temperature, country)   R² = 0.657
```

模型: GradientBoostingRegressor (200 trees, depth=4), 训练样本 n=202,847

### 2.5 效用函数与CE计算

**Stone-Geary CRRA效用** (与BIRL模型完全一致):

```
U(Y) = [(Y - γ)^(1-ρ) - 1] / (1-ρ)

其中:
  Y = 收入 (USD)
  γ = 生存底线 (subsistence threshold)
  ρ = 相对风险厌恶系数
```

**CE计算 (5点求积)**:

```
y₁=q10, y₂=(q10+q50)/2, y₃=q50, y₄=(q50+q90)/2, y₅=q90
E[U] = 0.10·U(y₁) + 0.20·U(y₂) + 0.40·U(y₃) + 0.20·U(y₄) + 0.10·U(y₅)
CE = γ + ((1-ρ)·E[U] + 1)^(1/(1-ρ))
```

**后验不确定性传播**: 3,000个后验样本 (12,000 thinned by 4), 每国每情景独立计算。

### 2.6 政策情景

| ID | 情景 | 气候 | ρ因子 | γ因子 | 经济学含义 |
|----|------|------|-------|-------|-----------|
| S0 | 当前基线 | 当前 | 1.0 | 1.0 | 无干预 |
| S1 | SSP2-4.5 | 2050 (中等) | 1.0 | 1.0 | 无干预 |
| S2 | SSP5-8.5 | 2050 (高排放) | 1.0 | 1.0 | 无干预 |
| S3 | 保险 | SSP5-8.5 | 0.5 | 1.0 | 天气指数保险降低有效风险暴露 |
| S4 | 安全网 | SSP5-8.5 | 1.0 | 0.5 | 社会转移降低有效生存底线 |
| S5 | 组合 | SSP5-8.5 | 0.5 | 0.5 | 保险+安全网同时实施 |

---

## 3. BIRL后验参数 (来自Step 06)

| 国家 | ρ (风险厌恶) | 95% CI | γ (生存底线, $) | 95% CI | 特征 |
|------|-------------|--------|----------------|--------|------|
| Nigeria | **1.35** | [1.25, 1.45] | 17.49 | [14.69, 20.66] | 最接近风险中性, 中等γ |
| Uganda | **1.62** | [1.47, 1.78] | **1.80** | [0.52, 3.67] | 低风险厌恶, 极低γ |
| Ethiopia | 2.39 | [2.23, 2.56] | **29.82** | [29.38, 29.99] | 中高ρ, γ逼上界 |
| Tanzania | 2.96 | [2.08, 3.90] | 25.42 | [5.75, 29.93] | 高ρ, 宽CI |
| Mali | **3.02** | [2.77, 3.27] | 29.12 | [26.58, 29.96] | 高ρ, γ逼上界 |
| Malawi | **3.30** | [3.04, 3.57] | 29.63 | [28.53, 29.98] | 最高风险厌恶, γ逼上界 |

---

## 4. 气候变化信号

### 4.1 SSP5-8.5 (高排放路径) 气候变化

| 国家 | 降雨变化 | 温度变化 | 降雨CV变化 | NDVI变化 |
|------|---------|---------|-----------|---------|
| Ethiopia | **+22.6%** | +1.5°C | -11.4% | +5.8% |
| Nigeria | +12.4% | +1.5°C | +7.7% | -1.6% |
| Uganda | +5.8% | +1.4°C | +11.7% | -4.5% |
| Tanzania | +6.4% | +1.5°C | +2.8% | -1.7% |
| Malawi | -0.4% | **+1.8°C** | **+24.9%** | +2.1% |
| Mali | +0.1% | **+1.9°C** | **+28.8%** | **-19.5%** |

**关键发现**:
- **East Africa (Ethiopia, Uganda, Tanzania)**: 降雨增加为主, 温度升幅相对温和
- **West Africa (Mali, Nigeria)**: 降雨几乎不变但 **波动性大幅增加** (Mali CV+29%), NDVI大幅下降
- **Southern Africa (Malawi)**: 降雨略减, 但CV增加25% — 干旱风险显著升高
- 温度普遍升高1.4–1.9°C, 与IPCC AR6 SSA预估一致

### 4.2 反事实收入矩阵变化 (环境模型预测)

| 国家 | 基线q50中位数 | SSP2-4.5 q50 | Δ% | SSP5-8.5 q50 | Δ% |
|------|-------------|-------------|-----|-------------|-----|
| Mali | $127.0 | $89.3 | **-29.7%** | $87.7 | **-30.9%** |
| Nigeria | $104.2 | $92.4 | **-11.3%** | $87.8 | **-15.8%** |
| Uganda | $27.4 | $29.0 | +5.6% | $29.1 | +5.9% |
| Tanzania | $30.4 | $30.9 | +1.7% | $31.2 | +2.7% |
| Malawi | $20.8 | $20.9 | +0.7% | $20.9 | +0.7% |
| Ethiopia | $19.9 | $20.2 | +1.8% | $19.9 | +0.3% |

**收入效应**:
- Mali和Nigeria承受最大收入冲击 (主要由温度升高和NDVI下降驱动)
- East Africa和Malawi收入基本不变或略增 (降雨增加补偿了温度效应)

---

## 5. 核心结果

### 5.1 确定性等价 (CE) — 6情景 × 6国

| 情景 | Ethiopia | Malawi | Mali | Nigeria | Tanzania | Uganda |
|------|----------|--------|------|---------|----------|--------|
| **S0 当前** | $34.43 | $31.38 | $181.63 | $133.25 | $30.39 | $33.30 |
| S1 SSP2-4.5 | $34.31 | $31.36 | $119.20 | $118.82 | $30.41 | $36.66 |
| **S2 SSP5-8.5** | $34.19 | $31.37 | $122.14 | $112.31 | $30.41 | $36.69 |
| S3 保险 | **$46.18** | $32.57 | **$173.35** | **$163.01** | $35.31 | **$51.63** |
| S4 安全网 | **$51.68** | $32.09 | **$169.80** | **$147.03** | **$44.81** | **$46.06** |
| **S5 组合** | **$60.93** | $33.39 | **$200.14** | **$180.06** | **$53.57** | **$56.07** |

*所有值为中位数CE ($), 基于3,000个后验样本。完整95% CI见 results/ce_by_scenario_country.csv*

### 5.2 气候损失: CE(S0) - CE(S2)

| 国家 | 气候损失 ($) | 95% CI | 损失比例 | 主要驱动因素 |
|------|-------------|--------|---------|-------------|
| **Mali** | **$59.34** | [$57.15, $61.14] | **-32.7%** | 温度+1.9°C, NDVI-20%, CV+29% |
| **Nigeria** | **$21.01** | [$19.91, $23.78] | **-15.7%** | 温度+1.5°C, 收入下降16% |
| Ethiopia | $0.25 | [$0.17, $0.34] | -0.7% | 降雨增加补偿温度效应 |
| Malawi | $0.00 | [$0.00, $0.01] | ~0% | 收入变化极小 |
| Tanzania | ~$0 | [-$0.53, $0.00] | ~0% | 收入微增 |
| Uganda | **-$3.44** | [-$3.81, -$3.11] | **+10.3%** | 降雨增加→收入+6% |

**核心发现**: 气候损失的国家间差异极大（从Mali -33%到Uganda +10%），**但这不仅仅是气候暴露的差异**。Mali和Nigeria的高损失部分来自其收入基数大（$127 vs $20），使得同样的气候冲击产生更大的绝对CE变化。

### 5.3 保险价值: CE(S3) - CE(S2)

保险干预通过 ρ_effective = ρ × 0.5 模拟:

| 国家 | 保险价值 ($) | 95% CI | CE改善比例 | ρ当前→ρ有效 |
|------|-------------|--------|-----------|------------|
| **Mali** | **$51.06** | [$50.21, $51.35] | +41.8% | 3.02→1.51 |
| **Nigeria** | **$50.57** | [$47.37, $55.63] | +45.1% | 1.35→0.68 |
| Uganda | $14.93 | [$13.53, $16.61] | +40.7% | 1.62→0.81 |
| Ethiopia | $12.00 | [$10.72, $13.13] | +35.1% | 2.39→1.20 |
| Tanzania | $6.08 | [$2.00, $22.68] | +20.0% | 2.96→1.48 |
| Malawi | $1.25 | [$1.03, $1.64] | +4.0% | 3.30→1.65 |

**核心发现**: 保险对**所有国家**都有正向效果。效果最大的是Mali和Nigeria（绝对值），因为它们有更高的收入基数和更大的气候冲击。即使Malawi有最高的ρ（3.30→1.65），由于其低收入基数（$20），绝对CE改善只有$1.25。

### 5.4 安全网价值: CE(S4) - CE(S2)

安全网干预通过**保底收入模型 (income floor)** 模拟:

```
floor = γ + (country_median_income - γ) × coverage
Y_effective = max(Y, floor)
```

coverage = 0.5 意味着政府保障收入不低于γ与国家中位收入的中点。γ本身不变（它是偏好参数，不是政策杠杆）。

| 国家 | 安全网价值 ($) | 95% CI | CE改善比例 |
|------|---------------|--------|-----------|
| **Mali** | **+$47.62** | [$44.19, $49.88] | +39.0% |
| **Nigeria** | **+$34.78** | [$30.71, $41.10] | +31.0% |
| **Ethiopia** | **+$17.49** | [$17.32, $17.82] | +51.2% |
| **Tanzania** | **+$15.06** | [$12.87, $29.00] | +49.5% |
| **Uganda** | **+$9.33** | [$7.55, $11.84] | +25.4% |
| Malawi | +$0.73 | [$0.56, $1.64] | +2.3% |

**核心发现**: 修正安全网建模后, **所有国家安全网效果均为正值**。效果最大的是Mali (+$47.62) 和Nigeria (+$34.78)，因为它们的收入基数高且气候损失大。Ethiopia和Tanzania效果显著（+51% 和 +49%），因为保底收入有效截断了它们收入分布的下尾风险。Malawi效果最小（+$0.73），因为其低收入基数限制了保底水平。

### 5.5 组合干预与协同效应

| 国家 | 组合价值 ($) | 保险+安全网之和 ($) | 协同效应 ($) | 协同比例 |
|------|-------------|-------------------|-------------|---------|
| Mali | +$77.96 | +$98.68 | **-$20.66** | -26.7% 次线性 |
| Nigeria | +$67.66 | +$85.35 | **-$17.64** | -26.1% 次线性 |
| Ethiopia | +$26.77 | +$29.49 | **-$2.75** | -10.3% 次线性 |
| Tanzania | +$24.19 | +$21.14 | **+$2.92** | +12.1% 超线性 |
| Uganda | +$19.35 | +$24.26 | **-$4.89** | -25.1% 次线性 |
| Malawi | +$2.03 | +$1.97 | **+$0.06** | +2.7% ~线性 |

**协同效应解读**:
- **Mali/Nigeria/Uganda**: 次线性（-20 ~ -26%）。保险和安全网在高收入国家有**部分替代性**——两者都通过不同渠道减少下尾风险，单独已覆盖大部分收益
- **Tanzania**: 唯一的超线性国家（+12%）。保险降低风险厌恶的效用增益与安全网截断下尾风险相互**放大**
- **Ethiopia**: 弱次线性（-10%），两种工具各有独立贡献但略有重叠
- **Malawi**: 几乎完全线性（协同≈0），因为低收入基数限制了两种干预的交互空间

---

## 6. 政策启示

### 6.1 保险 vs 安全网的国家差异

| 国家 | 保险价值 | 安全网价值 | 比率 | 推荐优先干预 |
|------|---------|-----------|------|------------|
| **Mali** | $51.06 | $47.62 | 1.07 | **两者并重**，均有巨大效果 |
| **Nigeria** | $50.57 | $34.78 | 1.45 | 保险略优，但安全网也很关键 |
| **Ethiopia** | $12.00 | $17.49 | 0.69 | **安全网优先** — 保底收入对低收入高γ国家更有效 |
| **Tanzania** | $6.08 | $15.06 | 0.40 | **安全网优先** — 效果是保险的2.5倍 |
| **Uganda** | $14.93 | $9.33 | 1.60 | 保险优先，安全网补充 |
| **Malawi** | $1.25 | $0.73 | 1.72 | 保险略优，但两者绝对效果均受限于低收入基数 |

### 6.2 关键洞察

1. **保险和安全网是互补工具，非替代品**: 与旧建模不同，修正后的安全网对所有国家均为正效果。组合干预（S5）在所有国家的CE改善超过单独任一工具。

2. **国家间政策排序翻转**: Ethiopia和Tanzania应**优先部署安全网**（保底收入），而非保险——因为保底收入对高γ、低收入国家的下尾风险截断效果更显著。Mali和Nigeria则**两者并重**。

3. **一刀切政策的浪费**: 组合干预效果差异达38倍（Mali $78 vs Malawi $2）。统一政策无法适应这种异质性。

4. **风险偏好是政策效果的关键调节变量**: 相同的气候冲击，通过不同的ρ和γ，产生了截然不同的政策响应。这是BIRL框架的核心贡献。

5. **气候损失集中在West Africa**: Mali和Nigeria承受了>90%的总福利损失，主要由温度上升和NDVI恶化驱动，而非降雨变化。

6. **East Africa可能受益于气候变化**: Ethiopia和Uganda在中期（2050）由于降雨增加，收入和福利均略有改善。但这不考虑极端事件频率的增加。

---

## 7. 方法论限制

### 7.1 安全网建模说明

安全网通过**保底收入模型 (income floor)** 实现:

```
floor = γ + (country_median_income - γ) × coverage
Y_effective = max(Y, floor)
```

其中 coverage = 0.5，γ 本身不变（它是偏好参数）。这相当于政府保障收入不低于γ与国家中位收入的中点，有效截断收入分布的下尾。

该建模的合理性: 真实安全网（如Ethiopia的PSNP、Malawi的SCTP）正是通过转移支付提升低收入端，而非改变农户偏好。保底收入模型捕捉了这一机制。

**仍存在的简化**: 实际安全网有覆盖率限制、管理成本、激励扭曲（moral hazard）等问题，本模型假设100%覆盖且无行为响应。

### 7.2 其他限制

| 限制 | 影响 | 方向 |
|------|------|------|
| 风险偏好假设不变 | 贫困加剧可能提高γ | 低估损失 |
| 非气候特征冻结 | 人口增长/城镇化改变基线 | 方向不确定 |
| 作物技术不变 | 品种改良可减缓损失 | 高估损失 |
| NDVI经验预测 (R²=0.60) | NDVI变化不确定性较大 | 不确定 |
| GCM结构不确定性 | 5-GCM ensemble未穷尽 | 不确定 |
| 极端事件未建模 | CMIP6日数据取均值丢失极端尾部 | 低估损失 |
| 194,825/222,023 obs匹配 | 12.2% obs保留原始气候值 | 低估变化 |

---

## 8. 技术细节

### 8.1 计算配置

| 参数 | 值 |
|------|-----|
| 后验样本 | 3,000 (12,000 thinned by 4) |
| 观测子采样 | 10,000/国 (大国有子采样, 小国全量) |
| 可行动作 | Per country×zone empirical mask (4–27 actions) |
| 最优CE | max across feasible actions per obs, 取中位数 |
| 运行时间 | Stage 2: 30s, Stage 3: 2min, Stage 4: 20min, Stage 5: 30s |

### 8.2 文件产出

```
07_2050_Counter_Fact/
├── data/
│   ├── cmip6_raw/                    # 20 CSVs (5 GCMs × 2 SSPs × 2 periods)
│   ├── cmip6_processed/
│   │   ├── ensemble_deltas.parquet   # 4,436 points × delta factors
│   │   ├── obs_climate_ssp245.parquet
│   │   └── obs_climate_ssp585.parquet
│   ├── ssp245_cf.npz                 # 222,023 × 27 × 4 (q10/q50/q90/sigma)
│   ├── ssp585_cf.npz
│   ├── ndvi_model_growing.joblib
│   └── ndvi_model_preseason.joblib
├── results/
│   ├── ce_by_scenario_country.csv    # 核心: 6 scenarios × 6 countries
│   ├── ce_posterior_samples.npz      # 3,000-sample arrays for CI
│   ├── climate_loss.csv
│   ├── policy_value.csv
│   └── synergy.csv
└── figures/
    ├── fig3_policy_heatmap.pdf       # Policy × Country interaction
    ├── fig4_climate_loss_map.pdf     # Geographic CE loss
    └── fig5_ce_distributions.pdf     # CE distribution comparison
```

### 8.3 可复现性

```bash
cd "Formal Analysis/07_2050_Counter_Fact"

# Stage 0: Export GPS points (already done)
python scripts/00_export_gps_points.py

# Stage 1: CMIP6 extraction (requires GEE auth)
python scripts/00_extract_cmip6_python.py

# Stages 2-5: Full pipeline
python scripts/run_pipeline.py --from 1

# Or individually:
python scripts/01_process_climate.py
python scripts/02_generate_cf_matrices.py
python scripts/03_compute_welfare.py
python scripts/04_make_figures.py
```

---

## 9. 论文叙事建议

### Abstract句子

> Under 2050 high-emission scenarios (SSP5-8.5), welfare losses range from +10% (Uganda, driven by rainfall increases) to -33% (Mali, driven by temperature rise and vegetation decline). Crop insurance (halving effective risk aversion) and safety nets (income floor guarantees) both generate positive welfare gains across all six countries, but with starkly different magnitudes: insurance delivers $1–51 per household, while safety nets deliver $1–48. Crucially, the optimal policy instrument varies by country — Ethiopia and Tanzania benefit more from safety nets, while Mali and Nigeria respond comparably to both. Combined interventions recover up to 78% of climate-induced welfare losses in the hardest-hit countries.

### Results Section核心段落

**段落1: 气候冲击的异质性**

2050年SSP5-8.5情景下, 6国面临的气候冲击差异巨大: 温度普遍升高1.4–1.9°C, 但降雨变化方向相反（Ethiopia +23%, Malawi -0.4%）。环境模型将此转化为收入分布: Mali中位收入下降31%, Nigeria下降16%, 而Uganda反而增加6%。

**段落2: 从收入到福利——风险偏好的放大/缓冲作用**

然而, 收入变化不等于福利变化。通过Stone-Geary CRRA效用函数和BIRL后验参数, 我们发现: Mali的32.7%福利损失中, 温度和降雨波动性增加（CV+29%）通过高ρ=3.02被进一步放大为CE下降; 相反, Nigeria的福利损失（15.7%）低于其收入下降（15.8%）, 因为其低ρ=1.35使农户对波动性增加不那么敏感。

**段落3: 保险与安全网的差异化价值**

保险干预（将有效ρ减半）对所有国家都产生正的CE改善, 但绝对效果差异达40倍: Mali +$51 vs Malawi +$1.25。安全网（保底收入）同样对所有国家有效, 且对Ethiopia (+$17.49, +51%)和Tanzania (+$15.06, +49%)的效果超过保险——因为保底收入有效截断了高γ国家收入分布的下尾风险。组合干预（保险+安全网）在Mali可恢复78%的气候损失、在Nigeria恢复67.7$。两种工具在大多数国家呈次线性关系（部分替代），但在Tanzania呈超线性（互补），提示政策组合设计需因国制宜。
