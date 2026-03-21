# Step 6: 2050 Climate Counterfactual Analysis — Complete Implementation Plan

**前置条件**: MCMC后验已完成 (hier_noalpha + R3, 0% divergence, all diagnostics PASS)
**目标**: 量化2050年气候变化下的福利损失，并比较保险vs安全网的政策价值
**预计工作量**: 5-7天
**核心输出**: Nature Food正文的2-3个核心Figure + 1个主要Results section

---

## 为什么2050分析是这篇论文的胜负手

当前结果的故事是描述性的："6国农户风险偏好不同"。
2050分析把它变成处方性的："在气候变化下，哪个国家需要什么政策"。

```
没有2050:  "Nigeria最不风险厌恶，Uganda最保守"
           → 审稿人："So what? 这和人均GDP排名有什么区别？"

有2050:    "在SSP5-8.5下，Malawi农户福利损失24%，
            但保险可挽回18%——因为Malawi的ρ=3.30极高。
            而Nigeria保险只挽回4%——因为ρ=1.35。
            安全网的模式相反：Ethiopia受益最大（γ/消费=150%）。
            这意味着一刀切的政策浪费了60%的预算。"
           → 审稿人："这改变了我对适应政策的理解。"
```

---

## 目录

1. [分析框架总览](#1-框架)
2. [CMIP6数据获取与处理](#2-cmip6)
3. [2050反事实收入矩阵生成](#3-反事实矩阵)
4. [政策情景设计](#4-政策情景)
5. [福利计算与不确定性传播](#5-福利计算)
6. [关键产出指标](#6-指标)
7. [可视化与Figure设计](#7-可视化)
8. [论文叙事与写作指南](#8-叙事)
9. [实施时间线](#9-时间线)
10. [文件结构](#10-文件)

---

<a id="1-框架"></a>
## 1. 分析框架总览

### 1.1 逻辑链

```
CMIP6 2050 气候预测
    │
    ▼
环境模型 (已训练的 LightGBM mu + sigma)
    │  输入: 2050气候特征 + 当前非气候特征
    │  输出: 2050年各action的收入分布
    ▼
反事实收入矩阵 (222,023 × 27)
    │  每个obs在2050气候下的q10, q50, q90, sigma
    ▼
BIRL后验 (ρ_c, γ_c, β per country)
    │  与2050收入矩阵结合
    ▼
CE计算 (per country × per scenario × per posterior sample)
    │
    ▼
政策价值 = CE(with intervention) - CE(baseline 2050)
气候损失 = CE(current climate) - CE(2050 climate)
```

### 1.2 关键假设

```
假设1: 风险偏好不变
  农户的ρ和γ在2050年与当前相同。
  理由: ρ和γ反映深层偏好和制度约束，短期内不随气候变化。
  讨论: 如果气候变化导致更多贫困→γ可能上升→福利损失更大。
        我们的估计是下界。

假设2: 非气候特征不变
  土壤、地形、家庭特征、市场接入保持当前水平。
  理由: 2050年的这些变量无法可靠预测。
  讨论: 实际中人口增长、城镇化会改变这些。
        我们隔离的是纯气候效应。

假设3: 作物技术不变
  环境模型学到的气候-产量关系在2050年仍成立。
  理由: 我们没有品种改良的预测数据。
  讨论: 新品种可能缓解部分气候损失。
        我们的估计是上界（对气候损失而言）。
```

### 1.3 分析不依赖β

```
所有2050分析基于CE (certainty equivalent)，不基于选择概率。

CE的计算:
  CE_c = γ_c + ((1-ρ_c) × E[U_c(Y)] + 1)^(1/(1-ρ_c))
  
  其中E[U_c(Y)]是给定收入分布下的期望效用。
  这只依赖ρ_c, γ_c和收入分布——不涉及β。

β=0.14意味着我们不预测选择变化（"2050年农户会改种什么"），
只预测福利变化（"2050年农户的确定性等价收入变多少"）。
这是一个更保守但更可信的分析。
```

---

<a id="2-cmip6"></a>
## 2. CMIP6数据获取与处理

### 2.1 数据源

```
来源: CMIP6 (Coupled Model Intercomparison Project Phase 6)
访问: 
  方案A: Google Cloud Public Datasets (推荐, 免费, 快速)
    gs://cmip6/CMIP6/ScenarioMIP/
  方案B: ESGF (Earth System Grid Federation)
    https://esgf-node.llnl.gov/search/cmip6/
  方案C: GEE (部分CMIP6数据已上架)
    ee.ImageCollection("NASA/GDDP-CMIP6")
    → 这是最简单的选项: NASA已经降尺度到0.25°并提供日数据
```

### 2.2 推荐使用NASA GDDP-CMIP6 (GEE方案)

NASA Global Daily Downscaled Projections (GDDP) 已经将CMIP6降尺度到0.25°×0.25°，
在GEE中可直接查询，省去大量预处理。

```
GEE Dataset: NASA/GDDP-CMIP6
分辨率: 0.25° (~25km)
时间: 1950-2100 (日数据)
变量: 
  pr: 降水 (mm/day)
  tasmax: 日最高温 (K)
  tasmin: 日最低温 (K)
  
可用GCM (取3-5个做ensemble):
  ACCESS-CM2        (Australia)
  MIROC6            (Japan)
  MRI-ESM2-0        (Japan)
  INM-CM5-0         (Russia)
  IPSL-CM6A-LR      (France)

情景:
  SSP2-4.5: 中等排放路径 ("middle of the road")
  SSP5-8.5: 高排放路径 ("fossil-fueled development")
```

### 2.3 时间窗口

```
当前基线: 2005-2023 (与你数据的实际时间范围一致)
近未来:   2040-2060 (取中间年2050作为代表)
远未来:   2080-2100 (可选, 如果想要更强的信号)

对2040-2060:
  取20年的月度均值 → 作为"2050年代的典型气候"
  这样做比取单年(如2050年)更稳定
```

### 2.4 GEE提取脚本

```python
"""
extract_cmip6_2050.py
从GEE提取CMIP6 2050年气候预测，对应7,019个GPS点。
"""

import ee
ee.Initialize()

# ── 配置 ──
GCMS = ['ACCESS-CM2', 'MIROC6', 'MRI-ESM2-0', 'INM-CM5-0', 'IPSL-CM6A-LR']
SCENARIOS = ['ssp245', 'ssp585']
PERIOD_FUTURE = ('2040-01-01', '2060-12-31')  # 2050年代
PERIOD_BASELINE = ('2005-01-01', '2023-12-31')  # 当前基线

# 生长季定义 (与你的config/growing_seasons.yaml一致)
GROWING_SEASONS = {
    'Nigeria':  {'months': [4,5,6,7,8,9,10]},
    'Ethiopia': {'months': [5,6,7,8,9,10,11,12]},
    'Malawi':   {'months': [11,12,1,2,3,4], 'cross_year': True},
    'Tanzania': {'months': [11,12,1,2,3,4,5], 'cross_year': True},
    'Uganda':   {'months': [3,4,5,6,7]},
    'Mali':     {'months': [6,7,8,9,10]},
}

def extract_for_gcm_scenario(gcm, scenario, period, points_fc):
    """
    对一个GCM×情景×时段，提取:
      - 生长季累计降雨 (mm)
      - 生长季均温 (°C)
      - 生长季最高温 (°C)
      - 降雨变异系数 (跨年)
    """
    cmip = (ee.ImageCollection('NASA/GDDP-CMIP6')
            .filter(ee.Filter.eq('model', gcm))
            .filter(ee.Filter.eq('scenario', scenario))
            .filterDate(period[0], period[1]))
    
    results = {}
    
    for country, gs in GROWING_SEASONS.items():
        months = gs['months']
        
        # 逐年计算生长季统计量
        years = range(int(period[0][:4]), int(period[1][:4])+1)
        annual_rain = []
        annual_temp = []
        
        for yr in years:
            # 按月过滤 (处理跨年)
            if gs.get('cross_year'):
                # e.g., Malawi: Nov(yr-1) to Apr(yr)
                month_filter = cmip.filter(
                    ee.Filter.Or(
                        ee.Filter.And(
                            ee.Filter.calendarRange(yr-1, yr-1, 'year'),
                            ee.Filter.calendarRange(months[0], 12, 'month')
                        ),
                        ee.Filter.And(
                            ee.Filter.calendarRange(yr, yr, 'year'),
                            ee.Filter.calendarRange(1, months[-1], 'month')
                        )
                    )
                )
            else:
                month_filter = cmip.filter(
                    ee.Filter.calendarRange(yr, yr, 'year')
                ).filter(
                    ee.Filter.calendarRange(months[0], months[-1], 'month')
                )
            
            # 降雨: 日→生长季总和 (mm/day → mm/season)
            rain = month_filter.select('pr').sum()
            # 温度: 日均值→生长季均值 (K → °C)
            tmax = month_filter.select('tasmax').mean().subtract(273.15)
            tmin = month_filter.select('tasmin').mean().subtract(273.15)
            tmean = tmax.add(tmin).divide(2)
            
            annual_rain.append(rain)
            annual_temp.append(tmean)
        
        # 跨年统计
        rain_collection = ee.ImageCollection(annual_rain)
        rain_mean = rain_collection.mean()
        rain_std = rain_collection.reduce(ee.Reducer.stdDev())
        rain_cv = rain_std.divide(rain_mean.max(0.1))
        
        temp_mean = ee.ImageCollection(annual_temp).mean()
        tmax_mean = ee.ImageCollection([...]).mean()  # 类似
        
        # 堆叠
        combined = ee.Image.cat([
            rain_mean.rename(f'{country}_rain_gs_mean'),
            rain_cv.rename(f'{country}_rain_gs_cv'),
            temp_mean.rename(f'{country}_tmean_gs'),
            tmax_mean.rename(f'{country}_tmax_gs'),
        ])
        
        # Reduce to points
        extracted = combined.reduceRegions(
            collection=points_fc.filter(ee.Filter.eq('country', country)),
            reducer=ee.Reducer.mean(),
            scale=25000,  # CMIP6 ~25km
        )
        results[country] = extracted
    
    return results

# ── 对每个GCM × 情景执行 ──
for gcm in GCMS:
    for scenario in SCENARIOS:
        for period_name, period in [('baseline', PERIOD_BASELINE), 
                                     ('2050', PERIOD_FUTURE)]:
            results = extract_for_gcm_scenario(gcm, scenario, period, points_fc)
            # Export to Drive
            for country, fc in results.items():
                task = ee.batch.Export.table.toDrive(
                    collection=fc,
                    description=f'cmip6_{gcm}_{scenario}_{period_name}_{country}',
                    folder='CMIP6_extraction',
                )
                task.start()
```

### 2.5 NDVI预测

CMIP6没有NDVI。需要用经验关系从降雨预测NDVI。

```python
"""
从当前数据拟合 NDVI = f(rainfall, temperature) 关系，
然后用2050年气候预测NDVI_2050。
"""

# 用当前数据拟合简单关系
from sklearn.ensemble import GradientBoostingRegressor

# 训练数据: 当前的(rainfall, temperature, country) → NDVI
X_ndvi = df[['rainfall_growing_sum_final', 'era5_tmean_growing_final', 
             'country_encoded']].dropna()
y_ndvi = df.loc[X_ndvi.index, 'ndvi_growing_mean_final']

ndvi_model = GradientBoostingRegressor(n_estimators=200, max_depth=4)
ndvi_model.fit(X_ndvi, y_ndvi)

r2_ndvi = ndvi_model.score(X_ndvi, y_ndvi)
print(f"NDVI ~ (rain, temp, country) R² = {r2_ndvi:.3f}")
# 预期: 0.5-0.7 (NDVI和降雨在SSA高度相关)

# 2050年NDVI预测
X_ndvi_2050 = pd.DataFrame({
    'rainfall_growing_sum_final': rain_2050,
    'era5_tmean_growing_final': temp_2050,
    'country_encoded': country_encoded,
})
ndvi_2050 = ndvi_model.predict(X_ndvi_2050)
```

### 2.6 处理GCM不确定性: Ensemble方法

```
不取单个GCM，取5个GCM的ensemble:

方案A (推荐): Ensemble mean
  rain_2050 = mean(rain_ACCESS, rain_MIROC, rain_MRI, rain_INM, rain_IPSL)
  → 一组2050气候值
  → 简单，后续分析清晰

方案B: GCM-level propagation
  对每个GCM独立跑一次2050分析
  → 5组结果
  → 报告GCM间的spread作为气候不确定性
  → 与BIRL后验不确定性叠加
  → 更完整但分析量×5

推荐: 主结果用方案A，附录用方案B展示GCM sensitivity。
```

### 2.7 气候变化的Delta方法 (更简单的替代)

如果GEE提取太慢，可以用Delta方法：

```
rain_2050 = rain_current × (rain_CMIP6_2050 / rain_CMIP6_baseline)

即: 用CMIP6的变化比例，而非绝对值。
优势: 消除GCM的系统性偏差 (bias correction内置)
劣势: 假设变化是乘性的

实际操作:
  1. 从GEE提取CMIP6的baseline (2005-2023) 和 2050 (2040-2060) 的生长季均值
  2. 计算变化因子: delta_rain = rain_2050 / rain_baseline
  3. 应用到你的实际数据: rain_adjusted = rain_current × delta_rain
  
这只需要提取国家×GCM级别的均值 (6国 × 5GCM = 30个值)，
而不是GPS点级别 (7019点 × 5GCM = 35,000个值)。

推荐: GPS点级别的delta方法——既修正了偏差，又保留了空间异质性。
```

---

<a id="3-反事实矩阵"></a>
## 3. 2050反事实收入矩阵生成

### 3.1 特征替换策略

```python
def generate_2050_counterfactual(model_mu, model_sigma, df, climate_2050, kappa):
    """
    用2050气候替换当前气候特征，其他特征不变，
    生成2050年的反事实收入矩阵。
    """
    
    # 需要替换的气候特征
    CLIMATE_FEATURES = [
        'rainfall_growing_sum_final',
        'rainfall_10yr_mean_final',
        'rainfall_10yr_cv_final',
        'ndvi_preseason_mean_final',
        'ndvi_growing_mean_final',
        'era5_tmean_growing_final',
        'era5_tmax_growing_final',
    ]
    
    # 不替换的特征 (保持当前值)
    # 土壤、地形、家庭特征、市场、冲突、DEA等
    
    # 构造2050特征矩阵
    X_2050 = df[feature_cols].copy()
    for col in CLIMATE_FEATURES:
        if col in climate_2050.columns:
            X_2050[col] = climate_2050[col].values
    
    # 10年均值和CV需要特殊处理
    # 2050的"10年均值"应该是2040-2050的均值
    # 如果用delta方法: rain_10yr_mean_2050 = rain_10yr_mean_current × delta
    # 如果直接提取: 需要2040-2050逐年数据
    
    # 生成反事实收入矩阵 (同Step 4的generate_counterfactual_matrix)
    q10_2050, q50_2050, q90_2050, sigma_2050 = \
        generate_counterfactual_matrix(
            model_mu, model_sigma, X_2050,
            feature_cols, action_features=['action_crop', 'input_intensity'],
            feasibility_mask=mask_array,
        )
    
    return q10_2050, q50_2050, q90_2050, sigma_2050
```

### 3.2 需要生成的矩阵

```
总共需要4组反事实矩阵 (每组222,023 × 27 × 4):

Matrix 1: current_climate (已有, env_model_output.npz)
  → 当前气候下各action的收入分布
  → 这是baseline

Matrix 2: ssp245_2050
  → SSP2-4.5情景下2050年气候
  → 中等排放路径

Matrix 3: ssp585_2050
  → SSP5-8.5情景下2050年气候
  → 高排放路径

Matrix 4: ssp585_2050_ndvi_adjusted (可选)
  → SSP5-8.5 + NDVI独立调整
  → 检验NDVI预测不确定性的影响

存储: 每组 ~56MB (float32 compressed)
总计: ~224MB
```

### 3.3 气候变化信号的量级检验

在跑完整分析之前，先检查气候变化信号是否足够大：

```python
# Delta factors by country (ensemble mean)
for country in countries:
    mask = df['country'] == country
    rain_current = df.loc[mask, 'rainfall_growing_sum_final'].median()
    rain_2050 = climate_2050.loc[mask, 'rainfall_growing_sum_2050'].median()
    temp_current = df.loc[mask, 'era5_tmean_growing_final'].median()
    temp_2050 = climate_2050.loc[mask, 'era5_tmean_growing_2050'].median()
    
    print(f"{country}: rain {rain_current:.0f}→{rain_2050:.0f}mm "
          f"({(rain_2050/rain_current-1)*100:+.1f}%), "
          f"temp {temp_current:.1f}→{temp_2050:.1f}°C "
          f"({temp_2050-temp_current:+.1f}°C)")

# 预期 (SSA, SSP5-8.5):
# 降雨: -10% to +5% (Sahel可能增加, East Africa可能减少)
# 温度: +1.5 to +2.5°C
# 降雨CV: +10 to +30% (极端事件频率增加)
```

如果delta factors太小（降雨变化<5%, 温度变化<1°C），环境模型预测的q50变化可能在噪声范围内。
这种情况下需要更长的时间窗口（2080-2100）来获得更强的信号。

---

<a id="4-政策情景"></a>
## 4. 政策情景设计

### 4.1 六个核心情景

```
┌────┬───────────────────┬──────────┬────────────────────────────┬─────────────────────┐
│ ID │ 情景名            │ 气候     │ 政策干预                   │ 对应参数变化        │
├────┼───────────────────┼──────────┼────────────────────────────┼─────────────────────┤
│ S0 │ Current baseline  │ 当前     │ 无                         │ (ρ_c, γ_c, β)不变  │
│ S1 │ 2050 no adapt     │ SSP2-4.5 │ 无                         │ 只换气候→换收入矩阵│
│ S2 │ 2050 high emiss   │ SSP5-8.5 │ 无                         │ 只换气候→换收入矩阵│
│ S3 │ 2050 + insurance  │ SSP5-8.5 │ 作物保险 (降低有效ρ)       │ ρ_eff = ρ_c × 0.5  │
│ S4 │ 2050 + safety net │ SSP5-8.5 │ 社会安全网 (降低有效γ)     │ γ_eff = γ_c × 0.5  │
│ S5 │ 2050 + combined   │ SSP5-8.5 │ 保险 + 安全网              │ ρ×0.5 AND γ×0.5    │
└────┴───────────────────┴──────────┴────────────────────────────┴─────────────────────┘
```

### 4.2 政策干预的经济学动机

#### 保险干预 (S3): ρ_eff = ρ_c × 0.5

```
机制: 天气指数保险在灾年赔付，降低收入的下行波动。
     等效于降低农户面对的有效风险 → 效用函数中ρ的有效值降低。

为什么×0.5:
  理想的保险消除所有风险 → ρ_eff = 0
  现实中基差风险(basis risk)、不完全覆盖、道德风险 → 约消除50%的风险
  文献参照: Cole et al. (2013), Karlan et al. (2014)
  
敏感性: ×0.3 (强保险) 和 ×0.7 (弱保险)

成本假设: 
  保费 = actuarially fair premium × 1.5 (50% loading)
  精算保费 ≈ E[max(0, γ - Y)] / E[Y] × 100%
  典型值: 年收入的 5-15%
  Per capita: $5-20/年 (与SSA已有保险项目一致)
```

#### 安全网干预 (S4): γ_eff = γ_c × 0.5

```
机制: 社会安全网(现金转移、食品援助)在极端情况下提供保底，
     等效于降低农户的有效生存底线。

为什么×0.5:
  完美安全网 → γ_eff = 0 (无生存风险)
  现实中覆盖不完全、targeting误差、行政延迟 → 约覆盖50%的缺口
  文献参照: Ethiopia PSNP, Malawi FISP, Nigeria NASSP

敏感性: ×0.3 和 ×0.7

成本假设:
  人均转移额 ≈ γ × 0.5 × 触发概率(~30%)
  Per capita: $10-50/年
  行政成本: 额外15-20%
```

#### 组合干预 (S5): ρ×0.5 AND γ×0.5

```
机制: 保险降低风险暴露 + 安全网降低生存底线
可能存在超线性协同:
  安全网降低γ → 农户"敢于"接受保险 (更接近风险中性)
  保险降低波动 → 安全网的触发频率降低 → 财政负担减轻
```

### 4.3 政策×国家的预期交互效应

```
高ρ国家 (Malawi 3.30, Mali 3.02):
  → 保险干预效果最大 (ρ×0.5 = 1.65/1.51, 大幅度降低)
  → 安全网效果较小 (γ已经逼bound)

低γ国家 (Uganda 1.80):
  → 安全网效果有限 (γ已经很低, ×0.5→$0.90)
  → 保险效果中等 (ρ=1.62×0.5=0.81)

高γ国家 (Ethiopia 29.82, Malawi 29.63):
  → 安全网效果可能最大 (γ×0.5≈$15, 显著降低生存压力)
  → 但注意γ逼bound的识别问题

Nigeria (低ρ=1.35, 中γ=17.49):
  → 两种干预效果都较小 (已经接近风险中性)
  → 但收入基数高, 绝对值变化可能仍显著
```

这种交互效应矩阵是Nature Food审稿人最想看到的——它直接回答"一刀切vs差异化政策"的问题。

---

<a id="5-福利计算"></a>
## 5. 福利计算与不确定性传播

### 5.1 CE计算公式

```python
def compute_ce(q10, q50, q90, rho, gamma, eps=1.0):
    """
    给定收入分布的分位数和偏好参数，计算certainty equivalent。
    
    CE是使 U(CE) = E[U(Y)] 的确定性收入。
    """
    # 5点求积计算E[U(Y)]
    y1 = q10
    y2 = 0.5 * (q10 + q50)
    y3 = q50
    y4 = 0.5 * (q50 + q90)
    y5 = q90
    
    def u(y, rho, gamma):
        surplus = np.maximum(y - gamma, eps)
        return (surplus ** (1 - rho) - 1) / (1 - rho)
    
    eu = 0.10*u(y1, rho, gamma) + 0.20*u(y2, rho, gamma) + \
         0.40*u(y3, rho, gamma) + 0.20*u(y4, rho, gamma) + \
         0.10*u(y5, rho, gamma)
    
    # 反求CE: U(CE) = EU → CE = γ + ((EU × (1-ρ) + 1))^(1/(1-ρ))
    inner = eu * (1 - rho) + 1
    inner = np.maximum(inner, 1e-10)  # 数值保护
    ce = gamma + inner ** (1 / (1 - rho))
    
    return ce
```

### 5.2 最优action的CE

```python
def compute_optimal_ce(cf_q10, cf_q50, cf_q90, rho, gamma, 
                        feasibility_mask, zones):
    """
    对每个obs，在所有feasible action中找到CE最高的action。
    注意: 不用softmax概率加权，直接取max。
    这反映了"如果农户完全理性(β→∞)"的福利上界。
    """
    N_obs, N_actions = cf_q50.shape
    ce_matrix = np.full((N_obs, N_actions), -np.inf)
    
    for a in range(N_actions):
        ce_matrix[:, a] = compute_ce(
            cf_q10[:, a], cf_q50[:, a], cf_q90[:, a],
            rho, gamma
        )
    
    # 只考虑feasible actions
    obs_masks = feasibility_mask[zones]  # (N_obs, N_actions)
    ce_matrix[~obs_masks] = -np.inf
    
    optimal_action = ce_matrix.argmax(axis=1)
    optimal_ce = ce_matrix.max(axis=1)
    
    return optimal_ce, optimal_action
```

### 5.3 Country-level CE汇总

```python
def compute_country_ce(cf_matrices, posterior_samples, df, 
                        feasibility_mask, country_col='country'):
    """
    对每个国家，用后验样本计算CE的分布。
    
    返回: {country: array of CE values, shape=(n_posterior_samples,)}
    """
    countries = df[country_col].unique()
    results = {}
    
    for country in countries:
        c_mask = df[country_col] == country
        c_idx = df.loc[c_mask].index
        
        # 该国家的反事实矩阵
        c_q10 = cf_matrices['q10'][c_mask]
        c_q50 = cf_matrices['q50'][c_mask]
        c_q90 = cf_matrices['q90'][c_mask]
        c_zones = df.loc[c_mask, 'zone_idx'].values
        
        ce_samples = []
        
        for k in range(len(posterior_samples['rho_c'])):
            # 后验样本k的参数
            rho_k = posterior_samples['rho_c'][k]  # 该国家的ρ
            gamma_k = posterior_samples['gamma_c'][k]  # 该国家的γ
            
            # 计算最优CE
            opt_ce, _ = compute_optimal_ce(
                c_q10, c_q50, c_q90, rho_k, gamma_k,
                feasibility_mask, c_zones
            )
            
            # 该国家的中位数CE
            ce_samples.append(np.median(opt_ce))
        
        results[country] = np.array(ce_samples)
    
    return results
```

### 5.4 不确定性传播

```
三重不确定性来源:

1. BIRL后验不确定性 (ρ, γ的12,000个后验样本)
   → 对每个样本独立计算CE
   → 报告CE的中位数和95% credible interval

2. GCM不确定性 (5个GCM的ensemble spread)
   → 主结果用ensemble mean
   → 附录展示5个GCM的individual结果

3. 环境模型不确定性 (R²=0.60, 40%未解释)
   → 通过σ（波动性模型）间接捕捉
   → CE计算已包含收入分布的宽度

实际操作: 只传播第1种不确定性 (BIRL后验)
  理由: 这是你能精确量化的不确定性
  GCM和env model不确定性在Discussion中定性讨论
```

### 5.5 完整的情景计算流程

```python
def run_all_scenarios(posterior_file, env_matrices, climate_2050_matrices,
                       df, feasibility_mask):
    """
    对6个情景 × 6国 × 12,000后验样本，计算CE。
    """
    # 加载后验
    posterior = load_posterior(posterior_file)
    n_samples = len(posterior['rho_c_Ethiopia'])
    
    # 情景定义
    scenarios = {
        'S0_current': {
            'climate': env_matrices,  # 当前
            'rho_factor': 1.0,
            'gamma_factor': 1.0,
        },
        'S1_ssp245': {
            'climate': climate_2050_matrices['ssp245'],
            'rho_factor': 1.0,
            'gamma_factor': 1.0,
        },
        'S2_ssp585': {
            'climate': climate_2050_matrices['ssp585'],
            'rho_factor': 1.0,
            'gamma_factor': 1.0,
        },
        'S3_insurance': {
            'climate': climate_2050_matrices['ssp585'],
            'rho_factor': 0.5,
            'gamma_factor': 1.0,
        },
        'S4_safety_net': {
            'climate': climate_2050_matrices['ssp585'],
            'rho_factor': 1.0,
            'gamma_factor': 0.5,
        },
        'S5_combined': {
            'climate': climate_2050_matrices['ssp585'],
            'rho_factor': 0.5,
            'gamma_factor': 0.5,
        },
    }
    
    # 计算
    all_results = {}
    for scenario_name, config in scenarios.items():
        ce_by_country = compute_country_ce(
            config['climate'], 
            posterior_samples={
                'rho_c': posterior['rho_c'] * config['rho_factor'],
                'gamma_c': posterior['gamma_c'] * config['gamma_factor'],
            },
            df=df,
            feasibility_mask=feasibility_mask,
        )
        all_results[scenario_name] = ce_by_country
    
    return all_results
```

---

<a id="6-指标"></a>
## 6. 关键产出指标

### 6.1 气候损失 (Climate Loss)

```
ΔCE_climate = CE(S0_current) - CE(S2_ssp585)

正值 = 气候变化导致福利下降
报告: 绝对值($) 和 相对值(%)

按国家分解:
  ΔCE_climate_c = CE_c(S0) - CE_c(S2)
  % loss = ΔCE / CE_c(S0) × 100

预期: 5-25%的福利损失 (依赖于降雨和温度变化幅度)
```

### 6.2 政策价值 (Policy Value)

```
Insurance value    = CE(S3_insurance) - CE(S2_ssp585)
Safety net value   = CE(S4_safety_net) - CE(S2_ssp585)
Combined value     = CE(S5_combined) - CE(S2_ssp585)
Synergy            = Combined - (Insurance + Safety_net)

正值 = 该政策改善了福利
Synergy > 0 = 超线性协同效应

按国家报告 + 95% CI
```

### 6.3 政策优先级指标

```
Insurance/Safety_net ratio = Insurance_value / Safety_net_value

> 1: 该国家应优先部署保险
< 1: 该国家应优先部署安全网
≈ 1: 两者同等重要

这个ratio直接决定Figure的核心信息。
```

### 6.4 成本效益比 (Cost-Effectiveness)

```
CE per dollar:
  Insurance CE per $ = Insurance_value / insurance_cost_per_capita
  Safety_net CE per $ = Safety_net_value / safety_net_cost_per_capita

成本假设:
  Insurance: $10/年/人 (SSA天气指数保险的典型保费)
  Safety net: $30/年/人 (SSA现金转移项目的典型额度)

注: 成本假设不确定性大，在Discussion中讨论。
```

---

<a id="7-可视化"></a>
## 7. 可视化与Figure设计

### 7.1 Figure 3 (Nature Food正文): 政策×国家交互热力图

```
这是最核心的Figure。

结构: 6行(国家) × 3列(保险/安全网/组合)
颜色: CE改善幅度 (绿=大改善, 红=小改善)
数字: 每格内标注ΔCE值和95% CI

     | Insurance | Safety Net | Combined |
     | (ρ×0.5)   | (γ×0.5)    | (both)   |
─────┼───────────┼────────────┼──────────┤
Uganda |   +$X     |    +$Y     |   +$Z   |
ETH    |   +$X     |    +$Y     |   +$Z   |
MWI    |   +$X     |    +$Y     |   +$Z   |
NGA    |   +$X     |    +$Y     |   +$Z   |
MLI    |   +$X     |    +$Y     |   +$Z   |
TZA    |   +$X     |    +$Y     |   +$Z   |

关键pattern (预期):
  Malawi行: 保险列颜色最深 (ρ=3.30最高)
  Ethiopia行: 安全网列颜色最深 (γ/消费=150%)
  Nigeria行: 两列都浅 (已近风险中性)
```

### 7.2 Figure 4 (Nature Food正文): 气候损失地图

```
6个panel (每国一个), 或一张大SSA地图。
每个GPS点的颜色 = CE(current) - CE(2050) = 气候损失。

红色 = 大损失 (需要干预)
绿色 = 小损失或改善
灰色 = 数据缺失

叠加: 等高线或hatching标记"保险优先区"vs"安全网优先区"
```

### 7.3 Figure 5 (Nature Food正文或ED): 2050 vs Current的CE分布

```
每国一个panel:
  x轴: CE ($)
  y轴: 密度
  蓝色曲线: Current climate CE分布
  红色曲线: 2050 SSP5-8.5 CE分布
  绿色曲线: 2050 + 最优政策 CE分布
  
  阴影: BIRL后验不确定性的95% CI带
```

### 7.4 Extended Data Figures

```
ED Fig: SSP2-4.5 vs SSP5-8.5对比
ED Fig: 5个GCM的individual结果 (GCM uncertainty)
ED Fig: 保险强度敏感性 (ρ×0.3/0.5/0.7)
ED Fig: 安全网强度敏感性 (γ×0.3/0.5/0.7)
ED Fig: Synergy map (组合干预的超线性效应)
```

---

<a id="8-叙事"></a>
## 8. 论文叙事与写作指南

### 8.1 Results Section 3: "2050 Climate Projections and Policy Value"

```
段落1: 气候变化的收入效应
  "Under SSP5-8.5, growing-season rainfall in [Sahel countries] is projected
   to [increase/decrease] by X%, while temperatures rise by Y°C.
   Our environment model translates these changes into income distributions:
   median plot-level income declines by Z% across the six countries,
   with the largest declines in [country] and smallest in [country]."

段落2: 福利损失的国家间异质性
  "However, welfare losses (measured by certainty equivalent) vary dramatically
   across countries — not because of different climate exposure, but because
   of different risk preferences. [Country A] faces only X% CE decline despite
   Y% income decline, because its low ρ=1.35 makes farmers relatively
   indifferent to increased volatility. In contrast, [Country B]'s CE declines
   by Z% — nearly double the income effect — because high ρ=3.30 amplifies
   the welfare cost of increased uncertainty."

段落3: 保险vs安全网的差异化价值
  "Insurance (modeled as ρ_effective × 0.5) is most valuable in countries
   with high risk aversion: [Country] gains $X in CE per $Y of premium,
   a cost-effectiveness ratio of Z. Safety nets (γ_effective × 0.5) show
   the opposite pattern: most effective where subsistence constraints bind
   tightly, particularly in [Country] where γ exceeds median consumption.
   The policy-country interaction explains W% of the variance in policy
   effectiveness — implying that uniform, continent-wide programs waste
   approximately V% of their budgets relative to targeted deployment."

段落4: 组合干预的协同效应
  "Combining insurance and safety nets yields [super/sub]-linear returns:
   the joint effect exceeds the sum of individual effects by X% in [countries],
   suggesting that [mechanism]. This synergy is strongest where both ρ and γ
   are high — precisely the most vulnerable populations."
```

### 8.2 摘要中的2050句子

```
"Under 2050 high-emission scenarios, welfare losses range from X% (Nigeria)
to Y% (Malawi), driven primarily by risk preferences rather than differential
climate exposure. Crop insurance is most cost-effective in high-risk-aversion
countries (Malawi, Mali), while social safety nets dominate where subsistence
constraints bind (Ethiopia) — uniform policy deployment wastes an estimated
Z% of adaptation budgets."
```

---

<a id="9-时间线"></a>
## 9. 实施时间线

```
Day 1: CMIP6数据获取 (4小时)
  ├─ 选择GEE NASA/GDDP-CMIP6方案
  ├─ 写GEE提取脚本 (5 GCM × 2 SSP × baseline/2050)
  ├─ 提交GEE任务
  └─ 同时: 训练NDVI预测模型

Day 2: 气候数据处理 (4小时)
  ├─ 下载GEE输出
  ├─ 计算Delta factors (per GPS point × GCM)
  ├─ 计算ensemble mean
  ├─ 检查气候变化信号量级
  └─ 预测2050 NDVI

Day 3: 反事实矩阵生成 (2小时)
  ├─ 替换气候特征
  ├─ 跑环境模型预测 (×2 SSP)
  ├─ 验证: 2050 q50 vs current q50的差异合理性
  └─ 保存: ssp245_cf.npz, ssp585_cf.npz

Day 4: 福利计算 (4小时)
  ├─ 实现CE计算函数
  ├─ 6情景 × 6国 × 12,000后验样本
  ├─ 计算所有指标 (climate loss, policy value, synergy)
  └─ 检查: 结果的量级和方向是否合理

Day 5: 可视化 (6小时)
  ├─ Figure 3: 政策×国家热力图
  ├─ Figure 4: 气候损失地图
  ├─ Figure 5: CE分布对比
  └─ ED Figures

Day 6-7: 写作 + 敏感性 (8小时)
  ├─ Results Section 3 draft
  ├─ GCM敏感性 (5个individual GCM)
  ├─ 政策强度敏感性 (×0.3/0.5/0.7)
  └─ 整合到论文主体
```

---

<a id="10-文件"></a>
## 10. 文件结构

```
Formal Analysis/07_2050_Analysis/
├── extract_cmip6.py                # GEE提取脚本
├── process_cmip6.py                # 气候数据处理+delta计算
├── predict_ndvi_2050.py            # NDVI经验预测
├── generate_2050_cf_matrices.py    # 2050反事实矩阵
├── compute_welfare.py              # CE计算 + 不确定性传播
├── run_all_scenarios.py            # 6情景完整pipeline
├── make_figures.py                 # 所有可视化
├── 2050_analysis_report.md         # 完整报告
│
├── data/
│   ├── cmip6_raw/                  # GEE输出CSV
│   ├── cmip6_processed/            # Delta factors
│   ├── ndvi_model.joblib           # NDVI预测模型
│   ├── ssp245_cf.npz               # SSP2-4.5反事实矩阵
│   └── ssp585_cf.npz               # SSP5-8.5反事实矩阵
│
├── results/
│   ├── ce_by_scenario_country.csv  # 核心结果表
│   ├── policy_value.csv            # 政策价值
│   ├── climate_loss.csv            # 气候损失
│   └── synergy.csv                 # 协同效应
│
└── figures/
    ├── fig3_policy_heatmap.pdf     # 核心Figure
    ├── fig4_climate_loss_map.pdf
    ├── fig5_ce_distributions.pdf
    ├── ed_gcm_sensitivity.pdf
    └── ed_policy_strength.pdf
```

---

## 附录: 关键文献

```
CMIP6:
  Eyring et al. (2016) — CMIP6 overview
  O'Neill et al. (2016) — SSP scenarios
  Thrasher et al. (2022) — NASA GDDP-CMIP6

SSA气候影响:
  Sultan & Gaetani (2016) — Agriculture in West Africa under climate change
  Thornton et al. (2011) — Agriculture and food systems in Sub-Saharan Africa
  Challinor et al. (2014) — Meta-analysis of crop yield under climate change

政策:
  Cole et al. (2013) — Weather index insurance in India
  Karlan et al. (2014) — Agricultural decisions after relaxing credit constraints
  Berhane et al. (2014) — Ethiopia PSNP evaluation
  Jensen et al. (2017) — Insurance demand and design
```