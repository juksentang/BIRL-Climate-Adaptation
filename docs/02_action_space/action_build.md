# Action Space Construction Documentation

**输入**: `birl_sample.parquet` — 222,023 obs, 6国, 15,644 HH
**输出**: 27个离散动作 (9 crops × 3 intensity tiers), 6个AEZ可行性zone
**日期**: 2026-03-16

---

## 1. 设计原则

### 1.1 决策单位 = Plot-Crop

BIRL的决策单位定义在**plot-crop level**（222,023 obs），而非household-wave-season level（42,318个决策点）。

**理由**：
- 农户面对每一块地独立决策"种什么、投入多少"。不同地块的土壤、面积、坡度不同，选择自然不同。
- LSMS-ISA问卷按plot收集数据，这是数据的原生粒度。
- 同一农户同一季的多个plot-crop共享行为参数 θ_i = (α_i, γ_i, λ_i)，联合约束参数识别。
- 有效观测数从42K增到222K，对参数精度有巨大提升。

**BIRL似然函数**：

```
L_i = ∏_t ∏_s ∏_p π(a_{itsp} | s_{itsp}; θ_i)
```

其中 t=wave, s=season, p=plot。θ_i 在农户 i 的所有 plot-crop 间共享。

**非独立性处理**：同一(hh, wave, season)下的多个plot选择不条件独立。主分析直接用plot-level，敏感性分析加decision-point level对比（取主作物），通过cluster bootstrap计算标准误。

### 1.2 动作 = 作物选择 × 投入强度

```
action = crop_choice (9 categories) × input_intensity (3 tiers)
       = 27 discrete actions
```

**间作(intercropped)作为状态变量**，不进入动作编码。理由：农户先选crop再安排种植方式，间作更多受地块大小和当地惯例约束，不是核心"投资决策"。BIRL可以从状态中学到"间作地块上的选择模式不同"。

---

## 2. 作物类别构建 (11 → 9)

### 2.1 映射规则

| 原始main_crop | → action_crop | 合并理由 |
|---------------|--------------|---------|
| MAIZE | **maize** | — |
| PERENNIAL/FRUIT | **tree_crops** | 多年生作物，生产周期长 |
| NUTS | **tree_crops** | 同为多年生 |
| TUBERS / ROOT CROPS | **tubers** | — |
| BEANS AND OTHER LEGUMES | **legumes** | — |
| SORGHUM | **sorghum_millet** | 同为热带旱地谷物 |
| MILLET | **sorghum_millet** | 同上 |
| RICE | **rice** | — |
| WHEAT | **wheat_barley** | 同为温带谷物，仅Ethiopia |
| BARLEY | **wheat_barley** | 同上 |
| OTHER (crop_name=TEFF) | **teff** | Ethiopia主粮，拆出OTHER |
| OTHER (其余) | **other** | 异质类（KALE, BANANA, PUMPKINS等） |

### 2.2 TEFF拆分

OTHER类25,158 obs中TEFF占5,522（21.9%）。TEFF是Ethiopia特有的主粮（Eragrostis tef），农学性质独特（极小粒种子、耐渍、高海拔适应），与OTHER中其他作物完全不同。拆出后OTHER内部更均匀（top crop MAIZE仅7.5%）。

### 2.3 最终分布

| action_crop | Obs | % | 主要国家 |
|-------------|----:|--:|---------|
| tree_crops | 52,633 | 23.7% | Uganda(banana/coffee), Ethiopia(enset), Nigeria(cocoa/plantain) |
| maize | 41,385 | 18.6% | 全部6国 |
| tubers | 31,079 | 14.0% | Nigeria(cassava/yam), Uganda, Tanzania |
| legumes | 30,933 | 13.9% | 全部6国 |
| sorghum_millet | 29,064 | 13.1% | Nigeria, Mali, Niger区, Ethiopia |
| other | 19,636 | 8.8% | 分散 |
| wheat_barley | 6,579 | 3.0% | 仅Ethiopia |
| teff | 5,522 | 2.5% | 仅Ethiopia |
| rice | 5,192 | 2.3% | Mali, Nigeria, Tanzania |

---

## 3. 投入强度分档

### 3.1 问题：施肥率的巨大异质性

不同作物的施肥率差异极大，无法用统一标准三档：

| 作物类型 | 零施肥率 | P33 fert/ha | P67 fert/ha |
|---------|-------:|----------:|----------:|
| Wheat/Barley | 32% | 5.4 | 45.0 |
| Maize | 56% | 0 | 26.9 |
| Rice | 60% | 0 | 22.4 |
| **Legumes** | **85%** | **0** | **0** |
| **Tubers** | **89%** | **0** | **0** |
| **Tree crops** | **93%** | **0** | **0** |

对低施肥作物，P33和P67都是0，三档PCA无意义。

### 3.2 自适应分档方案

#### 高施肥作物（maize, rice, wheat_barley, sorghum_millet, teff）

使用**PCA联合分档**：

```python
# 对每个 (country, crop) 组合
X = StandardScaler().fit_transform([fertilizer_kg_ha, labor_days_ha])
pc1 = PCA(n_components=1).fit_transform(X)
# pc1综合了施肥和劳动力的标准化信息

# 按P33/P67三分
low:    pc1 ≤ P33
medium: P33 < pc1 ≤ P67
high:   pc1 > P67
```

**效果**：这些作物的三档分布为 33.2-33.8%（几乎完美均匀），因为PCA是在连续变量上分位切割。

#### 低施肥作物（tubers, legumes, tree_crops, other）

使用**二元施肥 × 劳动力中位数**：

```python
used_fert = fertilizer_kg_ha > 0    # 二元
high_labor = labor_days_ha > median  # 二元

# 4种组合 → 合并为3档
low:    不施肥 + 低劳动
medium: 不施肥 + 高劳动  OR  施肥 + 低劳动
high:   施肥 + 高劳动
```

**效果**：low 42-48%, medium 48-50%, high 4-10%。High档偏小是因为低施肥作物中"施肥+高劳动"确实是少数行为。

### 3.3 各作物投入强度分布

| action_crop | Low | Medium | High | 分档方法 |
|-------------|----:|-------:|-----:|---------|
| maize | 33.4% | 33.3% | 33.3% | PCA三分 |
| sorghum_millet | 33.3% | 33.3% | 33.3% | PCA三分 |
| teff | 33.3% | 33.3% | 33.3% | PCA三分 |
| rice | 33.2% | 33.8% | 33.1% | PCA三分 |
| wheat_barley | 33.2% | 33.6% | 33.2% | PCA三分 |
| tree_crops | 47.7% | 48.4% | 3.9% | 二元×劳动 |
| tubers | 44.5% | 49.5% | 6.0% | 二元×劳动 |
| legumes | 42.9% | 49.2% | 7.9% | 二元×劳动 |
| other | 41.2% | 49.3% | 9.5% | 二元×劳动 |

---

## 4. 完整动作空间

### 4.1 27个动作

| ID | Label | Obs | % |
|---:|-------|----:|--:|
| 0 | maize_low | 13,805 | 6.2% |
| 1 | maize_medium | 13,789 | 6.2% |
| 2 | maize_high | 13,791 | 6.2% |
| 3 | tree_crops_low | 25,130 | 11.3% |
| 4 | tree_crops_medium | 25,454 | 11.5% |
| 5 | tree_crops_high | 2,049 | 0.9% |
| 6 | tubers_low | 13,829 | 6.2% |
| 7 | tubers_medium | 15,395 | 6.9% |
| 8 | tubers_high | 1,855 | 0.8% |
| 9 | legumes_low | 13,261 | 6.0% |
| 10 | legumes_medium | 15,216 | 6.9% |
| 11 | legumes_high | 2,456 | 1.1% |
| 12 | sorghum_millet_low | 9,691 | 4.4% |
| 13 | sorghum_millet_medium | 9,688 | 4.4% |
| 14 | sorghum_millet_high | 9,685 | 4.4% |
| 15 | teff_low | 1,841 | 0.8% |
| 16 | teff_medium | 1,840 | 0.8% |
| 17 | teff_high | 1,841 | 0.8% |
| 18 | other_low | 8,082 | 3.6% |
| 19 | other_medium | 9,688 | 4.4% |
| 20 | other_high | 1,866 | 0.8% |
| 21 | rice_low | 1,723 | 0.8% |
| 22 | rice_medium | 1,753 | 0.8% |
| 23 | rice_high | 1,716 | 0.8% |
| 24 | wheat_barley_low | 2,184 | 1.0% |
| 25 | wheat_barley_medium | 2,211 | 1.0% |
| 26 | wheat_barley_high | 2,184 | 1.0% |

**全部27个动作都≥50 obs**（最小1,716）。无需合并低频动作。

### 4.2 动作编码规则

```
action_id = crop_index × 3 + intensity_index

crop_index:     0=maize, 1=tree_crops, 2=tubers, 3=legumes,
                4=sorghum_millet, 5=teff, 6=other, 7=rice, 8=wheat_barley
intensity_index: 0=low, 1=medium, 2=high
```

---

## 5. AEZ可行性掩码

### 5.1 AEZ处理

**原始AEZ**：7个zone (FAO GAEZ编码 311-324)，94.5%覆盖。

**处理**：
- AEZ 311（648 obs，干旱热带）合并到 AEZ 312（半干旱热带），因为部分crop×zone cell <10 obs
- 缺失AEZ（5.5%）用 country × admin_1 的众数填补
- 最终：**6个zone**

### 5.2 Zone定义

| Zone | FAO描述 | Obs | 气候特征 |
|------|---------|----:|---------|
| 312 | 热带温暖-半干旱(含311干旱) | 39,356 | 萨赫勒、北Nigeria |
| 313 | 热带温暖-亚湿润 | 33,520 | Guinea savanna |
| 314 | 热带温暖-湿润 | 32,067 | 森林-草原过渡带 |
| 322 | 热带冷凉-半干旱 | 20,828 | Ethiopia高地干区 |
| 323 | 热带冷凉-亚湿润 | 46,029 | Ethiopia/Uganda高原 |
| 324 | 热带冷凉-湿润 | 38,049 | 高地湿润区 |

### 5.3 可行性掩码

对每个(zone, action_id)组合，如果该zone中该action出现≥10次，标记为feasible。

| Zone | Feasible动作数 (/27) | 说明 |
|:----:|:-------------------:|------|
| 312 | 21 | 缺teff、wheat_barley、部分rice |
| 313 | 21 | 类似312 |
| 314 | 21 | 缺teff、wheat_barley（热带低地无这些作物） |
| 322 | 24 | Ethiopia高地，大部分作物可行 |
| 323 | 27 | 全部可行（最多样化的zone） |
| 324 | 27 | 全部可行 |

掩码存储在 `action_space_config.json` 的 `feasibility_mask` 字段中。

---

## 6. 验证结果

| 检验项 | 结果 | 要求 |
|--------|------|------|
| 最小动作频率 | 1,716 | ≥50 ✅ |
| 最少zone可行动作 | 21 | ≥3 ✅ |
| 最高频动作占比 | 11.5% (tree_crops_medium) | <30% ✅ |
| 全部obs有action_id | 100% | 100% ✅ |
| 投入强度均衡 | 高施肥作物~33/33/33; 低施肥作物~45/48/7 | — |

---

## 7. 与BIRL模型的接口

### 7.1 状态向量 s_{itsp}

环境模型的输入特征包含plot-level和household-level两层：

**Plot-level**（因plot而异）：
- `plot_area_GPS`: 地块面积
- `intercropped`: 是否间作（状态，不是动作）
- `plot_owned`: 土地权属
- `irrigated`: 灌溉
- `slope_deg`, `elevation_m`: 地形
- `soil_fertility_index`: 问卷土壤质量

**Household-level**（同一HH所有plot共享）：
- 气候：`rainfall_growing_sum_final`, `ndvi_growing_mean_final`, `era5_tmean_growing_final`
- 人口：`age_manager`, `female_manager`, `hh_size`, `hh_asset_index`
- 冲突：`conflict_events_25km_12m`
- 市场：`travel_time_city_min`

### 7.2 动作向量 a_{itsp}

```python
action_id ∈ {0, 1, ..., 26}  # 27 discrete actions
# 可解码为:
action_crop = CROP_ORDER[action_id // 3]
input_intensity = ['low', 'medium', 'high'][action_id % 3]
```

### 7.3 可行动作集 A(s)

```python
feasible_actions = feasibility_mask[action_zone]
# action_zone ∈ {312, 313, 314, 322, 323, 324}
```

BIRL的softmax概率只在feasible动作上归一化：

```
π(a | s; θ) = exp(Q(s,a;θ)) / Σ_{a' ∈ A(s)} exp(Q(s,a';θ))
```

### 7.4 输出文件

| 文件 | 内容 |
|------|------|
| `birl_sample.parquet` | 新增列: action_crop, input_intensity, action_id, action_zone |
| `action_space_config.json` | action_labels, feasibility_mask, frequencies, zone list |

---

## 8. 设计决策记录

1. **Plot-crop level决策单位**：222K obs共享HH参数，vs 42K聚合decision points。选择前者，敏感性分析用后者对比。
2. **间作是状态不是动作**：不膨胀动作空间，让BIRL从状态中学习间作效应。
3. **TEFF单独类别**：从OTHER拆出（21.9%占比，Ethiopia主粮，农学独特）。
4. **自适应投入分档**：高施肥作物PCA三分，低施肥作物二元施肥×劳动力合并。避免80-93%零施肥作物的无意义三分。
5. **AEZ 311合并到312**：648 obs太少，某些crop×zone cell <10 obs。
6. **全部27个动作活跃**：最小频率1,716，无需合并。
