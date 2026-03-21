# BIRL数据缺口分析与补充方案

**生成日期**：2026-03-16
**数据源**：`outputs/final/all_countries_panel_with_conflict.parquet` (514,665 × 148)

---

## 一、BIRL可用样本量：逐步筛选结果

从514,665 obs / 56,141 HH出发，按BIRL最低要求逐步过滤：

| Step | Obs | HH | 累计% | 本步丢失 |
|------|----:|---:|------:|--------:|
| Base | 514,665 | 56,141 | 100% | — |
| GPS坐标 | 513,690 | 56,022 | 99.8% | 975 |
| 作物名称 | 511,329 | 55,959 | 99.4% | 2,361 |
| 收获>0 | 429,874 | 54,595 | 83.5% | 81,455 |
| 产量>0 | 405,165 | 51,951 | 78.7% | 24,709 |
| 面积>0.001 | 401,222 | 51,918 | 78.0% | 3,943 |
| 非极端产量 | 399,324 | 51,870 | 77.6% | 1,898 |
| **气候数据(rain>0)** | **223,579** | **30,889** | **43.4%** | **175,745** |
| **土壤数据(clay)** | **144,505** | **18,390** | **28.1%** | **79,074** |
| 冲突数据 | 144,505 | 18,390 | 28.1% | 0 |

**最终可用：144,505 obs / 18,390 HH（28.1%）**

### 面板深度（全部筛选后）

| 条件 | HH数 |
|------|-----:|
| ≥1 wave | 18,390 |
| **≥2 waves** | **5,643** |
| ≥3 waves | 2,922 |
| ≥4 waves | 454 |
| 每户均值 | 1.5 waves |

### 按国分布（全部筛选后）

| Country | Obs | HH | HH≥2w | Obs/HH |
|---------|----:|---:|------:|-------:|
| Ethiopia | 5,985 | 713 | 255 | 8.4 |
| Malawi | 2,318 | 730 | 12 | 3.2 |
| Mali | 24,706 | 4,883 | 1,372 | 5.1 |
| Niger | 7,574 | 1,786 | 0 | 4.2 |
| Nigeria | 31,580 | 4,365 | 2,254 | 7.2 |
| Tanzania | 19,941 | 4,176 | 0 | 4.8 |
| Uganda | 52,401 | 1,906 | 1,668 | 27.5 |

### 放松要求的影响

| 方案 | Obs | HH | HH≥2w |
|------|----:|---:|------:|
| 全部要求 | 144,505 | 18,390 | 5,643 |
| **去掉气候要求** | **273,222** | **34,403** | **10,501** |
| **去掉气候+土壤** | **399,324** | **51,870** | **15,794** |

**关键发现**：气候和土壤是最大瓶颈。气候单独砍掉175K obs（44%）。如果将气候/土壤作为可选协变量而非必需筛选条件，可用HH从5,643→15,794（≥2w），增加近3倍。

---

## 二、气候数据缺失根因诊断

### 总体：46%的obs缺失气候数据，原因有三

#### 根因1：后期waves未提取坐标（最大瓶颈，占70%）

coords文件只提取了部分waves，后期waves完全没有GPS坐标输出：

| Country | Panel waves | Coords waves | 丢失的waves | 影响 |
|---------|------------|-------------|------------|------|
| Nigeria | 1,2,3,4,**5** | 1,2,3,4 | **w5** | 10,554 obs = 0%覆盖 |
| Ethiopia | 1,2,3,4,5 | 1,2,3,4,5 | (另有merge问题) | 见根因2 |
| Niger | 1,**2** | 1 | **w2** | 0%覆盖 |
| Tanzania | 1,2,3,**4,5** | 1,2,3 | **w4,5** | 0%覆盖 |
| Uganda | 1,2,3,**4,5,7,8** | 1,2,3 | **w4,5,7,8** | 0%覆盖 |
| Malawi | 1,2,3,4 | 1,2,3,4 | 无丢失 ✓ | |
| Mali | 1,2 | 1,2 | 无丢失 ✓ | |

**Panel有81,289个HH-wave，coords只有59,312个。32,022个HH-wave有GPS但无coords提取。**

各国气候覆盖by wave：
```
Nigeria:   w1=99%  w2=98%  w3=100%  w4=100%  w5=0%
Ethiopia:  w1=2%   w2=9%   w3=9%    w4=17%   w5=0%
Niger:     w1=98%  w2=0%
Tanzania:  w1=91%  w2=100% w3=100%  w4=0%    w5=0%
Uganda:    w1=100% w2=99%  w3=97%   w4=0%    w5=0%  w7=0%  w8=0%
Malawi:    w1=100% w2=100% w3=100%  w4=100%
Mali:      w1=100% w2=100%
```

#### 根因2：Ethiopia的hh_id_merge类型不匹配

Ethiopia的hh_id_merge在coords中被转为int64（丢失前导零），导致merge时匹配率只有12.8%（1,549/12,137 HH-wave）。实际coords文件里有11,746行，但因为ID不匹配几乎全部丢失。这解释了Ethiopia即使在有coords的waves也只有2-17%覆盖。

#### 根因3：EA/district imputed GPS未进入coords

`gps_source = ea_median` 或 `district_median` 的观测（23%）完全没有气候数据：
- `plot_gps` (76.8% of obs): 70.3%有气候
- `ea_median` (13.3% of obs): **0%有气候**
- `district_median` (9.7% of obs): **0%有气候**

气候提取只覆盖了有原始GPS坐标的household，imputed坐标被跳过。

### 能否用GEE补救？

**可以，而且是最佳方案。** 具体路径：

1. **重新生成完整coords文件**：从panel的`gps_lat_final`/`gps_lon_final`提取所有unique (lat, lon, year)组合（包括imputed坐标），修复Ethiopia ID类型，覆盖所有waves
   - 当前coords: 59,312个HH-wave
   - 目标coords: ~81,000个HH-wave（+37%）
   - Unique GPS点: ~7,019个（远少于HH-wave，因为同一GPS点跨多个wave）

2. **GEE提取气候数据**：对这些坐标点，用GEE JavaScript/Python API提取：
   - **CHIRPS降雨**：`ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')`，按月聚合 → 月降雨mm
   - **MODIS NDVI**：`ee.ImageCollection('MODIS/061/MOD13A1')`，16天合成 → 月均NDVI
   - **ERA5温度**（可选）：`ee.ImageCollection('ECMWF/ERA5_LAND/MONTHLY_AGGR')` → 月均/最高/最低温
   - 时间范围：2008-2023（覆盖所有waves的种植-收获季节）

3. **GEE脚本已有雏形**：
   - `GEE/download_additional_CHIRPS.js` — CHIRPS下载脚本
   - `src/extractors/climate_extractor.py` — 现有的本地raster提取器（可改为GEE版）

4. **预期效果**：
   - 气候覆盖从 **54% → ~99.8%**（仅975个完全无GPS的obs丢失）
   - Ethiopia从 8% → ~100%
   - 后期waves全部补齐

### 工作量估计

| 步骤 | 工作量 | 说明 |
|------|--------|------|
| 重建coords（Python） | 0.5小时 | 从panel提取unique GPS×year |
| 写GEE提取脚本 | 2-3小时 | CHIRPS + MODIS point extraction for 7K points × 15 years |
| GEE运行 | 1-2小时 | GEE cloud computation |
| 聚合为growing season指标 | 1小时 | 月数据→生长季累计/均值 |
| 重新merge到panel | 0.5小时 | 替换现有climate intermediate |
| **总计** | **5-7小时** | |

### 本地raster vs GEE对比

| 方案 | 优势 | 劣势 |
|------|------|------|
| 本地raster（当前） | 无需网络/GEE账号 | 需要下载大量.tif文件（~50GB），只覆盖已下载的时间段 |
| **GEE提取（推荐）** | 无需下载数据，全时间覆盖，按点提取更高效 | 需要GEE账号，有配额限制（~5000请求/天） |

---

## 三、缺失变量清单与补充来源

### A. Plot_dataset.dta中有但panel中没有的关键变量（79个）

**来源**：`Nigeria/meta/Plot_dataset.dta`（263,195 rows × 115 cols）
**Merge key**：(country, hh_id_merge, wave, plot_id_merge) → 与panel多对一

#### BIRL核心需求变量（按类别）

| 类别 | 变量 | 完整度 | BIRL角色 |
|------|------|--------|----------|
| **人口统计** | `age_manager` | 93.2% | state vector |
| | `female_manager` | 93.4% | state vector |
| | `married_manager` | 93.1% | heterogeneity |
| | `formal_education_manager` | 92.8% | state vector |
| **劳动力** | `total_labor_days` | 94.5% | action vector |
| | `total_family_labor_days` | 92.9% | action vector |
| | `total_hired_labor_days` | 94.1% | action vector |
| | `hired_labor_value_LCU/USD` | 94.0% | cost |
| **施肥** | `inorganic_fertilizer` (0/1) | 95.2% | action vector |
| | `nitrogen_kg` | 94.4% | action vector |
| | `inorganic_fertilizer_value_LCU/USD` | 93.3% | cost |
| | `organic_fertilizer` (0/1) | 94.6% | action vector |
| **资产** | `ag_asset_index` | 92.4% | λ_i识别 |
| | `farm_size` | 96.0% | state vector |
| | `livestock` (0/1) | 98.0% | wealth proxy |
| | `plot_owned` | 96.5% | tenure |
| | `plot_certificate` | 90.3% | tenure |
| | `tractor` (0/1) | 87.2% | mechanization |
| **管理** | `intercropped` | 96.1% | action vector |
| | `irrigated` | 95.5% | action vector |
| | `main_crop` | 100% | action vector |
| | `nb_seasonal_crop` | 97.4% | diversification |
| **土壤(问卷)** | `soil_fertility_index` | 78.0% | state vector |
| | `nutrient_availability` | 78.0% | state vector |
| **市场** | `dist_market` | 38.3% | state vector |
| | `dist_popcenter` | 78.0% | state vector |

### B. Household_dataset.dta中有但panel中没有的变量（15个）

**来源**：`Nigeria/meta/Household_dataset.dta`（148,421 rows × 33 cols）
**Merge key**：(hh_id_merge, wave) → 与panel多对一
**匹配率**：99.8%（246 HH-wave未匹配）

| 变量 | 完整度 | BIRL角色 |
|------|--------|----------|
| `hh_size` | 99.6% | state vector（家庭规模） |
| `hh_asset_index` | 99.4% | λ_i识别（资产指数） |
| `hh_dependency_ratio` | 99.5% | state vector |
| `hh_formal_education` | 98.9% | heterogeneity |
| `hh_electricity_access` | 99.1% | wealth proxy |
| `nonfarm_enterprise` (0/1) | 95.4% | 非农收入 |
| `hh_shock` (0/1) | 95.9% | state vector |
| `HDDS` (饮食多样性) | 93.1% | welfare measure |
| `totcons_USD` | 81.0% | 总消费（λ_i识别关键） |
| `cons_quint` | 83.0% | 消费五分位 |
| `share_kg_sold` | 41.7% | 市场参与度 |
| `nb_plots` | 71.4% | 规模 |

### C. 仍然缺失的（两个dta都没有）

| 变量 | 状态 | 影响 |
|------|------|------|
| 信贷/借贷行为 | 原始CSV sect6中有，dta中无 | 风险管理行动缺失 |
| 保险购买 | 原始CSV中可能有 | 风险管理行动缺失 |
| 非农收入明细 | `nonfarm_enterprise`是0/1，无金额 | λ_i识别受限 |

---

## 四、Crop Category映射缺口

当前覆盖率：**42.4%**（218,002/514,665）

### 现状分析

- 99.5%的obs有crop_name（1,421个unique值）
- 映射文件：`Data/External/CropCategories/crop_category_mapping.csv`（525行）
- 已映射5个类别：cereals 46.5%, roots 23.9%, legumes 20.1%, cash_crops 8.2%, vegetables 1.3%

### 长尾分布

| 覆盖目标 | 需要映射的crop数 | 工作量 |
|----------|----------------:|--------|
| 75% | top 50 | 1-2小时 |
| **90%** | **top 109** | **2-3小时** |
| 95% | top 168 | 5-7小时 |
| 99% | top 348 | 大量手工 |

**未映射的主要原因**：法语名（MAÏS, SORGHO, ARACHIDE）、复数变体（COWPEAS→COWPEA）、拼写错误（ASSAVA→CASSAVA）、方言名（TEFF, ENSET）。

---

## 五、实施方案（优先级排序）

### Step 1：合并 Plot_dataset.dta + Household_dataset.dta（1-2小时）

```
Panel (148 cols, 514K rows)
  + Plot_dataset.dta (~30个BIRL变量) on (country, hh_id_merge, wave, plot_id_merge)
  + Household_dataset.dta (~12个HH变量) on (hh_id_merge, wave)
  → 输出: ~190 cols, 514K rows
```

**关键文件**：
- `Nigeria/meta/Plot_dataset.dta` — 施肥、劳动力、人口统计、资产
- `Nigeria/meta/Household_dataset.dta` — 家庭规模、资产指数、消费、非农收入
- `Nigeria/outputs/final/all_countries_panel_with_conflict.parquet` — 当前panel

**注意事项**：
- Plot_dataset是plot-level（263K rows），panel是plotcrop-level（514K rows），需要plot_id_merge去重后merge
- Household_dataset是HH-wave level（148K rows），直接多对一merge
- 合并后做完整性检查：行数不变、无多对多爆炸

### Step 2：GEE补齐气候数据（5-7小时）

1. 从panel提取完整coords（all HH-wave with GPS，修复Ethiopia ID类型）
2. 写GEE脚本提取CHIRPS+MODIS（对~7,019 unique GPS点 × 15年）
3. 聚合为growing season指标（rainfall_growing_sum, ndvi_growing_mean等）
4. 替换现有climate intermediate，重新merge

### Step 3：补全Crop Category映射（2-3小时）

- 扩展 `crop_category_mapping.csv`，覆盖top 109 crops
- 主要是法语翻译 + 拼写变体 + 复数统一
- 重跑 `panel_cleaner.py` 的crop mapping逻辑
- 目标：42% → 90%

### Step 4：重跑样本筛选 + 更新RP第3章

合并新变量+补齐气候后重新评估BIRL可用N：
- 预期：气候覆盖99.8%，不再是瓶颈
- 新硬筛选：yield>0 + crop_category非空 + 人口统计非空
- 预期最终：HH≥2w ~15,000-20,000

---

## 六、关键结论

1. **RP写的148,421户 vs 实际56,141户**：差距来自RP用的是Household_dataset行数（含所有HH-wave），实际unique HH是56,141
2. **BIRL严格筛选后只剩5,643户（≥2w）**：主要瓶颈是气候（-44%）和土壤（-15%），不是变量缺失
3. **气候缺失是pipeline bug，不是数据不存在**：后期waves未提取coords + Ethiopia ID类型错误 + imputed GPS被跳过
4. **GEE可以彻底解决气候缺口**：54% → 99.8%，工作量5-7小时
5. **缺失变量全部可从现有dta文件补充**：无需回原始CSV，1-2小时merge搞定
6. **Crop category可通过扩展mapping达到90%**：2-3小时手工映射
7. **修复后实测：HH≥2w从5,643 → 10,494（含土壤）/ 15,768（不含土壤）**，BIRL完全可行

---

## 七、修复后实测结果（2026-03-16）

### 已完成的修复

| Step | 结果 |
|------|------|
| dta合并 | 148→197列 (+49个BIRL变量: 人口/劳动力/施肥/资产/HH特征) |
| Coords重建 | 59,312→81,137 HH-waves (+37%), 7,018 unique GPS点 |
| GEE云端提取 | 1,431,672行 CHIRPS+NDVI, 0错误, 68分钟完成 |
| 季节聚合+merge | 气候覆盖 54%→**99.5%** |

### 气候覆盖修复前后对比

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| rainfall_growing_sum | 54.0% | **99.5%** |
| ndvi_growing_mean | 54.5% | **99.8%** |
| rainfall_10yr_mean | 54.5% | **99.5%** |

### 修复后BIRL样本筛选

| Step | Obs | HH | Cum% |
|------|----:|---:|-----:|
| Base | 514,665 | 56,141 | 100% |
| GPS | 513,690 | 56,022 | 99.8% |
| Crop name | 511,329 | 55,959 | 99.4% |
| Harvest>0 | 429,874 | 54,595 | 83.5% |
| Yield>0 | 405,165 | 51,951 | 78.7% |
| Area>0.001 | 401,222 | 51,918 | 78.0% |
| Not extreme | 399,324 | 51,870 | 77.6% |
| **NEW Climate** | **398,310** | **51,531** | **77.4%** |
| Soil | 272,825 | 34,259 | 53.0% |

**气候不再是瓶颈**：修复前气候砍掉44%数据，修复后只丢0.2%。

### 修复后面板深度

| 方案 | Obs | HH | HH≥2w |
|------|----:|---:|------:|
| 全筛选（含土壤） | 272,825 | 34,259 | **10,494** |
| 不含土壤 | 398,310 | 51,531 | **15,768** |
| **修复前（全筛选）** | **144,505** | **18,390** | **5,643** |

**HH≥2w从5,643提升到10,494（+86%），不含土壤达15,768（+180%）。**

### 修复后各国分布（全筛选含土壤）

| Country | Obs | HH | HH≥2w |
|---------|----:|---:|------:|
| Ethiopia | 61,623 | 5,718 | 3,208 |
| Malawi | 2,318 | 730 | 12 |
| Mali | 24,706 | 4,883 | 1,372 |
| Niger | 14,670 | 3,292 | 0 |
| Nigeria | 38,782 | 4,596 | 3,784 |
| Tanzania | 37,611 | 7,913 | 355 |
| Uganda | 93,115 | 7,332 | 1,672 |

Ethiopia从713 HH飙升到5,718 HH（ID类型修复+全waves补齐的效果）。
