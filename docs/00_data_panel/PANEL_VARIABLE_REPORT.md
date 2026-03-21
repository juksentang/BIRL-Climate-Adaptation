# BIRL Panel Variable Report

**Dataset**: `outputs/final/all_countries_panel_birl.parquet`
**Total**: 514,665 observations × 211 variables
**Countries**: Ethiopia, Malawi, Mali, Niger, Nigeria, Tanzania, Uganda (7)
**Waves**: [1, 2, 3, 4, 5, 7, 8]
**Time span**: 2008–2023
**Unique households**: 56,141
**Unique plots**: 192,552
**Unique GPS points**: 7,019
**Generated**: 2026-03-16 (v3 — growing season fix + CHIRPS 1997-2006)

## Sample Distribution by Country

| Country | Obs | % | Households | Plots | EAs |
|---------|----:|--:|----------:|------:|----:|
| Ethiopia | 98,415 | 19.1 | 7,808 | 54,665 | 669 |
| Malawi | 40,332 | 7.8 | 7,789 | 25,027 | 204 |
| Mali | 34,984 | 6.8 | 6,671 | 29,505 | 898 |
| Niger | 23,948 | 4.7 | 4,083 | 12,609 | 204 |
| Nigeria | 62,905 | 12.2 | 6,306 | 16,902 | 771 |
| Tanzania | 74,585 | 14.5 | 13,208 | 26,724 | 782 |
| Uganda | 179,496 | 34.9 | 12,542 | 27,386 | 293 |
| **Total** | **514,665** | **100.0** | **56,141** | **192,552** | **3,821** |

## Growing Season Definitions & Climate Window Assignment

Climate variables with `_final` suffix use a **three-layer** strategy to assign the correct growing season window to each observation:

1. **Layer 1 — Precise dates (37.9%)**: For observations with valid `planting_month` AND `harvest_end_month`, climate is aggregated over the actual planting-to-harvest period.
2. **Layer 2 — Revised calendar (50.3%)**: For observations without precise dates, a country×season calendar is used (see table below). Season 2 is now supported for Uganda, Ethiopia, and Tanzania.
3. **Layer 2 Tanzania special (11.8%)**: Tanzania has 0% planting date coverage. Season is inferred from `harvest_end_month` (harvest Jan-Jul → Masika; Aug-Oct → Vuli).

The `climate_source` column records which layer was used for each observation.

### Revised Calendar (Layer 2)

| Country | Season 1 | Pre-planting 1 | Season 2 | Pre-planting 2 | Cross-year |
|---------|----------|----------------|----------|----------------|:----------:|
| Nigeria | Apr–Oct (4–10) | Jan–Mar | — | — | No |
| Ethiopia | May–Dec (5–12) | Feb–Apr | Feb–Jun (Belg) | Nov–Jan | No/No |
| Malawi | Nov–Apr (11–4) | Aug–Oct | — | — | **Yes** |
| Tanzania | Nov–May (11–5) | Aug–Oct | Oct–Jan (Vuli) | Jul–Sep | **Yes**/**Yes** |
| Uganda | Mar–Jul (3–7) | Jan–Feb | Sep–Jan (9–1) | Jul–Aug | No/**Yes** |
| Niger | Jun–Sep (6–9) | Mar–May | — | — | No |
| Mali | Jun–Oct (6–10) | Mar–May | — | — | No |

### Changes from v1 Calendar

| Country | v1 (old) | v3 (new) | Rationale |
|---------|----------|----------|-----------|
| Ethiopia | Jun–Sep only | **May–Dec** + Belg Feb–Jun | 35% of obs planted in May; harvest extends to Dec |
| Uganda | Mar–Jun, Season 2 ignored | Mar–Jul + **Sep–Jan Season 2** | 47.5% of obs are Season 2; old window covered only 49% |
| Tanzania | Nov–May uniform | **Nov–May + Oct–Jan Vuli** inferred from harvest | Season assigned via harvest month (0% planting data) |

### 10-Year Rainfall Lookback

`rainfall_10yr_mean_final` and `rainfall_10yr_cv_final` use the same growing-season months for each of the 10 years prior to the observation's base year. CHIRPS data now covers **1997–2023** (supplemented from original 2007–2023), so **100% of observations have a full 10-year lookback**. `rainfall_10yr_nyears` records the actual number of years used.

## Complete Variable Dictionary

### Core Identifiers / 核心标识符

| # | Variable | Description | Type | Complete % | Unique | Min | Median | Max |
|--:|----------|-------------|------|----------:|-------:|----:|-------:|----:|
| 1 | `country` | 国家 | str | 100.0 | 7 | — | — | — |
| 2 | `wave` | 调查轮次 | float | 100.0 | 7 | 1.00 | 3.00 | 8.00 |
| 3 | `year` | 年份 | int | 100.0 | 15 | 2,008.0 | 2,014.0 | 2,023.0 |
| 4 | `season` | 季节(1=主,2=副) | float | 100.0 | 2 | 1.00 | 1.00 | 2.00 |
| 5 | `hh_id_merge` | 统一户ID | str | 100.0 | 56,141 | — | — | — |
| 6 | `hh_id_obs` | 原始户ID | float | 99.7 | 39,113 | 1,000,002 | 5,008,173 | 7,005,654 |
| 7 | `plot_id_merge` | 统一地块ID | str | 100.0 | 192,552 | — | — | — |
| 8 | `plot_id_obs` | 原始地块ID | float | 100.0 | 299,809 | 1,000,001 | 5,030,266 | 7,090,606 |
| 9 | `parcel_id_merge` | 统一宗地ID | str | 71.6 | 88,452 | — | — | — |
| 10 | `parcel_id_obs` | 原始宗地ID | float | 71.3 | 98,165 | 1,000,001 | 4,010,030 | 7,033,879 |
| 11 | `ea_id_merge` | 统一枚举区ID | str | 87.9 | 3,821 | — | — | — |
| 12 | `ea_id_obs` | 原始枚举区ID | float | 87.9 | 3,820 | 1,000,001 | 5,000,352 | 7,000,317 |
| 13 | `strataid` | 抽样层 | float | 90.9 | 64 | 1.00 | 6.00 | 99.00 |
| 14 | `pw` | 抽样权重 | float | 99.7 | 32,393 | 4.39 | 1,909.6 | 130,002.2 |
| 15 | `geocoords_id` | GPS坐标ID | float | 76.8 | 7,656 | 1,000,001 | 5,000,014 | 7,001,236 |
| 16 | `gps_id` | GPS点ID(lat_lon) | str | 100.0 | 7,019 | — | — | — |

### Administrative Geography / 行政区划

| # | Variable | Description | Type | Complete % | Unique | Min | Median | Max |
|--:|----------|-------------|------|----------:|-------:|----:|-------:|----:|
| 1 | `admin_1` | 一级行政区代码 | float | 99.8 | 32 | 0 | 3.00 | 55.00 |
| 2 | `admin_1_name` | 一级行政区名 | str | 94.8 | 148 | — | — | — |
| 3 | `admin_2` | 二级行政区代码 | float | 88.8 | 512 | 1.00 | 107.0 | 55,552.0 |
| 4 | `admin_2_name` | 二级行政区名 | str | 57.2 | 872 | — | — | — |
| 5 | `admin_3` | 三级行政区代码 | float | 80.8 | 3,720 | 1.00 | 3,102.0 | 55,552,341 |
| 6 | `admin_3_name` | 三级行政区名 | str | 47.7 | 2,031 | — | — | — |
| 7 | `urban` | 城乡(0=农村,1=城市) | float | 99.8 | 2 | 0 | 0 | 1.00 |

### GPS Coordinates / GPS坐标

| # | Variable | Description | Type | Complete % | Unique | Min | Median | Max |
|--:|----------|-------------|------|----------:|-------:|----:|-------:|----:|
| 1 | `gps_lat_final` | 最终纬度(imputation后) | float | 99.8 | 7,610 | -16.99 | 1.70 | 17.78 |
| 2 | `gps_lon_final` | 最终经度 | float | 99.8 | 7,165 | -11.98 | 32.65 | 44.36 |
| 3 | `lat_modified` | 原始纬度(隐私偏移) | float | 99.8 | 7,610 | -16.99 | 1.70 | 17.78 |
| 4 | `lon_modified` | 原始经度 | float | 99.8 | 7,165 | -11.98 | 32.65 | 44.36 |
| 5 | `ea_lat_median` | EA中位数纬度 | float | 87.8 | 3,682 | -16.99 | 3.09 | 17.78 |
| 6 | `ea_lon_median` | EA中位数经度 | float | 87.8 | 3,680 | -11.98 | 33.04 | 44.36 |
| 7 | `district_lat_median` | 区级中位数纬度 | float | 99.8 | 71 | -15.61 | 0.9614 | 17.24 |
| 8 | `district_lon_median` | 区级中位数经度 | float | 99.8 | 71 | -10.36 | 32.17 | 42.44 |
| 9 | `gps_lat_rounded` | 四舍五入纬度 | float | 99.8 | 6,415 | -16.99 | 1.70 | 17.78 |
| 10 | `gps_lon_rounded` | 四舍五入经度 | float | 99.8 | 6,824 | -11.98 | 32.65 | 44.36 |
| 11 | `gps_source` | GPS来源(plot/ea/district) | str | 100.0 | 4 | — | — | — |
| 12 | `gps_imputed` | 是否GPS imputation | bool | 100.0 | 2 | — | — | — |

### Crop Information / 作物信息

| # | Variable | Description | Type | Complete % | Unique | Min | Median | Max |
|--:|----------|-------------|------|----------:|-------:|----:|-------:|----:|
| 1 | `crop_name` | 原始作物名(1421种) | str | 99.5 | 1,421 | — | — | — |
| 2 | `crop_name_clean` | 清洗后作物名 | str | 99.5 | 1,421 | — | — | — |
| 3 | `crop_code` | 标准化作物代码 | float | 97.4 | 195 | 1,010.0 | 1,110.0 | 9,103.0 |
| 4 | `crop_category` | 作物分类(8类) | str | 97.4 | 8 | — | — | — |
| 5 | `main_crop` | 主作物(Plot级,12种) | str | 99.5 | 12 | — | — | — |
| 6 | `nb_seasonal_crop` | 季节作物数 | float | 97.8 | 16 | 0 | 2.00 | 16.00 |
| 7 | `maincrop_valueshare` | 主作物价值份额 | float | 92.4 | 50,052 | 0.1026 | 0.8182 | 1.00 |
| 8 | `intercropped` | 是否间作 | float | 96.8 | 2 | 0 | 1.00 | 1.00 |
| 9 | `agro_ecological_zone` | 农业生态区 | float | 70.5 | 8 | 311.0 | 314.0 | 324.0 |

### Yield & Harvest / 产量与收获

| # | Variable | Description | Type | Complete % | Unique | Min | Median | Max |
|--:|----------|-------------|------|----------:|-------:|----:|-------:|----:|
| 1 | `harvest_kg` | 收获量(kg) | float | 87.9 | 16,166 | 0 | 100.0 | 72,258,500 |
| 2 | `yield_kg_ha` | 产量(kg/ha) | float | 82.8 | 213,463 | 0 | 411.4 | 375,000,000 |
| 3 | `plot_area_GPS` | GPS地块面积(ha) | float | 94.2 | 75,945 | -0.0192 | 0.3197 | 11,618.5 |
| 4 | `farm_size` | 农场总面积(ha) | float | 95.2 | 51,009 | 0 | 1.18 | 20,977.7 |
| 5 | `harvest_value_LCU` | 收获价值(本币) | float | 86.5 | 62,302 | 0 | 22,585.7 | 52,260,000,000 |
| 6 | `harvest_value_USD` | 收获价值(USD) | float | 86.5 | 92,209 | 0 | 33.67 | 21,412,034 |
| 7 | `harvest_value_LCU_real2010` | 收获价值(2010不变本币) | float | 86.5 | 90,537 | 0 | 16,000.0 | 30,536,094,739 |
| 8 | `harvest_value_USD_real2010` | 收获价值(2010不变USD) | float | 86.5 | 92,203 | 0 | 21.69 | 8,916,540 |
| 9 | `revenue_per_ha` | 每公顷收入(USD/ha) | float | 81.6 | 312,687 | 0 | 125.0 | 224,322,604 |

### Agricultural Inputs / 农业投入

| # | Variable | Description | Type | Complete % | Unique | Min | Median | Max |
|--:|----------|-------------|------|----------:|-------:|----:|-------:|----:|
| 1 | `seed_kg` | 种子用量(kg) | float | 57.7 | 19,686 | 0 | 5.00 | 2,364,000 |
| 2 | `seed_value_LCU` | 种子价值(本币) | float | 49.7 | 41,665 | 0 | 1,500.0 | 7,900,000,000 |
| 3 | `seed_value_USD` | 种子价值(USD) | float | 49.7 | 52,374 | 0 | 3.86 | 57,107,523 |
| 4 | `seed_value_LCU_real2010` | 种子价值(2010本币) | float | 49.7 | 52,057 | 0 | 949.9 | 6,351,820,506 |
| 5 | `seed_value_USD_real2010` | 种子价值(2010USD) | float | 49.7 | 52,304 | 0 | 2.33 | 45,916,043 |
| 6 | `improved` | 改良品种(0/1) | float | 55.7 | 2 | 0 | 0 | 1.00 |
| 7 | `used_pesticides` | 使用农药 | float | 95.2 | 3 | 0 | 0 | 5.00 |
| 8 | `inorganic_fertilizer` | 无机肥(0/1) | float | 95.9 | 2 | 0 | 0 | 1.00 |
| 9 | `nitrogen_kg` | 施肥量(kg) | float | 95.0 | 5,108 | 0 | 0 | 1,242,020 |
| 10 | `inorganic_fertilizer_value_LCU` | 肥料价值(本币) | float | 94.4 | 14,509 | 0 | 0 | 1,350,052,000 |
| 11 | `inorganic_fertilizer_value_USD` | 肥料价值(USD) | float | 94.4 | 19,459 | 0 | 0 | 1,779,800 |
| 12 | `organic_fertilizer` | 有机肥(0/1) | float | 95.3 | 2 | 0 | 0 | 1.00 |

### Labor Inputs / 劳动力投入

| # | Variable | Description | Type | Complete % | Unique | Min | Median | Max |
|--:|----------|-------------|------|----------:|-------:|----:|-------:|----:|
| 1 | `total_labor_days` | 总劳动天数 | float | 94.5 | 4,962 | 0 | 54.00 | 511,452.0 |
| 2 | `total_family_labor_days` | 家庭劳动天数 | float | 92.6 | 4,085 | 0 | 48.00 | 197,906.0 |
| 3 | `total_hired_labor_days` | 雇工天数 | float | 94.2 | 1,051 | 0 | 0 | 510,788.0 |
| 4 | `hired_labor_value_LCU` | 雇工成本(本币) | float | 94.1 | 12,090 | 0 | 0 | 168,000,015,000 |
| 5 | `hired_labor_value_USD` | 雇工成本(USD) | float | 94.1 | 21,127 | 0 | 0 | 1,214,438,579 |

### Manager Demographics / 管理者人口统计

| # | Variable | Description | Type | Complete % | Unique | Min | Median | Max |
|--:|----------|-------------|------|----------:|-------:|----:|-------:|----:|
| 1 | `age_manager` | 管理者年龄 | float | 93.3 | 101 | 0 | 46.00 | 100.0 |
| 2 | `female_manager` | 女性管理者(0/1) | float | 93.5 | 2 | 0 | 0 | 1.00 |
| 3 | `married_manager` | 已婚管理者(0/1) | float | 93.2 | 2 | 0 | 1.00 | 1.00 |
| 4 | `formal_education_manager` | 正规教育(0/1) | float | 92.7 | 2 | 0 | 1.00 | 1.00 |
| 5 | `primary_education_manager` | 小学教育(0/1) | float | 88.0 | 2 | 0 | 0 | 1.00 |
| 6 | `age_respondent` | 受访者年龄 | float | 82.4 | 101 | 0 | 45.00 | 100.0 |
| 7 | `female_respondent` | 女性受访者(0/1) | float | 82.6 | 2 | 0 | 0 | 1.00 |

### Household Characteristics / 家庭特征

| # | Variable | Description | Type | Complete % | Unique | Min | Median | Max |
|--:|----------|-------------|------|----------:|-------:|----:|-------:|----:|
| 1 | `hh_size` | 家庭规模(人) | float | 99.6 | 74 | 1.00 | 6.00 | 84.00 |
| 2 | `hh_asset_index` | 家庭资产指数 | float | 99.4 | 26,332 | -13.49 | -0.3347 | 10.00 |
| 3 | `hh_dependency_ratio` | 抚养比 | float | 99.7 | 310 | 0 | 1.00 | 28.00 |
| 4 | `hh_formal_education` | 户正规教育(0/1) | float | 99.5 | 2 | 0 | 1.00 | 1.00 |
| 5 | `hh_primary_education` | 户小学教育(0/1) | float | 98.1 | 2 | 0 | 1.00 | 1.00 |
| 6 | `hh_electricity_access` | 通电(0/1) | float | 99.5 | 2 | 0 | 0 | 1.00 |
| 7 | `hh_shock` | 遭受冲击(0/1) | float | 96.2 | 2 | 0 | 1.00 | 1.00 |
| 8 | `nonfarm_enterprise` | 非农经营(0/1) | float | 96.4 | 2 | 0 | 0 | 1.00 |
| 9 | `HDDS` | 饮食多样性得分(0-12) | float | 94.7 | 13 | 0 | 8.00 | 12.00 |
| 10 | `totcons_LCU` | 总消费(本币) | float | 83.4 | 61,677 | 0 | 242,599.9 | 578,154,004 |
| 11 | `totcons_USD` | 总消费(USD) | float | 83.4 | 61,711 | 0 | 317.6 | 286,300.0 |
| 12 | `cons_quint` | 消费五分位 | float | 83.4 | 5 | 1.00 | 3.00 | 5.00 |
| 13 | `share_kg_sold` | 出售比例 | float | 55.9 | 11,587 | 0 | 0 | 1.00 |
| 14 | `nb_plots` | 地块数 | float | 65.0 | 53 | 0 | 4.00 | 60.00 |
| 15 | `nb_fallow_plots` | 休耕地块数 | float | 62.7 | 13 | 0 | 0 | 12.00 |

### Assets & Tenure / 资产与土地权属

| # | Variable | Description | Type | Complete % | Unique | Min | Median | Max |
|--:|----------|-------------|------|----------:|-------:|----:|-------:|----:|
| 1 | `ag_asset_index` | 农业资产指数 | float | 89.7 | 4,291 | -3.12 | 0.0479 | 55.20 |
| 2 | `livestock` | 牲畜(0/1) | float | 98.1 | 2 | 0 | 1.00 | 1.00 |
| 3 | `tractor` | 拖拉机(0/1) | float | 83.6 | 2 | 0 | 0 | 1.00 |
| 4 | `plot_owned` | 自有地(0/1) | float | 97.1 | 2 | 0 | 1.00 | 1.00 |
| 5 | `plot_certificate` | 有产权证(0/1) | float | 91.3 | 2 | 0 | 0 | 1.00 |
| 6 | `irrigated` | 灌溉(0/1) | float | 95.9 | 2 | 0 | 0 | 1.00 |

### Market Access / 市场接入

| # | Variable | Description | Type | Complete % | Unique | Min | Median | Max |
|--:|----------|-------------|------|----------:|-------:|----:|-------:|----:|
| 1 | `dist_market` | 到市场距离(km,仅ETH+NGA问卷) | float | 28.8 | 3,818 | 0.2800 | 60.10 | 291.5 |
| 2 | `dist_popcenter` | 到人口中心距离(km,问卷) | float | 70.5 | 9,591 | 0 | 27.60 | 231.0 |
| 3 | `travel_time_city_min` | 到最近城市旅行时间(分钟,GEE Nelson 2015) | float | 96.4 | 430 | 0 | 59.00 | 894.0 |

### Soil Quality — Survey / 土壤质量(问卷)

| # | Variable | Description | Type | Complete % | Unique | Min | Median | Max |
|--:|----------|-------------|------|----------:|-------:|----:|-------:|----:|
| 1 | `soil_fertility_index` | 土壤肥力指数(问卷) | float | 70.5 | 430 | -4.82 | 0.3226 | 1.01 |
| 2 | `nutrient_availability` | 养分可用性(问卷) | float | 70.5 | 2 | 0 | 0 | 1.00 |
| 3 | `erosion_protection` | 抗侵蚀(问卷) | float | 72.0 | 2 | 0 | 0 | 1.00 |

### Shocks / 冲击变量

| # | Variable | Description | Type | Complete % | Unique | Min | Median | Max |
|--:|----------|-------------|------|----------:|-------:|----:|-------:|----:|
| 1 | `crop_shock` | 作物受灾(0/1) | float | 87.4 | 2 | 0 | 0 | 1.00 |
| 2 | `pests_shock` | 虫害 | float | 56.7 | 3 | 0 | 0 | 12.00 |
| 3 | `drought_shock` | 旱灾 | float | 50.5 | 3 | 0 | 0 | 12.00 |
| 4 | `rain_shock` | 雨灾 | float | 29.2 | 3 | 0 | 0 | 12.00 |
| 5 | `flood_shock` | 洪灾 | float | 38.3 | 2 | 0 | 0 | 1.00 |

### Planting & Harvest Timing / 种植收获时间

| # | Variable | Description | Type | Complete % | Unique | Min | Median | Max |
|--:|----------|-------------|------|----------:|-------:|----:|-------:|----:|
| 1 | `planting_month` | 种植月份(datetime) | datetime | 60.9 | 443 | — | — | — |
| 2 | `harvest_end_month` | 收获月份(datetime) | datetime | 68.7 | 257 | — | — | — |

### Price Deflators / 价格平减

| # | Variable | Description | Type | Complete % | Unique | Min | Median | Max |
|--:|----------|-------------|------|----------:|-------:|----:|-------:|----:|
| 1 | `cpi` | 消费者物价指数 | float | 100.0 | 27 | 83.97 | 137.8 | 524.9 |
| 2 | `cpi_deflator` | CPI平减因子 | float | 100.0 | 27 | 19.05 | 72.58 | 119.1 |

### Climate — Corrected Final / 气候(修正后最终版)

| # | Variable | Description | Type | Complete % | Unique | Min | Median | Max |
|--:|----------|-------------|------|----------:|-------:|----:|-------:|----:|
| 1 | `rainfall_growing_sum_final` | 生长季累计降雨(mm,修正窗口) | float | 96.2 | 54,787 | 0 | 737.5 | 4,085.6 |
| 2 | `rainfall_10yr_mean_final` | 10年均月降雨(mm,修正窗口) | float | 96.2 | 59,782 | 0 | 123.0 | 416.8 |
| 3 | `rainfall_10yr_cv_final` | 降雨变异系数(修正窗口) | float | 96.2 | 57,995 | 0.0261 | 0.1264 | 2.00 |
| 4 | `rainfall_10yr_nyears` | 10年回溯实际年数 | float | 96.2 | 1 | 10.00 | 10.00 | 10.00 |
| 5 | `ndvi_preseason_mean_final` | 季前NDVI(修正窗口) | float | 96.4 | 25,795 | -0.1767 | 0.4330 | 0.8690 |
| 6 | `ndvi_growing_mean_final` | 生长季NDVI(修正窗口) | float | 96.4 | 55,583 | -0.1577 | 0.5862 | 0.8912 |
| 7 | `climate_source` | 气候窗口来源(precise/calendar_s1/calendar_s2/calendar_tz_s1/calendar_tz_s2) | str | 100.0 | 5 | — | — | — |

### Climate — Original Calendar (reference) / 气候(原始日历,参考用)

| # | Variable | Description | Type | Complete % | Unique | Min | Median | Max |
|--:|----------|-------------|------|----------:|-------:|----:|-------:|----:|
| 1 | `gee_rainfall_growing_sum` | [参考]生长季降雨(旧统一窗口) | float | 99.5 | 11,065 | 0 | 661.4 | 3,049.6 |
| 2 | `gee_rainfall_10yr_mean` | [参考]10年均降雨(旧) | float | 99.5 | 11,066 | 0.000000 | 132.1 | 447.2 |
| 3 | `gee_rainfall_10yr_cv` | [参考]降雨CV(旧) | float | 91.1 | 10,367 | 0.000724 | 0.1320 | 3.00 |
| 4 | `gee_ndvi_preseason_mean` | [参考]季前NDVI(旧) | float | 99.8 | 10,962 | -0.1732 | 0.4335 | 0.8529 |
| 5 | `gee_ndvi_growing_mean` | [参考]生长季NDVI(旧) | float | 99.8 | 12,094 | -0.1295 | 0.5939 | 0.8615 |

### Temperature — ERA5 Corrected / 温度(ERA5修正后)

| # | Variable | Description | Type | Complete % | Unique | Min | Median | Max |
|--:|----------|-------------|------|----------:|-------:|----:|-------:|----:|
| 1 | `era5_tmean_growing_final` | 生长季日均温(°C,修正窗口) | float | 91.0 | 53,004 | 10.01 | 22.27 | 33.91 |
| 2 | `era5_tmax_growing_final` | 生长季日最高温(°C,修正窗口) | float | 91.0 | 53,013 | 14.71 | 26.83 | 40.13 |
| 3 | `era5_tmin_growing_final` | 生长季日最低温(°C,修正窗口) | float | 91.0 | 52,988 | 5.23 | 18.37 | 28.18 |

### Temperature — ERA5 Original (reference) / 温度(ERA5原始,参考用)

| # | Variable | Description | Type | Complete % | Unique | Min | Median | Max |
|--:|----------|-------------|------|----------:|-------:|----:|-------:|----:|
| 1 | `era5_tmax_growing` | [参考]日最高温(旧窗口) | float | 91.0 | 9,000 | 14.65 | 27.02 | 41.84 |
| 2 | `era5_tmin_growing` | [参考]日最低温(旧窗口) | float | 91.0 | 8,978 | 7.02 | 18.68 | 30.14 |
| 3 | `era5_tmean_growing` | [参考]日均温(旧窗口) | float | 91.0 | 8,996 | 10.23 | 22.51 | 36.35 |

### Terrain — NASADEM / 地形

| # | Variable | Description | Type | Complete % | Unique | Min | Median | Max |
|--:|----------|-------------|------|----------:|-------:|----:|-------:|----:|
| 1 | `elevation_m` | 海拔(m) | float | 99.8 | 1,849 | 0 | 1,128.0 | 3,536.0 |
| 2 | `slope_deg` | 坡度(°) | float | 99.8 | 4,918 | 0 | 3.38 | 52.42 |
| 3 | `terrain_ruggedness_m` | 崎岖指数(m) | float | 99.8 | 4,344 | 0 | 2.85 | 78.96 |

### Soil Chemistry — ISRIC SoilGrids / 土壤化学(7属性×5深度=35变量)

| # | Variable | Description | Type | Complete % | Unique | Min | Median | Max |
|--:|----------|-------------|------|----------:|-------:|----:|-------:|----:|
| 1 | `bdod_0-5cm` | 容重g/cm³,0-5cm | float | 93.4 | 83 | 0.7900 | 1.28 | 1.65 |
| 2 | `bdod_5-15cm` | 容重g/cm³,5-15cm | float | 93.4 | 83 | 0.8000 | 1.31 | 1.66 |
| 3 | `bdod_15-30cm` | 容重g/cm³,15-30cm | float | 93.4 | 79 | 0.8100 | 1.33 | 1.68 |
| 4 | `bdod_30-60cm` | 容重g/cm³,30-60cm | float | 93.4 | 80 | 0.8000 | 1.36 | 1.73 |
| 5 | `bdod_60-100cm` | 容重g/cm³,60-100cm | float | 93.4 | 85 | 0.8300 | 1.37 | 1.75 |
| 6 | `clay_0-5cm` | 粘土%,0-5cm | float | 93.5 | 401 | 7.00 | 32.70 | 54.10 |
| 7 | `clay_5-15cm` | 粘土%,5-15cm | float | 93.5 | 416 | 6.80 | 32.80 | 57.70 |
| 8 | `clay_15-30cm` | 粘土%,15-30cm | float | 93.5 | 439 | 8.20 | 35.90 | 62.70 |
| 9 | `clay_30-60cm` | 粘土%,30-60cm | float | 93.5 | 475 | 8.10 | 40.00 | 70.80 |
| 10 | `clay_60-100cm` | 粘土%,60-100cm | float | 93.5 | 484 | 8.40 | 40.60 | 76.50 |
| 11 | `sand_0-5cm` | 沙%,0-5cm | float | 93.5 | 648 | 12.50 | 42.00 | 85.00 |
| 12 | `sand_5-15cm` | 沙%,5-15cm | float | 93.5 | 658 | 10.20 | 41.80 | 85.00 |
| 13 | `sand_15-30cm` | 沙%,15-30cm | float | 93.5 | 649 | 8.80 | 39.50 | 85.30 |
| 14 | `sand_30-60cm` | 沙%,30-60cm | float | 93.5 | 642 | 7.30 | 36.50 | 84.60 |
| 15 | `sand_60-100cm` | 沙%,60-100cm | float | 93.5 | 635 | 7.20 | 36.20 | 84.20 |
| 16 | `silt_0-5cm` | 粉砂%,0-5cm | float | 93.5 | 419 | 2.80 | 24.70 | 67.10 |
| 17 | `silt_5-15cm` | 粉砂%,5-15cm | float | 93.5 | 419 | 2.70 | 24.50 | 66.90 |
| 18 | `silt_15-30cm` | 粉砂%,15-30cm | float | 93.5 | 400 | 2.40 | 23.30 | 64.50 |
| 19 | `silt_30-60cm` | 粉砂%,30-60cm | float | 93.5 | 386 | 2.50 | 22.10 | 57.20 |
| 20 | `silt_60-100cm` | 粉砂%,60-100cm | float | 93.5 | 383 | 2.60 | 21.70 | 55.50 |
| 21 | `soc_0-5cm` | 有机碳g/kg,0-5cm | float | 93.4 | 614 | 3.10 | 32.90 | 101.4 |
| 22 | `soc_5-15cm` | 有机碳g/kg,5-15cm | float | 93.4 | 458 | 2.30 | 22.00 | 82.20 |
| 23 | `soc_15-30cm` | 有机碳g/kg,15-30cm | float | 93.4 | 326 | 1.80 | 15.60 | 71.00 |
| 24 | `soc_30-60cm` | 有机碳g/kg,30-60cm | float | 93.4 | 262 | 1.50 | 10.80 | 51.40 |
| 25 | `soc_60-100cm` | 有机碳g/kg,60-100cm | float | 93.4 | 235 | 1.00 | 7.10 | 51.90 |
| 26 | `nitrogen_0-5cm` | 总氮g/kg,0-5cm | float | 93.5 | 2,518 | 3.17 | 24.94 | 60.18 |
| 27 | `nitrogen_5-15cm` | 总氮g/kg,5-15cm | float | 93.5 | 2,132 | 1.29 | 16.48 | 51.67 |
| 28 | `nitrogen_15-30cm` | 总氮g/kg,15-30cm | float | 93.5 | 1,778 | 2.26 | 13.56 | 40.80 |
| 29 | `nitrogen_30-60cm` | 总氮g/kg,30-60cm | float | 93.5 | 1,402 | 1.59 | 9.71 | 42.85 |
| 30 | `nitrogen_60-100cm` | 总氮g/kg,60-100cm | float | 93.5 | 1,149 | 1.49 | 7.06 | 48.33 |
| 31 | `phh2o_0-5cm` | pH,0-5cm | float | 93.5 | 38 | 4.80 | 5.90 | 8.80 |
| 32 | `phh2o_5-15cm` | pH,5-15cm | float | 93.5 | 40 | 4.70 | 5.90 | 8.90 |
| 33 | `phh2o_15-30cm` | pH,15-30cm | float | 93.5 | 41 | 4.70 | 5.90 | 9.00 |
| 34 | `phh2o_30-60cm` | pH,30-60cm | float | 93.5 | 38 | 4.70 | 5.90 | 8.40 |
| 35 | `phh2o_60-100cm` | pH,60-100cm | float | 93.5 | 38 | 4.80 | 5.90 | 8.60 |

### Conflict — ACLED Primary / 冲突(主要4变量)

| # | Variable | Description | Type | Complete % | Unique | Min | Median | Max |
|--:|----------|-------------|------|----------:|-------:|----:|-------:|----:|
| 1 | `conflict_events_25km_12m` | 25km内12月事件数 | float | 99.8 | 88 | 0 | 0 | 108.0 |
| 2 | `conflict_fatal_25km_12m` | 25km内12月有致死(0/1) | float | 99.8 | 2 | 0 | 0 | 1.00 |
| 3 | `conflict_fatalities_25km_12m` | 25km内12月死亡人数 | float | 99.8 | 203 | 0 | 0 | 1,001.0 |
| 4 | `conflict_nearest_event_km` | 最近冲突距离(km) | float | 99.8 | 7,695 | 0.0131 | 8.74 | 134.3 |

### Conflict — ACLED Event Counts / 冲突事件数(9变量)

| # | Variable | Description | Type | Complete % | Unique | Min | Median | Max |
|--:|----------|-------------|------|----------:|-------:|----:|-------:|----:|
| 1 | `conflict_events_10km_3m` | 10km内3月事件数 | float | 99.8 | 14 | 0 | 0 | 13.00 |
| 2 | `conflict_events_10km_12m` | 10km内12月事件数 | float | 99.8 | 36 | 0 | 0 | 43.00 |
| 3 | `conflict_events_10km_24m` | 10km内24月事件数 | float | 99.8 | 66 | 0 | 0 | 97.00 |
| 4 | `conflict_events_25km_3m` | 25km内3月事件数 | float | 99.8 | 30 | 0 | 0 | 35.00 |
| 5 | `conflict_events_25km_12m` | 25km内12月事件数 | float | 99.8 | 88 | 0 | 0 | 108.0 |
| 6 | `conflict_events_25km_24m` | 25km内24月事件数 | float | 99.8 | 132 | 0 | 0 | 199.0 |
| 7 | `conflict_events_50km_3m` | 50km内3月事件数 | float | 99.8 | 67 | 0 | 0 | 90.00 |
| 8 | `conflict_events_50km_12m` | 50km内12月事件数 | float | 99.8 | 187 | 0 | 0 | 277.0 |
| 9 | `conflict_events_50km_24m` | 50km内24月事件数 | float | 99.8 | 286 | 0 | 1.00 | 510.0 |

### Conflict — ACLED Distance-Weighted / 冲突距离加权(9变量)

| # | Variable | Description | Type | Complete % | Unique | Min | Median | Max |
|--:|----------|-------------|------|----------:|-------:|----:|-------:|----:|
| 1 | `conflict_events_wt_10km_3m` | 10km内3月距离加权事件数 | float | 99.8 | 1,027 | 0 | 0 | 11.58 |
| 2 | `conflict_events_wt_10km_12m` | 10km内12月距离加权事件数 | float | 99.8 | 1,852 | 0 | 0 | 34.18 |
| 3 | `conflict_events_wt_10km_24m` | 10km内24月距离加权事件数 | float | 99.8 | 2,316 | 0 | 0 | 80.00 |
| 4 | `conflict_events_wt_25km_3m` | 25km内3月距离加权事件数 | float | 99.8 | 2,889 | 0 | 0 | 13.80 |
| 5 | `conflict_events_wt_25km_12m` | 25km内12月距离加权事件数 | float | 99.8 | 4,794 | 0 | 0 | 43.73 |
| 6 | `conflict_events_wt_25km_24m` | 25km内24月距离加权事件数 | float | 99.8 | 5,782 | 0 | 0 | 92.25 |
| 7 | `conflict_events_wt_50km_3m` | 50km内3月距离加权事件数 | float | 99.8 | 6,093 | 0 | 0 | 13.85 |
| 8 | `conflict_events_wt_50km_12m` | 50km内12月距离加权事件数 | float | 99.8 | 9,185 | 0 | 0 | 43.74 |
| 9 | `conflict_events_wt_50km_24m` | 50km内24月距离加权事件数 | float | 99.8 | 10,820 | 0 | 0.000142 | 92.27 |

### Conflict — ACLED Fatal Indicators / 冲突致死指标(9变量)

| # | Variable | Description | Type | Complete % | Unique | Min | Median | Max |
|--:|----------|-------------|------|----------:|-------:|----:|-------:|----:|
| 1 | `conflict_fatal_10km_3m` | 10km内3月有致死(0/1) | float | 99.8 | 2 | 0 | 0 | 1.00 |
| 2 | `conflict_fatal_10km_12m` | 10km内12月有致死(0/1) | float | 99.8 | 2 | 0 | 0 | 1.00 |
| 3 | `conflict_fatal_10km_24m` | 10km内24月有致死(0/1) | float | 99.8 | 2 | 0 | 0 | 1.00 |
| 4 | `conflict_fatal_25km_3m` | 25km内3月有致死(0/1) | float | 99.8 | 2 | 0 | 0 | 1.00 |
| 5 | `conflict_fatal_25km_12m` | 25km内12月有致死(0/1) | float | 99.8 | 2 | 0 | 0 | 1.00 |
| 6 | `conflict_fatal_25km_24m` | 25km内24月有致死(0/1) | float | 99.8 | 2 | 0 | 0 | 1.00 |
| 7 | `conflict_fatal_50km_3m` | 50km内3月有致死(0/1) | float | 99.8 | 2 | 0 | 0 | 1.00 |
| 8 | `conflict_fatal_50km_12m` | 50km内12月有致死(0/1) | float | 99.8 | 2 | 0 | 0 | 1.00 |
| 9 | `conflict_fatal_50km_24m` | 50km内24月有致死(0/1) | float | 99.8 | 2 | 0 | 0 | 1.00 |

### Conflict — ACLED Other / 冲突其他(2变量)

| # | Variable | Description | Type | Complete % | Unique | Min | Median | Max |
|--:|----------|-------------|------|----------:|-------:|----:|-------:|----:|
| 1 | `conflict_battles_25km_12m` | 25km内12月战斗数 | float | 99.8 | 43 | 0 | 0 | 47.00 |
| 2 | `conflict_vc_25km_12m` | 25km内12月对平民暴力数 | float | 99.8 | 52 | 0 | 0 | 65.00 |

### Quality Flags / 质量标记

| # | Variable | Description | True % |
|--:|----------|-------------|-------:|
| 1 | `is_gps_invalid` | 无效GPS | 0.00% |
| 2 | `is_gps_out_of_bounds` | GPS越界 | 0.00% |
| 3 | `is_gps_missing` | GPS缺失 | 0.19% |
| 4 | `is_area_too_small` | 面积<0.001ha | 1.49% |
| 5 | `is_area_very_large` | 面积>100ha | 0.03% |
| 6 | `is_area_missing` | 面积缺失 | 5.76% |
| 7 | `is_harvest_zero` | 零收获 | 3.93% |
| 8 | `is_harvest_very_low` | 收获<100g | 0.04% |
| 9 | `is_harvest_missing` | 收获缺失 | 12.08% |
| 10 | `is_intercrop_inconsistent` | 间作不一致 | 0.00% |
| 11 | `is_yield_extreme` | 产量极端值 | 0.43% |

## Per-Country Completeness (%) for Key Variables

| Variable | Ethiopia | Malawi | Mali | Niger | Nigeria | Tanzania | Uganda |
|----------|------:|------:|------:|------:|------:|------:|------:|
| `harvest_kg` | 90.2 | 90.0 | 98.3 | 86.8 | 86.3 | 83.0 | 86.9 |
| `yield_kg_ha` | 86.3 | 86.5 | 96.6 | 85.0 | 85.2 | 82.1 | 76.5 |
| `plot_area_GPS` | 96.0 | 94.8 | 98.0 | 89.3 | 98.1 | 99.1 | 89.7 |
| `crop_category` | 99.7 | 95.8 | 98.8 | 92.7 | 90.2 | 99.5 | 98.5 |
| `inorganic_fertilizer` | 97.8 | 83.6 | 96.3 | 88.2 | 93.8 | 95.1 | 99.7 |
| `nitrogen_kg` | 97.7 | 83.3 | 95.5 | 84.0 | 93.7 | 91.2 | 99.7 |
| `total_labor_days` | 98.5 | 83.7 | 98.6 | 89.8 | 97.4 | 91.5 | 94.8 |
| `hh_size` | 99.5 | 100.0 | 100.0 | 100.0 | 99.8 | 99.4 | 99.5 |
| `hh_asset_index` | 99.5 | 100.0 | 100.0 | 100.0 | 99.7 | 99.4 | 98.9 |
| `totcons_USD` | 96.1 | 31.4 | 29.3 | 98.5 | 83.1 | 86.2 | 95.6 |
| `age_manager` | 96.9 | 83.0 | 97.3 | 87.1 | 96.6 | 91.5 | 93.2 |
| `female_manager` | 96.9 | 83.5 | 97.5 | 87.1 | 97.5 | 91.5 | 93.3 |
| `rainfall_growing_sum_final` | 99.5 | 99.8 | 98.5 | 100.0 | 98.4 | 97.4 | 91.3 |
| `ndvi_growing_mean_final` | 99.5 | 99.8 | 98.5 | 100.0 | 98.4 | 99.0 | 91.3 |
| `era5_tmean_growing_final` | 84.4 | 99.8 | 98.5 | 100.0 | 81.9 | 95.4 | 91.3 |
| `travel_time_city_min` | 99.5 | 99.8 | 98.5 | 100.0 | 98.4 | 98.8 | 91.3 |
| `clay_0-5cm` | 98.6 | 96.2 | 95.5 | 96.4 | 93.7 | 93.7 | 89.3 |
| `elevation_m` | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 99.5 |
| `conflict_events_25km_12m` | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 99.5 |

## Descriptive Statistics — Key Continuous Variables

| Variable | N | Mean | Std | Min | P25 | Median | P75 | Max |
|----------|--:|-----:|----:|----:|----:|-------:|----:|----:|
| `yield_kg_ha` | 426,203 | 7,650.1 | 912,642.2 | 0 | 96.94 | 411.4 | 1,320.1 | 375,000,000 |
| `harvest_kg` | 452,508 | 1,057.0 | 127,575.8 | 0 | 30.00 | 100.0 | 350.0 | 72,258,500 |
| `plot_area_GPS` | 485,022 | 1.19 | 49.59 | -0.0192 | 0.1018 | 0.3197 | 0.8296 | 11,618.5 |
| `revenue_per_ha` | 420,114 | 4,040.4 | 448,750.4 | 0 | 28.34 | 125.0 | 424.8 | 224,322,604 |
| `nitrogen_kg` | 489,189 | 30.27 | 3,368.3 | 0 | 0 | 0 | 0 | 1,242,020 |
| `total_labor_days` | 486,440 | 205.5 | 3,075.2 | 0 | 23.00 | 54.00 | 119.0 | 511,452.0 |
| `seed_kg` | 297,069 | 135.3 | 8,360.3 | 0 | 1.00 | 5.00 | 15.00 | 2,364,000 |
| `hh_size` | 512,650 | 6.80 | 4.52 | 1.00 | 4.00 | 6.00 | 8.00 | 84.00 |
| `hh_asset_index` | 511,608 | -0.1170 | 0.7967 | -13.49 | -0.6173 | -0.3347 | 0.1420 | 10.00 |
| `totcons_USD` | 429,155 | 440.6 | 1,074.8 | 0 | 194.9 | 317.6 | 513.1 | 286,300.0 |
| `HDDS` | 487,400 | 7.56 | 2.16 | 0 | 6.00 | 8.00 | 9.00 | 12.00 |
| `rainfall_growing_sum_final` | 494,897 | 826.6 | 417.5 | 0 | 535.6 | 737.5 | 1,009.8 | 4,085.6 |
| `rainfall_10yr_mean_final` | 494,897 | 132.2 | 48.40 | 0 | 100.9 | 123.0 | 151.8 | 416.8 |
| `ndvi_growing_mean_final` | 496,136 | 0.5632 | 0.1430 | -0.1577 | 0.4843 | 0.5862 | 0.6708 | 0.8912 |
| `era5_tmean_growing_final` | 468,188 | 22.69 | 3.71 | 10.01 | 20.36 | 22.27 | 25.18 | 33.91 |
| `era5_tmax_growing_final` | 468,188 | 27.41 | 3.88 | 14.71 | 24.88 | 26.83 | 29.52 | 40.13 |
| `era5_tmin_growing_final` | 468,188 | 18.61 | 3.78 | 5.23 | 16.12 | 18.37 | 21.82 | 28.18 |
| `travel_time_city_min` | 495,989 | 83.67 | 88.06 | 0 | 27.00 | 59.00 | 110.0 | 894.0 |
| `elevation_m` | 513,690 | 1,085.8 | 647.3 | 0 | 477.0 | 1,128.0 | 1,487.0 | 3,536.0 |
| `slope_deg` | 513,690 | 6.15 | 6.58 | 0 | 2.08 | 3.38 | 7.23 | 52.42 |
| `clay_0-5cm` | 481,423 | 29.96 | 9.35 | 7.00 | 21.20 | 32.70 | 37.60 | 54.10 |
| `soc_0-5cm` | 480,936 | 31.08 | 16.37 | 3.10 | 16.00 | 32.90 | 42.90 | 101.4 |
| `nitrogen_0-5cm` | 481,423 | 24.41 | 12.36 | 3.17 | 12.71 | 24.94 | 33.34 | 60.18 |
| `phh2o_0-5cm` | 481,423 | 6.04 | 0.5157 | 4.80 | 5.70 | 5.90 | 6.20 | 8.80 |
| `conflict_events_25km_12m` | 513,690 | 0.8608 | 4.00 | 0 | 0 | 0 | 0 | 108.0 |
| `conflict_nearest_event_km` | 513,690 | 13.22 | 13.57 | 0.0131 | 4.46 | 8.74 | 17.11 | 134.3 |

## Growing Season Fix Verification

### Climate Source Distribution

| Source | Obs | % | Description |
|--------|----:|--:|-------------|
| precise | 195,159 | 37.9 | Layer 1: actual planting→harvest dates |
| calendar_s1 | 194,523 | 37.8 | Layer 2: revised calendar Season 1 |
| calendar_s2 | 64,229 | 12.5 | Layer 2: revised calendar Season 2 |
| calendar_tz_s1 | 52,602 | 10.2 | Layer 2: Tanzania Masika (inferred from harvest) |
| calendar_tz_s2 | 8,152 | 1.6 | Layer 2: Tanzania Vuli (inferred from harvest) |

### Old vs New Climate Comparison

| Country | Season | Mean diff (mm) | Std diff | N | Interpretation |
|---------|:------:|---------------:|----------|--:|----------------|
| Ethiopia | 1 | +252 | 273 | 97,896 | Extended window |
| Uganda | 1 | +140 | 280 | 86,124 | Extended window |
| Uganda | 2 | +118 | 233 | 77,700 | Extended window |
| Nigeria | 1 | -22 | 206 | 61,914 | Minimal change |
| Tanzania | 1 | -59 | 182 | 72,627 | Moderate shift |
| Malawi | 1 | +44 | 249 | 40,233 | Moderate shift |
| Mali | 1 | -57 | 87 | 34,457 | Moderate shift |
| Niger | 1 | +0 | 21 | 23,946 | Minimal change |

### Key Validation Metrics

- **Uganda Season 2 old↔new rain correlation**: 0.424 (low = seasons correctly differ)
- **10-year lookback depth**: 100% of obs have full 10-year data (CHIRPS 1997–2023)
- **Range checks**: Rain 0–4086mm, NDVI -0.16–0.89, Temp 10.0–33.9°C
- **OLS R² (log yield ~ rainfall)**: old=0.0468 → new=0.0541 (+15.6% improvement)

## Data Quality Notes

1. **Climate _final 96.2%**: Three-layer growing season assignment (precise dates 38% + calendar 50% + TZ harvest inference 12%). 3.8% gap from missing GPS coordinates.
2. **CHIRPS 1997–2023**: Supplemented from original 2007–2023 via GEE. All obs have full 10-year rainfall lookback.
3. **ERA5 temperature 91.0%**: Data covers 2000–2020. 2021–2023 waves (Ethiopia w5, Nigeria w5, Uganda w7-8) lack temperature.
4. **Soil (ISRIC) 93.5%**: Fixed gps_id precision mismatch (was 68.6%). Remaining gap: Malawi ~99.8% after fix.
5. **Travel time 96.4%**: Nelson et al. accessibility_to_cities_2015, GEE cloud extraction.
6. **Crop category 97.4%**: Case-insensitive mapping + 117 extra entries (was 42.4%).
7. **dist_market 28.8%**: Survey-only variable (ETH 98%, NGA 79%, others 0%). `travel_time_city_min` (96.4%) is the geospatial alternative.
8. **Fertilizer 95.9%, Labor 94.5%**: From Plot_dataset.dta. Previously missing.
9. **HH characteristics 99%+**: From Household_dataset.dta. Previously missing.
10. **Conflict 99.8%**: ACLED 1997–2025, 31 variables. 975 obs missing (no GPS).
11. **yield_kg_ha outliers**: Max=375M kg/ha. Use `is_yield_extreme` flag.
12. **Original `gee_*` columns preserved**: Old single-window climate values kept as reference for comparison.

## Changelog

| Date | Change | Result |
|------|--------|--------|
| 2026-03-16 v3 | Growing season fix (3-layer precise+calendar+TZ) | 40.5% mismatch fixed; OLS R² +15.6% |
| 2026-03-16 v3 | CHIRPS 1997-2006 supplement | 10yr lookback: incomplete→100% full |
| 2026-03-16 v3 | ERA5 per-obs window | Corrected temperature for dual-season countries |
| 2026-03-16 v2 | Soil merge fix (4dp lat/lon) | clay 68.6%→93.5% |
| 2026-03-16 v2 | ERA5 temperature added | +3 vars, 91.0% |
| 2026-03-16 v2 | Travel time (GEE Nelson) | +1 var, 96.4% |
| 2026-03-16 v1 | GEE climate extraction | CHIRPS+MODIS 54%→99.5% |
| 2026-03-16 v1 | Crop category remapping | 42.4%→97.4% |
| 2026-03-16 v1 | Merged Plot_dataset.dta | +34 vars (fertilizer, labor, demographics) |
| 2026-03-16 v1 | Merged Household_dataset.dta | +15 vars (hh_size, assets, consumption) |
| 2026-03-06 | ACLED conflict merge | +31 conflict variables, 99.8% |
| Base | Original pipeline | 117 variables, 7 countries, 30 waves |