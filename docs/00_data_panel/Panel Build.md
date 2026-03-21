# Panel Data Construction Documentation

**Final output**: `all_countries_panel_birl.parquet` — 514,665 obs × 211 variables
**Unit of observation**: Plot-crop-wave (each row = one crop on one plot in one survey wave)
**Coverage**: 7 countries, 30 waves, 2008–2023
**Last updated**: 2026-03-16 (v3)

---

## 1. Pipeline Overview

```
Stage 1: Load                   Stage 2: Clean                Stage 3: Extract (parallel)
┌─────────────────┐     ┌─────────────────────┐     ┌──────────────────────────┐
│ Plotcrop_dataset │     │ PanelCleaner        │     │ GEE: CHIRPS rainfall     │
│ Plot_dataset     │────▶│ • GPS validation    │────▶│ GEE: MODIS NDVI          │
│ Household_dataset│     │ • Yield flags       │     │ GEE: ISRIC SoilGrids     │
│ (World Bank .dta)│     │ • Area flags        │     │ NASADEM: terrain          │
└─────────────────┘     │ • Crop matching     │     │ ACLED: conflict           │
                        └─────────────────────┘     │ ERA5: temperature         │
                                                     │ Nelson: travel time       │
                                                     └──────────────────────────┘
                                                                │
                                    Stage 4: Merge              ▼
                              ┌─────────────────────────────────────┐
                              │ PanelMerger                         │
                              │ • Plotcrop + climate + terrain      │
                              │ • + soil + conflict                 │
                              │ • + Plot_dataset (inputs/demog)     │
                              │ • + Household_dataset (HH chars)    │
                              │ • + CPI deflators                   │
                              │ • + crop category mapping           │
                              │ • + growing season fix              │
                              └─────────────────────────────────────┘
                                                │
                                    Stage 5: Validate
                              ┌─────────────────────────────────────┐
                              │ Row count preservation              │
                              │ Range checks, flag rates            │
                              │ Cross-country consistency           │
                              └─────────────────────────────────────┘
```

### Data Sources

| Source | Type | Scope | Variables |
|--------|------|-------|-----------|
| `meta/Plotcrop_dataset.dta` | World Bank harmonized | 514,665 plot-crop obs | IDs, crop, harvest, area, GPS, shocks |
| `meta/Plot_dataset.dta` | World Bank harmonized | 263,195 plot obs | Fertilizer, labor, demographics, assets, soil(survey) |
| `meta/Household_dataset.dta` | World Bank harmonized | 148,421 HH-wave obs | hh_size, assets, consumption, HDDS, enterprise |
| CHIRPS v2.0 | GEE satellite | 1997–2023 monthly | Rainfall (mm) |
| MODIS MOD13A1 | GEE satellite | 2000–2023 16-day | NDVI (-0.2 to 1.0) |
| ERA5-Land | ECMWF reanalysis | 2000–2020 daily | Temperature (°C) |
| ISRIC SoilGrids v2.0 | GEE gridded | Static | 7 soil properties × 5 depths |
| NASADEM | GEE DEM | Static | Elevation, slope, ruggedness |
| ACLED | Event database | 1997–2025 | Conflict events, fatalities, types |
| Nelson et al. 2015 | GEE accessibility | Static (2015) | Travel time to nearest city |
| CPI series | World Bank WDI | 2000–2023 annual | Consumer Price Index |
| `crop_category_mapping.csv` | Manual curation | 542 entries | Crop code → category mapping |

---

## 2. Core Identifiers (16 variables)

### ID Naming Convention

| Suffix | Meaning | Example |
|--------|---------|---------|
| `_merge` | Harmonized across countries/waves, string type | `hh_id_merge = "4143000506"` |
| `_obs` | Original survey numeric ID | `hh_id_obs = 4143000506.0` |

`_merge` IDs are constructed by the World Bank LSMS-ISA harmonization team by concatenating state + EA + household codes into a unique string. They are stable across waves for panel tracking. `_obs` IDs are the raw numeric codes from the original country survey instrument.

### ID Hierarchy

```
country
  └── ea_id_merge (枚举区, 3,821 unique)
        └── hh_id_merge (户, 56,141 unique)
              └── parcel_id_merge (宗地, 88,452 unique)
                    └── plot_id_merge (地块, 192,552 unique)
                          └── crop_name (作物, 1,421 names)
                                = one observation (514,665 total)
```

### Special Variables

- **wave**: Survey round (1–8). Uganda skips wave 6.
- **year**: Mapped from wave via `config/countries_info.json`. E.g., Nigeria: w1→2010, w2→2012, w3→2015, w4→2018, w5→2023.
- **season**: 1=main season, 2=minor season. Only Uganda has season 2 in the survey data.
- **pw**: Sampling probability weight from survey design.
- **gps_id**: Constructed as `"{lat_rounded}_{lon_rounded}"` (6 decimal places). 7,019 unique GPS locations.

---

## 3. GPS Coordinates & Imputation (12 variables)

### Imputation Hierarchy

The pipeline applies a four-level fallback when plot-level GPS is missing or invalid:

```
Level 1: plot_gps        (76.8% of obs)  ← original survey GPS
Level 2: ea_median       (13.3%)         ← median lat/lon of all plots in same EA
Level 3: district_median  (9.7%)         ← median lat/lon of all plots in same district
Level 4: missing          (0.2%)         ← no GPS available at any level
```

**Implementation** (`src/extractors/gps_extractor.py` + `src/cleaners/panel_cleaner.py`):
1. Validate raw GPS: check `|lat| ≤ 90`, `|lon| ≤ 180`, not (0,0), within country bounding box ±0.5°
2. If invalid → try household GPS
3. If still invalid → compute EA median from all valid GPS in the same `ea_id_merge`
4. If EA has no valid GPS → use district median

### Output Columns

| Variable | Description | Source |
|----------|-------------|--------|
| `gps_lat_final` / `gps_lon_final` | Final coordinates used for all geospatial extraction | After imputation |
| `lat_modified` / `lon_modified` | Original survey coordinates (privacy-jittered by World Bank) | Raw survey |
| `ea_lat_median` / `ea_lon_median` | EA-level median coordinates | Computed |
| `district_lat_median` / `district_lon_median` | District-level median coordinates | Computed |
| `gps_lat_rounded` / `gps_lon_rounded` | Rounded for deduplication | Derived |
| `gps_source` | Which level was used: `plot_gps` / `ea_median` / `district_median` | Flag |
| `gps_imputed` | Boolean: was any imputation applied? | Flag |

### Country Bounding Boxes (from `config/countries_info.json`)

| Country | Lat range | Lon range |
|---------|-----------|-----------|
| Nigeria | 3.83–14.21 | 2.31–14.21 |
| Ethiopia | 3.24–15.00 | 33.00–48.00 |
| Tanzania | -11.75–-1.00 | 29.00–40.50 |
| Uganda | -1.50–4.25 | 29.50–35.00 |
| Malawi | -17.13–-9.37 | 32.67–35.92 |
| Mali | 10.00–25.00 | -12.50–4.50 |
| Niger | 11.50–23.50 | 0.00–16.00 |

---

## 4. Crop Information (9 variables)

### Harmonization Pipeline

```
Raw survey crop name (1,421 unique strings)
    │
    ▼ Step 1: Normalize text
    Uppercase, strip, remove punctuation, expand abbreviations
    (G/NUT→GROUNDNUT, S/BEANS→SOYA BEANS)
    Fix misspellings (PLAINTAIN→PLANTAIN)
    │
    ▼ Step 2: Case-insensitive match to mapping CSV (542 entries)
    crop_category_mapping.csv: [crop_code, crop_name, crop_category, subcategory]
    Match rate: 93.9% exact match
    │
    ▼ Step 3: Extra mappings for country-specific variants (+117 entries)
    French: MIL→MILLET, SORGHO→SORGHUM, ARACHIDE→GROUNDNUT
    Nigeria truncations: ASSAVA→CASSAVA, AIZE→MAIZE
    Uganda subtypes: BANANA FOOD, COFFEE ALL, (ARABICA)
    Malawi varieties: MAIZE LOCAL, GROUNDNUT CG7, PIGEONPEA(NANDOLO)
    Ethiopian: TEFF, ENSET, CHAT, GODERE, HORSE BEANS
    │
    ▼ Result: 97.4% coverage
```

### Category System

| crop_category | Obs | % | Examples |
|---------------|----:|--:|---------|
| cereals | 167,875 | 32.6 | MAIZE, SORGHUM, MILLET, RICE, TEFF, WHEAT |
| legumes | 96,174 | 18.7 | BEANS, GROUNDNUT, COWPEA, PIGEON PEA, SOYA |
| roots_tubers | 80,058 | 15.6 | CASSAVA, YAM, SWEET POTATO, ENSET, POTATO |
| fruits | 77,441 | 15.0 | BANANA, MANGO, AVOCADO, PLANTAIN, PAPAYA |
| cash_crops | 50,037 | 9.7 | COFFEE, COTTON, TOBACCO, SUGAR CANE, COCOA |
| vegetables | 22,147 | 4.3 | TOMATO, ONION, OKRA, PEPPER, KALE |
| non_crop | 4,349 | 0.8 | FALLOW, TIMBER, FIREWOOD |
| other | 3,196 | 0.6 | Unclassified |

### Additional Crop Variables (from Plot_dataset.dta)

- **main_crop**: Plot-level primary crop (12 coarse categories). 99.5% coverage.
- **nb_seasonal_crop**: Number of crops on the plot in the season.
- **maincrop_valueshare**: Share of plot revenue attributable to the main crop.
- **intercropped**: Binary — multiple crops sharing the plot (0/1).
- **agro_ecological_zone**: FAO GAEZ classification (see code table in variable report).

---

## 5. Yield & Harvest (9 variables)

### Yield Calculation

```
yield_kg_ha = harvest_kg / plot_area_GPS
```

Where:
- `harvest_kg`: Total weight of crop harvested (kg). From survey post-harvest module.
- `plot_area_GPS`: GPS-measured plot area (hectares). Priority over farmer-estimated area.

### Plot Area Processing

```
Priority 1: GPS-measured area (s11aq4d in raw CSV) → convert m² to ha (÷10,000)
Priority 2: Farmer estimate (s11aq4a) + unit conversion
  acres → ×0.404686
  plots/heaps/ridges → flag as uncertain (is_unit_conversion_uncertain)
```

### Harvest Value & Currency Conversion

```
harvest_value_USD = harvest_value_LCU / exchange_rate
harvest_value_USD_real2010 = harvest_value_USD × (cpi_deflator / 100)
revenue_per_ha = harvest_value_USD / plot_area_GPS
```

### Quality Flags & Thresholds

| Flag | Condition | Action |
|------|-----------|--------|
| `is_harvest_zero` | harvest_kg = 0 | Keep (crop failure/theft) |
| `is_harvest_very_low` | harvest_kg < 0.1 kg | Flag |
| `is_harvest_missing` | harvest_kg is NaN | Flag |
| `is_area_too_small` | plot_area_GPS < 0.001 ha | Flag |
| `is_area_very_large` | plot_area_GPS > 100 ha | Flag |
| `is_area_missing` | plot_area_GPS is NaN | Flag |
| `is_yield_extreme` | yield_kg_ha outside 1st/99th percentile within crop category | Flag |

**Crop-specific yield caps** (from `config/cleaning_rules.yaml`):

| Category | Max yield (kg/ha) |
|----------|------------------:|
| cereals | 20,000 |
| legumes | 5,000 |
| roots_tubers | 80,000 |
| vegetables | 50,000 |
| fruits | 30,000 |
| cash_crops | 10,000 |

**Policy**: Flag only, never winsorize or delete. Original values always preserved.

### Wave Mapping Corrections

62.8% of original variable name mappings in `wave_mappings.yaml` were incorrect (per audit). Critical fixes:
- Nigeria Wave 3: `harvest_kg` was mapped to year column (`sa3iq4a2` contained years 2015/2016). Corrected to `sa3iq5a`.
- Nigeria Wave 4: Same issue. Corrected to `sa3iq6i`.

---

## 6. Agricultural Inputs (12 variables)

### Source: `meta/Plot_dataset.dta` → merged via `merge_dta_to_panel.py`

**Merge key**: (country, hh_id_merge, wave, plot_id_merge) — many-to-one (multiple crops per plot share the same inputs).

### Fertilizer

| Variable | Description | Coverage |
|----------|-------------|---------|
| `inorganic_fertilizer` | Binary: applied inorganic fertilizer (0/1) | 95.9% |
| `nitrogen_kg` | Total inorganic fertilizer applied (kg, not just N content) | 95.0% |
| `inorganic_fertilizer_value_LCU` | Cost in local currency | 94.4% |
| `inorganic_fertilizer_value_USD` | Cost in USD | 94.4% |
| `organic_fertilizer` | Binary: applied organic fertilizer (0/1) | 95.3% |

### Labor

| Variable | Description | Coverage |
|----------|-------------|---------|
| `total_labor_days` | Total person-days (family + hired) | 94.5% |
| `total_family_labor_days` | Family labor person-days | 92.6% |
| `total_hired_labor_days` | Hired labor person-days | 94.2% |
| `hired_labor_value_LCU` / `_USD` | Wage bill for hired labor | 94.1% |

### Other Inputs

| Variable | Description | Coverage |
|----------|-------------|---------|
| `seed_kg` | Seed quantity (kg) | 57.7% |
| `improved` | Improved/hybrid seed variety (0/1) | 55.7% |
| `used_pesticides` | Applied pesticide/herbicide | 95.2% |
| `irrigated` | Plot irrigated (0/1) | 95.9% |

---

## 7. Manager Demographics (7 variables)

### Source: `meta/Plot_dataset.dta`

Manager = plot-level decision-maker (not necessarily household head or survey respondent).

| Variable | Description | Coverage |
|----------|-------------|---------|
| `age_manager` | Age (years) | 93.3% |
| `female_manager` | Female (0/1) | 93.5% |
| `married_manager` | Married (0/1) | 93.2% |
| `formal_education_manager` | Any formal education (0/1) | 92.7% |
| `primary_education_manager` | Completed primary (0/1) | 88.0% |
| `age_respondent` | Survey respondent age | 82.4% |
| `female_respondent` | Survey respondent female (0/1) | 82.6% |

**Note**: Manager ≠ respondent. For BIRL modeling, use manager variables (the actual decision-maker).

---

## 8. Household Characteristics (15 variables)

### Source: `meta/Household_dataset.dta` → merged via `merge_dta_to_panel.py`

**Merge key**: (hh_id_merge, wave) — many-to-one. Match rate: 99.8%.

| Variable | Description | Coverage | Notes |
|----------|-------------|---------|-------|
| `hh_size` | Household members | 99.6% | |
| `hh_asset_index` | PCA-based asset index | 99.4% | Radio, TV, phone, bicycle, motorcycle, etc. |
| `hh_dependency_ratio` | Dependents / working-age | 99.7% | |
| `hh_formal_education` | Head has formal education (0/1) | 99.5% | |
| `hh_primary_education` | Head completed primary (0/1) | 98.1% | |
| `hh_electricity_access` | Access to electricity (0/1) | 99.5% | |
| `hh_shock` | Experienced shock past year (0/1) | 96.2% | Any type |
| `nonfarm_enterprise` | Has off-farm business (0/1) | 96.4% | Binary only, no revenue |
| `HDDS` | Dietary Diversity Score (0–12) | 94.7% | Count of food groups consumed |
| `totcons_LCU` | Annual total consumption (local currency) | 83.4% | |
| `totcons_USD` | Annual total consumption (USD) | 83.4% | |
| `cons_quint` | Consumption quintile (1–5) | 83.4% | Country-specific |
| `share_kg_sold` | Share of harvest sold | 55.9% | 0=subsistence, 1=fully commercial |
| `nb_plots` | Number of cultivated plots | 65.0% | |
| `nb_fallow_plots` | Number of fallow plots | 62.7% | |

---

## 9. Assets & Tenure (6 variables)

### Source: `meta/Plot_dataset.dta`

| Variable | Description | Coverage |
|----------|-------------|---------|
| `ag_asset_index` | Agricultural asset score | 89.7% |
| `livestock` | Owns livestock (0/1) | 98.1% |
| `tractor` | Owns/rents tractor (0/1) | 83.6% |
| `plot_owned` | Plot is owned (vs. rented/borrowed) (0/1) | 97.1% |
| `plot_certificate` | Has formal land title (0/1) | 91.3% |
| `irrigated` | Plot is irrigated (0/1) | 95.9% |

---

## 10. Shocks (5 variables)

### Source: `meta/Plotcrop_dataset.dta`

| Variable | Description | Coverage | Coding |
|----------|-------------|---------|--------|
| `crop_shock` | Any crop shock (0/1) | 87.4% | Binary |
| `pests_shock` | Pest/disease attack | 56.7% | 0/1/severity |
| `drought_shock` | Drought | 50.5% | 0/1/severity |
| `rain_shock` | Excessive rain | 29.2% | 0/1/severity |
| `flood_shock` | Flood | 38.3% | Binary |

Variable coverage differs because not all country surveys asked all shock questions.

---

## 11. Climate — Three-Layer System (12 variables)

### Architecture

```
For each of the 514,665 observations:
  │
  ├── Has valid planting_month AND harvest_end_month? (year 2000-2025)
  │     YES → Layer 1: PRECISE (37.9% of obs)
  │           growing_months = [planting → harvest], preplanting = 3 months before
  │
  │     NO ──┬── Country = Tanzania AND has harvest_end_month?
  │          │     YES → Layer 2 TZ: INFER from harvest (11.8%)
  │          │           harvest Jan-Jul → Masika (Nov-May)
  │          │           harvest Aug-Oct → Vuli (Oct-Jan)
  │          │
  │          │     NO → Layer 2: REVISED CALENDAR (50.3%)
  │          │           Use country × season lookup table
  │          └──────────────────────────────────────────────
  │
  ▼
  climate_source = 'precise' | 'calendar_s1' | 'calendar_s2' | 'calendar_tz_s1' | 'calendar_tz_s2'
```

### Revised Calendar (Layer 2)

| Country | Season 1 | Season 2 | v1→v3 changes |
|---------|----------|----------|---------------|
| Ethiopia | May–Dec | Feb–Jun (Belg) | Extended from Jun–Sep; added Belg |
| Uganda | Mar–Jul | Sep–Jan (cross-year) | Extended from Mar–Jun; added S2 |
| Tanzania | Nov–May | Oct–Jan (Vuli) | Added Vuli via harvest inference |
| Nigeria | Apr–Oct | — | Unchanged |
| Malawi | Nov–Apr (cross-year) | — | Unchanged |
| Mali | Jun–Oct | — | Unchanged |
| Niger | Jun–Sep | — | Unchanged |

### Cross-Year Handling

For Malawi (Nov→Apr), Tanzania (Nov→May), Uganda S2 (Sep→Jan):
- Months > 6 use survey year
- Months ≤ 6 use survey year + 1
- planting_month year is extracted directly from the datetime (not assumed = survey year)
- Malawi: planting_year == survey_year only 8.3% of cases (most plant Nov/Dec of previous year)

### Data Sources & Processing

#### CHIRPS Rainfall

| Item | Detail |
|------|--------|
| **Source** | `UCSB-CHG/CHIRPS/DAILY` via GEE |
| **Resolution** | ~5 km |
| **Time range** | 1997–2023 (supplemented from original 2007–2023) |
| **Extraction** | `reduceRegions()` for 7,018 GPS points, batched at 2,000/batch |
| **Intermediate** | `outputs/intermediate/climate/gee_chirps_monthly.csv` (2.27M rows) |
| **Aggregation** | Sum monthly rainfall over growing season months |

**Output variables**:
- `rainfall_growing_sum_final`: Growing season total rainfall (mm)
- `rainfall_10yr_mean_final`: Mean of annual growing-season totals for 10 years prior (mm/month)
- `rainfall_10yr_cv_final`: Coefficient of variation of those 10 annual totals
- `rainfall_10yr_nyears`: Actual number of lookback years used (all = 10 after supplement)

#### MODIS NDVI

| Item | Detail |
|------|--------|
| **Source** | `MODIS/061/MOD13A1` via GEE |
| **Resolution** | 500 m, 16-day composite |
| **Time range** | 2007–2023 |
| **Scale factor** | ×0.0001 (raw integer → -0.2 to 1.0) |
| **Aggregation** | Mean of monthly values within the window |

**Output variables**:
- `ndvi_growing_mean_final`: Mean NDVI during growing season
- `ndvi_preseason_mean_final`: Mean NDVI during pre-planting months (soil moisture proxy)

#### ERA5 Temperature

| Item | Detail |
|------|--------|
| **Source** | ERA5-Land (pre-downloaded as 71 batch parquets, daily) |
| **Resolution** | ~9 km |
| **Time range** | 2000–2020 (2021–2023 gap: no ERA5 data) |
| **Processing** | Streaming: one batch at a time → monthly aggregation → growing season mean |
| **Matching** | lat4/lon4 (4 decimal places ≈ 11m) |

**Output variables**:
- `era5_tmean_growing_final`: Mean daily-average temperature during growing season (°C)
- `era5_tmax_growing_final`: Mean daily-maximum temperature (°C)
- `era5_tmin_growing_final`: Mean daily-minimum temperature (°C)

### Reference Columns (old single-calendar, preserved for comparison)

`gee_rainfall_growing_sum`, `gee_rainfall_10yr_mean`, `gee_rainfall_10yr_cv`, `gee_ndvi_preseason_mean`, `gee_ndvi_growing_mean`, `era5_tmax_growing`, `era5_tmin_growing`, `era5_tmean_growing`

---

## 12. Terrain — NASADEM (3 variables)

### Source

| Item | Detail |
|------|--------|
| **Dataset** | NASADEM (NASA + SRTM combined DEM) |
| **Resolution** | ~30 m |
| **Extraction** | GEE `reduceRegions()` at each GPS point |
| **Static** | Does not vary across waves |

### Variables

| Variable | Unit | Derivation |
|----------|------|------------|
| `elevation_m` | Meters above sea level | Direct DEM value |
| `slope_deg` | Degrees (0–90) | ∂elevation/∂distance, computed by GEE `ee.Terrain.slope()` |
| `terrain_ruggedness_m` | Meters | TRI = √(Σ(elevation_neighbor − elevation_center)² / 8) |

---

## 13. Soil Chemistry — ISRIC SoilGrids (35 variables)

### Source

| Item | Detail |
|------|--------|
| **Dataset** | ISRIC SoilGrids v2.0 |
| **Resolution** | 250 m |
| **Extraction** | GEE `sampleRegions()` |
| **Static** | Does not vary across waves |

### Variables: 7 properties × 5 depths

| Property | Variable prefix | Unit | Interpretation |
|----------|----------------|------|----------------|
| Bulk density | `bdod_` | g/cm³ | Soil compaction |
| Clay content | `clay_` | % | Texture (water retention) |
| Sand content | `sand_` | % | Texture (drainage) |
| Silt content | `silt_` | % | Texture |
| Soil organic carbon | `soc_` | g/kg | Fertility indicator |
| Total nitrogen | `nitrogen_` | g/kg | Nutrient availability |
| pH (water) | `phh2o_` | pH units | Acidity/alkalinity |

**Depths**: `0-5cm`, `5-15cm`, `15-30cm`, `30-60cm`, `60-100cm`

### Merge Fix (v2)

Original pipeline used `gps_id` string matching — varying decimal precision between soil extraction (4-6 decimals) and panel (6 decimals) caused 2,364/7,018 GPS points to fail matching. Fixed by matching on `round(lat, 4)` and `round(lon, 4)` (≈11m precision). Coverage: 68.6% → **93.5%**.

---

## 14. Conflict Exposure — ACLED (31 variables)

### Source

| Item | Detail |
|------|--------|
| **Dataset** | ACLED (Armed Conflict Location & Event Data) |
| **File** | `Data/External/ACLED/acled_events_all_countries.parquet` |
| **Events** | 55,481 conflict events, 7 countries, 1997–2025 |
| **Extraction** | `src/extractors/conflict_extractor.py` |

### Spatial Indexing

```python
# BallTree with haversine metric for efficient radius queries
tree = BallTree(np.deg2rad(acled[['latitude','longitude']]), metric='haversine')
# Query: all events within radius of each household GPS
```

### Variable Structure: 3 dimensions

| Dimension | Values | Purpose |
|-----------|--------|---------|
| **Buffer radius** | 10 km, 25 km, 50 km | Spatial scale |
| **Time window** | 3 months, 12 months, 24 months | Temporal scale |
| **Measure** | event count, distance-weighted count, fatal indicator, fatality count | Intensity |

### Naming Convention

```
conflict_{measure}_{buffer}km_{window}m
Examples:
  conflict_events_25km_12m      = count of events within 25km, 12 months before planting
  conflict_events_wt_50km_24m   = Gaussian distance-weighted count, 50km, 24 months
  conflict_fatal_10km_3m        = binary: any fatal event within 10km, 3 months
```

### Distance Weighting Formula

```
w(d) = exp(-d² / (2 × h²))    where h = 10 km (bandwidth)
weighted_count = Σ w(d_i)      for all events i within buffer
```

### Primary Variables (4)

| Variable | Description |
|----------|-------------|
| `conflict_events_25km_12m` | Total events within 25km, past 12 months |
| `conflict_fatal_25km_12m` | Any fatal event within 25km, past 12 months (0/1) |
| `conflict_fatalities_25km_12m` | Total fatalities within 25km, past 12 months |
| `conflict_nearest_event_km` | Distance to nearest event in 24-month window (km) |

### Additional Variables: `conflict_battles_25km_12m` (battles only), `conflict_vc_25km_12m` (violence against civilians only)

**Temporal anchoring**: Conflict windows are anchored to planting month (growing season start), not survey interview date. This ensures the treatment measures the conflict environment farmers faced when making planting decisions.

---

## 15. Market Access (3 variables)

| Variable | Source | Coverage | Description |
|----------|--------|---------|-------------|
| `dist_market` | Survey questionnaire | 28.8% | Distance to nearest market (km). Only Ethiopia (98%) and Nigeria (79%) asked this. |
| `dist_popcenter` | Survey questionnaire | 70.5% | Distance to population center (km). |
| `travel_time_city_min` | GEE Nelson et al. 2015 | 96.4% | Travel time to nearest city ≥50K population (minutes). Static raster, extracted via `reduceRegions()`. |

`travel_time_city_min` is the recommended market access proxy for cross-country analysis (96.4% vs 28.8% for survey-based `dist_market`).

---

## 16. Price Deflators (2 variables)

### Source: World Bank WDI Consumer Price Index

| Variable | Description |
|----------|-------------|
| `cpi` | Consumer Price Index (base varies by country) |
| `cpi_deflator` | Deflator to convert nominal to 2010 constant prices |

### Deflation Formula

```
cpi_deflator = (CPI_2010 / CPI_year) × 100

harvest_value_LCU_real2010 = harvest_value_LCU × (cpi_deflator / 100)
harvest_value_USD_real2010 = harvest_value_USD × (cpi_deflator / 100)
```

Applied to: harvest value, seed value, hired labor cost, fertilizer cost.

---

## 17. Planting & Harvest Timing (2 variables)

| Variable | Type | Coverage | Source |
|----------|------|---------|--------|
| `planting_month` | datetime64 | 60.9% | Survey: "When did you plant?" |
| `harvest_end_month` | datetime64 | 68.7% | Survey: "When did harvesting end?" |

Contains actual year and month (e.g., `2011-06-01`). Critical for Layer 1 climate assignment.

**Known issues**:
- 1.18% of dates are anomalous (year < 2000 or > 2025) → treated as missing, fallback to Layer 2
- Malawi: planting_year = survey_year only 8.3% of cases (Nov/Dec planting → previous year)
- Tanzania: planting_month has 0% coverage (harvest_end_month has 89.2%)

---

## 18. Quality Flags (11 variables)

### Design Principle: Minimal Intervention

The pipeline **flags** anomalies but **never modifies, winsorizes, or deletes** original values. All quality indicators are additive boolean columns. Researchers choose their own filtering strategy.

| Flag | Condition | Rate |
|------|-----------|-----:|
| `is_gps_invalid` | Coordinates (0,0) or outside ±90/±180 | 0.00% |
| `is_gps_out_of_bounds` | Outside country bounding box ±0.5° | 0.00% |
| `is_gps_missing` | No GPS at any imputation level | 0.19% |
| `is_area_too_small` | plot_area_GPS < 0.001 ha (10 m²) | 1.49% |
| `is_area_very_large` | plot_area_GPS > 100 ha | 0.03% |
| `is_area_missing` | plot_area_GPS is NaN | 5.76% |
| `is_harvest_zero` | harvest_kg = 0 | 3.93% |
| `is_harvest_very_low` | harvest_kg < 0.1 kg | 0.04% |
| `is_harvest_missing` | harvest_kg is NaN | 12.08% |
| `is_yield_extreme` | yield_kg_ha outside 1st/99th percentile within crop category | 0.43% |
| `is_intercrop_inconsistent` | Intercrop percentage sum > 105% | 0.00% |

---

## 19. Changelog

| Version | Date | Changes |
|---------|------|---------|
| Base | — | Original 5-stage pipeline: 117 variables, 7 countries |
| +ACLED | 2026-03-06 | Conflict merge: +31 variables, 99.8% coverage |
| v1 | 2026-03-16 | Plot_dataset.dta merge: +34 vars (fertilizer, labor, demographics, assets) |
| v1 | 2026-03-16 | Household_dataset.dta merge: +15 vars (hh_size, assets, consumption) |
| v1 | 2026-03-16 | GEE CHIRPS+MODIS extraction: 7,018 pts × 2007–2023, climate 54%→99.5% |
| v1 | 2026-03-16 | Crop category case-insensitive remapping: 42.4%→97.4% |
| v2 | 2026-03-16 | Soil merge precision fix (lat4/lon4): 68.6%→93.5% |
| v2 | 2026-03-16 | ERA5 temperature streaming: +3 vars, 91.0% |
| v2 | 2026-03-16 | Travel time to city (GEE Nelson): +1 var, 96.4% |
| v3 | 2026-03-16 | Growing season 3-layer fix: precise dates + revised calendar + TZ inference |
| v3 | 2026-03-16 | CHIRPS 1997–2006 supplement: 10yr lookback 100% complete |
| v3 | 2026-03-16 | ERA5 per-obs window: corrected for dual-season countries |

---

## 20. Key Design Decisions

1. **Minimal intervention**: Flag anomalies, preserve originals. No winsorizing, no deletion. Researchers choose their own strategy.
2. **Three-layer climate**: Precise dates (38%) → revised calendar (50%) → TZ harvest inference (12%). Tracks source in `climate_source`.
3. **GPS imputation hierarchy**: plot → EA median → district median, with full flag transparency.
4. **Conflict anchored to planting**: Not survey date. Measures the security environment at decision time.
5. **CPI base year 2010**: Consistent cross-country real price comparison.
6. **4-decimal lat/lon matching**: Avoids floating-point precision issues between GEE output and panel coordinates (~11m tolerance).
7. **Cross-year month handling**: Explicit year assignment for Nov→Apr seasons. planting_month datetime used directly, not assumed = survey year.
8. **Separate final vs reference columns**: Old `gee_*` columns preserved alongside corrected `*_final` columns for comparison.
