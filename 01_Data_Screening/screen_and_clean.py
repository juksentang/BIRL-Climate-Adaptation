#!/usr/bin/env python3
"""
BIRL Sample Screening and Cleaning
====================================
Filter a clean subsample satisfying BIRL analysis requirements from 514,665 panel observations.

Input:  all_countries_panel_birl.parquet (211 columns)
Output: birl_sample.parquet + sample_selection_report.md
"""

import logging
from pathlib import Path
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

BASE = Path(__file__).resolve().parent.parent  # Formal Analysis/
INPUT = BASE / "data" / "all_countries_panel_birl.parquet"
OUTPUT = BASE / "data" / "birl_sample.parquet"
REPORT = Path(__file__).resolve().parent / "sample_selection_report.md"

GLOBAL_CAPS = {
    'yield_kg_ha': 80_000,
    'harvest_value_USD': 50_000,
    'nitrogen_kg': 500,
    'total_labor_days': 5_000,
    'hired_labor_value_USD': 10_000,
    'seed_kg': 2_000,
}

WINSORIZE_VARS = ['yield_kg_ha', 'harvest_value_USD', 'nitrogen_kg',
                  'total_labor_days', 'hired_labor_value_USD', 'seed_kg']

report = []  # collect report lines


def rpt(line=""):
    report.append(line)
    if line:
        logger.info(line)


def main():
    rpt("# BIRL Sample Selection Report")
    rpt()
    rpt(f"Generated: 2026-03-16")
    rpt()

    # =========== LOAD ===========
    df = pd.read_parquet(INPUT)
    N0 = len(df)
    rpt(f"## 1. Selection Chain")
    rpt()
    rpt(f"Input: {N0:,} obs, {df['hh_id_merge'].nunique():,} HH, {df['country'].nunique()} countries")
    rpt()
    rpt("| Step | Obs | HH | Cumul% | Dropped |")
    rpt("|------|----:|---:|------:|--------:|")
    rpt(f"| Base | {N0:,} | {df['hh_id_merge'].nunique():,} | 100% | — |")

    def apply_filter(df, name, mask):
        prev = len(df)
        df = df[mask]
        hh = df['hh_id_merge'].nunique()
        drop = prev - len(df)
        rpt(f"| {name} | {len(df):,} | {hh:,} | {100*len(df)/N0:.1f}% | {drop:,} |")
        return df

    # =========== STEP 1: FILTERS ===========
    df = apply_filter(df, "harvest_value_USD notna", df['harvest_value_USD'].notna())
    df = apply_filter(df, "area 0.001–20 ha", (df['plot_area_GPS'] > 0.001) & (df['plot_area_GPS'] < 20) & df['plot_area_GPS'].notna())
    df = apply_filter(df, "crop_category + main_crop", df['crop_category'].notna() & df['main_crop'].notna())
    df = apply_filter(df, "rain + ndvi _final", df['rainfall_growing_sum_final'].notna() & df['ndvi_growing_mean_final'].notna())
    df = apply_filter(df, "hh_size + asset", df['hh_size'].notna() & df['hh_asset_index'].notna())
    df = apply_filter(df, "not yield_extreme", df['is_yield_extreme'] == False)
    df = apply_filter(df, "Exclude Niger", df['country'] != 'Niger')

    # Panel depth
    hh_waves = df.groupby('hh_id_merge')['wave'].nunique()
    panel_hh = set(hh_waves[hh_waves >= 2].index)
    prev = len(df)
    df = df[df['hh_id_merge'].isin(panel_hh)]
    rpt(f"| **HH ≥2 waves** | **{len(df):,}** | **{len(panel_hh):,}** | **{100*len(df)/N0:.1f}%** | **{prev-len(df):,}** |")
    rpt()

    # Zero harvest adjustment
    df['harvest_value_USD_adj'] = df['harvest_value_USD'].clip(lower=0.01)
    df['yield_kg_ha_adj'] = df['yield_kg_ha'].fillna(0).clip(lower=0.001)

    # =========== STEP 2: COUNTRY BREAKDOWN ===========
    rpt("## 2. Country Distribution")
    rpt()
    rpt("| Country | Obs | HH | Obs/HH |")
    rpt("|---------|----:|---:|-------:|")
    for c in sorted(df['country'].unique()):
        sub = df[df['country'] == c]
        hh = sub['hh_id_merge'].nunique()
        rpt(f"| {c} | {len(sub):,} | {hh:,} | {len(sub)/hh:.1f} |")
    rpt()

    # =========== STEP 3: PANEL DEPTH ===========
    rpt("## 3. Panel Depth")
    rpt()

    # Standard: by wave
    hh_depth = df.groupby('hh_id_merge')['wave'].nunique()
    rpt("### By Wave Count")
    rpt()
    rpt("| Wave depth | HH count | Share |")
    rpt("|:----------:|-----:|-----:|")
    total_hh = len(hh_depth)
    for n in range(2, 8):
        cnt = (hh_depth == n).sum()
        if cnt > 0:
            rpt(f"| {n} | {cnt:,} | {100*cnt/total_hh:.1f}% |")
    cnt5 = (hh_depth >= 5).sum()
    rpt(f"| 5+ | {cnt5:,} | {100*cnt5/total_hh:.1f}% |")
    rpt()

    # Effective decision points: wave × season
    rpt("### By Wave x Season Count (Effective Decision Points)")
    rpt()
    df['decision_id'] = df['hh_id_merge'] + '_' + df['wave'].astype(str) + '_' + df['season'].astype(str)
    hh_decisions = df.groupby('hh_id_merge')['decision_id'].nunique()
    rpt("| Decision points | HH count | Share |")
    rpt("|:--------:|-----:|-----:|")
    for n in range(2, 12):
        cnt = (hh_decisions == n).sum()
        if cnt > 0 and cnt >= 10:
            rpt(f"| {n} | {cnt:,} | {100*cnt/total_hh:.1f}% |")
    cnt_high = (hh_decisions >= 10).sum()
    rpt(f"| 10+ | {cnt_high:,} | {100*cnt_high/total_hh:.1f}% |")
    rpt()
    rpt(f"**Uganda dual-season effect**: Uganda HH mean decisions={hh_decisions[df[df['country']=='Uganda']['hh_id_merge'].unique()].mean():.1f}, "
        f"vs other countries mean={hh_decisions[df[df['country']!='Uganda']['hh_id_merge'].unique()].mean():.1f}")
    rpt()

    # =========== STEP 4: WINSORIZATION ===========
    rpt("## 4. Winsorization")
    rpt()

    # Track small groups
    small_groups = []

    for var in WINSORIZE_VARS:
        if var not in df.columns:
            continue
        # Global cap first
        cap = GLOBAL_CAPS.get(var, np.inf)
        df[f'{var}_w'] = df[var].clip(upper=cap)

        # Country × main_crop P1/P99
        for (c, crop), grp_idx in df.groupby(['country', 'main_crop']).groups.items():
            if len(grp_idx) < 50:
                small_groups.append((c, crop, len(grp_idx), var))
                continue
            vals = df.loc[grp_idx, f'{var}_w'].dropna()
            if len(vals) < 10:
                continue
            p1, p99 = vals.quantile(0.01), vals.quantile(0.99)
            df.loc[grp_idx, f'{var}_w'] = df.loc[grp_idx, f'{var}_w'].clip(lower=p1, upper=p99)

    # Report winsorization
    rpt("### Before vs After Comparison")
    rpt()
    rpt("| Variable | Raw P99 | Raw Max | Winsorized P99 | Winsorized Max |")
    rpt("|------|--------:|--------:|---------------:|---------------:|")
    for var in WINSORIZE_VARS:
        if var not in df.columns:
            continue
        orig = df[var].dropna()
        winz = df[f'{var}_w'].dropna()
        rpt(f"| `{var}` | {orig.quantile(0.99):,.1f} | {orig.max():,.1f} | {winz.quantile(0.99):,.1f} | {winz.max():,.1f} |")
    rpt()

    # Small groups report
    if small_groups:
        sg_df = pd.DataFrame(small_groups, columns=['country', 'crop', 'n_obs', 'variable'])
        sg_summary = sg_df.groupby(['country', 'crop'])['n_obs'].first().reset_index()
        rpt(f"### Small Groups (<50 obs, P1/P99 skipped): {len(sg_summary)} groups")
        rpt()
        rpt("| Country | Crop | N obs |")
        rpt("|---------|------|------:|")
        for _, r in sg_summary.sort_values('n_obs').iterrows():
            rpt(f"| {r['country']} | {r['crop']} | {r['n_obs']} |")
        rpt()

    # =========== STEP 5: MISSING VALUE HANDLING ===========
    rpt("## 5. Missing Value Handling")
    rpt()

    # nitrogen_kg: NaN → 0
    na_before = df['nitrogen_kg'].isna().sum()
    df['nitrogen_kg_w'] = df['nitrogen_kg_w'].fillna(0)
    rpt(f"- `nitrogen_kg`: {na_before:,} NaN -> 0 (76.8% already 0, NaN = not used)")

    # hired labor: NaN → 0
    na_before = df['total_hired_labor_days'].isna().sum()
    df['total_hired_labor_days'] = df['total_hired_labor_days'].fillna(0)
    if 'hired_labor_value_USD_w' in df.columns:
        df['hired_labor_value_USD_w'] = df['hired_labor_value_USD_w'].fillna(0)
    rpt(f"- `total_hired_labor_days`: {na_before:,} NaN -> 0")

    # total_labor_days: NaN → 0 (for _w version)
    df['total_labor_days_w'] = df['total_labor_days_w'].fillna(0)

    # Soil median imputation
    soil_vars = ['clay_0-5cm', 'soc_0-5cm', 'phh2o_0-5cm', 'nitrogen_0-5cm']
    for var in soil_vars:
        if var not in df.columns:
            continue
        na_before = df[var].isna().sum()
        if na_before > 0:
            df[f'{var}_missing'] = df[var].isna().astype(int)
            medians = df.groupby(['country', 'wave'])[var].transform('median')
            df[var] = df[var].fillna(medians)
            na_after = df[var].isna().sum()
            rpt(f"- `{var}`: {na_before:,} NaN -> country x wave median ({na_after:,} remaining)")

    rpt(f"- `seed_kg`: Keep NaN (data gap, not zero-filled)")
    rpt(f"- `era5_tmean_growing_final`: Keep NaN (handled natively by LightGBM)")
    rpt(f"- `soil_fertility_index`: Keep NaN (74.4% coverage too low)")
    rpt()

    # =========== STEP 6: DERIVED VARIABLES ===========
    rpt("## 6. Derived Variables")
    rpt()

    df['log_yield'] = np.log(df['yield_kg_ha_adj'] + 1)
    df['log_harvest_value'] = np.log(df['harvest_value_USD_adj'] + 1)
    df['log_area'] = np.log(df['plot_area_GPS'])
    df['log_labor'] = np.log(df['total_labor_days_w'] + 1)

    df['fertilizer_kg_ha'] = df['nitrogen_kg_w'] / (df['plot_area_GPS'] + 0.01)
    df['labor_days_ha'] = df['total_labor_days_w'] / (df['plot_area_GPS'] + 0.01)

    # Winsorize ratio variables
    for var in ['fertilizer_kg_ha', 'labor_days_ha']:
        p99 = df[var].quantile(0.99)
        df[var] = df[var].clip(upper=p99)

    df['used_fertilizer'] = (df['nitrogen_kg'].fillna(0) > 0).astype(int)
    df['used_hired_labor'] = (df['total_hired_labor_days'] > 0).astype(int)

    derived = ['log_yield', 'log_harvest_value', 'log_area', 'log_labor',
               'fertilizer_kg_ha', 'labor_days_ha', 'used_fertilizer', 'used_hired_labor',
               'harvest_value_USD_adj', 'yield_kg_ha_adj', 'decision_id']
    rpt("| Variable | Formula | Coverage |")
    rpt("|------|------|-------:|")
    for var in derived:
        nn = df[var].notna().sum()
        rpt(f"| `{var}` | — | {100*nn/len(df):.1f}% |")
    rpt()

    # =========== STEP 7: MALI NOTE ===========
    rpt("## 7. Mali Uncertainty Note")
    rpt()
    mali = df[df['country'] == 'Mali']
    mali_hh = mali['hh_id_merge'].nunique()
    mali_waves = mali.groupby('hh_id_merge')['wave'].nunique().mean()
    rpt(f"Mali: {mali_hh:,} HH, mean {mali_waves:.1f} waves/HH, {len(mali):,} obs.")
    rpt(f"Only 2 waves of data. In the three-parameter hierarchical model, Mali's country-level posterior will be wider than other countries.")
    rpt(f"Recommend reporting posterior width by country in the final report and explicitly noting Mali's uncertainty.")
    rpt()

    # =========== STEP 8: DESCRIPTIVE STATS ===========
    rpt("## 8. Key Variable Descriptive Statistics (Post-Screening)")
    rpt()
    stats_vars = ['yield_kg_ha_w', 'harvest_value_USD_w', 'nitrogen_kg_w', 'total_labor_days_w',
                  'plot_area_GPS', 'hh_size', 'hh_asset_index', 'rainfall_growing_sum_final',
                  'ndvi_growing_mean_final', 'fertilizer_kg_ha', 'labor_days_ha',
                  'travel_time_city_min', 'conflict_events_25km_12m']
    rpt("| Variable | N | Mean | Std | P1 | Median | P99 | Max |")
    rpt("|------|--:|-----:|----:|---:|-------:|----:|----:|")
    for var in stats_vars:
        if var not in df.columns:
            continue
        s = df[var].dropna()
        if len(s) == 0:
            continue

        def fmt(x):
            if abs(x) >= 1000:
                return f"{x:,.1f}"
            if abs(x) >= 1:
                return f"{x:.2f}"
            if abs(x) >= 0.01:
                return f"{x:.4f}"
            return f"{x:.6f}"

        rpt(f"| `{var}` | {len(s):,} | {fmt(s.mean())} | {fmt(s.std())} | {fmt(s.quantile(0.01))} | {fmt(s.median())} | {fmt(s.quantile(0.99))} | {fmt(s.max())} |")
    rpt()

    # =========== STEP 9: DESIGN DECISIONS ===========
    rpt("## 9. Key Design Decisions")
    rpt()
    rpt("1. **BIRL decision unit = household x wave x season**. Uganda dual-season observations are independent decision points.")
    rpt("2. **Zero harvest retained**. Assigned $0.01 for log transformation. The environment model needs to see the lower tail distribution.")
    rpt("3. **seed_kg NaN retained**. 38.6% missing is a data gap, not 'not used'.")
    rpt("4. **Niger excluded**. 1 wave, 0 panel HH. Final count: 6 countries.")
    rpt("5. **Intensity ratio variables**: denominator constant (0.01 ha) + P99 winsorization.")
    rpt("6. **Global cap as safety net** + country x crop P1/P99 winsorization as double insurance.")
    rpt()

    # =========== SAVE ===========
    df.to_parquet(OUTPUT, index=False)
    fsize = OUTPUT.stat().st_size / (1024**2)
    rpt(f"## Output")
    rpt()
    rpt(f"- `birl_sample.parquet`: {len(df):,} obs × {len(df.columns)} cols ({fsize:.1f} MB)")
    rpt(f"- 6 countries, {df['hh_id_merge'].nunique():,} HH, all with >=2 waves")

    # Write report
    with open(REPORT, 'w') as f:
        f.write('\n'.join(report))
    logger.info(f"Report: {REPORT}")
    logger.info(f"Done: {len(df):,} × {len(df.columns)}")


if __name__ == '__main__':
    main()
