#!/usr/bin/env python3
"""
02 Action Space Construction
============================
Discretize continuous crop×input decisions into 9 crops × 3 intensity tiers.
Decision unit = plot-crop (222K obs). Intercropped is state, not action.

Output: Updated birl_sample.parquet + action_space_config.json + report
"""

import json, logging
from pathlib import Path
import pandas as pd, numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

BASE = Path(__file__).resolve().parent.parent  # Formal Analysis/
INPUT = BASE / "data" / "birl_sample.parquet"
OUTPUT_PARQUET = BASE / "data" / "birl_sample.parquet"
CONFIG_OUT = Path(__file__).resolve().parent / "action_space_config.json"
REPORT_OUT = Path(__file__).resolve().parent / "action_space_report.md"

# Crop mapping: main_crop → action_crop
CROP_MAP = {
    'MAIZE': 'maize',
    'PERENNIAL/FRUIT': 'tree_crops',
    'NUTS': 'tree_crops',
    'TUBERS / ROOT CROPS': 'tubers',
    'BEANS AND OTHER LEGUMES': 'legumes',
    'SORGHUM': 'sorghum_millet',
    'MILLET': 'sorghum_millet',
    'RICE': 'rice',
    'WHEAT': 'wheat_barley',
    'BARLEY': 'wheat_barley',
    'OTHER': 'other',  # will be overridden for TEFF
}

CROP_ORDER = ['maize', 'tree_crops', 'tubers', 'legumes', 'sorghum_millet',
              'teff', 'other', 'rice', 'wheat_barley']

# High-fertilizer crops (use PCA-based 3-tier)
HIGH_FERT_CROPS = {'maize', 'rice', 'wheat_barley', 'sorghum_millet', 'teff'}
# Low-fertilizer crops (use binary fert × labor 2-tier → merge to 3)
LOW_FERT_CROPS = {'tubers', 'legumes', 'tree_crops', 'other'}

report = []


def rpt(line=""):
    report.append(line)
    if line and not line.startswith('|'):
        logger.info(line)


def main():
    rpt("# Action Space Construction Report")
    rpt()

    df = pd.read_parquet(INPUT)
    N = len(df)
    rpt(f"Input: {N:,} obs, {df['hh_id_merge'].nunique():,} HH")
    rpt()

    # ===== STEP 1: CROP CATEGORIES =====
    rpt("## 1. Crop Category Mapping (11 → 9)")
    rpt()

    # Map main_crop → action_crop
    df['action_crop'] = df['main_crop'].map(CROP_MAP)

    # Split TEFF from OTHER
    teff_mask = (df['main_crop'] == 'OTHER') & (df['crop_name'] == 'TEFF')
    df.loc[teff_mask, 'action_crop'] = 'teff'

    rpt("| action_crop | Source | Obs | % |")
    rpt("|-------------|------|----:|--:|")
    for crop in CROP_ORDER:
        n = (df['action_crop'] == crop).sum()
        rpt(f"| {crop} | — | {n:,} | {100*n/N:.1f}% |")
    rpt()

    # ===== STEP 2: INPUT INTENSITY =====
    rpt("## 2. Input Intensity Tiering")
    rpt()

    df['input_intensity'] = ''

    # --- High-fertilizer crops: PCA-based ---
    rpt("### High-Fertilizer Crops (PCA-based 3-tier)")
    rpt()
    rpt("| Country | Crop | N | P33 fert | P67 fert | P33 labor | P67 labor |")
    rpt("|---------|------|--:|------:|------:|-------:|-------:|")

    for crop in HIGH_FERT_CROPS:
        crop_mask = df['action_crop'] == crop
        for country in sorted(df.loc[crop_mask, 'country'].unique()):
            mask = crop_mask & (df['country'] == country)
            sub = df.loc[mask].copy()
            if len(sub) < 30:
                # Too few obs: assign all to 'medium'
                df.loc[mask, 'input_intensity'] = 'medium'
                continue

            # Standardize fertilizer_kg_ha and labor_days_ha
            X = sub[['fertilizer_kg_ha', 'labor_days_ha']].fillna(0).values
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # PCA first component
            pca = PCA(n_components=1)
            pc1 = pca.fit_transform(X_scaled).flatten()

            # Tercile split
            p33 = np.percentile(pc1, 33.33)
            p67 = np.percentile(pc1, 66.67)

            intensity = np.where(pc1 <= p33, 'low',
                        np.where(pc1 <= p67, 'medium', 'high'))
            df.loc[mask, 'input_intensity'] = intensity

            # Report thresholds
            fert = sub['fertilizer_kg_ha']
            labor = sub['labor_days_ha']
            rpt(f"| {country} | {crop} | {len(sub):,} | {fert.quantile(0.33):.1f} | {fert.quantile(0.67):.1f} | {labor.quantile(0.33):.1f} | {labor.quantile(0.67):.1f} |")

    rpt()

    # --- Low-fertilizer crops: binary fert × labor median ---
    rpt("### Low-Fertilizer Crops (Binary Fert x Labor Median)")
    rpt()
    rpt("| Country | Crop | N | % fert>0 | Labor P50 |")
    rpt("|---------|------|--:|--------:|----------:|")

    for crop in LOW_FERT_CROPS:
        crop_mask = df['action_crop'] == crop
        for country in sorted(df.loc[crop_mask, 'country'].unique()):
            mask = crop_mask & (df['country'] == country)
            sub = df.loc[mask].copy()
            if len(sub) < 30:
                df.loc[mask, 'input_intensity'] = 'medium'
                continue

            used_fert = sub['fertilizer_kg_ha'] > 0
            labor_med = sub['labor_days_ha'].median()
            high_labor = sub['labor_days_ha'] > labor_med

            # 3-tier:
            # low: no fert + low labor
            # medium: (no fert + high labor) OR (fert + low labor)
            # high: fert + high labor
            intensity = np.where(~used_fert & ~high_labor, 'low',
                        np.where(used_fert & high_labor, 'high', 'medium'))
            df.loc[mask, 'input_intensity'] = intensity

            rpt(f"| {country} | {crop} | {len(sub):,} | {100*used_fert.mean():.1f}% | {labor_med:.1f} |")

    rpt()

    # ===== STEP 3: ACTION ENCODING =====
    rpt("## 3. Action Encoding")
    rpt()

    crop_to_idx = {c: i for i, c in enumerate(CROP_ORDER)}
    intensity_to_idx = {'low': 0, 'medium': 1, 'high': 2}

    df['crop_index'] = df['action_crop'].map(crop_to_idx)
    df['intensity_index'] = df['input_intensity'].map(intensity_to_idx)
    df['action_id'] = df['crop_index'] * 3 + df['intensity_index']

    # Action labels
    action_labels = {}
    for crop in CROP_ORDER:
        ci = crop_to_idx[crop]
        for intensity in ['low', 'medium', 'high']:
            ii = intensity_to_idx[intensity]
            aid = ci * 3 + ii
            action_labels[aid] = f"{crop}_{intensity}"

    # Frequency table
    rpt("| action_id | Label | Obs | % |")
    rpt("|:---------:|-------|----:|--:|")
    freq = df['action_id'].value_counts().sort_index()
    active_actions = []
    for aid in sorted(action_labels.keys()):
        n = freq.get(aid, 0)
        label = action_labels[aid]
        status = "" if n >= 50 else " ⚠️<50"
        rpt(f"| {aid} | {label} | {n:,} | {100*n/N:.1f}%{status} |")
        if n >= 50:
            active_actions.append(aid)
    rpt()
    rpt(f"**Theoretical actions: {len(action_labels)}, active actions (>=50 obs): {len(active_actions)}**")
    rpt()

    # Merge low-frequency actions into nearest neighbor
    low_freq = [aid for aid in freq.index if freq[aid] < 50]
    if low_freq:
        rpt(f"### Low-Frequency Action Merging")
        rpt()
        for aid in low_freq:
            crop_idx = aid // 3
            # Merge to medium intensity of same crop
            target = crop_idx * 3 + 1  # medium
            if target == aid:
                target = crop_idx * 3 + 0  # low
            rpt(f"- action {aid} ({action_labels[aid]}, n={freq.get(aid,0)}) → merged to {target} ({action_labels[target]})")
            df.loc[df['action_id'] == aid, 'action_id'] = target
            df.loc[df['action_id'] == target, 'input_intensity'] = 'medium'  # relabel
        rpt()

    # ===== STEP 4: AEZ FEASIBILITY MASK =====
    rpt("## 4. AEZ Feasibility Mask")
    rpt()

    # Merge AEZ 311 → 312
    df['action_zone'] = df['agro_ecological_zone'].copy()
    df.loc[df['action_zone'] == 311, 'action_zone'] = 312

    # Fill missing AEZ with country×admin_1 mode
    missing_aez = df['action_zone'].isna()
    if missing_aez.sum() > 0:
        mode_by_ca = df.dropna(subset=['action_zone']).groupby(['country', 'admin_1'])['action_zone'].agg(
            lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else np.nan
        )
        for idx in df[missing_aez].index:
            key = (df.loc[idx, 'country'], df.loc[idx, 'admin_1'])
            if key in mode_by_ca.index:
                df.loc[idx, 'action_zone'] = mode_by_ca[key]
    df['action_zone'] = df['action_zone'].fillna(312)  # final fallback
    df['action_zone'] = df['action_zone'].astype(int)

    rpt(f"AEZ missing imputation: {missing_aez.sum():,} obs -> filled with country x admin_1 mode")
    rpt()

    zones = sorted(df['action_zone'].unique())
    final_actions = sorted(df['action_id'].unique())

    # Build mask
    mask = {}
    rpt("| Zone | Feasible actions | Total obs |")
    rpt("|:----:|:----------------:|----------:|")
    for zone in zones:
        zone_df = df[df['action_zone'] == zone]
        zone_freq = zone_df['action_id'].value_counts()
        feasible = [int(a) for a in final_actions if zone_freq.get(a, 0) >= 10]
        mask[int(zone)] = feasible
        rpt(f"| {zone} | {len(feasible)} | {len(zone_df):,} |")
    rpt()

    # ===== STEP 5: VALIDATION =====
    rpt("## 5. Validation")
    rpt()

    # Check 1: all actions ≥50
    final_freq = df['action_id'].value_counts()
    min_freq = final_freq.min()
    rpt(f"- Min action frequency: {min_freq} (required >=50): {'PASS' if min_freq >= 50 else 'FAIL'}")

    # Check 2: each zone ≥3 feasible
    min_feasible = min(len(v) for v in mask.values())
    rpt(f"- Min feasible actions per zone: {min_feasible} (required >=3): {'PASS' if min_feasible >= 3 else 'FAIL'}")

    # Check 3: max action share <30%
    max_share = final_freq.max() / N
    rpt(f"- Max action share: {100*max_share:.1f}% (required <30%): {'PASS' if max_share < 0.30 else 'FAIL'}")

    # Check 4: all obs have action_id
    all_assigned = df['action_id'].notna().all()
    rpt(f"- All obs have action_id: {'PASS' if all_assigned else 'FAIL'}")

    # Check 5: intensity roughly balanced per crop
    rpt()
    rpt("### Input Intensity Balance")
    rpt()
    rpt("| Crop | Low % | Med % | High % |")
    rpt("|------|------:|------:|-------:|")
    for crop in CROP_ORDER:
        sub = df[df['action_crop'] == crop]
        for intensity in ['low', 'medium', 'high']:
            pct = 100 * (sub['input_intensity'] == intensity).sum() / len(sub)
            if intensity == 'low':
                row = f"| {crop} | {pct:.1f}% |"
            elif intensity == 'medium':
                row += f" {pct:.1f}% |"
            else:
                row += f" {pct:.1f}% |"
                rpt(row)
    rpt()

    # ===== STEP 6: SAVE =====
    # Drop temp columns
    df = df.drop(columns=['crop_index', 'intensity_index'], errors='ignore')

    df.to_parquet(OUTPUT_PARQUET, index=False)
    rpt(f"## Output")
    rpt()
    rpt(f"- `birl_sample.parquet`: {len(df):,} × {len(df.columns)} cols")
    rpt(f"- Actions: {len(final_actions)} (theoretical 27, active {len(active_actions)})")
    rpt(f"- AEZ zones: {len(zones)}")

    # Save config JSON
    config = {
        'crop_order': CROP_ORDER,
        'action_labels': {str(k): v for k, v in action_labels.items()},
        'active_actions': [int(a) for a in final_actions],
        'n_actions': len(final_actions),
        'feasibility_mask': {str(k): v for k, v in mask.items()},
        'zones': [int(z) for z in zones],
        'action_frequencies': {str(int(k)): int(v) for k, v in final_freq.items()},
        'high_fert_crops': list(HIGH_FERT_CROPS),
        'low_fert_crops': list(LOW_FERT_CROPS),
    }
    with open(CONFIG_OUT, 'w') as f:
        json.dump(config, f, indent=2)
    rpt(f"- `action_space_config.json`: saved")

    # Save report
    with open(REPORT_OUT, 'w') as f:
        f.write('\n'.join(report))
    logger.info(f"Report: {REPORT_OUT}")
    logger.info(f"Done: {len(final_actions)} actions, {len(zones)} zones")


if __name__ == '__main__':
    main()
