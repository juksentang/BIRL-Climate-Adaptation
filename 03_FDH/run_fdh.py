#!/usr/bin/env python3
"""
03 Order-m FDH Frontier Estimation
===================================
Output-oriented Order-m FDH (Free Disposal Hull).
No LP needed — pure numpy max operations.

Cazals, Florens & Simar (2002): For each obs i,
  1. Find dominated set {j : X_j ≤ X_i element-wise}
  2. Sample m from dominated set
  3. θ_b = max(Y_sampled) / Y_i
  4. Repeat B times, θ_i = mean(θ)

3 inputs (land, fertilizer, labor) × 1 output (harvest value USD).
Stratified by (country × main_crop).
"""

import logging
from pathlib import Path
import time
import pandas as pd
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

BASE = Path(__file__).resolve().parent.parent
INPUT = BASE / "data" / "birl_sample.parquet"
OUTPUT = BASE / "data" / "birl_sample.parquet"
REPORT = Path(__file__).resolve().parent / "fdh_report.md"

X_COLS = ['plot_area_GPS', 'nitrogen_kg_w', 'total_labor_days_w']
Y_COL = 'harvest_value_USD_w'

MERGE_MAP = {
    ('Tanzania', 'MILLET'): ('Tanzania', 'SORGHUM'),
    ('Mali', 'TUBERS / ROOT CROPS'): ('Mali', 'OTHER'),
    ('Malawi', 'MILLET'): ('Malawi', 'SORGHUM'),
    ('Malawi', 'RICE'): ('Malawi', 'OTHER'),
    ('Tanzania', 'WHEAT'): ('Tanzania', 'OTHER'),
    ('Nigeria', 'WHEAT'): ('Nigeria', 'OTHER'),
    ('Nigeria', 'NUTS'): ('Nigeria', 'OTHER'),
}

M = 25
B = 200
MIN_LAYER = 50

report = []


def rpt(line=""):
    report.append(line)
    if line and not line.startswith('|'):
        logger.info(line)


def order_m_fdh(X, Y, m=M, B_reps=B, seed=42):
    """
    Order-m FDH efficiency (output-oriented, single output).
    No LP — pure numpy.

    Returns: (scores, fallback_count)
      scores: (N,) efficiency, can be >1 (super-efficient)
      fallback_count: obs with dominated set < m
    """
    rng = np.random.RandomState(seed)
    N, K = X.shape
    scores = np.ones(N, dtype=np.float64)
    fallback = 0

    if N <= 25000:
        # Vectorized domination matrix: dom[i,j] = True if X[j] <= X[i] for all k
        # Shape (N, N, K) → (N, N)
        dom_matrix = np.all(X[np.newaxis, :, :] <= X[:, np.newaxis, :], axis=2)

        for i in tqdm(range(N), desc="  FDH (matrix)", leave=False):
            dom_idx = np.where(dom_matrix[i])[0]
            n_dom = len(dom_idx)

            if n_dom < 2:
                scores[i] = 1.0
                continue

            Y_dom = Y[dom_idx]
            sample_size = min(m, n_dom)
            if n_dom < m:
                fallback += 1

            all_samples = rng.choice(n_dom, size=(B_reps, sample_size), replace=True)
            max_y = Y_dom[all_samples].max(axis=1)
            scores[i] = Y[i] / max(max_y.mean(), 1e-10)

    else:
        for i in tqdm(range(N), desc="  FDH (loop)", leave=False):
            dominated = np.all(X <= X[i], axis=1)
            dom_idx = np.where(dominated)[0]
            n_dom = len(dom_idx)

            if n_dom < 2:
                scores[i] = 1.0
                continue

            Y_dom = Y[dom_idx]
            sample_size = min(m, n_dom)
            if n_dom < m:
                fallback += 1

            all_samples = rng.choice(n_dom, size=(B_reps, sample_size), replace=True)
            max_y = Y_dom[all_samples].max(axis=1)
            scores[i] = Y[i] / max(max_y.mean(), 1e-10)

    return scores, fallback


def main():
    rpt("# DEA Frontier Estimation Report (Order-m FDH)")
    rpt()

    df = pd.read_parquet(INPUT)
    N_total = len(df)
    rpt(f"Input: {N_total:,} obs")
    rpt()

    # === Step 1: Prepare ===
    rpt("## 1. Data Preparation")
    rpt()

    n_zero_out = (df[Y_COL] <= 0).sum()
    rpt(f"- Zero output excluded: {n_zero_out:,} obs ({100*n_zero_out/N_total:.1f}%)")

    df['dea_group'] = df['country'] + '_' + df['main_crop']
    for (c, crop), (tc, tcrop) in MERGE_MAP.items():
        mask = (df['country'] == c) & (df['main_crop'] == crop)
        df.loc[mask, 'dea_group'] = tc + '_' + tcrop
        n = mask.sum()
        if n > 0:
            rpt(f"- Merged: {c}x{crop} ({n}) -> {tc}x{tcrop}")
    rpt()

    # === Step 2: Order-m FDH by layer ===
    rpt("## 2. Order-m FDH (m=25, B=200)")
    rpt()
    rpt("| Layer | Obs | Time(s) | Mean eff | Median | Super% | Fallback |")
    rpt("|-------|----:|--------:|------:|-------:|-------:|---------:|")

    dea_mask = df[Y_COL] > 0
    df['dea_efficiency'] = np.nan
    total_fb = 0
    total_processed = 0
    t_total = time.time()

    layers = df[dea_mask].groupby('dea_group')

    for group_name, group in layers:
        if len(group) < MIN_LAYER:
            rpt(f"| {group_name} | {len(group)} | — | SKIP | | | |")
            continue

        X = group[X_COLS].values.astype(np.float64)
        Y = group[Y_COL].values.astype(np.float64)

        t0 = time.time()
        scores, fb = order_m_fdh(X, Y, m=M, B_reps=B)
        elapsed = time.time() - t0

        df.loc[group.index, 'dea_efficiency'] = scores
        total_fb += fb
        total_processed += len(group)

        mean_eff = np.mean(scores)
        med_eff = np.median(scores)
        super_pct = 100 * np.mean(scores > 1.0)

        rpt(f"| {group_name} | {len(group):,} | {elapsed:.1f} | {mean_eff:.3f} | {med_eff:.3f} | {super_pct:.1f}% | {fb:,} |")

    elapsed_total = time.time() - t_total
    rpt()
    rpt(f"**Total: {total_processed:,} obs, {elapsed_total:.0f}s, fallback: {total_fb:,} ({100*total_fb/max(total_processed,1):.1f}%)**")
    rpt()

    # === Step 3: Derived variables ===
    rpt("## 3. Derived Variables")
    rpt()

    valid = df['dea_efficiency'].notna() & (df['dea_efficiency'] > 0)
    df.loc[valid, 'dea_frontier_value_USD'] = df.loc[valid, Y_COL] / df.loc[valid, 'dea_efficiency']
    df['dea_gap'] = np.where(valid, 1 - df['dea_efficiency'], np.nan)
    df.loc[valid, 'frontier_yield_kg_ha'] = df.loc[valid, 'dea_frontier_value_USD'] / df.loc[valid, 'plot_area_GPS']

    for col in ['dea_efficiency', 'dea_frontier_value_USD', 'dea_gap', 'frontier_yield_kg_ha']:
        nn = df[col].notna().sum()
        rpt(f"- `{col}`: {nn:,} ({100*nn/N_total:.1f}%)")
    rpt()

    # === Step 4: Survival thresholds ===
    rpt("## 4. Survival Thresholds")
    rpt()

    # ---- Survival thresholds ----
    # Primary: Data-driven conditional percentiles within (country, action_crop)
    #   P25 = main result, P10/P50 = sensitivity analysis
    #   Ref: Fafchamps (1992), Dercon (1996), Roy (1952) Safety-First
    # Secondary: Poverty line (World Bank/FAO anchored) for robustness

    plot_share = (df['plot_area_GPS'] / df['farm_size'].clip(lower=0.01)).clip(upper=1.0)

    # --- Primary system: Conditional percentiles (P10, P25, P50) ---
    for (c, crop), grp in df[valid].groupby(['country', 'action_crop']):
        if len(grp) < 20:
            continue
        p10 = grp[Y_COL].quantile(0.10)
        p25 = grp[Y_COL].quantile(0.25)
        p50 = grp[Y_COL].quantile(0.50)
        df.loc[grp.index, 'survival_threshold_P10'] = p10
        df.loc[grp.index, 'survival_threshold_P25'] = p25
        df.loc[grp.index, 'survival_threshold_P50'] = p50

    # --- Secondary: Poverty line (robustness) ---
    poverty_annual = 2.15 * 365  # $784.75/year/person
    ag_share = 0.7
    n_plots = df['nb_plots'].fillna(df.groupby('country')['nb_plots'].transform('median')).clip(lower=1)
    df['survival_threshold_poverty'] = poverty_annual * df['hh_size'] * plot_share * ag_share / n_plots

    for col in ['survival_threshold_P10', 'survival_threshold_P25', 'survival_threshold_P50', 'survival_threshold_poverty']:
        nn = df[col].notna().sum()
        med = df[col].median() if nn > 0 else 0
        below = (df[Y_COL] < df[col]).sum() / max(nn, 1) * 100
        rpt(f"- `{col}`: {nn:,} obs, median=${med:.1f}, trigger={below:.1f}%")
    rpt()

    # === Step 5: Validation ===
    rpt("## 5. Validation")
    rpt()

    eff = df['dea_efficiency'].dropna()
    rpt(f"- Mean efficiency: {eff.mean():.3f}")
    rpt(f"- Median efficiency: {eff.median():.3f}")
    rpt(f"- Super-efficient (>1) share: {100*(eff>1).mean():.1f}%")
    rpt(f"- Near-frontier (>=0.95) share: {100*(eff>=0.95).mean():.1f}%")
    rpt()

    rpt("### Efficiency by Country")
    rpt()
    rpt("| Country | Mean | Median | Super% | N |")
    rpt("|---------|-----:|-------:|-------:|--:|")
    for c in sorted(df['country'].unique()):
        sub = df[(df['country'] == c) & df['dea_efficiency'].notna()]
        if len(sub) == 0:
            continue
        e = sub['dea_efficiency']
        rpt(f"| {c} | {e.mean():.3f} | {e.median():.3f} | {100*(e>1).mean():.1f}% | {len(sub):,} |")
    rpt()

    # === Save ===
    df.to_parquet(OUTPUT, index=False)
    fsize = OUTPUT.stat().st_size / 1024**2
    rpt(f"## Output")
    rpt()
    rpt(f"- `birl_sample.parquet`: {len(df):,} × {len(df.columns)} ({fsize:.1f} MB)")
    rpt(f"- DEA efficiency coverage: {len(eff):,} obs ({100*len(eff)/N_total:.1f}%)")
    rpt(f"- Total computation time: {elapsed_total:.0f}s")

    with open(REPORT, 'w') as f:
        f.write('\n'.join(report))
    logger.info(f"Report: {REPORT}")
    logger.info(f"Done: {len(eff):,} obs, {elapsed_total:.0f}s")


if __name__ == '__main__':
    main()
