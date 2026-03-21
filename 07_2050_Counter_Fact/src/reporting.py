"""
Console reporting utilities for welfare analysis results.

Extracted from visualization.py to decouple text reporting from matplotlib plotting.
"""

import pandas as pd


def print_summary_table(
    results_df: pd.DataFrame,
    climate_loss_df: pd.DataFrame,
    policy_value_df: pd.DataFrame,
    synergy_df: pd.DataFrame,
):
    """Print comprehensive results table to console."""
    print("\n" + "=" * 80)
    print("2050 COUNTERFACTUAL ANALYSIS — SUMMARY RESULTS")
    print("=" * 80)

    print("\n--- CE by Scenario ---")
    for scenario in results_df["scenario"].unique():
        sub = results_df[results_df["scenario"] == scenario]
        print(f"\n  {scenario}:")
        for _, row in sub.iterrows():
            print(f"    {row['country']:<12} CE = ${row['ce_median']:.2f} "
                  f"[${row['ce_q025']:.2f}, ${row['ce_q975']:.2f}]")

    print("\n--- Climate Loss (CE_current - CE_2050_SSP585) ---")
    if not climate_loss_df.empty:
        for _, row in climate_loss_df.iterrows():
            print(f"  {row['country']:<12} Loss = ${row['loss_ssp585_median']:.2f} "
                  f"({row['loss_ssp585_pct_median']:.1f}%)")

    print("\n--- Policy Value (CE_policy - CE_2050_SSP585) ---")
    if not policy_value_df.empty:
        for _, row in policy_value_df.iterrows():
            print(f"  {row['country']:<12} Ins=${row['insurance_median']:.2f} "
                  f" SN=${row['safety_net_median']:.2f} "
                  f" Comb=${row['combined_median']:.2f} "
                  f" Ratio={row['policy_ratio']:.2f}")

    print("\n--- Synergy ---")
    if not synergy_df.empty:
        for _, row in synergy_df.iterrows():
            print(f"  {row['country']:<12} Synergy = ${row['synergy_median']:.2f} "
                  f"({row['synergy_pct']:.1f}%)")

    print("\n" + "=" * 80)
