"""
Robustness summary: Spearman rank correlations between main and variant posteriors.
"""

import logging

import numpy as np
import pandas as pd

log = logging.getLogger("birl")


def compute_robustness_summary(main_np, robustness_results, hh_to_idx, r9_hh_list, out_dir):
    """Compute Spearman r between main and each variant. Returns DataFrame."""
    try:
        from scipy.stats import spearmanr
    except ImportError:
        log.warning("  scipy not available, skipping Spearman correlations")
        return None

    main_alpha = main_np["alpha_i"].mean(0)
    main_rho = main_np.get("rho_i", np.zeros(1))
    main_rho = main_rho.mean(0) if main_rho.ndim > 1 else None
    main_gamma = main_np.get("gamma_i", np.zeros(1))
    main_gamma = main_gamma.mean(0) if main_gamma.ndim > 1 else None

    summary_rows = []
    for vname, np_v in robustness_results.items():
        row = {"variant": vname, "alpha_r": np.nan, "rho_r": np.nan, "gamma_r": np.nan}
        if np_v is None:
            summary_rows.append(row)
            continue

        # For R9, match HHs by ID
        if vname == "R9":
            r9_in_main = np.array([hh_to_idx.get(h, -1) for h in r9_hh_list])
            valid = r9_in_main >= 0
            m_alpha = main_alpha[r9_in_main[valid]]
            m_rho = main_rho[r9_in_main[valid]] if main_rho is not None else None
            m_gamma = main_gamma[r9_in_main[valid]] if main_gamma is not None else None
            v_slice = valid
        else:
            m_alpha, m_rho, m_gamma = main_alpha, main_rho, main_gamma
            v_slice = slice(None)

        if "alpha_i" in np_v:
            v_alpha = np_v["alpha_i"].mean(0)[v_slice]
            if v_alpha.std() > 0 and m_alpha.std() > 0:
                row["alpha_r"] = float(spearmanr(m_alpha, v_alpha)[0])

        if "rho_i" in np_v and m_rho is not None:
            v_rho = np_v["rho_i"].mean(0)[v_slice]
            if v_rho.std() > 0 and m_rho.std() > 0:
                row["rho_r"] = float(spearmanr(m_rho, v_rho)[0])

        if "gamma_i" in np_v and m_gamma is not None:
            v_gamma = np_v["gamma_i"].mean(0)[v_slice]
            if v_gamma.std() > 0 and m_gamma.std() > 0:
                row["gamma_r"] = float(spearmanr(m_gamma, v_gamma)[0])

        summary_rows.append(row)

    sum_df = pd.DataFrame(summary_rows)
    sum_df.to_csv(out_dir / "robustness_summary.csv", index=False)

    log.info(f"\n  Robustness Summary:")
    log.info(f"  {'Variant':20s} {'alpha_r':>8s} {'rho_r':>8s} {'gamma_r':>8s}")
    log.info(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8}")
    for _, r in sum_df.iterrows():
        def fmt(v):
            return f"{v:.3f}" if pd.notna(v) and not np.isnan(v) else "  N/A "
        log.info(f"  {r['variant']:20s} {fmt(r['alpha_r']):>8s} "
                 f"{fmt(r['rho_r']):>8s} {fmt(r['gamma_r']):>8s}")

    n_robust = sum_df["alpha_r"].dropna().gt(0.85).sum()
    total = sum_df["alpha_r"].notna().sum()
    log.info(f"\n  alpha robust (r > 0.85): {n_robust}/{total} variants")
    return sum_df
