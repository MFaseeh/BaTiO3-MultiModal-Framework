"""
05_significance_tests.py
========================
Paired t-tests to assess statistical significance of stacking improvements.

Comparisons
-----------
  1. Stacking (All Modalities, Ridge) vs Experimental Only (XGB)
  2. Stacking (All Modalities, Ridge) vs Early Fusion (XGB)

A paired test is appropriate because both configurations are evaluated
on the same 25 fold splits (5 repeats × 5-fold CV), making the R²
scores paired by construction.

Output
------
  4. Stacking_Fusion/Significance_Tests.csv
  Printed to console
"""

from __future__ import annotations

import os

import pandas as pd
from scipy import stats

from src.config import PATHS, TARGETS
from src.utils import get_logger

logger = get_logger("significance_tests", "05_significance_tests.log")

# ======================================================
# LOAD RAW RESULTS
# ======================================================
raw = pd.read_csv(PATHS["stacking_raw"])

# ======================================================
# PAIRED T-TESTS
# ======================================================
comparisons = [
    (
        "Stacking (All Modalities)", "Ridge",
        "Experimental Only",         "XGB",
        "Stacking (Ridge) vs Exp Only (XGB)",
    ),
    (
        "Stacking (All Modalities)", "Ridge",
        "Early Fusion",              "XGB",
        "Stacking (Ridge) vs Early Fusion (XGB)",
    ),
]

results = []

for exp_a, meta_a, exp_b, meta_b, label in comparisons:
    print(f"\n=== {label} ===")
    for target in TARGETS:
        a = raw[
            (raw["experiment"]   == exp_a) &
            (raw["meta_learner"] == meta_a) &
            (raw["target"]       == target)
        ]["R2"].values

        b = raw[
            (raw["experiment"]   == exp_b) &
            (raw["meta_learner"] == meta_b) &
            (raw["target"]       == target)
        ]["R2"].values

        if len(a) == 0 or len(b) == 0:
            print(f"  {target}: insufficient data.")
            continue

        # Align lengths (both should be 25 from 5×5 CV)
        n = min(len(a), len(b))
        t_stat, p_val = stats.ttest_rel(a[:n], b[:n])
        mean_diff     = a[:n].mean() - b[:n].mean()
        significant   = "✓" if p_val < 0.05 else "✗"

        print(
            f"  {target:<25} "
            f"ΔR²={mean_diff:+.4f}  "
            f"t={t_stat:.3f}  p={p_val:.4f}  "
            f"{'significant' if p_val < 0.05 else 'not significant'} {significant}"
        )

        results.append({
            "Comparison":   label,
            "Target":       target,
            "Mean_R2_A":    a[:n].mean(),
            "Mean_R2_B":    b[:n].mean(),
            "Delta_R2":     mean_diff,
            "t_statistic":  t_stat,
            "p_value":      p_val,
            "Significant":  p_val < 0.05,
            "n_pairs":      n,
        })

# ======================================================
# SAVE
# ======================================================
results_df = pd.DataFrame(results)
out_path   = os.path.join(PATHS["stacking_output"], "Significance_Tests.csv")
results_df.to_csv(out_path, index=False)
logger.info(f"Significance test results saved → {out_path}")
