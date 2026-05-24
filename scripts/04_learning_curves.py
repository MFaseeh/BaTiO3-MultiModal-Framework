"""
04_learning_curves.py
=====================
Generate learning curves for the stacking framework.

Evaluates R² vs training set size (10%–100%) for:
  - Stacking (All Modalities, Ridge)
  - Experimental Only (XGB)

Sub-sampling is performed inside each training fold.
Test folds always use original, unperturbed samples.

Output
------
  4. Stacking_Fusion/Learning_Curves/
      Learning_Curve_Raw.csv
      Learning_Curve_Aggregated.csv
      Fig_Learning_Curves.pdf / .png
"""

from __future__ import annotations

import os
import warnings

import pandas as pd

from src.config import MODALITY_SUFFIXES, PATHS, TARGETS
from src.learning_curve import plot_learning_curves, run_learning_curves
from src.utils import get_logger

warnings.filterwarnings("ignore")

logger   = get_logger("learning_curves", "04_learning_curves.log")
OUT_DIR  = PATHS["lc_output"]
os.makedirs(OUT_DIR, exist_ok=True)

# ======================================================
# LOAD DATA
# ======================================================
logger.info("Loading dataset …")
df = pd.read_csv(PATHS["final_dataset"])
df.columns = df.columns.str.strip().str.lower()
Y  = df[TARGETS].copy()

modality_features = {
    mod: [c for c in df.columns if c.endswith(suf)]
    for mod, suf in MODALITY_SUFFIXES.items()
}
exp_features = modality_features["Experimental"]

logger.info(f"  Samples  : {len(df)}")
logger.info(f"  Features : {len([c for c in df.columns if c not in TARGETS])}")

# ======================================================
# RUN LEARNING CURVES
# ======================================================
lc_agg = run_learning_curves(
    df=df,
    Y=Y,
    modality_features=modality_features,
    exp_features=exp_features,
    output_dir=OUT_DIR,
    augment=True,
    logger=logger,
)

# ======================================================
# PLOT
# ======================================================
plot_learning_curves(
    lc_agg=lc_agg,
    n_total_samples=len(df),
    output_dir=OUT_DIR,
    logger=logger,
)

logger.info("=" * 65)
logger.info("LEARNING CURVES — COMPLETE")
logger.info(f"    Output : {OUT_DIR}")
logger.info("=" * 65)
