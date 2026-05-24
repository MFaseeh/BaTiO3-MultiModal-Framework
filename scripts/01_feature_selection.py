"""
01_feature_selection.py
=======================
Run the full four-stage consensus feature selection pipeline.

Stages
------
  Stage 1: All four methods independently on each source modality
  Stage 2: Build hybrid-fused dataset with weighted targets
  Stage 3: All four methods on hybrid-fused dataset
  Stage 4: Consensus voting → Final_Selected_Dataset.csv

Output
------
  2. Feature Selection/Unified_FS/
      Final_Selected_Dataset.csv   ← input for all downstream scripts
      Master_FS_Summary.csv
      env_metadata.csv
      Stage1_Independent/          ← per-source results
      Stage3_Hybrid/               ← hybrid-fused results + consensus
"""

from __future__ import annotations

import os
import warnings

import lightgbm
import numpy as np
import pandas as pd
import shap
import sklearn
import xgboost

from src.config import MODALITY_WEIGHTS, PATHS, TARGETS
from src.feature_selection import (
    consensus_voting,
    lgbm_fs,
    mi_fs,
    shap_fs,
    xgb_fs,
)
from src.utils import (
    add_suffix_safe,
    get_logger,
    load_and_clean,
)

warnings.filterwarnings("ignore")

# ======================================================
# SETUP
# ======================================================
logger  = get_logger("feature_selection", "01_feature_selection.log")
BASE_DIR = PATHS["fs_output"]
os.makedirs(BASE_DIR, exist_ok=True)

# ======================================================
# ENVIRONMENT METADATA
# ======================================================
pd.DataFrame([{
    "lightgbm":  lightgbm.__version__,
    "xgboost":   xgboost.__version__,
    "sklearn":   sklearn.__version__,
    "pandas":    pd.__version__,
    "numpy":     np.__version__,
    "shap":      shap.__version__,
}]).to_csv(os.path.join(BASE_DIR, "env_metadata.csv"), index=False)

# ======================================================
# LOAD DATA
# ======================================================
logger.info("Loading input datasets …")
exp_df  = load_and_clean(PATHS["experimental"],  "Experimental", logger)
comp_df = load_and_clean(PATHS["computational"], "Computational", logger)
nlpt_df = load_and_clean(PATHS["nlp"],           "NLP",          logger)

# Align to shortest modality
min_len = min(len(exp_df), len(comp_df), len(nlpt_df))
exp_df  = exp_df.iloc[:min_len].reset_index(drop=True)
comp_df = comp_df.iloc[:min_len].reset_index(drop=True)
nlpt_df = nlpt_df.iloc[:min_len].reset_index(drop=True)
logger.info(f"All datasets aligned to {min_len} rows.")

source_map = {
    "Experimental":  exp_df,
    "Computational": comp_df,
    "NLPT":          nlpt_df,
}

# ======================================================
# STAGE 1 — Independent per-source selection
# ======================================================
logger.info("=" * 65)
logger.info("STAGE 1 — Independent FS on each source (all 4 methods)")
logger.info("=" * 65)

stage1_results: dict[str, dict[str, list[str]]] = {}

for src_name, src_df in source_map.items():
    src_dir = os.path.join(BASE_DIR, "Stage1_Independent", src_name)
    logger.info(f"  Running all methods on {src_name} …")

    _, mi_sel   = mi_fs  (src_df, src_name, os.path.join(src_dir, "MI"),   logger=logger)
    _, lgbm_sel = lgbm_fs(src_df, src_name, os.path.join(src_dir, "LGBM"), logger=logger)
    _, shap_sel = shap_fs(src_df, src_name, os.path.join(src_dir, "SHAP"), logger=logger)
    _, xgb_sel  = xgb_fs (src_df, src_name, os.path.join(src_dir, "XGB"),  logger=logger)

    all_feats = [c for c in src_df.columns if c not in TARGETS]
    stage1_results[src_name] = {
        "MI": mi_sel, "LGBM": lgbm_sel, "SHAP": shap_sel, "XGB": xgb_sel,
    }
    consensus_voting(
        stage1_results[src_name],
        all_feats,
        src_name,
        os.path.join(src_dir, "Consensus"),
        logger=logger,
    )

# ======================================================
# STAGE 2 — Build hybrid-fused dataset
# ======================================================
logger.info("=" * 65)
logger.info("STAGE 2 — Building hybrid-fused dataset")
logger.info("=" * 65)

w = MODALITY_WEIGHTS
weighted_y = pd.DataFrame({
    t: w["exp"] * exp_df[t] + w["comp"] * comp_df[t] + w["nlpt"] * nlpt_df[t]
    for t in TARGETS
})

hybrid_df = pd.concat([
    add_suffix_safe(exp_df,  "_exp" ).drop(columns=TARGETS, errors="ignore"),
    add_suffix_safe(comp_df, "_comp").drop(columns=TARGETS, errors="ignore"),
    add_suffix_safe(nlpt_df, "_nlpt").drop(columns=TARGETS, errors="ignore"),
    weighted_y,
], axis=1).reset_index(drop=True)

logger.info(
    f"Hybrid-fused dataset — shape: {hybrid_df.shape} "
    f"({hybrid_df.shape[1] - len(TARGETS)} features | {len(TARGETS)} targets)"
)

# ======================================================
# STAGE 3 — All methods on hybrid-fused dataset
# ======================================================
logger.info("=" * 65)
logger.info("STAGE 3 — All 4 methods on Hybrid-Fused dataset")
logger.info("=" * 65)

hybrid_dir   = os.path.join(BASE_DIR, "Stage3_Hybrid")
hybrid_feats = [c for c in hybrid_df.columns if c not in TARGETS]

_, h_mi_sel   = mi_fs  (hybrid_df, "Hybrid_Fused", os.path.join(hybrid_dir, "MI"),   logger=logger)
_, h_lgbm_sel = lgbm_fs(hybrid_df, "Hybrid_Fused", os.path.join(hybrid_dir, "LGBM"), logger=logger)
_, h_shap_sel = shap_fs(hybrid_df, "Hybrid_Fused", os.path.join(hybrid_dir, "SHAP"), logger=logger)
_, h_xgb_sel  = xgb_fs (hybrid_df, "Hybrid_Fused", os.path.join(hybrid_dir, "XGB"),  logger=logger)

hybrid_selections = {
    "MI":   h_mi_sel,
    "LGBM": h_lgbm_sel,
    "SHAP": h_shap_sel,
    "XGB":  h_xgb_sel,
}

# ======================================================
# STAGE 4 — Final consensus vote
# ======================================================
logger.info("=" * 65)
logger.info("STAGE 4 — Consensus voting on Hybrid-Fused selections")
logger.info("=" * 65)

final_selected = consensus_voting(
    hybrid_selections,
    hybrid_feats,
    "Hybrid_Fused",
    os.path.join(hybrid_dir, "Consensus"),
    logger=logger,
)

# Save final dataset — uses ORIGINAL (non-weighted) targets
final_df = hybrid_df[final_selected + TARGETS].copy()
# Restore original target values from exp_df
for t in TARGETS:
    final_df[t] = exp_df[t].values
final_df.to_csv(PATHS["final_dataset"], index=False)
logger.info(
    f"Final dataset saved — {len(final_selected)} features | "
    f"{len(final_df)} samples → {PATHS['final_dataset']}"
)

# ======================================================
# MASTER SUMMARY
# ======================================================
summary_rows = []
for src_name, method_sels in stage1_results.items():
    src_df  = source_map[src_name]
    n_feats = len([c for c in src_df.columns if c not in TARGETS])
    for method, sel in method_sels.items():
        summary_rows.append({
            "Stage":             "Stage1_Independent",
            "Dataset":           src_name,
            "Method":            method,
            "Total_Features":    n_feats,
            "Selected_Features": len(sel),
            "Retention_%":       f"{100*len(sel)/max(1,n_feats):.2f}%",
        })
for method, sel in hybrid_selections.items():
    summary_rows.append({
        "Stage":             "Stage3_Hybrid",
        "Dataset":           "Hybrid_Fused",
        "Method":            method,
        "Total_Features":    len(hybrid_feats),
        "Selected_Features": len(sel),
        "Retention_%":       f"{100*len(sel)/max(1,len(hybrid_feats)):.2f}%",
    })
summary_rows.append({
    "Stage":             "Stage4_Consensus",
    "Dataset":           "Hybrid_Fused",
    "Method":            "Consensus (>=2/4 votes)",
    "Total_Features":    len(hybrid_feats),
    "Selected_Features": len(final_selected),
    "Retention_%":       f"{100*len(final_selected)/max(1,len(hybrid_feats)):.2f}%",
})
pd.DataFrame(summary_rows).to_csv(
    os.path.join(BASE_DIR, "Master_FS_Summary.csv"), index=False
)

logger.info("=" * 65)
logger.info("FEATURE SELECTION PIPELINE — COMPLETE")
logger.info(f"    Final features : {len(final_selected)} / {len(hybrid_feats)}")
logger.info(f"    Output dir     : {BASE_DIR}")
logger.info("=" * 65)
