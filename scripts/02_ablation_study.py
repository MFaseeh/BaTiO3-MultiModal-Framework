"""
02_ablation_study.py
====================
Run the four ablation experiments (A–D) plus single-modality baselines.

Experiments
-----------
  A — Feature set ablation: vote threshold sweep (V>=1 to V>=4)
  B — Modality ablation: drop one modality at a time
  C — Selection method ablation: single method vs consensus
  D — Vote threshold sensitivity

All experiments use the anti-leakage CV engine with in-fold augmentation.
Results are saved as CSV and visualised as grouped bar charts.

Output
------
  3. Ablation_Study/
      A_Feature_Set_Ablation/
      B_Modality_Ablation/
      C_Method_Ablation/
      D_Threshold_Sensitivity/
      Figures/
      All_Experiments_Raw.csv
      Master_Ablation_Summary.csv
"""

from __future__ import annotations

import os
import warnings

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Optional

from src.config import (
    COLORS,
    PATHS,
    TARGET_LABELS,
    TARGETS,
    VOTE_GROUPS,
)
from src.evaluation import aggregate_results, evaluate_feature_set
from src.utils import get_logger

warnings.filterwarnings("ignore")

# ======================================================
# SETUP
# ======================================================
logger  = get_logger("ablation_study", "02_ablation_study.log")
BASE_DIR = PATHS["ablation_output"]
os.makedirs(BASE_DIR, exist_ok=True)
plt.style.use("seaborn-v0_8-whitegrid")

PALETTE  = COLORS
EXP_SUF  = "_exp"
COMP_SUF = "_comp"
NLPT_SUF = "_nlpt"

# ======================================================
# LOAD DATA
# ======================================================
logger.info("Loading datasets …")
final_df = pd.read_csv(PATHS["final_dataset"])
full_df  = pd.read_csv(PATHS["full_dataset"])
votes_df = pd.read_csv(PATHS["votes_csv"], index_col=0)

for df in [final_df, full_df]:
    df.columns = df.columns.str.strip().str.lower()

Y_final = final_df[TARGETS].copy()
Y_full  = full_df[TARGETS].copy()
feats_all   = [c for c in full_df.columns if c not in TARGETS]
feats_final = [c for c in final_df.columns if c not in TARGETS]

# ======================================================
# FIGURE HELPER
# ======================================================
def _model_color(name: str) -> str:
    return PALETTE.get(name, "#555555")


def save_fig(path_no_ext: str) -> None:
    for fmt in ["pdf", "png"]:
        plt.savefig(f"{path_no_ext}.{fmt}", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Figure saved → {path_no_ext}")


def plot_experiment_grouped(
    raw: pd.DataFrame,
    experiments: list[str],
    metric: str,
    title: str,
    save_path: str,
    highlight_exp: Optional[str] = None,
) -> None:
    models   = list(PALETTE.keys())
    n_exp    = len(experiments)
    n_models = len(models)
    x        = np.arange(n_exp)
    width    = 0.18
    offsets  = np.linspace(-(n_models-1)/2, (n_models-1)/2, n_models) * width

    fig, axes = plt.subplots(1, len(TARGETS), figsize=(16, 5), sharey=False)
    fig.suptitle(title, fontsize=12, fontweight="bold")

    for ax, target in zip(axes, TARGETS):
        for model, offset in zip(models, offsets):
            means, stds = [], []
            for exp in experiments:
                sub = raw[
                    (raw["experiment"] == exp) &
                    (raw["model"]      == model) &
                    (raw["target"]     == target)
                ]
                means.append(sub[metric].mean() if not sub.empty else 0)
                stds.append( sub[metric].std()  if not sub.empty else 0)
            ax.bar(
                x + offset, means, width=width,
                yerr=stds, label=model,
                color=_model_color(model), capsize=3,
                edgecolor="white", linewidth=0.5,
                error_kw={"elinewidth": 1.0, "ecolor": "#444444"},
            )
        if highlight_exp and highlight_exp in experiments:
            idx = experiments.index(highlight_exp)
            ax.axvspan(idx - 0.45, idx + 0.45, alpha=0.08, color="green", zorder=0)
        ax.set_title(TARGET_LABELS[target], fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [e.replace(" (", "\n(") for e in experiments],
            fontsize=7, rotation=15, ha="right",
        )
        ax.set_ylabel(metric, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)

    handles = [mpatches.Patch(color=_model_color(m), label=m) for m in models]
    fig.legend(
        handles=handles, loc="lower center", ncol=n_models,
        fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.06),
    )
    plt.tight_layout()
    save_fig(save_path)


# ======================================================
# EXPERIMENT A — Feature Set Ablation
# ======================================================
logger.info("=" * 65)
logger.info("EXPERIMENT A — Feature Set Ablation")
logger.info("=" * 65)

exp_a_dir = os.path.join(BASE_DIR, "A_Feature_Set_Ablation")
os.makedirs(exp_a_dir, exist_ok=True)
raw_a = []

for label, v_min in VOTE_GROUPS.items():
    selected = votes_df[votes_df["total_votes"] >= v_min].index.tolist()
    selected = [f for f in selected if f in full_df.columns]
    logger.info(f"  [A] {label} — {len(selected)} features")
    raw_a.append(evaluate_feature_set(full_df[selected], Y_full, label, logger=logger))

# Baseline: all 24 features
raw_a.append(
    evaluate_feature_set(full_df[feats_all], Y_full, "No Selection (24 feat)", logger=logger)
)
raw_a_df = pd.concat(raw_a, ignore_index=True)
raw_a_df.to_csv(os.path.join(exp_a_dir, "ExpA_Raw.csv"), index=False)
aggregate_results(raw_a_df).to_csv(os.path.join(exp_a_dir, "ExpA_Aggregated.csv"))

# ======================================================
# EXPERIMENT B — Modality Ablation
# ======================================================
logger.info("=" * 65)
logger.info("EXPERIMENT B — Modality Ablation")
logger.info("=" * 65)

exp_b_dir = os.path.join(BASE_DIR, "B_Modality_Ablation")
os.makedirs(exp_b_dir, exist_ok=True)

modality_experiments = {
    "All Modalities (F*)":    feats_final,
    "No Experimental":        [f for f in feats_final if not f.endswith(EXP_SUF)],
    "No Computational":       [f for f in feats_final if not f.endswith(COMP_SUF)],
    "No NLP":                 [f for f in feats_final if not f.endswith(NLPT_SUF)],
    "Experimental Only":      [f for f in feats_final if f.endswith(EXP_SUF)],
    "Computational Only":     [f for f in feats_final if f.endswith(COMP_SUF)],
    "NLP Only":               [f for f in feats_final if f.endswith(NLPT_SUF)],
}
raw_b = []
for label, feats in modality_experiments.items():
    if not feats:
        logger.warning(f"  [B] {label} — empty, skipping.")
        continue
    logger.info(f"  [B] {label} — {len(feats)} features")
    raw_b.append(evaluate_feature_set(final_df[feats], Y_final, label, logger=logger))

raw_b_df = pd.concat(raw_b, ignore_index=True)
raw_b_df.to_csv(os.path.join(exp_b_dir, "ExpB_Raw.csv"), index=False)
aggregate_results(raw_b_df).to_csv(os.path.join(exp_b_dir, "ExpB_Aggregated.csv"))

# ======================================================
# EXPERIMENT C — Selection Method Ablation
# ======================================================
logger.info("=" * 65)
logger.info("EXPERIMENT C — Selection Method Ablation")
logger.info("=" * 65)

exp_c_dir = os.path.join(BASE_DIR, "C_Method_Ablation")
os.makedirs(exp_c_dir, exist_ok=True)

method_experiments = {
    f"{m} Only": votes_df[votes_df[m] == 1].index.tolist()
    for m in ["MI", "LGBM", "SHAP", "XGB"]
}
method_experiments["Consensus F*"] = feats_final

raw_c = []
for label, feats in method_experiments.items():
    feats = [f for f in feats if f in full_df.columns]
    if not feats:
        continue
    logger.info(f"  [C] {label} — {len(feats)} features")
    raw_c.append(evaluate_feature_set(full_df[feats], Y_full, label, logger=logger))

raw_c_df = pd.concat(raw_c, ignore_index=True)
raw_c_df.to_csv(os.path.join(exp_c_dir, "ExpC_Raw.csv"), index=False)
aggregate_results(raw_c_df).to_csv(os.path.join(exp_c_dir, "ExpC_Aggregated.csv"))

# ======================================================
# EXPERIMENT D — Vote Threshold Sensitivity
# ======================================================
logger.info("=" * 65)
logger.info("EXPERIMENT D — Vote Threshold Sensitivity")
logger.info("=" * 65)

exp_d_dir = os.path.join(BASE_DIR, "D_Threshold_Sensitivity")
os.makedirs(exp_d_dir, exist_ok=True)
raw_d = []

for v_min in [1, 2, 3, 4]:
    selected = votes_df[votes_df["total_votes"] >= v_min].index.tolist()
    selected = [f for f in selected if f in full_df.columns]
    label    = f"V>={v_min} ({len(selected)} features)"
    logger.info(f"  [D] {label}")
    raw_d.append(evaluate_feature_set(full_df[selected], Y_full, label, logger=logger))

raw_d_df = pd.concat(raw_d, ignore_index=True)
raw_d_df.to_csv(os.path.join(exp_d_dir, "ExpD_Raw.csv"), index=False)
aggregate_results(raw_d_df).to_csv(os.path.join(exp_d_dir, "ExpD_Aggregated.csv"))

# ======================================================
# COMBINED RAW + VISUALISATIONS
# ======================================================
all_raw = pd.concat([raw_a_df, raw_b_df, raw_c_df, raw_d_df], ignore_index=True)
all_raw.to_csv(os.path.join(BASE_DIR, "All_Experiments_Raw.csv"), index=False)

VIZ_DIR = os.path.join(BASE_DIR, "Figures")
os.makedirs(VIZ_DIR, exist_ok=True)

exp_a_order = ["No Selection (24 feat)", "V>=1 (16 feat)", "V>=2 (15 feat)", "V>=3 (10 feat)", "V>=4 (6 feat)"]
exp_b_order = [e for e in modality_experiments if e in raw_b_df["experiment"].unique()]
exp_c_order = [e for e in ["MI Only","LGBM Only","SHAP Only","XGB Only","Consensus F*"] if e in raw_c_df["experiment"].unique()]
exp_d_order = sorted(raw_d_df["experiment"].unique(), key=lambda s: int(s.split(">=")[1][0]))

for metric in ["R2", "RMSE", "MAE"]:
    plot_experiment_grouped(raw_a_df, exp_a_order, metric,
        f"Experiment A — Feature Set Ablation: {metric}",
        os.path.join(VIZ_DIR, f"ExpA_FeatureSet_{metric}"), "V>=2 (15 feat)")
    plot_experiment_grouped(raw_b_df, exp_b_order, metric,
        f"Experiment B — Modality Ablation: {metric}",
        os.path.join(VIZ_DIR, f"ExpB_Modality_{metric}"), "All Modalities (F*)")
    plot_experiment_grouped(raw_c_df, exp_c_order, metric,
        f"Experiment C — Selection Method Ablation: {metric}",
        os.path.join(VIZ_DIR, f"ExpC_Method_{metric}"), "Consensus F*")

for metric in ["R2", "RMSE"]:
    plot_experiment_grouped(raw_d_df, exp_d_order, metric,
        f"Experiment D — Vote Threshold Sensitivity: {metric}",
        os.path.join(VIZ_DIR, f"ExpD_Threshold_{metric}"))

logger.info("=" * 65)
logger.info("ABLATION STUDY — COMPLETE")
logger.info(f"    Output : {BASE_DIR}")
logger.info("=" * 65)
