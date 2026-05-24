"""
03_stacking_fusion.py
=====================
Run the full two-level stacking pipeline:
  - All stacking configurations (All / No Exp / No Comp / No NLP)
  - Single-modality baselines (Exp Only / Comp Only / NLP Only)
  - Early fusion baseline
  - Meta-learner comparison (Ridge / RF / LGBM / XGB / MLP)

Output
------
  4. Stacking_Fusion/
      Stacking_All_Raw.csv
      Stacking_Aggregated.csv
      Stacking_Master_Summary.csv
      Figures/
"""

from __future__ import annotations

import os
import warnings

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import (
    COLORS,
    MODALITY_SUFFIXES,
    PATHS,
    TARGET_LABELS,
    TARGETS,
)
from src.evaluation import aggregate_results, evaluate_feature_set, stacking_cv
from src.utils import get_logger

warnings.filterwarnings("ignore")

# ======================================================
# SETUP
# ======================================================
logger  = get_logger("stacking_fusion", "03_stacking_fusion.log")
BASE_DIR = PATHS["stacking_output"]
FIG_DIR  = os.path.join(BASE_DIR, "Figures")
os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)
plt.style.use("seaborn-v0_8-whitegrid")

# ======================================================
# LOAD DATA
# ======================================================
logger.info("Loading dataset …")
df = pd.read_csv(PATHS["final_dataset"])
df.columns = df.columns.str.strip().str.lower()
Y  = df[TARGETS].copy()

modality_features: dict[str, list[str]] = {
    mod: [c for c in df.columns if c.endswith(suf)]
    for mod, suf in MODALITY_SUFFIXES.items()
}
all_features = [c for c in df.columns if c not in TARGETS]
exp_features = modality_features["Experimental"]

logger.info(f"Total samples : {len(df)}")
logger.info(f"Total features: {len(all_features)}")
for mod, feats in modality_features.items():
    logger.info(f"  {mod}: {len(feats)} features")


# ======================================================
# HELPER
# ======================================================
def save_fig(path_no_ext: str) -> None:
    for fmt in ["pdf", "png"]:
        plt.savefig(f"{path_no_ext}.{fmt}", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Figure saved → {path_no_ext}")


# ======================================================
# STACKING CONFIGURATIONS
# ======================================================
stacking_configs = {
    "Stacking (All Modalities)":  modality_features,
    "Stacking (No Experimental)": {
        k: v for k, v in modality_features.items() if k != "Experimental"
    },
    "Stacking (No Computational)": {
        k: v for k, v in modality_features.items() if k != "Computational"
    },
    "Stacking (No NLP)": {
        k: v for k, v in modality_features.items() if k != "NLP"
    },
}

raw_stacking = []
for name, mods in stacking_configs.items():
    logger.info(f"Running: {name}")
    res = stacking_cv(mods, df, Y, name, augment=True, logger=logger)
    raw_stacking.append(res)

# ======================================================
# SINGLE-MODALITY BASELINES
# ======================================================
logger.info("=" * 65)
logger.info("Single-modality and early fusion baselines …")
logger.info("=" * 65)

baseline_configs = {
    "Experimental Only":  exp_features,
    "Computational Only": modality_features["Computational"],
    "NLP Only":           modality_features["NLP"],
    "Early Fusion":       all_features,
}

raw_baselines = []
for name, feats in baseline_configs.items():
    if not feats:
        logger.warning(f"  {name} — empty feature set, skipping.")
        continue
    logger.info(f"  {name} — {len(feats)} features")
    res = evaluate_feature_set(df[feats], Y, name, augment=True, logger=logger)
    # Rename 'model' column to 'meta_learner' for consistent schema
    res = res.rename(columns={"model": "meta_learner"})
    raw_baselines.append(res)

# ======================================================
# COMBINE AND SAVE
# ======================================================
all_raw = pd.concat(raw_stacking + raw_baselines, ignore_index=True)
all_raw.to_csv(os.path.join(BASE_DIR, "Stacking_All_Raw.csv"), index=False)

agg = (
    all_raw
    .groupby(["experiment", "meta_learner", "target"])[["RMSE", "MAE", "R2"]]
    .agg(["mean", "std"])
    .round(4)
)
agg.to_csv(os.path.join(BASE_DIR, "Stacking_Aggregated.csv"))

# ======================================================
# FIGURE 1 — Stacking vs Baselines (R²)
# ======================================================
def plot_stacking_vs_baselines(raw: pd.DataFrame, save_path: str) -> None:
    configurations = [
        ("Computational Only",         "MLP",   COLORS["MLP"],   "Comp Only (MLP)"),
        ("NLP Only",                   "MLP",   COLORS["MLP"],   "NLP Only (MLP)"),
        ("Early Fusion",               "XGB",   COLORS["XGB"],   "Early Fusion (XGB)"),
        ("Experimental Only",          "XGB",   COLORS["XGB"],   "Exp Only (XGB)"),
        ("Stacking (No Experimental)", "Ridge", COLORS["Ridge"], "Stk No Exp (Ridge)"),
        ("Stacking (No Computational)","Ridge", COLORS["Ridge"], "Stk No Comp (Ridge)"),
        ("Stacking (No NLP)",          "Ridge", COLORS["Ridge"], "Stk No NLP (Ridge)"),
        ("Stacking (All Modalities)",  "Ridge", "#1B2A4A",       "Stk All (Ridge)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        "Stacking vs Baselines — Mean $R^2$ ± Std\n"
        "(25 independent train-test splits)",
        fontsize=12, fontweight="bold",
    )
    for ax, target in zip(axes, TARGETS):
        labels, means, stds, colors = [], [], [], []
        for exp, meta, col, lbl in configurations:
            sub = raw[
                (raw["experiment"]   == exp) &
                (raw["meta_learner"] == meta) &
                (raw["target"]       == target)
            ]
            if sub.empty:
                continue
            labels.append(lbl)
            means.append(sub["R2"].mean())
            stds.append(sub["R2"].std())
            colors.append(col)

        x    = np.arange(len(labels))
        bars = ax.bar(x, means, yerr=stds, color=colors, capsize=4,
                      edgecolor="white", linewidth=0.6,
                      error_kw={"elinewidth": 1.2, "ecolor": "#333"})
        for bar, m, s in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width()/2,
                    max(0, m) + s + 0.005,
                    f"{m:.3f}", ha="center", va="bottom", fontsize=7)

        ax.set_title(TARGET_LABELS[target], fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7, rotation=30, ha="right")
        ax.set_ylabel("Mean $R^2$", fontsize=9)
        ax.set_ylim(bottom=-0.15)
        ax.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    save_fig(save_path)


plot_stacking_vs_baselines(
    all_raw,
    os.path.join(FIG_DIR, "Fig1_Stacking_vs_Baselines_R2"),
)

# ======================================================
# FIGURE 2 — Meta-Learner Comparison
# ======================================================
def plot_meta_learner_comparison(raw: pd.DataFrame, save_path: str) -> None:
    stacking_raw = raw[raw["experiment"] == "Stacking (All Modalities)"]
    meta_learners = ["Ridge", "MLP", "RF", "LGBM", "XGB"]
    x = np.arange(len(meta_learners))
    width = 0.25

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "Meta-Learner Comparison — Stacking (All Modalities)\n"
        "Mean $R^2$ ± Std over 25 splits",
        fontsize=12, fontweight="bold",
    )
    for ax, target in zip(axes, TARGETS):
        means = [
            stacking_raw[
                (stacking_raw["meta_learner"] == m) &
                (stacking_raw["target"] == target)
            ]["R2"].mean()
            for m in meta_learners
        ]
        stds = [
            stacking_raw[
                (stacking_raw["meta_learner"] == m) &
                (stacking_raw["target"] == target)
            ]["R2"].std()
            for m in meta_learners
        ]
        colors = [COLORS.get(m, "#555555") for m in meta_learners]
        bars = ax.bar(x, means, yerr=stds, color=colors, capsize=4,
                      edgecolor="white", linewidth=0.6,
                      error_kw={"elinewidth": 1.2, "ecolor": "#333"})
        for bar, m, s in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width()/2,
                    m + s + 0.004, f"{m:.3f}",
                    ha="center", va="bottom", fontsize=8)
        ax.set_title(TARGET_LABELS[target], fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(meta_learners, fontsize=9)
        ax.set_ylabel("Mean $R^2$", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    save_fig(save_path)


plot_meta_learner_comparison(
    all_raw,
    os.path.join(FIG_DIR, "Fig2_Meta_Learner_Comparison"),
)

# ======================================================
# FIGURE 3 — Stacking Modality Ablation
# ======================================================
def plot_stacking_ablation(raw: pd.DataFrame, save_path: str) -> None:
    stk_exps = [
        "Stacking (All Modalities)",
        "Stacking (No Experimental)",
        "Stacking (No Computational)",
        "Stacking (No NLP)",
    ]
    labels = ["All", "No Exp", "No Comp", "No NLP"]
    full_mean_by_target: dict[str, float] = {}

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(
        "Stacking Modality Ablation — Ridge Meta-Learner\n"
        "Mean $R^2$ ± Std (bar annotations show Δ vs All Modalities)",
        fontsize=12, fontweight="bold",
    )
    for ax, target in zip(axes, TARGETS):
        means, stds, colors_list = [], [], []
        for exp in stk_exps:
            sub = raw[
                (raw["experiment"]   == exp) &
                (raw["meta_learner"] == "Ridge") &
                (raw["target"]       == target)
            ]
            m = sub["R2"].mean() if not sub.empty else 0
            s = sub["R2"].std()  if not sub.empty else 0
            means.append(m)
            stds.append(s)
            colors_list.append(COLORS["Ridge"] if exp == "Stacking (All Modalities)" else "#888888")

        full_mean = means[0]
        bars = ax.bar(
            range(len(labels)), means, yerr=stds,
            color=colors_list, capsize=4,
            edgecolor="white", linewidth=0.6,
            error_kw={"elinewidth": 1.2, "ecolor": "#333"},
        )
        for bar, m in zip(bars, means):
            drop = full_mean - m
            ax.text(
                bar.get_x() + bar.get_width()/2,
                max(0, m)/2,
                f"-{drop:.3f}" if drop > 0 else f"+{abs(drop):.3f}",
                ha="center", va="center", fontsize=7,
                color="white", fontweight="bold",
            )
        ax.set_title(TARGET_LABELS[target], fontsize=10)
        ax.set_ylabel("$R^2$", fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=8, rotation=15, ha="right")
        ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    save_fig(save_path)


plot_stacking_ablation(all_raw, os.path.join(FIG_DIR, "Fig3_Stacking_Modality_Ablation"))

# ======================================================
# FIGURE 4 — R² Heatmap
# ======================================================
def plot_r2_heatmap(raw: pd.DataFrame, save_path: str) -> None:
    rows = []
    for exp in raw["experiment"].unique():
        sub = raw[raw["experiment"] == exp]
        best_meta = sub.groupby("meta_learner")["R2"].mean().idxmax()
        row = {"Configuration": f"{exp}\n({best_meta})"}
        for target in TARGETS:
            m = sub[
                (sub["meta_learner"] == best_meta) &
                (sub["target"] == target)
            ]["R2"].mean()
            row[TARGET_LABELS[target]] = round(m, 3)
        rows.append(row)

    hdf = (
        pd.DataFrame(rows)
        .set_index("Configuration")
        .sort_values(TARGET_LABELS["d33"], ascending=False)
    )
    fig, ax = plt.subplots(figsize=(8, max(5, len(hdf) * 0.6)))
    sns.heatmap(hdf, ax=ax, cmap="YlGnBu", annot=True, fmt=".3f",
                vmin=-0.15, vmax=1.0, linewidths=0.5, annot_kws={"size": 9})
    ax.set_title("Mean $R^2$ Summary — Best Model per Configuration",
                 fontsize=11, fontweight="bold", pad=10)
    ax.tick_params(axis="y", labelsize=8, rotation=0)
    ax.tick_params(axis="x", labelsize=9)
    plt.tight_layout()
    save_fig(save_path)


plot_r2_heatmap(all_raw, os.path.join(FIG_DIR, "Fig4_R2_Heatmap_All_Configs"))

# ======================================================
# FIGURE 5 — Boxplot
# ======================================================
def plot_boxplot_comparison(raw: pd.DataFrame, save_path: str) -> None:
    compare = [
        ("Experimental Only",        "XGB",   COLORS["XGB"],   "Exp Only (XGB)"),
        ("Early Fusion",             "XGB",   COLORS["MLP"],   "Early Fusion (XGB)"),
        ("Stacking (All Modalities)","Ridge", "#1B2A4A",       "Stacking (Ridge)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "R² Distribution — Stacking vs Baselines (25 fold scores)",
        fontsize=12, fontweight="bold",
    )
    for ax, target in zip(axes, TARGETS):
        data_plot, labels, colors = [], [], []
        for exp, meta, col, lbl in compare:
            sub = raw[
                (raw["experiment"]   == exp) &
                (raw["meta_learner"] == meta) &
                (raw["target"]       == target)
            ]["R2"].values
            if len(sub):
                data_plot.append(sub)
                labels.append(lbl)
                colors.append(col)
        bp = ax.boxplot(
            data_plot, patch_artist=True,
            medianprops={"color": "white", "linewidth": 2},
            whiskerprops={"linewidth": 1.2},
            capprops={"linewidth": 1.2},
            flierprops={"marker": "o", "markersize": 3, "alpha": 0.5},
        )
        for patch, col in zip(bp["boxes"], colors):
            patch.set_facecolor(col)
            patch.set_alpha(0.75)
        ax.set_xticks(range(1, len(labels)+1))
        ax.set_xticklabels(labels, fontsize=8, rotation=10, ha="right")
        ax.set_title(TARGET_LABELS[target], fontsize=10)
        ax.set_ylabel("$R^2$", fontsize=9)
        ax.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    save_fig(save_path)


plot_boxplot_comparison(all_raw, os.path.join(FIG_DIR, "Fig5_Boxplot_Stacking_vs_Baselines"))

# ======================================================
# MASTER SUMMARY TABLE
# ======================================================
summary_rows = []
for exp in all_raw["experiment"].unique():
    sub = all_raw[all_raw["experiment"] == exp]
    best_meta = sub.groupby("meta_learner")["R2"].mean().idxmax()
    row = {"Configuration": exp, "Best_Model": best_meta}
    for target in TARGETS:
        t_sub = sub[(sub["meta_learner"] == best_meta) & (sub["target"] == target)]
        for metric in ["R2", "RMSE", "MAE"]:
            short_t = target.replace("dielectric constant", "Diel").replace("density", "Dens")
            row[f"{short_t}_{metric}"] = f"{t_sub[metric].mean():.4f}±{t_sub[metric].std():.4f}"
    summary_rows.append(row)

pd.DataFrame(summary_rows).to_csv(
    os.path.join(BASE_DIR, "Stacking_Master_Summary.csv"), index=False
)

logger.info("=" * 65)
logger.info("STACKING FUSION PIPELINE — COMPLETE")
logger.info(f"    Output : {BASE_DIR}")
logger.info("=" * 65)
