"""
utils.py
========
Shared utilities: logging setup, plotting helpers, normalisation,
and short-name registry. Imported by all other modules.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import FIGURE, TARGETS, TARGET_LABELS


# ======================================================
# LOGGING
# ======================================================

def get_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """Return a configured logger. Adds a FileHandler if log_file given."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s — %(levelname)s — %(message)s")

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_file:
        fh = logging.FileHandler(log_file, mode="w")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# ======================================================
# NORMALISATION
# ======================================================

def safe_norm(mat: pd.DataFrame) -> pd.DataFrame:
    """
    Min-max normalise each column of mat to [0, 1].
    Columns with max == 0 are left as zeros (avoid division by zero).
    This is the normalisation applied before aggregating heterogeneous
    importance scores (split-count, SHAP, permutation) — making them
    dimensionally commensurable before arithmetic averaging.
    """
    col_min = mat.min()
    col_max = mat.max()
    denom   = col_max - col_min
    denom[denom == 0] = 1          # prevent division by zero
    return (mat - col_min) / denom


# ======================================================
# SHORT-NAME REGISTRY
# ======================================================

def build_shortname_registry(names: list[str]) -> dict[str, str]:
    """
    Build {full_name: short_name} with guaranteed uniqueness.
    Used for axis labels and legend entries in all figures.
    """
    registry: dict[str, str] = {}
    seen:     dict[str, int] = {}
    for name in names:
        words = name.split()
        base  = (
            "".join(w[0].upper() for w in words)
            if len(words) > 1
            else name[:4].upper()
        )
        if base in seen:
            seen[base] += 1
            short = f"{base}{seen[base]}"
        else:
            seen[base] = 0
            short = base
        registry[name] = short
    return registry


# ======================================================
# FIGURE HELPERS
# ======================================================

def save_figure(path_no_ext: str) -> None:
    """Save current figure in all configured formats."""
    for fmt in FIGURE["formats"]:
        plt.savefig(
            f"{path_no_ext}.{fmt}",
            dpi=FIGURE["dpi"],
            bbox_inches="tight",
        )


def save_fig_and_close(path_no_ext: str, logger: Optional[logging.Logger] = None) -> None:
    save_figure(path_no_ext)
    plt.close()
    if logger:
        logger.info(f"Figure saved → {path_no_ext}")


def add_legend(
    ax: plt.Axes,
    short_to_full: dict[str, str],
    title: str = "Legend",
) -> None:
    """Add a legend mapping short IDs to full feature names."""
    patches = [
        mpatches.Patch(color="none", label=f"{s}  →  {f}")
        for s, f in short_to_full.items()
    ]
    ax.legend(
        handles=patches,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=3,
        frameon=False,
        title=title,
        fontsize=7,
        title_fontsize=8,
    )


def plot_heatmap(
    data: pd.DataFrame,
    short_reg: dict[str, str],
    title: str,
    save_path: str,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Feature × target (or method) heatmap with short-name labels."""
    plot_df         = data.copy()
    plot_df.index   = [short_reg.get(f, f) for f in data.index]
    plot_df.columns = [short_reg.get(c, c) for c in data.columns]

    fig, ax = plt.subplots(
        figsize=(
            max(8, len(data.columns) * 2),
            max(6, len(data.index)   * 0.4),
        )
    )
    sns.heatmap(
        plot_df,
        cmap="YlGnBu",
        annot=True,
        fmt=".2f",
        annot_kws={"size": 7},
        linewidths=0.4,
        ax=ax,
    )
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=7)
    add_legend(
        ax,
        {short_reg[f]: f for f in data.index if f in short_reg},
        "Feature Abbreviations",
    )
    plt.tight_layout()
    save_fig_and_close(save_path, logger)


def plot_bar(
    scores: pd.Series,
    short_reg: dict[str, str],
    title: str,
    xlabel: str,
    save_path: str,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Horizontal bar chart of feature importance scores."""
    sorted_s    = scores.sort_values(ascending=False)
    short_names = [short_reg.get(f, f) for f in sorted_s.index]

    fig, ax = plt.subplots(figsize=(9, max(5, len(short_names) * 0.35)))
    sns.barplot(x=sorted_s.values, y=short_names, palette="viridis", ax=ax)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel("Feature Abbreviation", fontsize=10)
    ax.tick_params(axis="y", labelsize=7)
    add_legend(
        ax,
        {short_reg[f]: f for f in sorted_s.index if f in short_reg},
        "Feature Abbreviations",
    )
    plt.tight_layout()
    save_fig_and_close(save_path, logger)


def plot_shap_beeswarm(
    shap_vals: "np.ndarray",
    X_display: pd.DataFrame,
    short_reg: dict[str, str],
    title: str,
    save_path: str,
    logger: Optional[logging.Logger] = None,
) -> None:
    """SHAP beeswarm summary plot with short-name column labels."""
    import shap as shap_lib
    disp = X_display.rename(
        columns={c: short_reg.get(c, c) for c in X_display.columns}
    )
    plt.figure(figsize=(9, max(5, len(disp.columns) * 0.35)))
    shap_lib.summary_plot(shap_vals, disp, show=False, plot_size=None, color_bar=True)
    plt.title(title, fontsize=11)
    plt.tight_layout()
    save_fig_and_close(save_path, logger)


# ======================================================
# DATA LOADING HELPERS
# ======================================================

def validate_dataframe(
    df: pd.DataFrame,
    name: str,
    required_cols: list[str],
    logger: Optional[logging.Logger] = None,
) -> None:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"[{name}] Missing columns: {missing}")
    if df.empty:
        raise ValueError(f"[{name}] DataFrame is empty.")
    if logger:
        logger.info(f"[{name}] Validated — shape: {df.shape}")


def load_and_clean(
    path: str,
    name: str,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """Load CSV, strip/lowercase column names, validate targets present."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    validate_dataframe(df, name, TARGETS, logger)
    return df


def add_suffix_safe(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    """Add suffix to all non-target columns."""
    return df.rename(
        columns={c: f"{c}{suffix}" for c in df.columns if c not in TARGETS}
    )


def save_method_report(
    output_path: str,
    dataset_name: str,
    method: str,
    features: list[str],
    selected: list[str],
    extra: Optional[dict] = None,
) -> None:
    """Save a one-row CSV report for a single FS method result."""
    row: dict = {
        "Dataset":           dataset_name,
        "Method":            method,
        "Total_Features":    len(features),
        "Selected_Features": len(selected),
        "Dropped_Features":  len(features) - len(selected),
        "Retention_%":       f"{100 * len(selected) / max(1, len(features)):.2f}%",
    }
    if extra:
        row.update(extra)
    pd.DataFrame([row]).to_csv(
        os.path.join(output_path, f"{dataset_name}_{method}_Report.csv"),
        index=False,
    )
