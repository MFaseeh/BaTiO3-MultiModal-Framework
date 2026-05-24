"""
learning_curve.py
=================
Learning curve analysis for the BaTiO3 stacking framework.

Evaluates R² vs training set size for:
  - Stacking (All Modalities, Ridge meta-learner)
  - Experimental Only (XGB baseline)

Sub-sampling is performed INSIDE each training fold — consistent with
the anti-leakage protocol. Test folds always use full original data.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.augmentation import augment_train_fold
from src.config import (
    AUGMENTATION,
    COLORS,
    CV,
    LEARNING_CURVE,
    MODALITY_SUFFIXES,
    PATHS,
    RANDOM_STATE,
    TARGET_LABELS,
    TARGETS,
    XGB_PARAMS,
)
from src.models import get_level1_models
from src.utils import get_logger, save_fig_and_close

from xgboost import XGBRegressor

RS        = RANDOM_STATE
N_SPLITS  = LEARNING_CURVE["n_splits"]
N_REPEATS = LEARNING_CURVE["n_repeats"]
FRACTIONS = LEARNING_CURVE["train_fractions"]
MIN_FLOOR = LEARNING_CURVE["min_train_floor"]


def _get_xgb_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model",  XGBRegressor(**XGB_PARAMS)),
    ])


def _get_ridge_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model",  Ridge(alpha=1.0)),
    ])


def run_learning_curves(
    df: pd.DataFrame,
    Y: pd.DataFrame,
    modality_features: dict[str, list[str]],
    exp_features: list[str],
    output_dir: str,
    augment: bool = True,
    logger=None,
) -> pd.DataFrame:
    """
    Compute learning curves for Stacking (Ridge) and Experimental Only (XGB).

    For each training fraction:
      - Sub-sample train_idx to the target size
      - Optionally augment sub-sampled training fold
      - Evaluate stacking and XGB on original test fold

    Returns aggregated DataFrame with columns:
        [fraction, target, stk_mean, stk_std, exp_mean, exp_std]
    """
    if logger is None:
        logger = get_logger(__name__)

    os.makedirs(output_dir, exist_ok=True)
    records = []
    level1_names = list(get_level1_models().keys())

    for repeat in range(N_REPEATS):
        seed     = RS + repeat * 100
        outer_kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)

        for fold_idx, (train_idx, test_idx) in enumerate(
            outer_kf.split(np.arange(len(df))), start=1
        ):
            for frac in FRACTIONS:

                # Sub-sample training indices
                n_use = max(
                    int(len(train_idx) * frac),
                    MIN_FLOOR,
                )
                rng     = np.random.default_rng(seed + fold_idx)
                sub_idx = rng.choice(train_idx, size=n_use, replace=False)

                for target in TARGETS:
                    y_np   = Y[target].values
                    y_test = y_np[test_idx]   # always original

                    # ── Optionally augment sub-sampled train ──────────────
                    if augment:
                        X_sub_orig = df[exp_features].fillna(0).iloc[sub_idx].reset_index(drop=True)
                        y_sub_orig = pd.DataFrame(y_np[sub_idx], columns=[target])
                        # augment all features together for stacking
                        X_full_orig = df[[c for c in df.columns if c not in TARGETS]].fillna(0).iloc[sub_idx].reset_index(drop=True)
                        X_full_aug, y_aug_df = augment_train_fold(
                            X_full_orig, y_sub_orig,
                            seed=seed + fold_idx + int(frac * 100),
                        )
                        y_aug = y_aug_df[target].values
                    else:
                        X_full_aug = df[[c for c in df.columns if c not in TARGETS]].fillna(0).iloc[sub_idx].reset_index(drop=True)
                        y_aug      = y_np[sub_idx]

                    # ── Stacking (Ridge, All Modalities) ─────────────────
                    train_meta_cols = []
                    test_meta_cols  = []

                    for mod, feats in modality_features.items():
                        if not feats:
                            continue
                        # Extract modality columns from augmented frame
                        mod_cols_present = [f for f in feats if f in X_full_aug.columns]
                        if not mod_cols_present:
                            continue

                        X_mod_tr = X_full_aug[mod_cols_present]
                        X_mod_te = df[mod_cols_present].fillna(0).iloc[test_idx]

                        n_aug  = len(X_mod_tr)
                        oof    = np.zeros((n_aug, len(level1_names)))
                        n_inner = min(4, n_aug // 2) if n_aug < 8 else 4
                        if n_inner < 2:
                            # Too few samples for inner CV — use full train for OOF
                            for m_idx, m_name in enumerate(level1_names):
                                pipe = get_level1_models()[m_name]
                                pipe.fit(X_mod_tr, y_aug)
                                oof[:, m_idx] = pipe.predict(X_mod_tr)
                        else:
                            ikf = KFold(n_splits=n_inner, shuffle=True, random_state=seed + fold_idx)
                            for m_idx, m_name in enumerate(level1_names):
                                col = np.zeros(n_aug)
                                for in_tr, in_val in ikf.split(X_mod_tr):
                                    pipe = get_level1_models()[m_name]
                                    pipe.fit(
                                        X_mod_tr.iloc[in_tr], y_aug[in_tr]
                                    )
                                    col[in_val] = pipe.predict(X_mod_tr.iloc[in_val])
                                oof[:, m_idx] = col

                        # Retrain on full augmented train → test preds
                        test_preds = np.zeros((len(test_idx), len(level1_names)))
                        for m_idx, m_name in enumerate(level1_names):
                            pipe = get_level1_models()[m_name]
                            pipe.fit(X_mod_tr, y_aug)
                            test_preds[:, m_idx] = pipe.predict(X_mod_te)

                        train_meta_cols.append(oof)
                        test_meta_cols.append(test_preds)

                    stacked_train = np.hstack(train_meta_cols)
                    stacked_test  = np.hstack(test_meta_cols)

                    meta = _get_ridge_pipeline()
                    meta.fit(stacked_train, y_aug)
                    r2_stk = r2_score(y_test, meta.predict(stacked_test))

                    # ── Experimental Only (XGB) ───────────────────────────
                    exp_cols_present = [f for f in exp_features if f in X_full_aug.columns]
                    X_exp_tr = X_full_aug[exp_cols_present]
                    X_exp_te = df[exp_cols_present].fillna(0).iloc[test_idx]

                    xgb = _get_xgb_pipeline()
                    xgb.fit(X_exp_tr, y_aug)
                    r2_xgb = r2_score(y_test, xgb.predict(X_exp_te))

                    records.append({
                        "repeat":      repeat + 1,
                        "fold":        fold_idx,
                        "fraction":    frac,
                        "n_train":     n_use,
                        "target":      target,
                        "R2_Stacking": r2_stk,
                        "R2_ExpOnly":  r2_xgb,
                    })

        logger.info(f"[Learning Curve] Repeat {repeat+1}/{N_REPEATS} complete.")

    lc_df = pd.DataFrame(records)
    lc_df.to_csv(os.path.join(output_dir, "Learning_Curve_Raw.csv"), index=False)

    lc_agg = (
        lc_df
        .groupby(["fraction", "target"])[["R2_Stacking", "R2_ExpOnly"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    lc_agg.columns = [
        "fraction", "target",
        "stk_mean", "stk_std",
        "exp_mean", "exp_std",
    ]
    lc_agg.to_csv(
        os.path.join(output_dir, "Learning_Curve_Aggregated.csv"), index=False
    )
    logger.info("[Learning Curve] Results saved.")
    return lc_agg


def plot_learning_curves(
    lc_agg: pd.DataFrame,
    n_total_samples: int,
    output_dir: str,
    logger=None,
) -> None:
    """Generate the three-panel learning curve figure."""
    if logger is None:
        logger = get_logger(__name__)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    fig.suptitle(
        "Learning Curves: Stacking (Ridge) vs. Experimental Only (XGB)\n"
        f"Mean $R^2$ $\\pm$ Std over "
        f"{LEARNING_CURVE['n_repeats']} repeats "
        f"$\\times$ {LEARNING_CURVE['n_splits']} folds",
        fontsize=13, fontweight="bold", y=1.02,
    )

    stk_color = COLORS.get("LGBM", "#2E9E7B")
    exp_color = COLORS.get("XGB",  "#E07B39")

    for ax, target in zip(axes, TARGETS):
        sub = lc_agg[lc_agg["target"] == target].sort_values("fraction")

        # x-axis: approximate number of training samples
        train_frac = (CV["n_outer_splits"] - 1) / CV["n_outer_splits"]
        x = sub["fraction"] * n_total_samples * train_frac

        ax.plot(
            x, sub["stk_mean"], "o-",
            color=stk_color, linewidth=2, markersize=5,
            label="Stacking (Ridge)",
        )
        ax.fill_between(
            x,
            sub["stk_mean"] - sub["stk_std"],
            sub["stk_mean"] + sub["stk_std"],
            alpha=0.15, color=stk_color,
        )

        ax.plot(
            x, sub["exp_mean"], "s--",
            color=exp_color, linewidth=2, markersize=5,
            label="Experimental Only (XGB)",
        )
        ax.fill_between(
            x,
            sub["exp_mean"] - sub["exp_std"],
            sub["exp_mean"] + sub["exp_std"],
            alpha=0.15, color=exp_color,
        )

        ax.set_title(TARGET_LABELS[target], fontsize=11)
        ax.set_xlabel("Training Set Size", fontsize=9)
        ax.set_ylabel("$R^2$", fontsize=9)
        ax.legend(fontsize=8, frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)

    plt.tight_layout()
    save_fig_and_close(
        os.path.join(output_dir, "Fig_Learning_Curves"), logger
    )
