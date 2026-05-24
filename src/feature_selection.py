"""
feature_selection.py
====================
Four complementary feature selection methods + consensus voting.

Methods
-------
  mi_fs    — Bootstrap-stabilised Mutual Information
  lgbm_fs  — Cross-validated LightGBM (split + SHAP + permutation)
  shap_fs  — Bootstrap SHAP with cumulative + stability criteria
  xgb_fs   — Cross-validated XGBoost (f-score + SHAP + permutation)

All three tree-based aggregations use min-max normalisation (safe_norm)
BEFORE averaging, making the heterogeneous scores dimensionally
commensurable. See paper §3.3 for the statistical justification.

Consensus Voting
----------------
  consensus_voting — Democratic vote across all four methods.
  A feature is retained if ≥ V_min methods select it (default V_min=2).
"""

from __future__ import annotations

import os
import warnings
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from sklearn.feature_selection import mutual_info_regression
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src.config import (
    LGBM_PARAMS,
    RANDOM_STATE,
    SELECTION,
    TARGETS,
    XGB_PARAMS,
)
from src.utils import (
    add_legend,
    build_shortname_registry,
    get_logger,
    plot_bar,
    plot_heatmap,
    plot_shap_beeswarm,
    safe_norm,
    save_fig_and_close,
    save_method_report,
)

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

RS          = RANDOM_STATE
N_SPLITS    = SELECTION["n_splits"]
N_BOOTSTRAP = SELECTION["n_bootstrap"]


# ======================================================
# METHOD 1 — MUTUAL INFORMATION
# ======================================================

def mi_fs(
    df: pd.DataFrame,
    dataset_name: str,
    output_path: str,
    rel_threshold: float = SELECTION["rel_threshold"],
    logger=None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Bootstrap-stabilised Mutual Information Feature Selection.

    Runs MI N_BOOTSTRAP times with distinct seeds and averages scores.
    A feature is retained if its score >= rel_threshold * max_score
    for ANY target (union rule).
    """
    if logger is None:
        logger = get_logger(__name__)
    os.makedirs(output_path, exist_ok=True)

    features  = [c for c in df.columns if c not in TARGETS]
    short_reg = build_shortname_registry(features + TARGETS)
    X         = df[features].fillna(0)

    mi_bootstrap: dict[str, list[np.ndarray]] = {t: [] for t in TARGETS}

    for b in range(N_BOOTSTRAP):
        for target in TARGETS:
            y = df[target].fillna(0)
            if y.std() == 0:
                mi_bootstrap[target].append(np.zeros(len(features)))
                continue
            scores = mutual_info_regression(
                X, y, random_state=RS + b, n_neighbors=5
            )
            mi_bootstrap[target].append(scores)

    mi_mean = pd.DataFrame(index=features)
    for target in TARGETS:
        mi_mean[target] = np.mean(mi_bootstrap[target], axis=0)

    mi_norm = safe_norm(mi_mean)
    mi_mean["mean_importance"] = mi_norm.mean(axis=1)

    # Visualisations
    plot_heatmap(
        mi_norm, short_reg,
        f"{dataset_name} — MI Scores (Bootstrap avg, {N_BOOTSTRAP} runs)",
        os.path.join(output_path, f"{dataset_name}_MI_Heatmap"),
        logger,
    )
    plot_bar(
        mi_mean["mean_importance"], short_reg,
        f"{dataset_name}: Mean MI Score",
        "Mean Normalised MI Score",
        os.path.join(output_path, f"{dataset_name}_MI_BarPlot"),
        logger,
    )

    # Selection: union across targets
    selected: set[str] = set()
    for target in TARGETS:
        col_max = mi_mean[target].max()
        if col_max == 0:
            continue
        selected.update(
            mi_mean[mi_mean[target] >= rel_threshold * col_max].index.tolist()
        )
    selected = list(selected) if selected else features.copy()

    reduced_df = df[selected + TARGETS].copy()
    mi_mean.to_csv(os.path.join(output_path, f"{dataset_name}_MI_Scores.csv"))
    reduced_df.to_csv(
        os.path.join(output_path, f"{dataset_name}_MI_Selected.csv"), index=False
    )
    save_method_report(output_path, dataset_name, "MI", features, selected, {
        "Criterion":      f"MI >= {rel_threshold*100:.1f}% of max (union)",
        "Bootstrap_Runs": N_BOOTSTRAP,
    })
    logger.info(
        f"[MI | {dataset_name}] {len(selected)}/{len(features)} features retained."
    )
    return reduced_df, selected


# ======================================================
# METHOD 2 — LIGHTGBM
# ======================================================

def lgbm_fs(
    df: pd.DataFrame,
    dataset_name: str,
    output_path: str,
    rel_threshold: float = SELECTION["rel_threshold"],
    logger=None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Cross-validated LightGBM Feature Selection.

    Aggregated score = mean of min-max normalised
    (split-count + SHAP + permutation) over K folds.

    Min-max normalisation before averaging makes the three metrics
    dimensionally commensurable despite their different mathematical
    definitions (paper §3.3).
    """
    if logger is None:
        logger = get_logger(__name__)
    os.makedirs(output_path, exist_ok=True)

    features  = [c for c in df.columns if c not in TARGETS]
    short_reg = build_shortname_registry(features + TARGETS)

    X_raw    = df[features].fillna(0)
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    lgbm_imp = pd.DataFrame(0.0, index=features, columns=TARGETS)
    shap_imp = pd.DataFrame(0.0, index=features, columns=TARGETS)
    perm_imp = pd.DataFrame(0.0, index=features, columns=TARGETS)

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RS)

    for target in TARGETS:
        y = df[target].fillna(0).values
        if y.std() == 0:
            logger.warning(
                f"[LGBM | {dataset_name}] Target '{target}' zero variance — skipped."
            )
            continue

        fold_lgbm, fold_shap, fold_perm = [], [], []
        for train_idx, val_idx in kf.split(X_scaled):
            model = lgb.LGBMRegressor(**LGBM_PARAMS)
            model.fit(
                X_scaled[train_idx], y[train_idx],
                eval_set=[(X_scaled[val_idx], y[val_idx])],
                eval_metric="l2",
                callbacks=[
                    lgb.early_stopping(
                        SELECTION["early_stopping_rounds"], verbose=False
                    )
                ],
            )
            fold_lgbm.append(model.feature_importances_)
            exp = shap.TreeExplainer(model)
            fold_shap.append(
                np.abs(exp.shap_values(X_scaled[val_idx])).mean(axis=0)
            )
            pi = permutation_importance(
                model, X_scaled[val_idx], y[val_idx],
                n_repeats=5, random_state=RS,
            )
            fold_perm.append(np.clip(pi.importances_mean, 0, None))

        lgbm_imp[target] = np.mean(fold_lgbm, axis=0)
        shap_imp[target] = np.mean(fold_shap, axis=0)
        perm_imp[target] = np.mean(fold_perm, axis=0)

    # Normalise then aggregate
    lgbm_norm = safe_norm(lgbm_imp)
    shap_norm = safe_norm(shap_imp)
    perm_norm = safe_norm(perm_imp)
    agg_norm  = (lgbm_norm + shap_norm + perm_norm) / 3.0
    agg_norm["mean_importance"] = agg_norm[TARGETS].mean(axis=1)

    # Visualisations
    X_df = pd.DataFrame(X_scaled, columns=features)
    for mat, label in [
        (lgbm_norm, "LGBM_Split"),
        (shap_norm, "SHAP"),
        (perm_norm, "Perm"),
        (agg_norm[TARGETS], "Aggregated"),
    ]:
        plot_heatmap(
            mat, short_reg,
            f"{dataset_name} — {label} Importance (KFold avg)",
            os.path.join(output_path, f"{dataset_name}_LGBM_{label}_Heatmap"),
            logger,
        )
    plot_bar(
        agg_norm["mean_importance"], short_reg,
        f"{dataset_name}: Aggregated LGBM Importance",
        "Mean Normalised Importance",
        os.path.join(output_path, f"{dataset_name}_LGBM_Agg_BarPlot"),
        logger,
    )
    # SHAP beeswarm per target
    for target in TARGETS:
        y_full = df[target].fillna(0).values
        if y_full.std() == 0:
            continue
        X_tr, X_vl, y_tr, y_vl = train_test_split(
            X_scaled, y_full, test_size=0.2, random_state=RS
        )
        m = lgb.LGBMRegressor(**LGBM_PARAMS)
        m.fit(
            X_tr, y_tr,
            eval_set=[(X_vl, y_vl)],
            eval_metric="l2",
            callbacks=[
                lgb.early_stopping(SELECTION["early_stopping_rounds"], verbose=False)
            ],
        )
        sv = shap.TreeExplainer(m).shap_values(X_df)
        safe_t = target.replace(" ", "_")
        plot_shap_beeswarm(
            sv, X_df, short_reg,
            f"{dataset_name} — SHAP Beeswarm for '{target}'",
            os.path.join(output_path, f"{dataset_name}_SHAP_{safe_t}"),
            logger,
        )

    # Selection
    max_imp  = agg_norm["mean_importance"].max()
    selected = (
        agg_norm[
            agg_norm["mean_importance"] >= rel_threshold * max_imp
        ].index.tolist()
        if max_imp > 0
        else features.copy()
    )
    selected = selected or features.copy()

    reduced_df = df[selected + TARGETS].copy()
    for mat, name in [
        (lgbm_imp, "LGBM"), (shap_imp, "SHAP"),
        (perm_imp, "Perm"), (agg_norm, "Agg"),
    ]:
        mat.to_csv(
            os.path.join(output_path, f"{dataset_name}_{name}_Importances.csv")
        )
    reduced_df.to_csv(
        os.path.join(output_path, f"{dataset_name}_LGBM_Selected.csv"), index=False
    )
    save_method_report(output_path, dataset_name, "LGBM", features, selected, {
        "Criterion":        f"Agg >= {rel_threshold*100:.1f}% of max",
        "CV_Folds":         N_SPLITS,
        "Importance_Types": "LightGBM + SHAP + Permutation (min-max normalised)",
    })
    logger.info(
        f"[LGBM | {dataset_name}] {len(selected)}/{len(features)} features retained."
    )
    return reduced_df, selected


# ======================================================
# METHOD 3 — SHAP BOOTSTRAP
# ======================================================

def shap_fs(
    df: pd.DataFrame,
    dataset_name: str,
    output_path: str,
    cumulative_threshold: float = SELECTION["cumulative_threshold"],
    stability_threshold:  float = SELECTION["stability_threshold"],
    logger=None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Bootstrap SHAP Feature Selection.

    Retention criteria (both must be satisfied):
      1. Cumulative: feature lies within top cumulative_threshold of total SHAP mass
      2. Stability:  feature ranks above run median in >= stability_threshold
                     fraction of bootstrap runs
    """
    if logger is None:
        logger = get_logger(__name__)
    os.makedirs(output_path, exist_ok=True)

    features  = [c for c in df.columns if c not in TARGETS]
    short_reg = build_shortname_registry(features + TARGETS)

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(df[features].fillna(0))
    run_means: list[pd.Series] = []

    for b in range(N_BOOTSTRAP):
        run_imp = pd.DataFrame(index=features)
        for target in TARGETS:
            y = df[target].fillna(0).values
            if y.std() == 0:
                run_imp[target] = 0.0
                continue
            X_tr, X_vl, y_tr, y_vl = train_test_split(
                X_scaled, y,
                test_size=SELECTION.get("test_size", 0.2),
                random_state=RS + b,
            )
            model = lgb.LGBMRegressor(**{**LGBM_PARAMS, "random_state": RS + b})
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_vl, y_vl)],
                eval_metric="l2",
                callbacks=[
                    lgb.early_stopping(SELECTION["early_stopping_rounds"], verbose=False)
                ],
            )
            exp = shap.TreeExplainer(model)
            sv  = exp.shap_values(X_vl)
            run_imp[target] = np.abs(sv).mean(axis=0)

        run_imp_norm = safe_norm(run_imp)
        run_means.append(run_imp_norm.mean(axis=1).rename(f"run_{b+1}"))

    shap_matrix = pd.concat(run_means, axis=1)
    shap_mean   = shap_matrix.mean(axis=1)

    med_per_run    = shap_matrix.median(axis=0)
    stability_frac = (
        shap_matrix.gt(med_per_run, axis=1).sum(axis=1) / N_BOOTSTRAP
    )

    combined = pd.DataFrame({
        "mean_importance":    shap_mean,
        "stability_fraction": stability_frac,
    }).sort_values("mean_importance", ascending=False)
    combined["cumulative"] = (
        combined["mean_importance"].cumsum()
        / combined["mean_importance"].sum()
    )

    selected = combined[
        (combined["cumulative"]         <= cumulative_threshold) &
        (combined["stability_fraction"] >= stability_threshold)
    ].index.tolist()
    selected = selected or features.copy()

    # Visualisations
    plot_heatmap(
        shap_matrix.rename(
            columns={c: f"Run {i+1}" for i, c in enumerate(shap_matrix.columns)}
        ),
        short_reg,
        f"{dataset_name} — SHAP Bootstrap Importance ({N_BOOTSTRAP} runs)",
        os.path.join(output_path, f"{dataset_name}_SHAP_Bootstrap_Heatmap"),
        logger,
    )
    plot_bar(
        combined["mean_importance"], short_reg,
        f"{dataset_name}: Mean SHAP Importance",
        "Mean Normalised SHAP Value",
        os.path.join(output_path, f"{dataset_name}_SHAP_BarPlot"),
        logger,
    )
    # Stability scatter
    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(
        combined["mean_importance"],
        combined["stability_fraction"],
        c=combined["cumulative"],
        cmap="viridis",
        edgecolors="k",
        linewidths=0.4,
        s=50,
    )
    ax.axhline(
        stability_threshold, ls="--", color="red", lw=1,
        label=f"Stability >= {stability_threshold}",
    )
    boundary_feats = combined[combined["cumulative"] <= cumulative_threshold]
    if not boundary_feats.empty:
        ax.axvline(
            boundary_feats["mean_importance"].min(),
            ls="--", color="blue", lw=1,
            label=f"Cumulative <= {cumulative_threshold}",
        )
    plt.colorbar(sc, ax=ax, label="Cumulative Importance")
    ax.set_xlabel("Mean SHAP Importance")
    ax.set_ylabel("Stability Fraction")
    ax.set_title(f"{dataset_name}: SHAP Stability vs Importance")
    ax.legend(fontsize=8)
    plt.tight_layout()
    save_fig_and_close(
        os.path.join(output_path, f"{dataset_name}_SHAP_Stability_Scatter"), logger
    )

    reduced_df = df[selected + TARGETS].copy()
    combined.to_csv(
        os.path.join(output_path, f"{dataset_name}_SHAP_Combined.csv")
    )
    reduced_df.to_csv(
        os.path.join(output_path, f"{dataset_name}_SHAP_Selected.csv"), index=False
    )
    save_method_report(output_path, dataset_name, "SHAP", features, selected, {
        "Cumulative_Threshold": f"{cumulative_threshold*100:.0f}%",
        "Stability_Threshold":  f"{stability_threshold*100:.0f}%",
        "Bootstrap_Runs":       N_BOOTSTRAP,
    })
    logger.info(
        f"[SHAP | {dataset_name}] {len(selected)}/{len(features)} features retained."
    )
    return reduced_df, selected


# ======================================================
# METHOD 4 — XGBOOST
# ======================================================

def xgb_fs(
    df: pd.DataFrame,
    dataset_name: str,
    output_path: str,
    rel_threshold: float = SELECTION["rel_threshold"],
    logger=None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Cross-validated XGBoost Feature Selection.

    Aggregated score = mean of min-max normalised
    (f-score + SHAP + permutation) over K folds.
    """
    if logger is None:
        logger = get_logger(__name__)
    os.makedirs(output_path, exist_ok=True)

    features  = [c for c in df.columns if c not in TARGETS]
    short_reg = build_shortname_registry(features + TARGETS)

    X_raw    = df[features].fillna(0)
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    X_df     = pd.DataFrame(X_scaled, columns=features)

    xgb_imp  = pd.DataFrame(0.0, index=features, columns=TARGETS)
    shap_imp = pd.DataFrame(0.0, index=features, columns=TARGETS)
    perm_imp = pd.DataFrame(0.0, index=features, columns=TARGETS)

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RS)

    for target in TARGETS:
        y = df[target].fillna(0).values
        if y.std() == 0:
            logger.warning(
                f"[XGB | {dataset_name}] Target '{target}' zero variance — skipped."
            )
            continue

        fold_xgb, fold_shap, fold_perm = [], [], []
        for train_idx, val_idx in kf.split(X_scaled):
            model = XGBRegressor(**XGB_PARAMS)
            model.fit(
                X_scaled[train_idx], y[train_idx],
                eval_set=[(X_scaled[val_idx], y[val_idx])],
                verbose=False,
            )
            fold_xgb.append(model.feature_importances_)
            exp = shap.TreeExplainer(model)
            fold_shap.append(
                np.abs(exp.shap_values(X_scaled[val_idx])).mean(axis=0)
            )
            pi = permutation_importance(
                model, X_scaled[val_idx], y[val_idx],
                n_repeats=5, random_state=RS,
            )
            fold_perm.append(np.clip(pi.importances_mean, 0, None))

        xgb_imp[target]  = np.mean(fold_xgb,  axis=0)
        shap_imp[target] = np.mean(fold_shap, axis=0)
        perm_imp[target] = np.mean(fold_perm, axis=0)

    # Normalise then aggregate
    xgb_norm  = safe_norm(xgb_imp)
    shap_norm = safe_norm(shap_imp)
    perm_norm = safe_norm(perm_imp)
    agg_norm  = (xgb_norm + shap_norm + perm_norm) / 3.0
    agg_norm["mean_importance"] = agg_norm[TARGETS].mean(axis=1)

    # Visualisations
    for mat, label in [
        (xgb_norm, "XGB_Fscore"),
        (shap_norm, "SHAP"),
        (perm_norm, "Perm"),
        (agg_norm[TARGETS], "Aggregated"),
    ]:
        plot_heatmap(
            mat, short_reg,
            f"{dataset_name} — {label} Importance (KFold avg)",
            os.path.join(output_path, f"{dataset_name}_XGB_{label}_Heatmap"),
            logger,
        )
    plot_bar(
        agg_norm["mean_importance"], short_reg,
        f"{dataset_name}: Aggregated XGBoost Importance",
        "Mean Normalised Importance",
        os.path.join(output_path, f"{dataset_name}_XGB_Agg_BarPlot"),
        logger,
    )
    for target in TARGETS:
        y_full = df[target].fillna(0).values
        if y_full.std() == 0:
            continue
        X_tr, X_vl, y_tr, y_vl = train_test_split(
            X_scaled, y_full, test_size=0.2, random_state=RS
        )
        m = XGBRegressor(**XGB_PARAMS)
        m.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], verbose=False)
        sv = shap.TreeExplainer(m).shap_values(X_df)
        safe_t = target.replace(" ", "_")
        plot_shap_beeswarm(
            sv, X_df, short_reg,
            f"{dataset_name} — XGB SHAP Beeswarm for '{target}'",
            os.path.join(output_path, f"{dataset_name}_XGB_SHAP_{safe_t}"),
            logger,
        )

    # Selection
    max_imp  = agg_norm["mean_importance"].max()
    selected = (
        agg_norm[
            agg_norm["mean_importance"] >= rel_threshold * max_imp
        ].index.tolist()
        if max_imp > 0
        else features.copy()
    )
    selected = selected or features.copy()

    reduced_df = df[selected + TARGETS].copy()
    for mat, name in [
        (xgb_imp, "XGB"), (shap_imp, "SHAP"),
        (perm_imp, "Perm"), (agg_norm, "Agg"),
    ]:
        mat.to_csv(
            os.path.join(output_path, f"{dataset_name}_{name}_Importances.csv")
        )
    reduced_df.to_csv(
        os.path.join(output_path, f"{dataset_name}_XGB_Selected.csv"), index=False
    )
    save_method_report(output_path, dataset_name, "XGB", features, selected, {
        "Criterion":        f"Agg >= {rel_threshold*100:.1f}% of max",
        "CV_Folds":         N_SPLITS,
        "Importance_Types": "XGBoost + SHAP + Permutation (min-max normalised)",
    })
    logger.info(
        f"[XGB | {dataset_name}] {len(selected)}/{len(features)} features retained."
    )
    return reduced_df, selected


# ======================================================
# CONSENSUS VOTING
# ======================================================

def consensus_voting(
    selections:  dict[str, list[str]],
    all_features: list[str],
    dataset_name: str,
    output_path: str,
    min_votes: int = SELECTION["consensus_min_votes"],
    logger=None,
) -> list[str]:
    """
    Retain features selected by >= min_votes out of len(selections) methods.

    This democratic mechanism is agnostic to the magnitude of individual
    importance scores and mitigates the structural biases of any single
    estimator (paper §3.4).

    Vote count V_j in {0,1,2,3,4} is saved to CSV for transparency.
    """
    if logger is None:
        logger = get_logger(__name__)
    os.makedirs(output_path, exist_ok=True)

    methods = list(selections.keys())
    vote_df = pd.DataFrame(0, index=all_features, columns=methods)
    for method, sel in selections.items():
        vote_df.loc[vote_df.index.isin(sel), method] = 1

    vote_df["total_votes"] = vote_df[methods].sum(axis=1)
    consensus = vote_df[vote_df["total_votes"] >= min_votes].index.tolist()
    if not consensus:
        logger.warning(
            f"[Consensus | {dataset_name}] No features reached {min_votes} votes "
            f"— falling back to full feature set."
        )
        consensus = all_features.copy()

    short_reg = build_shortname_registry(all_features + methods + ["total_votes"])

    plot_heatmap(
        vote_df[methods], short_reg,
        f"{dataset_name} — Method Agreement Matrix (1=selected, 0=dropped)",
        os.path.join(output_path, f"{dataset_name}_Consensus_Heatmap"),
        logger,
    )
    plot_bar(
        vote_df["total_votes"], short_reg,
        f"{dataset_name}: Feature Vote Counts ({min_votes}/{len(methods)} required)",
        "Number of Methods Selecting Feature",
        os.path.join(output_path, f"{dataset_name}_Consensus_BarPlot"),
        logger,
    )

    vote_df.to_csv(
        os.path.join(output_path, f"{dataset_name}_Consensus_Votes.csv")
    )
    pd.DataFrame([{
        "Dataset":            dataset_name,
        "Total_Features":     len(all_features),
        "Consensus_Selected": len(consensus),
        "Dropped":            len(all_features) - len(consensus),
        "Retention_%":        f"{100*len(consensus)/max(1,len(all_features)):.2f}%",
        "Min_Votes_Required": min_votes,
        "Methods":            ", ".join(methods),
    }]).to_csv(
        os.path.join(output_path, f"{dataset_name}_Consensus_Report.csv"), index=False
    )

    logger.info(
        f"[Consensus | {dataset_name}] {len(consensus)}/{len(all_features)} "
        f"features retained (>={min_votes}/{len(methods)} votes)."
    )
    return consensus
