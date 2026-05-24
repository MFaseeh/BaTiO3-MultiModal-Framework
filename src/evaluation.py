"""
evaluation.py
=============
Anti-leakage cross-validated evaluation engines.

Two engines:
  evaluate_feature_set  — repeated K-Fold for ablation study (Experiments A–E)
  stacking_cv           — two-level stacking with OOF meta-features

Both engines integrate data augmentation INSIDE the training fold,
addressing the reviewer concern (R1-Major-1).

Anti-Leakage Protocol (three layers)
-------------------------------------
Layer 1 — Outer fold isolation:
    The outer KFold defines the test partition. No fitted parameter
    (scaler, model weights, meta-learner) is estimated on test samples.

Layer 2 — Augmentation isolation:
    augment_train_fold() is called AFTER train_idx / test_idx are
    determined, operating only on X[train_idx]. The test fold always
    contains original, unperturbed samples.

Layer 3 — Pipeline isolation:
    StandardScaler is inside every sklearn Pipeline, fitted only on
    the (augmented) training partition of each fold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

from src.augmentation import augment_train_fold
from src.config import CV, MODALITY_SUFFIXES, TARGETS
from src.models import get_level1_models, get_meta_learners, get_models
from src.utils import get_logger


RS         = CV["random_state"]
N_SPLITS   = CV["n_outer_splits"]
N_REPEATS  = CV["n_repeats"]
N_INNER    = CV["n_inner_splits"]


# ======================================================
# METRIC HELPER
# ======================================================

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE":  mean_absolute_error(y_true, y_pred),
        "R2":   r2_score(y_true, y_pred),
    }


def aggregate_results(raw: pd.DataFrame) -> pd.DataFrame:
    """Collapse fold/repeat → mean ± std per (experiment, model, target)."""
    return (
        raw
        .groupby(["experiment", "model", "target"])[["RMSE", "MAE", "R2"]]
        .agg(["mean", "std"])
        .round(4)
    )


# ======================================================
# ABLATION ENGINE — evaluate_feature_set
# ======================================================

def evaluate_feature_set(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    experiment_name: str,
    augment: bool = True,
    logger=None,
) -> pd.DataFrame:
    """
    Repeated K-Fold evaluation for the ablation study.

    Augmentation is applied inside each training fold when augment=True.
    Test folds always use original, unperturbed samples.

    Returns DataFrame with columns:
        [experiment, model, target, repeat, fold, RMSE, MAE, R2]
    """
    if logger is None:
        logger = get_logger(__name__)

    records = []

    for repeat in range(N_REPEATS):
        seed = RS + repeat * 100
        kf   = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)

        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X), start=1):

            # ── Original train/test partitions ────────────────────────────
            X_tr_orig = X.iloc[train_idx]
            X_te      = X.iloc[test_idx]          # TEST: always original
            Y_tr_orig = Y.iloc[train_idx]
            Y_te      = Y.iloc[test_idx]

            # ── Augment training fold only ─────────────────────────────────
            # augment_train_fold never sees test_idx data.
            if augment:
                X_tr, Y_tr = augment_train_fold(
                    X_tr_orig, Y_tr_orig,
                    seed=seed + fold_idx,
                )
            else:
                X_tr, Y_tr = X_tr_orig.copy(), Y_tr_orig.copy()

            # ── Train and evaluate each model on each target ───────────────
            for model_name in get_models():
                pipe = get_models()[model_name]   # fresh pipeline

                for target in TARGETS:
                    pipe.fit(X_tr, Y_tr[target].values)
                    y_pred = pipe.predict(X_te)

                    records.append({
                        "experiment": experiment_name,
                        "model":      model_name,
                        "target":     target,
                        "repeat":     repeat + 1,
                        "fold":       fold_idx,
                        **compute_metrics(Y_te[target].values, y_pred),
                    })

    logger.info(
        f"[{experiment_name}] Complete — "
        f"{N_REPEATS} repeats × {N_SPLITS} folds × "
        f"{len(get_models())} models × {len(TARGETS)} targets"
        + (" (with in-fold augmentation)" if augment else "")
    )
    return pd.DataFrame(records)


# ======================================================
# STACKING ENGINE — OOF generation
# ======================================================

def _generate_oof_predictions(
    X_mod: pd.DataFrame,
    y_np:  np.ndarray,
    train_idx: np.ndarray,
    seed: int,
    inner_n_splits: int = N_INNER,
) -> np.ndarray:
    """
    Generate OOF predictions for one modality on the training fold only.

    - Called exclusively with X_mod.iloc[train_idx] data
    - Test fold is never visible at this stage
    - Each inner fold fits scaler + model only on inner-train rows
    - Returns shape: (len(train_idx), n_level1_models)
    """
    X_tr = X_mod.iloc[train_idx].reset_index(drop=True)
    y_tr = y_np[train_idx]
    n    = len(X_tr)

    oof      = np.zeros((n, len(get_level1_models())))
    inner_kf = KFold(n_splits=inner_n_splits, shuffle=True, random_state=seed)

    for m_idx, m_name in enumerate(get_level1_models()):
        oof_col = np.zeros(n)
        for in_tr, in_val in inner_kf.split(X_tr):
            pipe = get_level1_models()[m_name]
            pipe.fit(X_tr.iloc[in_tr], y_tr[in_tr])
            oof_col[in_val] = pipe.predict(X_tr.iloc[in_val])
        oof[:, m_idx] = oof_col

    return oof


def _train_level1_on_full_train(
    X_mod: pd.DataFrame,
    y_np:  np.ndarray,
    train_idx: np.ndarray,
) -> dict:
    """Refit each Level-1 model on the full training fold for test prediction."""
    fitted = {}
    X_tr   = X_mod.iloc[train_idx]
    y_tr   = y_np[train_idx]
    for m_name in get_level1_models():
        pipe = get_level1_models()[m_name]
        pipe.fit(X_tr, y_tr)
        fitted[m_name] = pipe
    return fitted


# ======================================================
# STACKING ENGINE — stacking_cv
# ======================================================

def stacking_cv(
    modality_features: dict[str, list[str]],
    df: pd.DataFrame,
    Y: pd.DataFrame,
    experiment_name: str,
    augment: bool = True,
    n_repeats: int = N_REPEATS,
    n_splits:  int = N_SPLITS,
    logger=None,
) -> pd.DataFrame:
    """
    Full two-level stacking pipeline with repeated outer K-Fold.

    Protocol (strictly anti-leakage)
    ---------------------------------
    Outer fold splits data into TRAIN (80%) and TEST (20%).

    On TRAIN only (after augmentation of train fold):
      For each modality:
        Inner 4-fold generates OOF predictions
        → stacked_train_meta: (n_train_aug, n_mods × n_l1)

    Meta-learner fitted on stacked_train_meta → y_train.

    On TEST fold (always original, unperturbed):
      Level-1 models (retrained on full TRAIN) generate test predictions
      → stacked_test_meta: (n_test, n_mods × n_l1)
      Meta-learner predicts → metrics computed on original test samples.

    Returns DataFrame with per-repeat/fold/meta-learner/target metrics.
    """
    if logger is None:
        logger = get_logger(__name__)

    records       = []
    level1_names  = list(get_level1_models().keys())
    meta_learners = get_meta_learners()

    # Build modality DataFrames (original, unaugmented)
    X_mods = {
        mod: df[feats].fillna(0).reset_index(drop=True)
        for mod, feats in modality_features.items()
        if feats
    }
    mod_names = list(X_mods.keys())

    logger.info(
        f"[{experiment_name}] Modalities: {mod_names} | "
        f"L1 models: {level1_names} | "
        f"Meta-learners: {list(meta_learners.keys())}"
        + (" | in-fold augmentation ON" if augment else "")
    )

    for repeat in range(n_repeats):
        seed     = RS + repeat * 100
        outer_kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

        for fold_idx, (train_idx, test_idx) in enumerate(
            outer_kf.split(np.arange(len(df))), start=1
        ):
            for target in TARGETS:
                y_np   = Y[target].values
                y_test = y_np[test_idx]     # original, unperturbed

                train_meta_cols: list[np.ndarray] = []
                test_meta_cols:  list[np.ndarray] = []

                for mod in mod_names:
                    X_mod = X_mods[mod]

                    # ── Augment training fold ──────────────────────────────
                    # Only X_mod.iloc[train_idx] is augmented.
                    # test_idx is never touched here.
                    if augment:
                        X_tr_orig = X_mod.iloc[train_idx].reset_index(drop=True)
                        y_tr_orig = pd.DataFrame(
                            y_np[train_idx], columns=[target]
                        )
                        X_tr_aug, y_tr_aug_df = augment_train_fold(
                            X_tr_orig, y_tr_orig,
                            seed=seed + fold_idx,
                        )
                        y_tr_aug = y_tr_aug_df[target].values

                        # OOF on augmented training fold (inner CV)
                        n_aug = len(X_tr_aug)
                        oof   = np.zeros((n_aug, len(level1_names)))
                        inner_kf = KFold(
                            n_splits=N_INNER, shuffle=True,
                            random_state=seed + fold_idx,
                        )
                        for m_idx, m_name in enumerate(level1_names):
                            oof_col = np.zeros(n_aug)
                            for in_tr, in_val in inner_kf.split(X_tr_aug):
                                pipe = get_level1_models()[m_name]
                                pipe.fit(
                                    X_tr_aug.iloc[in_tr],
                                    y_tr_aug[in_tr],
                                )
                                oof_col[in_val] = pipe.predict(
                                    X_tr_aug.iloc[in_val]
                                )
                            oof[:, m_idx] = oof_col

                        # Retrain on full augmented train → test predictions
                        fitted_l1 = {}
                        for m_name in level1_names:
                            pipe = get_level1_models()[m_name]
                            pipe.fit(X_tr_aug, y_tr_aug)
                            fitted_l1[m_name] = pipe

                    else:
                        # No augmentation — original train fold
                        oof = _generate_oof_predictions(
                            X_mod, y_np, train_idx,
                            seed=seed + fold_idx,
                        )
                        fitted_l1 = _train_level1_on_full_train(
                            X_mod, y_np, train_idx
                        )

                    # Test predictions — always on original test fold
                    test_preds = np.column_stack([
                        fitted_l1[m].predict(X_mod.iloc[test_idx])
                        for m in level1_names
                    ])

                    train_meta_cols.append(oof)
                    test_meta_cols.append(test_preds)

                stacked_train = np.hstack(train_meta_cols)
                stacked_test  = np.hstack(test_meta_cols)

                # Determine y_train for meta-learner
                # (augmented size when augment=True)
                if augment:
                    factor     = 1 + __import__("src.config", fromlist=["AUGMENTATION"]).AUGMENTATION["augment_factor"]
                    n_aug_full = len(train_idx) * factor
                    y_train_meta = np.tile(y_np[train_idx], factor)[:stacked_train.shape[0]]
                else:
                    y_train_meta = y_np[train_idx]

                # ── Train and evaluate each meta-learner ──────────────────
                for meta_name in meta_learners:
                    meta_pipe = get_meta_learners()[meta_name]
                    meta_pipe.fit(stacked_train, y_train_meta)
                    y_pred = meta_pipe.predict(stacked_test)

                    records.append({
                        "experiment":  experiment_name,
                        "meta_learner": meta_name,
                        "target":      target,
                        "repeat":      repeat + 1,
                        "fold":        fold_idx,
                        **compute_metrics(y_test, y_pred),
                    })

        logger.info(
            f"[{experiment_name}] Repeat {repeat+1}/{n_repeats} complete."
        )

    return pd.DataFrame(records)
