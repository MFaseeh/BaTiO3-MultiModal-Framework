"""
models.py
=========
Base learner and meta-learner registries for the BaTiO3 framework.

All models are wrapped in sklearn Pipelines with StandardScaler inside —
this guarantees the scaler is fitted only on the training partition of
each fold and never touches validation or test data (anti-leakage).

Each call to get_level1_models() / get_models() / get_meta_learners()
returns FRESH instances — never reuse a fitted pipeline across folds.
"""

from __future__ import annotations

import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src.config import (
    LGBM_PARAMS,
    MLP_PARAMS,
    RF_PARAMS,
    RIDGE_ALPHA,
    XGB_PARAMS,
)


# ======================================================
# LEVEL-1 BASE LEARNERS
# ======================================================

def get_level1_models() -> dict[str, Pipeline]:
    """
    Return a fresh dict of name → Pipeline for the four base learners.

    Architectural diversity:
      RF   — variance-reducing bagging
      LGBM — leaf-wise gradient boosting
      XGB  — regularised level-wise gradient boosting
      MLP  — non-linear neural approximation

    StandardScaler inside Pipeline: fitted only on inner-train rows
    during OOF generation, never on validation or test rows.
    """
    return {
        "RF": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  RandomForestRegressor(**RF_PARAMS)),
        ]),
        "LGBM": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  lgb.LGBMRegressor(**LGBM_PARAMS)),
        ]),
        "XGB": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  XGBRegressor(**XGB_PARAMS)),
        ]),
        "MLP": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  MLPRegressor(**MLP_PARAMS)),
        ]),
    }


# Alias: ablation study uses get_models()
get_models = get_level1_models


# ======================================================
# LEVEL-2 META-LEARNERS
# ======================================================

def get_meta_learners() -> dict[str, Pipeline]:
    """
    Return a fresh dict of name → Pipeline for five meta-learner candidates.

    Ridge is the primary choice:
      (i)  meta-feature space has only 12 columns → non-linear meta-learners
           prone to overfitting
      (ii) Level-1 predictions are highly collinear → ℓ2 regularisation
           provides stable weight estimates
      (iii) learned weights are directly interpretable as modality-level
            reliability scores

    The remaining four are evaluated for comparison in Section 6.2.
    """
    return {
        "Ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  Ridge(alpha=RIDGE_ALPHA)),
        ]),
        "RF": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  RandomForestRegressor(
                n_estimators=200,
                random_state=RF_PARAMS["random_state"],
                n_jobs=-1,
            )),
        ]),
        "LGBM": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  lgb.LGBMRegressor(
                n_estimators=200,
                learning_rate=LGBM_PARAMS["learning_rate"],
                random_state=LGBM_PARAMS["random_state"],
                n_jobs=-1,
                verbose=-1,
            )),
        ]),
        "XGB": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  XGBRegressor(
                n_estimators=200,
                learning_rate=XGB_PARAMS["learning_rate"],
                random_state=XGB_PARAMS["random_state"],
                verbosity=0,
            )),
        ]),
        "MLP": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  MLPRegressor(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                max_iter=300,
                random_state=MLP_PARAMS["random_state"],
                early_stopping=True,
            )),
        ]),
    }
