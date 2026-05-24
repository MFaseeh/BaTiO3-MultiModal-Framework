"""
augmentation.py
===============
Data augmentation for the BaTiO3 framework.

CRITICAL DESIGN RULE
--------------------
Augmentation is applied EXCLUSIVELY inside each CV training fold.
It is NEVER called on the full dataset before splitting.

The public API is a single function:

    augment_train_fold(X_tr, y_tr, seed, continuous_cols, cat_cols)

which is called from evaluation.py inside the CV loop, after
train_idx / test_idx have been determined.

This guarantees:
  - Test folds always contain only original, unperturbed samples.
  - No augmented variant of a base sample can appear in a test fold
    simultaneously with its source.
  - There is no information leakage through near-duplicate proximity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import AUGMENTATION, TARGETS


def augment_train_fold(
    X_tr: pd.DataFrame,
    y_tr: pd.DataFrame,
    seed: int,
    continuous_cols: list[str] | None = None,
    cat_cols:        list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply controlled Gaussian perturbation to the training fold only.

    Parameters
    ----------
    X_tr : pd.DataFrame
        Feature matrix for the training fold (original, unperturbed).
    y_tr : pd.DataFrame
        Target matrix for the training fold (original, unperturbed).
    seed : int
        Random seed — should be varied per fold/repeat for diversity.
    continuous_cols : list[str] | None
        Columns to perturb. If None, all columns not in cat_cols are treated
        as continuous.
    cat_cols : list[str] | None
        Categorical columns — copied without perturbation.

    Returns
    -------
    X_aug : pd.DataFrame
        Concatenation of original + augmented training samples.
    y_aug : pd.DataFrame
        Corresponding target matrix.

    Notes
    -----
    - Augment factor: each base sample produces AUGMENTATION["augment_factor"]
      additional copies, giving a total of (1 + factor) × n_train samples.
    - Noise std: AUGMENTATION["noise_std"] fraction of each feature's std,
      reflecting ±5% instrument calibration uncertainty.
    - If AUGMENTATION["clip_to_range"] is True, augmented values are clipped
      to the [min, max] range observed in X_tr to prevent out-of-distribution
      generation.
    - Categorical columns and target values are perturbed by the same
      proportional noise to maintain consistency, or left unchanged if
      AUGMENTATION["perturb_targets"] is False.
    """
    if not AUGMENTATION["enabled"]:
        return X_tr.copy(), y_tr.copy()

    rng    = np.random.default_rng(seed)
    factor = AUGMENTATION["augment_factor"]
    std    = AUGMENTATION["noise_std"]

    # Identify continuous columns
    if cat_cols is None:
        cat_cols = []
    if continuous_cols is None:
        continuous_cols = [c for c in X_tr.columns if c not in cat_cols]

    # Observed ranges for clipping
    col_min = X_tr[continuous_cols].min()
    col_max = X_tr[continuous_cols].max()
    col_std = X_tr[continuous_cols].std().replace(0, 1)   # avoid zero std

    aug_X_parts: list[pd.DataFrame] = [X_tr.copy()]
    aug_y_parts: list[pd.DataFrame] = [y_tr.copy()]

    for _ in range(factor):
        X_copy = X_tr.copy()

        # Perturb continuous features
        noise = rng.normal(
            loc=0.0,
            scale=std * col_std.values,
            size=(len(X_tr), len(continuous_cols)),
        )
        X_copy[continuous_cols] = X_copy[continuous_cols].values + noise

        # Clip to observed range
        if AUGMENTATION["clip_to_range"]:
            X_copy[continuous_cols] = X_copy[continuous_cols].clip(
                lower=col_min, upper=col_max, axis=1
            )

        aug_X_parts.append(X_copy)

        # Perturb targets consistently
        if AUGMENTATION["perturb_targets"]:
            y_copy  = y_tr.copy()
            y_std   = y_tr.std().replace(0, 1)
            y_noise = rng.normal(
                loc=0.0,
                scale=std * y_std.values,
                size=y_tr.shape,
            )
            y_copy = pd.DataFrame(
                y_tr.values + y_noise,
                columns=y_tr.columns,
                index=y_tr.index,
            )
            aug_y_parts.append(y_copy)
        else:
            aug_y_parts.append(y_tr.copy())

    X_aug = pd.concat(aug_X_parts, ignore_index=True)
    y_aug = pd.concat(aug_y_parts, ignore_index=True)

    return X_aug, y_aug
