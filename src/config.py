"""
config.py
=========
Centralised configuration for the BaTiO3 multi-modal prediction framework.
All hyperparameters, paths, and thresholds are defined here.
Import this module in every other script — never hard-code values elsewhere.
"""

from __future__ import annotations

# ======================================================
# INPUT / OUTPUT PATHS
# ======================================================
PATHS: dict = {
    "experimental":   "1. Input/Experimental.csv",
    "computational":  "1. Input/Computational.csv",
    "nlp":            "1. Input/NLP.csv",

    "fs_output":      "2. Feature Selection/Unified_FS",
    "ablation_output": "3. Ablation_Study",
    "stacking_output": "4. Stacking_Fusion",
    "lc_output":       "4. Stacking_Fusion/Learning_Curves",

    # These are written by 01_feature_selection.py and read by later scripts
    "final_dataset":  "2. Feature Selection/Unified_FS/Final_Selected_Dataset.csv",
    "full_dataset":   "2. Feature Selection/Unified_FS/Stage3_Hybrid/Hybrid_Fused_LGBM_Selected.csv",
    "votes_csv":      "2. Feature Selection/Unified_FS/Stage3_Hybrid/Consensus/Hybrid_Fused_Consensus_Votes.csv",
    "stacking_raw":   "4. Stacking_Fusion/Stacking_All_Raw.csv",
}

# ======================================================
# DOMAIN
# ======================================================
TARGETS: list[str] = ["density", "dielectric constant", "d33"]

TARGET_LABELS: dict[str, str] = {
    "density":             "Density $\\rho$",
    "dielectric constant": "Dielectric Constant $\\varepsilon_r$",
    "d33":                 "Piezoelectric $d_{33}$",
}

# Modality weights — used only for composite target construction
# during feature selection (Stage 2). NOT used for stacking weights.
MODALITY_WEIGHTS: dict[str, float] = {
    "exp":  0.5,
    "comp": 0.3,
    "nlpt": 0.2,
}

# Modality column suffixes
MODALITY_SUFFIXES: dict[str, str] = {
    "Experimental":  "_exp",
    "Computational": "_comp",
    "NLP":           "_nlpt",
}

# ======================================================
# RANDOM STATE
# ======================================================
RANDOM_STATE: int = 42

# ======================================================
# DATA AUGMENTATION
# ======================================================
# Applied EXCLUSIVELY inside each training fold — never globally.
AUGMENTATION: dict = {
    "enabled":          True,
    "noise_std":        0.05,   # ±5% Gaussian perturbation of continuous features
    "augment_factor":   3,      # Each base sample produces this many augmented copies
    "perturb_targets":  True,   # Perturb target values consistently
    "clip_to_range":    True,   # Never generate samples outside observed min/max
}

# ======================================================
# CROSS-VALIDATION
# ======================================================
CV: dict = {
    "n_outer_splits":  5,
    "n_inner_splits":  4,   # for OOF generation in stacking
    "n_repeats":       5,
    "random_state":    RANDOM_STATE,
}

# ======================================================
# FEATURE SELECTION
# ======================================================
SELECTION: dict = {
    "rel_threshold":        0.05,   # MI / LGBM / XGB: % of max importance
    "cumulative_threshold": 0.90,   # SHAP: cumulative importance cutoff
    "stability_threshold":  0.70,   # SHAP: bootstrap stability fraction
    "consensus_min_votes":  2,      # Min methods that must agree (out of 4)
    "n_bootstrap":          5,      # Bootstrap runs for MI / SHAP
    "n_splits":             5,      # KFold for LGBM / XGB importance
    "early_stopping_rounds": 30,
}

# ======================================================
# BASE LEARNER HYPERPARAMETERS
# ======================================================
LGBM_PARAMS: dict = {
    "n_estimators":  300,
    "learning_rate": 0.05,
    "num_leaves":    31,
    "random_state":  RANDOM_STATE,
    "n_jobs":        -1,
    "verbose":       -1,
}

XGB_PARAMS: dict = {
    "n_estimators":  300,
    "learning_rate": 0.05,
    "objective":     "reg:squarederror",
    "tree_method":   "hist",
    "eval_metric":   "rmse",
    "verbosity":     0,
    "random_state":  RANDOM_STATE,
}

RF_PARAMS: dict = {
    "n_estimators": 300,
    "max_features": "sqrt",
    "random_state": RANDOM_STATE,
    "n_jobs":       -1,
}

MLP_PARAMS: dict = {
    "hidden_layer_sizes": (128, 64, 32),
    "activation":         "relu",
    "max_iter":           500,
    "random_state":       RANDOM_STATE,
    "early_stopping":     True,
    "validation_fraction": 0.1,
}

# Meta-learner Ridge regularisation
RIDGE_ALPHA: float = 1.0

# ======================================================
# ABLATION EXPERIMENT VOTE GROUPS
# ======================================================
VOTE_GROUPS: dict[str, int] = {
    "V>=1 (16 feat)": 1,
    "V>=2 (15 feat)": 2,   # F* — proposed consensus set
    "V>=3 (10 feat)": 3,
    "V>=4 (6 feat)":  4,
}

# ======================================================
# LEARNING CURVE
# ======================================================
LEARNING_CURVE: dict = {
    "train_fractions": [0.10, 0.20, 0.30, 0.40, 0.50,
                        0.60, 0.70, 0.80, 0.90, 1.00],
    "n_repeats":       5,
    "n_splits":        5,
    "min_train_floor": 30,   # minimum training samples regardless of fraction
}

# ======================================================
# FIGURE OUTPUT
# ======================================================
FIGURE: dict = {
    "formats": ["pdf", "png"],
    "dpi":     150,
    "style":   "seaborn-v0_8-whitegrid",
}

# ======================================================
# COLOUR PALETTE
# ======================================================
COLORS: dict[str, str] = {
    "RF":    "#3A7DC9",
    "LGBM":  "#2E9E7B",
    "XGB":   "#E07B39",
    "MLP":   "#9B72C0",
    "Ridge": "#1B2A4A",
}
