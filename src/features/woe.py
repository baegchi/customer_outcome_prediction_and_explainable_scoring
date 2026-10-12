"""
Weight of Evidence (WoE) and Information Value (IV) transformation pipeline.
Implements the exact contract specified in docs/woe_iv.md:
  - fit_woe: strictly fitted on TRAINING data only
  - transform_woe: applies frozen BinArtifact without target access or refitting
  - save_woe_bins / load_woe_bins: JSON serialization for pipeline reproducibility
  - compute_iv_table: feature selection audit trail
  - validate_no_oot_leakage: immutability assertion test helper
"""

from __future__ import annotations

import copy
import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from optbinning import OptimalBinning

# Type alias matching docs/woe_iv.md specification
BinArtifact = Dict[str, Dict[str, Any]]


# -----------------------------------------------------------------------------
# Core Fitting API (TRAIN ONLY)
# -----------------------------------------------------------------------------

def fit_woe(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    features: List[str],
    cat_features: Optional[List[str]] = None,
    binary_features: Optional[List[str]] = None,
    max_bins: int = 10,
    min_bin_pct: float = 0.05,
    rare_pct: float = 0.01,
    iv_threshold: float = 0.02,
    monotonic_trend: str = "auto",
    random_state: int = 42,
) -> BinArtifact:
    """
    Fit WoE bins on TRAINING data only.

    MUST be called exclusively on training data (TRAIN set or CV fold train portion).
    Never call this on DEV or OOT data, or on any dataset that includes DEV/OOT rows.

    Args:
        X_train:          Training features DataFrame. TRAIN set only.
        y_train:          Binary target Series (0/1). TRAIN set only.
        features:         All feature names to bin (numeric + categorical + binary).
        cat_features:     Subset of `features` that are categorical (object dtype).
        binary_features:  Subset of `features` that are binary (0/1 only).
        max_bins:         Maximum number of bins for numeric features. Default: 10.
        min_bin_pct:      Minimum fraction of TRAIN rows per bin. Default: 0.05 (5%).
        rare_pct:         Fraction threshold below which categorical values are grouped
                          as '_RARE_'. Default: 0.01 (1%).
        iv_threshold:     IV threshold for feature selection. Default: 0.02.
        monotonic_trend:  Monotonicity constraint for numeric binning. "auto" infers
                          direction; "ascending"/"descending" forces direction.
        random_state:     Random seed.

    Returns:
        BinArtifact: Dictionary mapping feature name -> bin definition.
    """
    # 1. Strict input validation
    if len(X_train) != len(y_train):
        raise ValueError(
            f"X_train and y_train have different lengths: {len(X_train)} vs {len(y_train)}."
        )

    y_clean = pd.to_numeric(y_train, errors="coerce")
    unique_targets = set(y_clean.dropna().unique())
    if not unique_targets.issubset({0, 1}):
        raise ValueError(
            f"y_train contains invalid target values: {unique_targets}. Must contain only 0 and 1."
        )

    missing_cols = set(features) - set(X_train.columns)
    if missing_cols:
        raise ValueError(f"Features not found in X_train columns: {sorted(missing_cols)}")

    n_events_total = float((y_clean == 1).sum())
    n_nonevents_total = float((y_clean == 0).sum())

    if n_events_total == 0 or n_nonevents_total == 0:
        raise ValueError("y_train must contain both positive (1) and negative (0) classes.")

    cat_set = set(cat_features) if cat_features is not None else set()
    bin_set = set(binary_features) if binary_features is not None else set()

    bins: BinArtifact = {}

    for feat in features:
        col_data = X_train[feat]

        # Determine feature type if not explicitly passed
        is_binary = feat in bin_set or (
            pd.api.types.is_numeric_dtype(col_data)
            and set(col_data.dropna().unique()).issubset({0, 1, 0.0, 1.0})
        )
        is_cat = feat in cat_set or (
            pd.api.types.is_object_dtype(col_data)
            or pd.api.types.is_string_dtype(col_data)
            or isinstance(col_data.dtype, pd.CategoricalDtype)
        )

        if is_binary:
            bins[feat] = _fit_binary_woe(
                col_data=col_data,
                y=y_clean,
                n_events_total=n_events_total,
                n_nonevents_total=n_nonevents_total,
                iv_threshold=iv_threshold,
            )
        elif is_cat:
            bins[feat] = _fit_categorical_woe(
                col_data=col_data,
                y=y_clean,
                rare_pct=rare_pct,
                min_bin_pct=min_bin_pct,
                iv_threshold=iv_threshold,
                n_events_total=n_events_total,
                n_nonevents_total=n_nonevents_total,
            )
        else:
            bins[feat] = _fit_numerical_woe(
                col_data=col_data,
                y=y_clean,
                max_bins=max_bins,
                min_bin_pct=min_bin_pct,
                monotonic_trend=monotonic_trend,
                iv_threshold=iv_threshold,
                n_events_total=n_events_total,
                n_nonevents_total=n_nonevents_total,
            )

    return bins


def _fit_binary_woe(
    col_data: pd.Series,
    y: pd.Series,
    n_events_total: float,
    n_nonevents_total: float,
    iv_threshold: float,
) -> Dict[str, Any]:
    """Fit exact WoE for binary (0/1) features with Laplace smoothing."""
    woe_map: Dict[str, float] = {}
    total_iv = 0.0

    for val in [0, 1]:
        mask = (col_data == val)
        n_ev = float((mask & (y == 1)).sum())
        n_nev = float((mask & (y == 0)).sum())

        # Laplace smoothing (+0.5) per docs/woe_iv.md §8
        p_ev = (n_ev + 0.5) / (n_events_total + 1.0)
        p_nev = (n_nev + 0.5) / (n_nonevents_total + 1.0)

        woe = float(np.log(p_ev / p_nev))
        iv_bin = float((p_ev - p_nev) * woe)

        woe_map[str(val)] = round(woe, 6)
        total_iv += iv_bin

    # Handle missing values if any
    miss_mask = col_data.isna()
    if miss_mask.sum() > 0:
        n_ev_m = float((miss_mask & (y == 1)).sum())
        n_nev_m = float((miss_mask & (y == 0)).sum())
        p_ev_m = (n_ev_m + 0.5) / (n_events_total + 1.0)
        p_nev_m = (n_nev_m + 0.5) / (n_nonevents_total + 1.0)
        missing_woe = round(float(np.log(p_ev_m / p_nev_m)), 6)
        total_iv += float((p_ev_m - p_nev_m) * missing_woe)
    else:
        missing_woe = 0.0

    return {
        "type": "binary",
        "splits": [0.5],
        "woe_map": woe_map,
        "iv": round(total_iv, 6),
        "selected": bool(total_iv >= iv_threshold),
        "rare_categories": [],
        "unseen_woe": 0.0,
        "missing_woe": missing_woe,
        "n_bins": 2,
    }


def _fit_numerical_woe(
    col_data: pd.Series,
    y: pd.Series,
    max_bins: int,
    min_bin_pct: float,
    monotonic_trend: str,
    iv_threshold: float,
    n_events_total: float,
    n_nonevents_total: float,
) -> Dict[str, Any]:
    """Fit optimal monotonic WoE binning for numerical features using optbinning."""
    clean_vals = pd.to_numeric(col_data, errors="coerce")
    valid_mask = clean_vals.notna()

    # Edge case: feature has < 2 unique values
    if clean_vals[valid_mask].nunique() < 2:
        return {
            "type": "numerical",
            "splits": [],
            "woe_map": {"0": 0.0},
            "iv": 0.0,
            "selected": False,
            "rare_categories": [],
            "unseen_woe": 0.0,
            "missing_woe": 0.0,
            "n_bins": 1,
        }

    # Attempt CP solver with specified monotonicity
    trend = "auto_asc_desc" if monotonic_trend == "auto" else monotonic_trend
    binner = OptimalBinning(
        name=str(col_data.name),
        dtype="numerical",
        solver="cp",
        monotonic_trend=trend,
        max_n_bins=max_bins,
        min_bin_size=min_bin_pct,
    )

    try:
        binner.fit(clean_vals.values, y.values)
        if binner.status != "OPTIMAL":
            # Fallback to unrestricted trend per docs/woe_iv.md §8
            binner = OptimalBinning(
                name=str(col_data.name),
                dtype="numerical",
                solver="cp",
                monotonic_trend="none",
                max_n_bins=max_bins,
                min_bin_size=min_bin_pct,
            )
            binner.fit(clean_vals.values, y.values)
    except Exception:
        # Fallback to unconstrained
        binner = OptimalBinning(
            name=str(col_data.name),
            dtype="numerical",
            solver="cp",
            monotonic_trend="none",
            max_n_bins=max_bins,
            min_bin_size=min_bin_pct,
        )
        binner.fit(clean_vals.values, y.values)

    table = binner.binning_table.build()

    # Extract splits and WoE values
    splits = [float(s) for s in binner.splits]
    woe_map: Dict[str, float] = {}

    # Standard numeric bins: indices 0 to len(splits)
    n_splits = len(binner.splits)
    bin_rows = table.iloc[: n_splits + 1].reset_index(drop=True)
    for idx, row in bin_rows.iterrows():
        woe_val = row["WoE"]
        woe_float = float(woe_val) if woe_val != "" and pd.notna(woe_val) else 0.0
        woe_map[str(idx)] = round(woe_float, 6)

    # Missing WoE
    is_missing = table["Bin"].apply(lambda b: isinstance(b, str) and b == "Missing")
    miss_rows = table[is_missing]
    missing_woe = 0.0
    if not miss_rows.empty:
        m_val = miss_rows["WoE"].iloc[0]
        if m_val != "" and pd.notna(m_val):
            missing_woe = float(m_val)

    # Total IV
    is_totals = table["Bin"].apply(lambda b: isinstance(b, str) and (b == "" or b == "Totals"))
    totals_rows = table[is_totals]
    total_iv = 0.0
    if not totals_rows.empty:
        t_val = totals_rows["IV"].iloc[-1]
        if t_val != "" and pd.notna(t_val):
            total_iv = float(t_val)

    return {
        "type": "numerical",
        "splits": splits,
        "woe_map": woe_map,
        "iv": round(total_iv, 6),
        "selected": bool(total_iv >= iv_threshold),
        "rare_categories": [],
        "unseen_woe": 0.0,
        "missing_woe": round(missing_woe, 6),
        "n_bins": len(splits) + 1,
    }


def _fit_categorical_woe(
    col_data: pd.Series,
    y: pd.Series,
    rare_pct: float,
    min_bin_pct: float,
    iv_threshold: float,
    n_events_total: float,
    n_nonevents_total: float,
) -> Dict[str, Any]:
    """Fit categorical WoE with rare-category grouping."""
    str_vals = col_data.astype(str).replace({"nan": np.nan, "None": np.nan})
    n_total = len(str_vals)

    # Identify rare categories (< rare_pct * total)
    counts = str_vals.value_counts(dropna=True)
    rare_cats = set(counts[counts < (rare_pct * n_total)].index)

    # Group rare categories into "_RARE_"
    x_grouped = str_vals.copy()
    if rare_cats:
        x_grouped = x_grouped.apply(lambda v: "_RARE_" if v in rare_cats else v)

    binner = OptimalBinning(
        name=str(col_data.name),
        dtype="categorical",
        solver="mip",
        min_bin_size=min_bin_pct,
        cat_cutoff=rare_pct,
    )

    try:
        binner.fit(x_grouped.values, y.values)
    except Exception:
        # Fallback to cp solver
        binner = OptimalBinning(
            name=str(col_data.name),
            dtype="categorical",
            solver="cp",
            min_bin_size=min_bin_pct,
            cat_cutoff=rare_pct,
        )
        binner.fit(x_grouped.values, y.values)

    table = binner.binning_table.build()

    # Build category-to-woe map
    cat_woe_map: Dict[str, float] = {}
    rare_woe = 0.0
    n_cat_splits = len(binner.splits)
    bin_rows = table.iloc[:n_cat_splits].reset_index(drop=True)

    for idx, row in bin_rows.iterrows():
        w_val = row["WoE"]
        woe_val = round(float(w_val), 6) if w_val != "" and pd.notna(w_val) else 0.0
        bin_cats = row["Bin"]
        if isinstance(bin_cats, (list, tuple, np.ndarray)):
            for c in bin_cats:
                cat_woe_map[str(c)] = woe_val
                if str(c) in ["_RARE_", "Other", "others"]:
                    rare_woe = woe_val
        else:
            cat_woe_map[str(bin_cats)] = woe_val
            if str(bin_cats) in ["_RARE_", "Other", "others"]:
                rare_woe = woe_val

    # Missing row
    is_missing = table["Bin"].apply(lambda b: isinstance(b, str) and b == "Missing")
    miss_rows = table[is_missing]
    missing_woe = 0.0
    if not miss_rows.empty:
        m_val = miss_rows["WoE"].iloc[0]
        if m_val != "" and pd.notna(m_val):
            missing_woe = float(m_val)

    # Totals row
    is_totals = table["Bin"].apply(lambda b: isinstance(b, str) and (b == "" or b == "Totals"))
    totals_rows = table[is_totals]
    total_iv = 0.0
    if not totals_rows.empty:
        t_val = totals_rows["IV"].iloc[-1]
        if t_val != "" and pd.notna(t_val):
            total_iv = float(t_val)

    return {
        "type": "categorical",
        "splits": [list(g) if isinstance(g, (list, tuple, np.ndarray)) else [str(g)] for g in binner.splits],
        "woe_map": cat_woe_map,
        "iv": round(total_iv, 6),
        "selected": bool(total_iv >= iv_threshold),
        "rare_categories": sorted(list(rare_cats)),
        "unseen_woe": round(rare_woe, 6),
        "missing_woe": round(missing_woe, 6),
        "n_bins": len(bin_rows),
    }


# -----------------------------------------------------------------------------
# Transformation API (TRAIN, DEV, OR OOT)
# -----------------------------------------------------------------------------

def transform_woe(
    X: pd.DataFrame,
    bins: BinArtifact,
    selected_only: bool = True,
) -> pd.DataFrame:
    """
    Transform features to WoE values using pre-fitted bins.

    Can be called on TRAIN, DEV, or OOT — the bins are never modified.
    The target variable is NOT an argument — this function has no access to labels.

    Args:
        X:             DataFrame to transform. May be TRAIN, DEV, or OOT.
        bins:          BinArtifact returned by fit_woe(). Never refit or modified here.
        selected_only: If True (default), return only features where bins[f]['selected']
                       is True (IV >= threshold). If False, return all binned features.

    Returns:
        DataFrame with WoE-transformed features.
    """
    # 1. Validation
    features_to_transform = [
        f for f, meta in bins.items() if not selected_only or meta["selected"]
    ]

    missing_cols = [f for f in features_to_transform if f not in X.columns]
    if missing_cols:
        raise KeyError(
            f"Features required by WoE bins not present in DataFrame: {missing_cols}"
        )

    out_df = pd.DataFrame(index=X.index)

    for feat in features_to_transform:
        meta = bins[feat]
        col_data = X[feat]
        f_type = meta["type"]
        missing_woe = meta["missing_woe"]
        unseen_woe = meta["unseen_woe"]
        woe_map = meta["woe_map"]

        if f_type == "binary":
            # Map 0 and 1, missing to missing_woe
            woe_0 = woe_map.get("0", unseen_woe)
            woe_1 = woe_map.get("1", unseen_woe)

            res = np.where(
                col_data.isna(),
                missing_woe,
                np.where(col_data == 1, woe_1, woe_0),
            )
            out_df[feat] = pd.Series(res, index=X.index, dtype=np.float32)

        elif f_type == "numerical":
            splits = meta["splits"]
            numeric_vals = pd.to_numeric(col_data, errors="coerce").values
            nan_mask = np.isnan(numeric_vals)

            if len(splits) == 0:
                res = np.full_like(numeric_vals, fill_value=woe_map.get("0", 0.0), dtype=np.float32)
            else:
                # np.digitize: val < splits[0] -> 0; splits[0] <= val < splits[1] -> 1; ...
                bin_indices = np.digitize(numeric_vals, splits)
                # Lookup each bin index
                res = np.array([woe_map.get(str(b), unseen_woe) for b in bin_indices], dtype=np.float32)

            res[nan_mask] = missing_woe
            out_df[feat] = pd.Series(res, index=X.index, dtype=np.float32)

        elif f_type == "categorical":
            str_vals = col_data.astype(str).replace({"nan": np.nan, "None": np.nan})
            rare_set = set(meta.get("rare_categories", []))

            transformed = []
            for val in str_vals:
                if pd.isna(val):
                    transformed.append(missing_woe)
                elif val in woe_map:
                    transformed.append(woe_map[val])
                elif val in rare_set or "_RARE_" in woe_map:
                    transformed.append(woe_map.get("_RARE_", unseen_woe))
                else:
                    transformed.append(unseen_woe)

            out_df[feat] = pd.Series(transformed, index=X.index, dtype=np.float32)

    return out_df


# -----------------------------------------------------------------------------
# Persistence & Audit API
# -----------------------------------------------------------------------------

def save_woe_bins(bins: BinArtifact, path: Union[str, Path]) -> None:
    """Serialize the BinArtifact to a JSON file for reproducibility."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(bins, f, indent=2, default=float)


def load_woe_bins(path: Union[str, Path]) -> BinArtifact:
    """Load a BinArtifact from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_iv_table(bins: BinArtifact) -> pd.DataFrame:
    """Return a sorted summary table of features, their IVs, and selection status."""
    rows = [
        {
            "feature": feat,
            "iv": float(meta["iv"]),
            "n_bins": int(meta["n_bins"]),
            "type": str(meta["type"]),
            "selected": bool(meta["selected"]),
        }
        for feat, meta in bins.items()
    ]
    df_iv = (
        pd.DataFrame(rows)
        .sort_values("iv", ascending=False)
        .reset_index(drop=True)
    )

    # Check for suspiciously high IV (potential leakage)
    suspicious = df_iv[df_iv["iv"] >= 0.5]
    if not suspicious.empty:
        warnings.warn(
            f"SUSPICIOUSLY HIGH IV (>= 0.5) DETECTED in {len(suspicious)} feature(s)! "
            f"Review for potential data leakage: {list(suspicious['feature'])}",
            UserWarning,
        )

    return df_iv


def validate_no_oot_leakage(
    bins_before: BinArtifact,
    bins_after: BinArtifact,
) -> None:
    """Assert that two BinArtifacts are identical (for leakage testing)."""
    assert bins_before == bins_after, (
        "BinArtifact was mutated during transform_woe(). "
        "The bins object must be read-only during transformation."
    )


if __name__ == "__main__":
    from src.data.load import load_raw
    from src.data.clean import clean_data
    from src.data.split import chronological_split
    from src.features.build import build_features

    print("Loading and preparing 3,000 rows for WoE demonstration...")
    raw = load_raw(nrows=3000)
    cleaned = clean_data(raw, drop_unresolved_target=True)
    train_df, dev_df, oot_df = chronological_split(cleaned, target_col="target", verbose=False)

    train_feat = build_features(train_df, drop_collinear_raw=True)
    dev_feat = build_features(dev_df, drop_collinear_raw=True)

    target_col = "target"
    candidate_features = [
        "fico_mid", "credit_age_months", "loan_to_income", "revol_bal_to_income",
        "log_annual_inc", "has_delinq_2y", "has_pub_rec", "has_bankruptcy",
        "mths_since_last_delinq_imputed", "bureau_extended_missing", "term_months",
    ]

    print("Fitting WoE on TRAIN only...")
    fitted_bins = fit_woe(
        X_train=train_feat,
        y_train=train_feat[target_col],
        features=candidate_features,
    )

    print("\nIV Summary Table:")
    iv_table = compute_iv_table(fitted_bins)
    print(iv_table)

    print("\nTransforming DEV without refitting...")
    bins_copy = copy.deepcopy(fitted_bins)
    dev_woe = transform_woe(dev_feat, fitted_bins)
    validate_no_oot_leakage(bins_copy, fitted_bins)
    print(f"DEV WoE shape: {dev_woe.shape}")
    print("SUCCESS: WoE fitted on TRAIN and applied immutably to DEV!")
