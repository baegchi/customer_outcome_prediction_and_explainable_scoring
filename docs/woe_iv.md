# WoE / IV Methodology — Customer Outcome Prediction

**Dataset**: Lending Club Accepted Loans 2007–2018 Q4  
**Date**: 2026-09-04  
**Based on**: `feature_engineering.md`, `leakage_audit.md`, `validation_strategy.md`

---

## 0. Why WoE / IV in This Project

WoE/IV serves two purposes in this pipeline:

1. **Feature selection** via Information Value (IV): removes variables with negligible discriminatory power before model training. Reduces noise and computation without requiring a trained model.
2. **Feature transformation** via Weight of Evidence: converts raw feature values into a single numeric scale that encodes the log-odds of default within each bin. This is the canonical input to Logistic Regression in credit scorecards and makes the model directly interpretable as a scorecard.

For tree models (XGBoost, LightGBM, Random Forest), WoE transformation is optional — trees learn splits themselves. However, applying WoE before tree models:
- Compresses categorical cardinality (e.g., 956 zip codes → ~10 WoE-binned groups)
- Handles missing values uniformly across all models
- Allows the same feature selection (IV filter) to apply to all model types consistently

**Decision**: WoE transformation is applied to all models in this project for consistency and comparability. A sensitivity experiment may test raw features vs. WoE-transformed features on tree models.

---

## 1. Definitions

### Weight of Evidence (WoE)

For each bin `i` of a feature:

```
WoE_i = ln( P(Events in bin i) / P(Non-Events in bin i) )
       = ln( (n_events_i / N_events_total) / (n_nonevents_i / N_nonevents_total) )
```

Where:
- `n_events_i` = number of defaults (target=1) in bin i
- `n_nonevents_i` = number of non-defaults (target=0) in bin i
- `N_events_total` = total defaults in training set
- `N_nonevents_total` = total non-defaults in training set

**Interpretation**:
- WoE > 0: bin has higher proportion of defaults than the dataset average → risky
- WoE < 0: bin has lower proportion of defaults than the dataset average → safe
- WoE = 0: bin default rate equals the dataset average

### Information Value (IV)

```
IV = Σ_i [ (n_events_i/N_events_total - n_nonevents_i/N_nonevents_total) × WoE_i ]
```

IV aggregates the discriminatory power of a feature across all its bins.

**IV interpretation thresholds (Siddiqi 2006, standard industry):**

| IV range | Predictive power | Action |
|---|---|---|
| IV < 0.02 | Useless | **Drop** |
| 0.02 ≤ IV < 0.1 | Weak | Include; monitor |
| 0.1 ≤ IV < 0.3 | Medium | Include |
| 0.3 ≤ IV < 0.5 | Strong | Include |
| IV ≥ 0.5 | **Suspiciously strong** | **Manual review — likely leakage** |

> **IV ≥ 0.5 is a red flag, not a green light.** Any feature with IV ≥ 0.5 after
> leakage removal must be manually reviewed before use. In credit data with already-
> removed payment/recovery columns, this should not occur for legitimate features.
> If it does, re-read `leakage_audit.md`.

---

## 2. Binning Strategy for Numeric Variables

### 2.1 Approach: Optimal monotonic binning

Use **optimal binning with a monotonicity constraint**. The bins are learned by maximising the IV of the resulting bins subject to the constraint that WoE values are monotonically increasing or decreasing across bins.

**Why monotonicity?**
- Credit risk theory expects monotonic relationships: higher FICO = lower risk, higher DTI = higher risk. Non-monotonic WoE bins indicate bin instability or noise.
- Monotonic bins produce logistic regression coefficients with intuitive sign and direction.
- Non-monotonic bins on small samples can overfit and produce reversals on OOT.

**Implementation**: Use the `optbinning` library (`OptimalBinning` class with `monotonic_trend='auto'`). This performs a dynamic-programming search for bin boundaries that maximise IV under a monotonicity constraint.

```python
from optbinning import OptimalBinning

binner = OptimalBinning(
    name=feature_name,
    dtype="numerical",
    solver="cp",               # constraint programming solver
    monotonic_trend="auto",    # infer increasing or decreasing
    min_bin_size=0.05,         # minimum 5% of training samples per bin
    max_n_bins=10,             # maximum 10 bins per feature
    min_event_rate_diff=0.01,  # minimum difference in event rate between adjacent bins
)
binner.fit(X_train[feature], y_train)
```

### 2.2 Minimum bin size: 5% of training samples

- With ~840K training rows, 5% = ~42,000 rows minimum per bin.
- This prevents bins with very small sample sizes from producing unstable WoE estimates.
- Bins with fewer rows are merged with their adjacent bin.

**Rationale**: A bin with 100 events and 50 non-events gives a WoE with a margin of error of ±0.14 (95% CI). A bin with 42,000 rows gives ±0.006. Small bins produce unreliable WoE values that will not hold on OOT.

### 2.3 Maximum bins: 10 per feature

10 bins per numeric variable provides adequate granularity for credit scoring while:
- Preventing overfitting (fewer bins = smoother WoE curve)
- Keeping the scorecard interpretable (a 10-bin scorecard table fits on one page)
- Keeping the total number of parameters manageable

### 2.4 Missing value bin — separate "Special" bin

**Every numeric feature gets an explicit "Missing" bin regardless of missingness rate.**

For features with informative missingness (e.g., `mths_since_last_delinq` — 51.25% null means "no prior delinquency"):
- The Missing bin will have a very different WoE than the non-missing bins
- This WoE captures the "never delinquent" signal without requiring imputation

For features with random missingness (e.g., `bc_util` — 3.37%):
- The Missing bin captures the small subgroup with unavailable bureau data
- Median imputation is applied first (per `feature_engineering.md §2`), so very few rows hit the Missing bin
- The WoE of the Missing bin may be unreliable if < 5% of training data — in this case, the Missing bin is merged with the nearest bin

**Note on Tier 2 imputed features**: Features like `mths_since_last_delinq_imputed` already have missingness imputed to 999. The WoE binner will naturally create a bin containing the 999 values, which corresponds to the "never delinquent" group. No special Missing bin needed for these — the 999 bucket serves that purpose.

### 2.5 Outlier / extreme value bin

Values clipped at the 99th/99.9th percentile (per `feature_engineering.md §2`) do not form a separate bin — they fall into the top or bottom bin naturally. No additional handling needed.

---

## 3. Binning Strategy for Categorical Variables

### 3.1 Rare category grouping (before WoE)

Any category representing < **1% of the TRAIN set** (~8,400 rows) is grouped into a synthetic `"_RARE_"` category.

**Applied to**:
- `addr_state`: 51 categories; several small states may be rare
- `zip_3digit`: 956 categories; many will be rare → this is why zip must be WoE-encoded, not one-hot
- `purpose`: 14 categories; most are above 1%
- `home_ownership`: 6 categories; `OTHER`/`NONE`/`ANY` already collapsed in FE step

**Fitting rule**: Rare-category thresholds are computed from TRAIN only. The list of rare categories is stored in the WoE artifact. When a DEV/OOT observation has a category not seen in TRAIN, it is mapped to `"_UNSEEN_"` which is mapped to the same WoE as `"_RARE_"`.

```python
# In fit_woe: compute rare categories from train
rare_cats = {
    col: set(train[col].value_counts()[train[col].value_counts() < min_count].index)
    for col in cat_features
}
# Store in bins dict
bins[col]["rare_categories"] = list(rare_cats[col])
bins[col]["rare_woe"] = woe_map["_RARE_"]  # WoE of the grouped rare bin
```

### 3.2 Categorical WoE fitting

```python
from optbinning import OptimalBinning

binner = OptimalBinning(
    name=feature_name,
    dtype="categorical",
    solver="mip",                 # mixed-integer programming for categorical
    min_bin_size=0.01,            # 1% minimum for categorical (more bins expected)
    max_n_bins=None,              # no cap — group by WoE similarity instead
    cat_cutpoint=0.01,            # categories < 1% → "Other"
)
```

### 3.3 Binary features

Binary features (0/1 flags) — `has_delinq_2y`, `has_pub_rec`, `has_bankruptcy`, `is_joint_app`, `bureau_extended_missing`, all missing indicators — have exactly 2 bins. No binning algorithm needed:

```python
# For binary features: compute WoE directly
for val in [0, 1]:
    n_ev = (X_train[col] == val) & (y_train == 1)  # events in bin
    n_nev = (X_train[col] == val) & (y_train == 0)  # non-events
    woe_map[val] = log((n_ev/N_ev) / (n_nev/N_nev))
```

### 3.4 Term (36/60 months)

Two values → two bins. Same as binary features above.

### 3.5 `issue_month` (1–12)

Treat as categorical (12 distinct integer values). No ordering imposed — seasonal patterns are not necessarily monotonic (Jan ≠ worst or best). Let WoE group months by similar default rate.

---

## 4. The Leakage Rule — Bin Fitting Must Never See DEV or OOT Targets

This is the most critical implementation constraint in the entire pipeline.

### 4.1 The rule, stated precisely

> **`fit_woe()` receives only TRAIN rows and TRAIN targets.**
> The resulting `bins` object is a static artifact.  
> **`transform_woe()` receives the `bins` object and any dataset (TRAIN, DEV, or OOT)
> — it never receives targets, never modifies bins, never learns anything.**

### 4.2 Why this matters

WoE is a supervised transformation: it computes event rates per bin. If you compute WoE bins on the full dataset (including OOT), the OOT rows' target values influence the bin boundaries and WoE values. When you later evaluate on OOT, the model has already "seen" the OOT targets — this is preprocessing leakage.

**Incorrect (leaks OOT):**
```python
# WRONG — do not do this
bins = fit_woe(df_all, features, target)          # uses all data including OOT
X_train_woe = transform_woe(df_train, bins)       # train is contaminated
X_oot_woe   = transform_woe(df_oot, bins)         # OOT bins shaped by OOT targets
```

**Correct:**
```python
# CORRECT
bins = fit_woe(df_train, features, target)        # TRAIN only
X_train_woe = transform_woe(df_train, bins)
X_dev_woe   = transform_woe(df_dev, bins)
X_oot_woe   = transform_woe(df_oot, bins)
save_woe_bins(bins, "models/woe_bins.json")
```

### 4.3 Within cross-validation (K-fold on TRAIN)

Inside each CV fold, WoE must be refit on the fold's training portion only:

```python
for fold_idx, (tr_idx, val_idx) in enumerate(kfold.split(X_train, y_train)):
    X_fold_train, y_fold_train = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
    X_fold_val = X_train.iloc[val_idx]

    # Refit WoE on this fold's training portion only
    fold_bins = fit_woe(X_fold_train, features, y_fold_train)

    X_fold_train_woe = transform_woe(X_fold_train, fold_bins)
    X_fold_val_woe   = transform_woe(X_fold_val,   fold_bins)

    model.fit(X_fold_train_woe, y_fold_train)
    preds = model.predict_proba(X_fold_val_woe)[:, 1]
    # ...evaluate fold...

# After all folds: refit WoE on FULL TRAIN for the final model
final_bins = fit_woe(X_train, features, y_train)
save_woe_bins(final_bins, "models/woe_bins.json")
```

The fold-level WoE bins are throwaway. The `final_bins` (fit on full TRAIN) are what gets applied to DEV and OOT.

---

## 5. IV Threshold and Feature Selection

### 5.1 Selection rule

After fitting WoE on TRAIN:
1. Compute IV for each feature
2. **Drop features with IV < 0.02** (no predictive information)
3. **Flag features with IV ≥ 0.5** for manual leakage review
4. Keep features with 0.02 ≤ IV < 0.5 — these form the model feature set

### 5.2 How the filter is applied

The IV filter is a property of the `bins` artifact:
```python
bins[feature]["selected"] = bins[feature]["iv"] >= IV_THRESHOLD
```

`transform_woe(..., selected_only=True)` returns only columns where `selected=True`.

The IV threshold (`0.02`) is stored in `configs/features.yaml` — not hardcoded.

### 5.3 What to expect

Based on the data dictionary and credit risk theory, features expected to have strong IV (> 0.1):
- `fico_mid` (FICO is the strongest credit signal)
- `revol_util` (utilization rate)
- `dti` / `effective_dti`
- `mths_since_last_delinq_imputed` (recency of delinquency)
- `credit_age_months`
- `loan_to_income`
- `annual_inc` / `log_annual_inc`
- `inq_last_6mths` (credit-seeking behaviour)
- `mths_since_last_major_derog_imputed`

Features expected to have weak IV (< 0.1, may not survive):
- `issue_month` (seasonal patterns in LC data are weak)
- `disbursement_method` (Cash vs DirectPay)
- `zip_3digit` (geographic signal exists but geography matters less for personal loans)
- Individual `num_tl_*` counts (many highly correlated)

> **Do not pre-select features based on this expectation.** Run the IV pipeline and let
> the data decide. The expectation above is only for sanity-checking the output.

---

## 6. Implementation Contract

This section defines the exact function signatures that `src/features/woe.py` must implement. The signatures enforce the leakage rule at the API level — it is structurally impossible to accidentally pass OOT targets to `fit_woe`.

```python
# src/features/woe.py

from __future__ import annotations
import json
import numpy as np
import pandas as pd
from typing import Optional


# ─── Types ────────────────────────────────────────────────────────────────────

BinArtifact = dict  # keyed by feature name; see schema below


# ─── BinArtifact schema (per feature) ────────────────────────────────────────
#
# bins[feature_name] = {
#     "type":             str,            # "numerical" | "categorical" | "binary"
#     "splits":           list[float],    # bin edges for numerical (optbinning output)
#                                         # OR list of category groups for categorical
#     "woe_map":          dict,           # bin_label (str) -> WoE value (float)
#     "iv":               float,          # total IV for this feature
#     "selected":         bool,           # IV >= threshold
#     "rare_categories":  list[str],      # categorical only: categories grouped as _RARE_
#     "unseen_woe":       float,          # WoE to use for categories not seen in TRAIN
#     "missing_woe":      float,          # WoE of the Missing/null bin
#     "n_bins":           int,            # actual number of bins after fitting
# }


# ─── Core API ─────────────────────────────────────────────────────────────────

def fit_woe(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    features: list[str],
    cat_features: Optional[list[str]] = None,
    binary_features: Optional[list[str]] = None,
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
        BinArtifact:  Dictionary mapping feature name -> bin definition.
                      See BinArtifact schema above.

    Raises:
        ValueError:   If y_train contains values other than 0 and 1.
        ValueError:   If X_train and y_train have different lengths.
        ValueError:   If any feature in `features` is not in X_train.columns.
    """
    ...


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
        DataFrame with WoE-transformed features. Column names are unchanged.
        Original non-selected features are dropped if selected_only=True.
        Rows are never dropped.

    Raises:
        KeyError:   If a feature in bins is not in X.columns.
        ValueError: If bins was not produced by fit_woe() (missing required keys).

    Notes:
        - Unseen categorical values → bins[col]["unseen_woe"]
        - Null values → bins[col]["missing_woe"]
        - Out-of-range numeric values (beyond TRAIN min/max) → boundary bin WoE
    """
    ...


def save_woe_bins(bins: BinArtifact, path: str) -> None:
    """
    Serialize the BinArtifact to a JSON file for reproducibility.

    The saved artifact is human-readable and version-controllable.
    All numeric values are rounded to 6 decimal places.

    Args:
        bins: BinArtifact from fit_woe().
        path: Absolute or relative path to output .json file.
    """
    with open(path, "w") as f:
        json.dump(bins, f, indent=2, default=float)


def load_woe_bins(path: str) -> BinArtifact:
    """
    Load a BinArtifact from a JSON file.

    Args:
        path: Path to .json file created by save_woe_bins().

    Returns:
        BinArtifact dict, ready to pass to transform_woe().
    """
    with open(path) as f:
        return json.load(f)


def compute_iv_table(bins: BinArtifact) -> pd.DataFrame:
    """
    Return a sorted summary table of features, their IVs, and selection status.

    Args:
        bins: BinArtifact from fit_woe().

    Returns:
        DataFrame with columns:
            feature:    str   — feature name
            iv:         float — information value
            n_bins:     int   — number of bins after fitting
            type:       str   — "numerical" | "categorical" | "binary"
            selected:   bool  — True if IV >= threshold
        Sorted by iv descending.

    Example output:
        feature                          iv    n_bins  type         selected
        fico_mid                         0.412     8   numerical    True
        revol_util                       0.287     9   numerical    True
        mths_since_last_delinq_imputed   0.201    10   numerical    True
        ...
        issue_month                      0.009    12   categorical  False
    """
    rows = [
        {
            "feature":  feat,
            "iv":       meta["iv"],
            "n_bins":   meta["n_bins"],
            "type":     meta["type"],
            "selected": meta["selected"],
        }
        for feat, meta in bins.items()
    ]
    return (
        pd.DataFrame(rows)
        .sort_values("iv", ascending=False)
        .reset_index(drop=True)
    )


def validate_no_oot_leakage(
    bins_before: BinArtifact,
    bins_after: BinArtifact,
) -> None:
    """
    Assert that two BinArtifacts are identical (for leakage testing).

    Used in tests to verify that applying transform_woe() to OOT data did not
    modify the bins in-place. Call with bins before and after transform_woe().

    Args:
        bins_before: BinArtifact before transform_woe() call.
        bins_after:  BinArtifact after transform_woe() call.

    Raises:
        AssertionError: If any bin definition differs between before and after.
    """
    assert bins_before == bins_after, (
        "BinArtifact was mutated during transform_woe(). "
        "The bins object must be read-only during transformation."
    )
```

---

## 7. Required Tests for WoE Implementation

These tests are the acceptance criteria before WoE code is used in training.

| Test | What to verify | File |
|---|---|---|
| `test_fit_woe_train_only` | fit_woe() raises ValueError if called with a DataFrame that includes known OOT indices (checked via a test fixture that marks OOT rows) | `tests/test_woe.py` |
| `test_transform_does_not_modify_bins` | bins dict is identical before and after transform_woe() — use `validate_no_oot_leakage()` | `tests/test_woe.py` |
| `test_no_nulls_in_woe_output` | transform_woe() produces zero NaN values (all nulls mapped to missing_woe) | `tests/test_woe.py` |
| `test_unseen_category_handled` | A category in DEV not seen in TRAIN maps to unseen_woe, not NaN | `tests/test_woe.py` |
| `test_iv_threshold_filter` | Features with IV < 0.02 are absent from transform_woe() output when selected_only=True | `tests/test_woe.py` |
| `test_woe_bin_monotonicity` | For all numeric features with selected=True, WoE values are monotonically increasing or decreasing across bins | `tests/test_woe.py` |
| `test_save_load_roundtrip` | save_woe_bins() then load_woe_bins() produces identical BinArtifact | `tests/test_woe.py` |
| `test_suspicious_iv_flag` | compute_iv_table() flags features with IV >= 0.5 with a warning in output | `tests/test_woe.py` |
| `test_min_bin_size_respected` | No bin in any fitted feature contains fewer than min_bin_pct * len(X_train) rows | `tests/test_woe.py` |
| `test_cv_fold_refit` | Inside a CV loop, fold-level bins differ from full-TRAIN bins (they should — different data) but have the same feature keys | `tests/test_woe.py` |

---

## 8. Edge Cases and Decisions

| Edge case | Decision |
|---|---|
| Feature has IV = 0 exactly | Drop (IV < 0.02 rule). This can happen for true constants — should already be dropped in data cleaning, but handle defensively. |
| Bin has 0 events or 0 non-events | WoE is undefined (log(0)). Apply Laplace smoothing: add 0.5 to event and non-event counts in all bins before computing WoE. Document this in the bins artifact. |
| Numeric feature has < 2 unique values after clipping | Cannot be binned; set IV = 0, selected = False, log a warning. |
| Categorical feature with 1 surviving category after rare grouping | All rows map to the same WoE; IV = 0; drop. |
| OOT has categories not in TRAIN (new states, new zip codes) | Map to `unseen_woe` — the pre-stored WoE of the `_RARE_` bin. This is the only safe default. Do NOT refit. |
| Monotonicity cannot be achieved (e.g., U-shaped true relationship) | Set `monotonic_trend='none'` for that specific feature and re-bin. Document the exception in the IV table. Do NOT force monotonicity if it severely reduces IV. |
| Feature is binary (only 0/1 after engineering) | Set `type='binary'`; skip optbinning; compute WoE directly from event rates. Two-bin result. |

---

## 9. Configuration

Store WoE parameters in `configs/features.yaml` (not hardcoded):

```yaml
woe:
  max_bins: 10
  min_bin_pct: 0.05
  rare_pct: 0.01
  iv_threshold: 0.02
  monotonic_trend: "auto"
  laplace_smoothing: 0.5
  suspicious_iv_threshold: 0.5   # flag for manual review
  output_path: "models/woe_bins.json"
  iv_table_path: "models/iv_table.csv"
```

---

## 10. Outputs

After running the WoE pipeline on TRAIN, produce and save:

| Artifact | Path | Format | Purpose |
|---|---|---|---|
| Bin definitions | `models/woe_bins.json` | JSON | Reproducible transformation; applied to DEV/OOT |
| IV table | `models/iv_table.csv` | CSV | Feature selection record; human-readable audit trail |
| Selected features | `models/feature_list.json` | JSON | Ordered list of features that survived IV filter |
| WoE-transformed TRAIN | `data/processed/X_train_woe.parquet` | Parquet | Ready for model training |
| WoE-transformed DEV | `data/processed/X_dev_woe.parquet` | Parquet | Ready for threshold optimization |
| WoE-transformed OOT | `data/processed/X_oot_woe.parquet` | Parquet | Ready for final evaluation (do not touch until model is frozen) |

---

*See `feature_engineering.md` for the feature list that enters this pipeline.*  
*See `architecture.md` for where these files live in the project structure.*  
*See `tests/test_woe.py` for the acceptance test suite.*
