# Validation Strategy — Customer Outcome Prediction

**Dataset**: Lending Club Accepted Loans 2007–2018 Q4  
**issue_d range**: June 2007 – December 2018 (139 unique months)  
**Audit date**: 2026-09-04

---

## 1. The Temporal Constraint

The Lending Club dataset has a fundamental property that makes random splitting incorrect: **loans are issued over time, and later observations depend on the real-world economic environment of earlier ones**. A model trained on randomly shuffled data will see future borrower profiles mixed into its training set, which inflates validation metrics and does not reflect real deployment performance.

**All splits must be strictly chronological on `issue_d`.**

---

## 2. The Maturity Problem

The dataset is a snapshot as of December 2018. Loans issued close to that date have not had time to resolve:

| Vintage | Time elapsed by Dec 2018 | Likely resolution |
|---|---|---|
| 2007–2013 | 5–11 years | Nearly all resolved (36-mo and 60-mo matured) |
| 2014 | ~4–5 years | Virtually all resolved |
| 2015 | ~3–4 years | 36-month mostly resolved; 60-month largely resolved |
| 2016 | ~2–3 years | 36-month at/near maturity; 60-month still running |
| 2017 | ~1–2 years | Many still Current; early defaults visible |
| 2018 | 0–12 months | Overwhelmingly Current |

This means **the 2017 and 2018 vintages are dominated by "Current" loans** that cannot be used in supervised training (their outcome is unknown). After applying the target encoding from `project_specification.md` (excluding Current, Late, In Grace Period), the usable pool shrinks substantially for recent vintages.

**Issue_d counts by year (full dataset):**

| Year | Total loans |
|---|---:|
| 2007 | 603 |
| 2008 | 2,393 |
| 2009 | 5,281 |
| 2010 | 12,537 |
| 2011 | 21,721 |
| 2012 | 53,367 |
| 2013 | 134,814 |
| 2014 | 235,629 |
| 2015 | 421,095 |
| 2016 | 434,407 |
| 2017 | 443,579 |
| 2018 | 495,242 |

The modeling set (1,348,099 resolved loans) is dominated by pre-2017 vintages. Post-2016 vintages contribute mostly their fast-resolving defaults (borrowers who charged off within 1-2 years) and early payoffs, not a representative sample.

---

## 3. Proposed Split Design

### Split boundaries (all on issue_d)

```
June 2007 ─────────────────────────────────── December 2018
│                              │              │
│←──────── TRAIN ─────────────→│←── DEV/VAL ─→│←── OOT ──→│
│  Jun 2007 – Dec 2015         │ Jan 2016     │ Jan 2017   │
│                              │ – Dec 2016   │ – Dec 2018 │
│                              │              │            │
│   ~8.5 years                 │   1 year     │   2 years  │
```

### Boundaries in code

```python
TRAIN_END   = "2015-12-31"   # issue_d <= this date -> TRAIN
DEV_START   = "2016-01-01"
DEV_END     = "2016-12-31"   # issue_d in [DEV_START, DEV_END] -> DEV
OOT_START   = "2017-01-01"
OOT_END     = "2018-12-31"   # issue_d >= OOT_START -> OOT
```

All comparisons use the parsed issue_d (datetime), not string comparison.

---

## 4. Justification for These Boundaries

### Why TRAIN ends at December 2015

- By December 2018 (the dataset snapshot), a 36-month loan issued in December 2015 has exactly 36 months of seasoning — it is at maturity. A 60-month loan from Dec 2015 is at 36 months, which means many will have either defaulted (visible) or are still Current (excluded). This vintage is the latest that has reasonable resolution rate.
- The 2007–2015 window contains 867,239 total loans — after excluding unresolved, this is the core training population.
- Early years (2007–2011) are small (32,000 total) but provide out-of-distribution examples from the pre-crisis and early post-crisis era.

### Why DEV is 2016

- 2016 is one full calendar year with a substantial loan count (~434K total).
- Loans from 2016: 36-month matures Dec 2019 (slightly beyond snapshot), but early defaults (2016–2018) are captured. The resolved DEV set is a truncated but valid cross-section.
- DEV serves as in-time held-out data for model selection and threshold optimization. It must not be used for hyperparameter tuning (use CV on TRAIN only).

### Why OOT is 2017–2018

- This is the genuine out-of-time test, mimicking production: "the model was built on pre-2016 data, does it still work in 2017-2018?"
- 2017-2018 is the most recent period and will show the most distribution shift.
- **Important caveat**: After excluding Current loans, the OOT resolved set is heavily biased toward early defaults (borrowers who charged off within 12-24 months of a 2017-2018 issue date). The non-default class in OOT will be underrepresented (most fully-paid loans from 2018 haven't paid off yet). This means OOT metrics will overstate default rate vs. reality. **Document this caveat prominently in the final report.**

---

## 5. Expected Split Sizes (Estimates)

The exact resolved counts per vintage require a query, but estimated:

| Split | issue_d range | Total loans | Est. resolved (excl. Current/Late/Grace) | Est. default rate |
|---|---|---:|---:|---:|
| **TRAIN** | Jun 2007 – Dec 2015 | ~867K | ~840K | ~18–20% |
| **DEV** | Jan 2016 – Dec 2016 | ~434K | ~280K | ~18–21% |
| **OOT** | Jan 2017 – Dec 2018 | ~939K | ~230K | ~22–28%* |

*OOT default rate is expected to be elevated because Current (non-default) loans from 2017-2018 are excluded, artificially increasing the proportion of defaults in the resolved OOT subset.

> **Action required**: Run the actual resolved-count query by vintage after target encoding
> to confirm these estimates. Replace estimates with actuals in this document before training.

---

## 6. Cross-Validation Within TRAIN

For hyperparameter tuning and model selection, use **5-fold stratified k-fold** on the TRAIN set only.

- Use `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
- Stratification preserves the ~20% default rate in each fold
- Report mean ± std across folds for all primary metrics
- DEV and OOT are held completely out during CV

> Note: Strict temporal CV (TimeSeriesSplit) is an alternative but splits unevenly
> given the exponential growth in loan volume (2015 alone is ~50% of pre-2016 data).
> For this dataset size and the fact that we are not predicting time-series outcomes
> (we are predicting loan outcome given features at a point in time), stratified k-fold
> on the TRAIN window is acceptable and produces more stable estimates.
> A sensitivity check using TimeSeriesSplit is recommended as a secondary experiment.

---

## 7. Preprocessing Discipline

| Operation | Fitted on | Applied to |
|---|---|---|
| Imputers (median, mode) | TRAIN only | TRAIN, DEV, OOT |
| Scalers (StandardScaler, MinMaxScaler) | TRAIN only | TRAIN, DEV, OOT |
| WoE/IV binning | TRAIN folds only (never on full TRAIN before splitting) | TRAIN, DEV, OOT |
| Rare category grouping | TRAIN only | TRAIN, DEV, OOT |
| Feature selection (IV thresholds) | TRAIN only | TRAIN, DEV, OOT |
| Threshold optimization | DEV only | OOT (applied, not re-optimized) |

**The OOT set must be treated as if it were production data seen for the first time.**

---

## 8. Expected Distribution Shift

The Lending Club loan portfolio changed materially between 2007 and 2018:

- **Loan volume**: Exponential growth (~600 loans in 2007 → ~495K in 2018)
- **Loan amounts**: Rising mean loan amounts over time
- **Purpose mix**: Debt consolidation became increasingly dominant
- **FICO distribution**: Post-2012 tightening after LC regulatory changes
- **Term mix**: 60-month loans grew as a share of the book
- **Macroeconomic context**: 2007-2009 loans originated in a pre-crisis environment; 2010-2015 in the recovery; 2016-2018 in expansion

**Expected behaviour**: The model may show reduced discrimination on OOT if:
1. The 2017-2018 economic environment differs from the training window
2. LC's underwriting criteria changed, shifting the approved population
3. The OOT truncation bias (mostly early-resolving loans) creates a non-representative sample

**Monitoring**: Compare feature distributions between TRAIN and OOT using Population Stability Index (PSI) after the data pipeline is built.

---

## 9. Reproducing the Reported Metrics

The original project claimed:
- ROC-AUC: 64.99% on chronological OOT
- Gini: 29.98%
- KS: 21.92

These numbers are consistent with the OOT being the 2016 or 2017 vintage. Without the original code, the exact split boundary is unknown. The most likely scenario is that the original project used:
- Train: pre-2016
- OOT: 2016 or a subset thereof

This project's split (OOT = 2017-2018) may produce different numbers — possibly lower due to later-vintage truncation bias. If numbers differ materially, document the methodological difference.

---

*See also: `project_specification.md` for target definition, `leakage_audit.md` for feature exclusions.*
