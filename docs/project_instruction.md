# Project Development Instructions — Customer Outcome Prediction & Explainable Scoring

**Project:** Customer Outcome Prediction and Explainable Scoring
**Timeframe:** Aug 2025 – Nov 2025
**Type:** Machine Learning Course / Portfolio Project

---

I want you to help me develop the following Machine Learning project **end-to-end as a serious portfolio/course project**, not as a toy ML notebook.

## Project

**Customer Outcome Prediction and Explainable Scoring**

The project is based on a **2.26M+ record tabular dataset** and should cover the complete machine-learning lifecycle:

* Data cleaning and quality analysis
* Data leakage detection
* Feature engineering
* WoE/IV binning and feature selection
* Baseline Logistic Regression
* Random Forest
* XGBoost
* LightGBM
* Stratified cross-validation
* Hyperparameter optimization
* Class-imbalance handling
* Decision-threshold optimization
* Precision, Recall, F1
* ROC-AUC
* PR-AUC
* Confusion matrices
* Chronological / out-of-time validation
* SHAP explainability
* Feature importance analysis
* Model comparison
* Final model selection
* Reproducible inference pipeline
* Documentation and final report

The original project description claims that the final model achieved on chronological out-of-time validation:

* ROC-AUC: **64.99%**
* Gini: **29.98%**
* KS: **21.92**

Do **not** assume these numbers are correct. They must be reproduced from the actual data and methodology. If reproduction differs, document the difference and explain why.

### Technology Stack

Use Python and, where appropriate:

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* LightGBM
* SHAP
* Optuna
* SciPy
* Matplotlib

You may recommend additional libraries only when there is a clear technical reason.

---

# IMPORTANT: HOW I WANT YOU TO WORK

Do NOT immediately start writing the complete implementation.

I want this project developed using a **documentation-first, staged engineering process**.

The project should first be completely understood and specified.

Only after the specification and architecture are sufficiently complete should implementation begin.

Think like a senior ML engineer / ML researcher reviewing a real project.

Your priorities are:

1. Correctness
2. Reproducibility
3. Statistical validity
4. Avoiding data leakage
5. Clear architecture
6. Explainability
7. Maintainability
8. Good documentation
9. Computational efficiency for a 2.26M+ row dataset
10. Interview-readiness

Do not optimize merely for producing code quickly.

---

# PHASE 0 — PROJECT DISCOVERY

Before designing anything, inspect the repository/files/dataset that I provide.

Determine:

* What files exist
* Dataset format
* Dataset size
* Number of rows
* Number of columns
* Target variable
* Feature types
* Numerical features
* Categorical features
* Dates/time columns
* IDs
* Potential leakage columns
* Missing-value patterns
* Duplicate patterns
* Class distribution
* Cardinality of categorical variables
* Potential train/test indicators
* Temporal information
* Any existing notebooks/code
* Existing preprocessing
* Existing models
* Existing results

Do not make assumptions when the data can answer the question.

Create a concise **Project Discovery Report** before proceeding.

---

# PHASE 1 — REQUIREMENTS & PROJECT SPECIFICATION

Create a detailed project specification before implementation.

Create documentation covering:

## 1. Project objective

Clearly define:

* What prediction problem we are solving
* What exactly constitutes the target/outcome
* What one row represents
* What a prediction means
* What the model is expected to optimize
* Who would theoretically use the prediction
* What decisions the score could support

## 2. ML problem formulation

Define:

* Binary/multiclass/regression problem
* Input X
* Target y
* Prediction output
* Probability interpretation
* Classification threshold
* Positive/negative class

## 3. Success criteria

Define:

### Primary metrics

* ROC-AUC
* PR-AUC
* KS
* Gini

### Secondary metrics

* Precision
* Recall
* F1
* Confusion matrix
* Calibration where appropriate

Explain why each metric matters.

## 4. Constraints

Document:

* 2.26M+ records
* Class imbalance
* Computational constraints
* Memory considerations
* Temporal validation
* Leakage prevention
* Reproducibility

---

# PHASE 2 — DATA UNDERSTANDING SPECIFICATION

Before writing preprocessing code, create a complete data dictionary.

For every feature, document where possible:

| Feature | Type | Meaning | Missing % | Cardinality | Leakage Risk | Treatment |
| ------- | ---- | ------- | --------: | ----------: | ------------ | --------- |

Also identify:

* Identifier columns
* Timestamp columns
* Target column
* Constant columns
* Near-constant columns
* Duplicate columns
* Highly correlated columns
* Suspicious post-outcome variables
* Variables that would not be available at prediction time

## Leakage audit

This is extremely important.

For every suspicious feature, ask:

> "Would this information genuinely be available at the exact moment the prediction is supposed to be made?"

If not, mark it as leakage and exclude it.

Do not merely search for obvious target leakage. Consider:

* Temporal leakage
* Aggregation leakage
* Duplicate/entity leakage
* Future information
* Post-event variables
* Train/test contamination
* Target-derived variables
* Preprocessing leakage

Document every leakage decision.

---

# PHASE 3 — DATA SPLITTING STRATEGY

Design the splitting methodology before model training.

We need to distinguish between:

### Development data

Used for:

* preprocessing
* feature engineering
* cross-validation
* hyperparameter tuning
* model selection

### Out-of-time validation

Used as a realistic final evaluation.

If the dataset contains a meaningful time variable, design a chronological split.

Do NOT randomly mix future observations into training if this would not reflect the real prediction setting.

Explain:

* Why the split was chosen
* Date boundaries
* Train size
* Validation size
* OOT size
* Class distribution in each split
* Potential distribution shift

All preprocessing operations that learn from data must be fitted only on the appropriate training folds.

---

# PHASE 4 — FEATURE ENGINEERING DESIGN

Before implementation, create a feature-engineering specification.

Consider:

* Missing-value indicators
* Numeric transformations
* Log transformations where justified
* Ratios
* Aggregations
* Date/time features
* Recency/frequency features if applicable
* Categorical encoding
* Rare-category handling
* Binning
* Interaction features where justified

Every engineered feature should have:

* Name
* Formula/logic
* Motivation
* Input columns
* Leakage risk
* Expected data type
* Handling of missing values

Avoid creating thousands of arbitrary features without justification.

---

# PHASE 5 — WoE / IV METHODOLOGY

Design and document the WoE/IV pipeline carefully.

Explain:

* What WoE means
* What IV means
* Why it is being used
* Binning strategy
* Numerical variable binning
* Categorical variable treatment
* Missing-value treatment
* Special bins
* Monotonicity if applicable
* Rare categories
* IV thresholds
* How bins are learned only from training data
* How validation/OOT data is transformed

Do not calculate bins using the complete dataset before splitting.

Create a reproducible WoE transformation pipeline.

Document all selected variables and their IV values.

---

# PHASE 6 — BASELINE MODEL

Implement a simple baseline first.

Start with:

**Logistic Regression**

The baseline should establish:

* ROC-AUC
* PR-AUC
* Gini
* KS
* Precision
* Recall
* F1
* Confusion matrix
* Training time
* Inference time

Do not jump directly to XGBoost/LightGBM.

The baseline is important for understanding whether more complex models actually provide meaningful improvement.

---

# PHASE 7 — TREE MODEL BENCHMARK

Implement and compare:

1. Logistic Regression
2. Random Forest
3. XGBoost
4. LightGBM

Use the same evaluation framework wherever scientifically appropriate.

Create a standard experiment interface so models can be compared consistently.

For every model record:

* Configuration
* Features
* Training dataset
* Validation methodology
* Hyperparameters
* Training time
* ROC-AUC
* PR-AUC
* Gini
* KS
* Precision
* Recall
* F1
* Confusion matrix

---

# PHASE 8 — CROSS-VALIDATION

Use stratified cross-validation where appropriate for development experiments.

Clearly distinguish:

**Cross-validation performance**

from

**Final chronological out-of-time performance.**

Do not use OOT data for model selection or hyperparameter tuning.

Record:

* Mean metric
* Standard deviation
* Fold-level results

Explain variance between folds.

---

# PHASE 9 — CLASS IMBALANCE

Analyze class imbalance first.

Do not blindly apply oversampling or class weights.

Compare appropriate approaches such as:

* Class weights
* Scale positive weight
* Threshold optimization
* Sampling strategies where justified

Evaluate whether the approach improves the metrics relevant to the business problem.

---

# PHASE 10 — HYPERPARAMETER OPTIMIZATION

Use Optuna where appropriate.

Define:

* Search space
* Objective metric
* Number of trials
* Cross-validation strategy
* Early stopping
* Random seed
* Computational constraints

Do not optimize on the OOT set.

Save the best configuration and experiment metadata.

Make experiments reproducible.

---

# PHASE 11 — DECISION THRESHOLD OPTIMIZATION

Do not automatically use threshold = 0.5.

Analyze the probability distribution.

Evaluate thresholds and their impact on:

* Precision
* Recall
* F1
* TP
* FP
* TN
* FN

Select a threshold according to the actual project objective.

Clearly distinguish:

**Probability ranking quality**

from

**classification threshold performance.**

---

# PHASE 12 — FINAL MODEL EVALUATION

After model selection is frozen, evaluate exactly once on the chronological OOT dataset.

Produce a final evaluation report containing:

* ROC-AUC
* PR-AUC
* Gini
* KS
* Precision
* Recall
* F1
* Confusion matrix
* ROC curve
* Precision-Recall curve
* Score/probability distribution
* Calibration analysis if appropriate

Calculate:

**Gini = 2 × ROC-AUC − 1**

and verify the reported:

* ROC-AUC ≈ 64.99%
* Gini ≈ 29.98%
* KS ≈ 21.92

Again, do not manipulate the methodology merely to reproduce these numbers.

If the results differ, report the actual results.

---

# PHASE 13 — EXPLAINABILITY

Use SHAP to explain the final model.

Produce:

### Global explanations

* SHAP feature importance
* Mean absolute SHAP values
* Summary plot
* Feature impact direction where meaningful

### Local explanations

For selected individual predictions:

* Prediction probability
* Top positive contributors
* Top negative contributors
* Original feature values
* SHAP contributions

Explain the difference between:

* Feature importance
* SHAP importance
* Correlation
* Causation

Do not make causal claims from SHAP.

---

# PHASE 14 — MODEL COMPARISON

Create a final model comparison table.

Example:

| Model | ROC-AUC | PR-AUC | Gini | KS | Precision | Recall | F1 | Training Time |
| ----- | ------: | -----: | ---: | -: | --------: | -----: | -: | ------------: |

Then explain:

* Which model won
* Why it won
* Whether improvement justified complexity
* Whether interpretability changed the decision
* Whether OOT performance degraded
* Whether there was evidence of overfitting

---

# PHASE 15 — SOFTWARE ARCHITECTURE

Do not keep everything inside one huge notebook.

Design a clean project structure such as:

```text
customer-outcome-prediction/
│
├── README.md
├── requirements.txt
├── pyproject.toml
│
├── configs/
│   ├── data.yaml
│   ├── model.yaml
│   └── experiment.yaml
│
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_data_quality.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_woe_iv.ipynb
│   ├── 05_baseline.ipynb
│   ├── 06_model_benchmark.ipynb
│   ├── 07_hyperparameter_tuning.ipynb
│   ├── 08_threshold_optimization.ipynb
│   └── 09_explainability.ipynb
│
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── evaluation/
│   ├── explainability/
│   └── utils/
│
├── tests/
│
├── experiments/
│
├── models/
│
├── reports/
│
└── docs/
    ├── project_specification.md
    ├── data_dictionary.md
    ├── leakage_audit.md
    ├── validation_strategy.md
    ├── feature_engineering.md
    ├── woe_iv.md
    ├── model_experiments.md
    ├── explainability.md
    ├── final_results.md
    └── architecture.md
```

Adapt this structure based on the actual repository and dataset.

---

# PHASE 16 — DOCUMENTATION

Documentation is a first-class deliverable.

Create detailed documentation for:

1. Project overview
2. Business/ML problem
3. Dataset
4. Data dictionary
5. Data cleaning
6. Leakage analysis
7. Feature engineering
8. WoE/IV
9. Validation strategy
10. Baseline model
11. Model benchmarking
12. Hyperparameter optimization
13. Threshold optimization
14. Final evaluation
15. Explainability
16. Architecture
17. Reproducibility
18. Limitations
19. Future improvements

Every major technical decision should answer:

> What did we do?

> Why did we do it?

> What alternatives did we consider?

> What evidence justified the decision?

---

# PHASE 17 — TESTING

Create tests for important functionality.

At minimum test:

* Data loading
* Schema validation
* Missing-value handling
* Feature engineering
* WoE transformation
* Train/validation separation
* Metric calculations
* Gini calculation
* KS calculation
* Threshold optimization
* Prediction pipeline

Add sanity checks for leakage.

Example:

A transformation fitted on training data must produce identical transformations when applied later to validation/OOT data without refitting.

---

# PHASE 18 — REPRODUCIBILITY

Everything should be reproducible.

Control:

* Random seeds
* Dataset version
* Feature configuration
* Model configuration
* Optuna configuration
* Environment dependencies
* Experiment IDs

Save:

* Model artifacts
* Preprocessing artifacts
* Feature lists
* WoE bin definitions
* Hyperparameters
* Metrics
* Threshold
* Training metadata

A fresh environment should be able to reproduce the final pipeline.

---

# PHASE 19 — PERFORMANCE / LARGE DATASET CONSIDERATIONS

The dataset contains 2.26M+ records.

Do not write inefficient code casually.

Consider:

* Memory usage
* Appropriate dtypes
* Chunked loading where needed
* Avoiding unnecessary dataframe copies
* Efficient categorical handling
* Vectorized operations
* Sparse representations where useful
* Efficient cross-validation
* Model training time
* Disk usage

Before running an expensive experiment, estimate its computational cost.

---

# PHASE 20 — DEVELOPMENT WORKFLOW

Work in small, verifiable stages.

For each stage:

1. Explain the objective.
2. State assumptions.
3. Identify relevant files.
4. Implement the smallest correct version.
5. Run/describe validation.
6. Check outputs.
7. Document the result.
8. Only then proceed.

Do not silently modify unrelated parts of the project.

Do not rewrite working code unnecessarily.

When changing an existing implementation, explain:

* What changed
* Why it changed
* What could break
* How it was validated

---

# CRITICAL RULES

## Rule 1 — Do not hallucinate

If something is unknown, say:

> "I need to inspect the dataset/repository to determine this."

Do not invent column names, target definitions, dataset statistics, or results.

## Rule 2 — Never fake results

Never claim a metric was achieved unless it was actually calculated from the data.

## Rule 3 — Leakage is more important than performance

A suspiciously high score should trigger an investigation rather than celebration.

## Rule 4 — OOT must remain untouched

Do not use chronological OOT data for:

* Feature selection
* Hyperparameter tuning
* Threshold optimization
* Model selection

unless explicitly designing a separate experiment and clearly labeling it.

## Rule 5 — Prefer simple solutions when performance is comparable

If Logistic Regression performs almost as well as LightGBM, explain whether the additional complexity is justified.

## Rule 6 — Explain every major ML decision

I should be able to explain this project in an interview.

## Rule 7 — Keep code modular

Avoid one giant notebook containing the entire project.

## Rule 8 — Documentation and implementation must stay synchronized

When implementation changes, update the relevant documentation.

---

# HOW I WANT YOU TO COMMUNICATE

Act as a senior ML engineer mentoring me.

For every major decision, explain:

### Decision

What are we doing?

### Reason

Why are we doing it?

### Alternatives

What else could we have done?

### Risk

What could go wrong?

### Validation

How will we know it worked?

Keep explanations technically rigorous but understandable.

---

# FIRST TASK — DO NOT IMPLEMENT YET

Your FIRST response should NOT contain a huge amount of implementation code.

Instead:

## Step 1

Inspect the project/repository/dataset that I provide.

## Step 2

Create:

```text
docs/project_specification.md
docs/data_dictionary.md
docs/leakage_audit.md
docs/validation_strategy.md
docs/architecture.md
```

or propose the exact contents of these documents if the repository is not yet available.

## Step 3

Create a complete project roadmap with phases and dependencies.

## Step 4

Identify uncertainties/questions that can only be answered after inspecting the actual data.

## Step 5

Give me the proposed project architecture.

## Step 6

Give me a list of implementation stages.

For each stage specify:

* Objective
* Inputs
* Outputs
* Files changed
* Tests
* Expected result
* Completion criteria

## Step 7

STOP.

Do not begin implementing the entire project until I approve the architecture/plan.

After I approve it, proceed one stage at a time.

---

# IMPORTANT: DO NOT OVERENGINEER

This is a Machine Learning course/portfolio project, not a production enterprise platform.

The architecture should demonstrate strong engineering practices without introducing unnecessary infrastructure.

Prioritize:

**correct ML methodology > reproducibility > clarity > maintainability > complexity.**

The final project should be something I can confidently put on my resume and defend in a technical interview.