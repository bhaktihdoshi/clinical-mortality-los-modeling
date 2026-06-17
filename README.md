# Clinical Mortality and Length-of-Stay Modeling

Leakage-safe machine learning pipeline for hospital mortality classification and length-of-stay regression on MIMIC-style admissions data.

## Why This Project Matters

The original exploratory workflow imputed missing values before splitting data into train, validation, and test sets. That leaks distributional information from held-out patients into training. This repo fixes that by splitting first at the patient level, then fitting imputers and encoders only on the training split.

The result is a cleaner, recruiter-friendly project that demonstrates:

- clinical ML data hygiene and leakage prevention
- subject-level train/validation/test splitting
- train-only imputation and categorical encoding
- reproducible artifacts and audit files
- standardized model comparison outputs for XGBoost, LSTM, Transformer, and BioGPT experiments

## Repository Layout

```text
.
├── src/preprocess_split_first.py      # Correct split-first preprocessing pipeline
├── tests/                             # Regression tests for leakage safeguards
├── data/raw/                          # Local input data, ignored by git
├── data/processed/                    # Generated split/model matrices, ignored by git
├── reports/                           # Optional exported summaries
├── figures/                           # Optional exported plots
├── build_team_model_comparison.py     # Combines standardized model metric files
├── make_corrected_tabular_splits.py   # Legacy split repair utility, now path-configurable
└── final_model_artifact_manifest.*    # Record of official model artifacts
```

## Quickstart

Create an environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place your source admissions CSV in `data/raw/`. The pipeline expects `subject_id`, `hadm_id`, mortality status, admission/discharge timestamps or a length-of-stay target, and clinical feature columns.

Run leakage-safe preprocessing:

```bash
python -m src.preprocess_split_first \
  --input data/raw/unified_patient_admissions_ccs.csv \
  --output-dir data/processed \
  --random-state 42
```

Outputs:

- `train_model_matrix.csv`, `val_model_matrix.csv`, `test_model_matrix.csv`
- `train_raw_split.csv`, `val_raw_split.csv`, `test_raw_split.csv`
- `preprocessor_train_only.joblib`
- `split_summary.csv`
- `preprocessing_audit.json`

Run tests:

```bash
pytest
```

## Leakage Controls

- Admissions are split by `subject_id`, so one patient cannot appear in multiple splits.
- Imputers are fit only on the training split.
- One-hot category levels are learned only from training data; unseen validation/test categories are ignored safely.
- Mortality features exclude post-outcome/discharge columns such as LOS, discharge time, death time, and target flags.
- The fitted preprocessing object is saved as `preprocessor_train_only.joblib` for reproducible transforms.

## Notes on Data

Patient-level datasets and generated matrices are intentionally ignored by git. Keep source data local under `data/raw/` and regenerate outputs with the commands above.

## Portfolio Framing

This project is best presented as a clinical ML reproducibility and leakage-prevention case study. A strong resume bullet:

> Built a leakage-safe clinical ML pipeline for mortality and length-of-stay prediction, using patient-level splitting, train-only preprocessing, reproducible audits, and standardized model comparison across tree-based, sequence, transformer, and LLM baselines.
