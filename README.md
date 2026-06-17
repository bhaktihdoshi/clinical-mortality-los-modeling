# Clinical Mortality and Length-of-Stay Modeling

Leakage-safe clinical machine learning project for in-hospital mortality triage and length-of-stay (LOS) planning on MIMIC-style electronic health record (EHR) admissions data.

This repository packages a reproducible, end-to-end portfolio workflow behind a broader clinical ML study. The public notebook runs leakage-safe preprocessing, trains XGBoost and a lightweight Transformer-style tabular model, and compares mortality and LOS metrics.

## Why This Project Matters

Hospitals need risk models that support two related decisions: which patients may require urgent clinical escalation, and which admissions may create capacity pressure through longer stays. Mortality prediction is a rare-event problem, where accuracy alone is misleading; LOS prediction is a resource-planning problem, where large errors on long stays are operationally expensive.

The original exploratory workflow imputed missing values before splitting data into train, validation, and test sets. That leaks distributional information from held-out patients into training. This repo fixes that by splitting first at the patient level, then fitting imputers and encoders only on the training split.

For job applications, the project demonstrates practical clinical ML judgement:

- preventing patient and preprocessing leakage before modeling
- treating mortality as an imbalanced clinical safety task, not a generic accuracy problem
- using LOS as an interpretable regression target for operational planning
- translating ICD diagnosis codes into clinically meaningful CCS groupings
- training and evaluating XGBoost and Transformer-style models from local raw admissions data
- documenting model-development assumptions through reproducible audit files and metrics tables
- separating public code from patient-level/generated data

## Repository Layout

```text
.
├── src/preprocess_split_first.py      # Correct split-first preprocessing pipeline
├── src/modeling.py                    # XGBoost, Transformer, and metric helpers
├── notebooks/clinical_ml_pipeline.ipynb # Runnable walkthrough notebook
├── tests/                             # Regression tests for leakage safeguards
├── data/raw/                          # Local input data, ignored by git
├── data/processed/                    # Generated split/model matrices, ignored by git
└── requirements.txt                   # Minimal Python dependencies
```

## Project Context

The original study audited four model families on the same EHR cohort:

- **XGBoost:** structured tabular baseline for interpretable mortality prediction
- **LSTM:** recurrent sequence baseline over diagnosis-history tokens
- **Multimodal Transformer:** fusion of CCS diagnosis sequences and tabular admission features
- **BioGPT:** biomedical language-model comparator using narrative-style admission representations

The modeling objective was not to crown one universal model. It was to test architecture-task fit:

- XGBoost is a strong candidate when sparse tabular comorbidity and diagnosis-burden features drive mortality risk and explainability matters.
- Multimodal Transformer is better aligned with LOS prediction when diagnosis history and structured admission context jointly influence recovery time.
- BioGPT is promising as a recall-oriented second reader, but needs stronger fine-tuning, calibration, and explainability before primary use.

This public repo keeps the job-application version focused: it includes leakage-safe patient-level preprocessing plus runnable XGBoost and Transformer-style modeling. LSTM and BioGPT remain part of the broader study context, but are intentionally left out of this repository to keep the code concise and reproducible for reviewers.

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

## Notebook

For the complete coding workflow, open:

```text
notebooks/clinical_ml_pipeline.ipynb
```

The notebook starts from `data/raw/unified_patient_admissions_ccs.csv` and runs:

- leakage-safe 70/15/15 patient-level splitting
- train-only imputation and one-hot encoding
- XGBoost mortality classification and LOS regression
- lightweight tabular Transformer mortality classification and LOS regression
- metric comparison saved to `reports/model_comparison_metrics.csv`
- final no-patient-overlap leakage check

## Leakage Controls

- Admissions are split by `subject_id`, so one patient cannot appear in multiple splits.
- Imputers are fit only on the training split.
- One-hot category levels are learned only from training data; unseen validation/test categories are ignored safely.
- Mortality features exclude post-outcome/discharge columns such as LOS, discharge time, death time, and target flags.
- High-cardinality text/code-sequence columns are excluded from the tabular matrix unless deliberately represented.
- The fitted preprocessing object is saved as `preprocessor_train_only.joblib` for reproducible transforms.

## Evaluation Mindset

The broader study used task-specific evaluation rather than one-size-fits-all metrics:

- **Mortality:** AUPRC, AUROC, F1, recall/sensitivity, specificity, and validation-selected thresholds. AUPRC and recall are emphasized because mortality is rare and missed deaths are clinically costly.
- **LOS:** RMSE, MAE, and R². RMSE is useful because large long-stay errors matter for bed and staffing capacity.
- **Robustness:** patient-grouped validation to prevent admission-level leakage.
- **Fairness:** subgroup recall checks across demographic groups, with caution around unstable estimates when positive cases are sparse.

These choices reflect the deployment reality: a clinically useful model must be robust, interpretable, calibrated, and fair enough for its intended decision point.

## Notes on Data

Patient-level datasets and generated matrices are intentionally ignored by git. Keep source data local under `data/raw/` and regenerate outputs with the commands above.

## Portfolio Framing

This project is best presented as a clinical ML reproducibility and model-audit case study. Strong resume bullets:

- Built an end-to-end clinical ML notebook for mortality and length-of-stay prediction, using patient-level splitting, train-only imputation/encoding, XGBoost, and a Transformer-style tabular neural model.
- Audited architecture-task fit across XGBoost and Transformer-style modeling approaches for rare-event mortality classification and LOS regression, with broader study context covering LSTM and BioGPT comparators.
- Applied clinical ML evaluation principles including AUPRC for imbalanced mortality, RMSE/MAE for LOS, grouped validation, subgroup fairness checks, and explainability-focused model selection.
