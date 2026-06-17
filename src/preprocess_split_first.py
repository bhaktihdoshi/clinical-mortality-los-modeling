"""Create leakage-safe train/validation/test modeling matrices.

This pipeline intentionally splits admissions before fitting any imputation or
categorical encoding step. The fitted preprocessing artifact is learned from
the training split only, then applied unchanged to validation and test.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ID_COLUMNS = ["subject_id", "hadm_id"]
MORTALITY_TARGET = "target_hospital_expire_flag"
LOS_TARGET = "length_of_stay_days"
POST_OUTCOME_COLUMNS = {
    "length_of_stay_days",
    "los_days_raw",
    "dischtime",
    "deathtime",
    "hospital_expire_flag",
    "target_hospital_expire_flag",
}
DEFAULT_CATEGORICAL_COLUMNS = {
    "gender",
    "race",
    "admission_type",
    "admission_location",
    "insurance",
    "language",
    "marital_status",
    "primary_ccs_lvl1_label",
}
MAX_CATEGORICAL_CARDINALITY = 50


@dataclass(frozen=True)
class SplitConfig:
    validation_size: float = 0.15
    test_size: float = 0.15
    random_state: int = 42


def normalize_categorical(series: pd.Series) -> pd.Series:
    cleaned = series.fillna("Unknown").astype(str).str.strip()
    return cleaned.str.replace(r"\s+", " ", regex=True)


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if "admittime" in result.columns and "dischtime" in result.columns:
        admit = pd.to_datetime(result["admittime"], errors="coerce")
        discharge = pd.to_datetime(result["dischtime"], errors="coerce")
        los_days_raw = ((discharge - admit).dt.total_seconds() / 86400).round(2)
        result["los_days_raw"] = los_days_raw
        result[LOS_TARGET] = los_days_raw.clip(lower=0)

    if MORTALITY_TARGET not in result.columns and "hospital_expire_flag" in result.columns:
        result[MORTALITY_TARGET] = result["hospital_expire_flag"]

    missing_targets = [col for col in [MORTALITY_TARGET, LOS_TARGET] if col not in result.columns]
    if missing_targets:
        raise ValueError(f"Missing required target columns: {missing_targets}")

    result[MORTALITY_TARGET] = pd.to_numeric(result[MORTALITY_TARGET], errors="coerce").fillna(0).astype(int)
    result[LOS_TARGET] = pd.to_numeric(result[LOS_TARGET], errors="coerce")
    return result


def subject_level_labels(df: pd.DataFrame) -> pd.DataFrame:
    labels = (
        df.groupby("subject_id", as_index=False)[MORTALITY_TARGET]
        .max()
        .rename(columns={MORTALITY_TARGET: "subject_mortality"})
    )
    return labels


def split_subjects(df: pd.DataFrame, config: SplitConfig) -> dict[str, set[int]]:
    labels = subject_level_labels(df)
    train_subjects, temp_subjects = train_test_split(
        labels,
        test_size=config.validation_size + config.test_size,
        random_state=config.random_state,
        stratify=labels["subject_mortality"],
    )

    relative_test_size = config.test_size / (config.validation_size + config.test_size)
    val_subjects, test_subjects = train_test_split(
        temp_subjects,
        test_size=relative_test_size,
        random_state=config.random_state,
        stratify=temp_subjects["subject_mortality"],
    )

    return {
        "train": set(train_subjects["subject_id"].astype(int)),
        "val": set(val_subjects["subject_id"].astype(int)),
        "test": set(test_subjects["subject_id"].astype(int)),
    }


def choose_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    excluded = set(ID_COLUMNS) | POST_OUTCOME_COLUMNS
    candidate_columns = [col for col in df.columns if col not in excluded]
    numeric_columns = [
        col
        for col in candidate_columns
        if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col])
    ]
    categorical_columns = []
    dropped_categorical_columns = []
    for col in [col for col in candidate_columns if col not in numeric_columns]:
        unique_count = df[col].nunique(dropna=True)
        if col in DEFAULT_CATEGORICAL_COLUMNS or unique_count <= MAX_CATEGORICAL_CARDINALITY:
            categorical_columns.append(col)
        else:
            dropped_categorical_columns.append(col)
    return numeric_columns, categorical_columns, dropped_categorical_columns


def make_preprocessor(numeric_columns: list[str], categorical_columns: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("numeric", SimpleImputer(strategy="median"), numeric_columns),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical_columns,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def transformed_frame(preprocessor: ColumnTransformer, df: pd.DataFrame) -> pd.DataFrame:
    matrix = preprocessor.transform(df)
    columns = preprocessor.get_feature_names_out()
    return pd.DataFrame(matrix, columns=columns, index=df.index)


def build_splits(input_csv: Path, output_dir: Path, config: SplitConfig) -> dict[str, object]:
    df = add_targets(pd.read_csv(input_csv))
    if "subject_id" not in df.columns or "hadm_id" not in df.columns:
        raise ValueError("Input data must include subject_id and hadm_id columns.")

    split_subject_map = split_subjects(df, config)
    splits = {
        name: df[df["subject_id"].astype(int).isin(subjects)].copy()
        for name, subjects in split_subject_map.items()
    }

    numeric_columns, categorical_columns, dropped_categorical_columns = choose_feature_columns(df)
    for col in categorical_columns:
        for split_df in splits.values():
            split_df[col] = normalize_categorical(split_df[col])

    preprocessor = make_preprocessor(numeric_columns, categorical_columns)
    preprocessor.fit(splits["train"])

    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, output_dir / "preprocessor_train_only.joblib")

    summary_rows = []
    for split_name, split_df in splits.items():
        features = transformed_frame(preprocessor, split_df)
        final_df = pd.concat(
            [
                split_df[ID_COLUMNS].reset_index(drop=True),
                split_df[[MORTALITY_TARGET, LOS_TARGET]].reset_index(drop=True),
                features.reset_index(drop=True),
            ],
            axis=1,
        )
        final_df.to_csv(output_dir / f"{split_name}_model_matrix.csv", index=False)
        split_df.to_csv(output_dir / f"{split_name}_raw_split.csv", index=False)
        summary_rows.append(
            {
                "split": split_name,
                "rows": int(len(split_df)),
                "subjects": int(split_df["subject_id"].nunique()),
                "mortality_rate": float(split_df[MORTALITY_TARGET].mean()),
                "los_median_days": float(split_df[LOS_TARGET].median()),
            }
        )

    audit = {
        "input_csv": str(input_csv),
        "random_state": config.random_state,
        "validation_size": config.validation_size,
        "test_size": config.test_size,
        "fit_policy": "split_first_train_only_imputation_and_encoding",
        "id_columns": ID_COLUMNS,
        "targets": [MORTALITY_TARGET, LOS_TARGET],
        "excluded_from_features": sorted(POST_OUTCOME_COLUMNS),
        "numeric_feature_count": len(numeric_columns),
        "categorical_feature_count": len(categorical_columns),
        "dropped_high_cardinality_categoricals": dropped_categorical_columns,
        "max_categorical_cardinality": MAX_CATEGORICAL_CARDINALITY,
        "split_summary": summary_rows,
    }
    (output_dir / "preprocessing_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    pd.DataFrame(summary_rows).to_csv(output_dir / "split_summary.csv", index=False)
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Raw or harmonized admissions CSV.")
    parser.add_argument("--output-dir", default=Path("data/processed"), type=Path)
    parser.add_argument("--validation-size", default=0.15, type=float)
    parser.add_argument("--test-size", default=0.15, type=float)
    parser.add_argument("--random-state", default=42, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = build_splits(
        input_csv=args.input,
        output_dir=args.output_dir,
        config=SplitConfig(
            validation_size=args.validation_size,
            test_size=args.test_size,
            random_state=args.random_state,
        ),
    )
    print(json.dumps(audit["split_summary"], indent=2))
    print(f"Saved leakage-safe splits to {args.output_dir}")


if __name__ == "__main__":
    main()
