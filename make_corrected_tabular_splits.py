"""Create corrected tabular split files that restore the 13 missing admissions.

The original tabular split files contain 11,815 admissions, while the corrected
transformer split files contain 11,828. This script adds the 13 admissions back
to the tabular train/validation/test files using the corrected transformer split
membership, fills known columns from the transformer master files, and imputes
engineered-only columns with training-derived defaults.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


SPLITS = {
    "train": ("final_train_tabular.csv", "transformer_train_master.csv", "final_train_tabular_corrected.csv"),
    "val": ("final_val.csv", "transformer_val_master.csv", "final_val_corrected.csv"),
    "test": ("final_test.csv", "transformer_test_master.csv", "final_test_corrected.csv"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        default=Path("."),
        type=Path,
        help="Directory containing original tabular and transformer split CSVs.",
    )
    parser.add_argument(
        "--output-dir",
        default=Path("."),
        type=Path,
        help="Directory where corrected split CSVs and summary will be written.",
    )
    return parser.parse_args()


def fill_defaults(train_df: pd.DataFrame) -> dict[str, object]:
    defaults: dict[str, object] = {}
    for col in train_df.columns:
        if col in {"subject_id", "hadm_id"}:
            defaults[col] = np.nan
        elif pd.api.types.is_numeric_dtype(train_df[col]):
            defaults[col] = train_df[col].median()
            if pd.isna(defaults[col]):
                defaults[col] = 0
        else:
            mode = train_df[col].mode(dropna=True)
            defaults[col] = mode.iloc[0] if not mode.empty else "UNKNOWN"
    return defaults


def build_row_from_master(
    master_row: pd.Series,
    final_columns: list[str],
    defaults: dict[str, object],
    numeric_columns: set[str],
) -> dict[str, object]:
    row = {col: defaults.get(col, np.nan) for col in final_columns}
    for col in final_columns:
        if col in master_row.index and pd.notna(master_row[col]):
            value = master_row[col]
            if col in numeric_columns:
                numeric_value = pd.to_numeric(value, errors="coerce")
                if pd.notna(numeric_value):
                    row[col] = numeric_value
            else:
                row[col] = value

    if "los_days" in row and pd.notna(row["los_days"]):
        row["log_los"] = np.log1p(float(row["los_days"]))
        row["los_hours"] = float(row["los_days"]) * 24.0

    if "hospital_expire_flag" in row and pd.notna(row["hospital_expire_flag"]):
        row["hospital_expire_flag"] = int(row["hospital_expire_flag"])

    return row


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    train_reference = pd.read_csv(input_dir / "final_train_tabular.csv")
    final_columns = list(train_reference.columns)
    defaults = fill_defaults(train_reference)
    numeric_columns = {
        col for col in final_columns if pd.api.types.is_numeric_dtype(train_reference[col])
    }

    summary_rows = []
    for split_name, (tabular_name, transformer_name, output_name) in SPLITS.items():
        tabular_df = pd.read_csv(input_dir / tabular_name)
        transformer_df = pd.read_csv(input_dir / transformer_name)

        existing_hadm = set(tabular_df["hadm_id"].astype(str))
        missing_master = transformer_df[~transformer_df["hadm_id"].astype(str).isin(existing_hadm)].copy()

        new_rows = [
            build_row_from_master(master_row, final_columns, defaults, numeric_columns)
            for _, master_row in missing_master.iterrows()
        ]
        if new_rows:
            add_df = pd.DataFrame(new_rows, columns=final_columns)
            corrected_df = pd.concat([tabular_df, add_df], ignore_index=True)
        else:
            corrected_df = tabular_df.copy()

        corrected_df = corrected_df[final_columns]
        corrected_df.to_csv(output_dir / output_name, index=False)

        summary_rows.append(
            {
                "split": split_name,
                "original_rows": int(len(tabular_df)),
                "corrected_rows": int(len(corrected_df)),
                "rows_added": int(len(new_rows)),
                "output_file": output_name,
                "imputation_policy": "training_split_defaults_only",
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "corrected_tabular_split_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"\nSaved corrected split files to {output_dir}")


if __name__ == "__main__":
    main()
