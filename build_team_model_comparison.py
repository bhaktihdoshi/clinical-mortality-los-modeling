"""Build one reusable team comparison table from standardized model outputs.

Run this after the final cells of the model notebooks have created their
`*_team_comparison_metrics.csv` files.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


TEAM_METRIC_COLUMNS = [
    "model",
    "model_variant",
    "task",
    "target",
    "dataset",
    "AUROC",
    "AUPRC",
    "F1",
    "Precision",
    "Recall",
    "Sensitivity",
    "Specificity",
    "threshold_used",
    "RMSE",
    "R2",
    "MAE",
    "scale",
    "test_n",
    "test_positives",
    "test_negatives",
]


def candidate_output_dirs() -> list[Path]:
    dirs = [Path.cwd()]
    downloads = Path("/Users/bhaktidoshi/Downloads")
    if downloads.exists() and downloads not in dirs:
        dirs.append(downloads)
    kaggle = Path("/kaggle/working")
    if kaggle.exists() and kaggle not in dirs:
        dirs.insert(0, kaggle)
    return dirs


def find_metric_file(filename: str) -> Path | None:
    for directory in candidate_output_dirs():
        path = directory / filename
        if path.exists():
            return path
    return None


def main() -> None:
    metric_files = [
        "xgb_team_comparison_metrics.csv",
        "lstm_team_comparison_metrics.csv",
        "transformer_team_comparison_metrics.csv",
        "biogpt_team_comparison_metrics.csv",
    ]

    frames = []
    used_files = []
    for filename in metric_files:
        path = find_metric_file(filename)
        if path is None:
            print(f"Missing: {filename}")
            continue
        frame = pd.read_csv(path)
        frame = frame.reindex(columns=TEAM_METRIC_COLUMNS)
        frames.append(frame)
        used_files.append(path)
        print(f"Loaded: {path}")

    if not frames:
        raise FileNotFoundError(
            "No standardized team comparison files were found. "
            "Run the final export cell in each model notebook first."
        )

    comparison = pd.concat(frames, ignore_index=True)
    comparison = comparison.sort_values(["task", "model"]).reset_index(drop=True)

    out_dir = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)

    all_path = out_dir / "team_model_comparison_metrics.csv"
    mortality_path = out_dir / "team_mortality_comparison_metrics.csv"
    los_path = out_dir / "team_los_comparison_metrics.csv"

    comparison.to_csv(all_path, index=False)
    comparison[comparison["task"] == "mortality_classification"].to_csv(mortality_path, index=False)
    comparison[comparison["task"] == "los_regression"].to_csv(los_path, index=False)

    print("\nSaved reusable team comparison outputs:")
    print(f" - {all_path}")
    print(f" - {mortality_path}")
    print(f" - {los_path}")

    print("\nFiles used:")
    for path in used_files:
        print(f" - {path}")


if __name__ == "__main__":
    main()
