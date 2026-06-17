from pathlib import Path

import joblib
import pandas as pd

from src.preprocess_split_first import SplitConfig, build_splits


def make_synthetic_input(path: Path) -> None:
    rows = []
    for i in range(80):
        rows.append(
            {
                "subject_id": i,
                "hadm_id": 1000 + i,
                "anchor_age": 40 + i if i < 56 else None,
                "gender": "F" if i % 2 else "M",
                "race": "Known",
                "admission_type": "EW EMER.",
                "diagnosis_count": i % 5,
                "hospital_expire_flag": 1 if i % 10 == 0 else 0,
                "admittime": "2020-01-01 00:00:00",
                "dischtime": "2020-01-03 00:00:00",
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_imputer_is_fit_on_training_split_only(tmp_path: Path) -> None:
    input_csv = tmp_path / "synthetic.csv"
    output_dir = tmp_path / "processed"
    make_synthetic_input(input_csv)

    build_splits(input_csv, output_dir, SplitConfig(random_state=7))

    train_raw = pd.read_csv(output_dir / "train_raw_split.csv")
    train_matrix = pd.read_csv(output_dir / "train_model_matrix.csv")
    val_matrix = pd.read_csv(output_dir / "val_model_matrix.csv")
    test_matrix = pd.read_csv(output_dir / "test_model_matrix.csv")
    audit = (output_dir / "preprocessing_audit.json").read_text(encoding="utf-8")

    train_median = train_raw["anchor_age"].median()
    assert "split_first_train_only_imputation_and_encoding" in audit
    assert train_matrix["anchor_age"].isna().sum() == 0
    assert val_matrix["anchor_age"].isna().sum() == 0
    assert test_matrix["anchor_age"].isna().sum() == 0
    assert set(val_matrix.loc[val_matrix["hadm_id"] >= 1056, "anchor_age"]) <= {train_median}
    assert set(test_matrix.loc[test_matrix["hadm_id"] >= 1056, "anchor_age"]) <= {train_median}

    preprocessor = joblib.load(output_dir / "preprocessor_train_only.joblib")
    assert hasattr(preprocessor, "transform")


def test_subjects_do_not_cross_splits(tmp_path: Path) -> None:
    input_csv = tmp_path / "synthetic.csv"
    output_dir = tmp_path / "processed"
    make_synthetic_input(input_csv)

    build_splits(input_csv, output_dir, SplitConfig(random_state=13))

    subject_sets = {
        split: set(pd.read_csv(output_dir / f"{split}_raw_split.csv")["subject_id"])
        for split in ["train", "val", "test"]
    }
    assert subject_sets["train"].isdisjoint(subject_sets["val"])
    assert subject_sets["train"].isdisjoint(subject_sets["test"])
    assert subject_sets["val"].isdisjoint(subject_sets["test"])
