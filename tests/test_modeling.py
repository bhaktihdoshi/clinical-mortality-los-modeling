from pathlib import Path

import numpy as np
import pandas as pd

from src.modeling import load_model_matrices, train_tabular_transformer_models, train_xgboost_models


def write_matrix(path: Path, rows: int, offset: int = 0) -> None:
    rng = np.random.default_rng(42 + offset)
    df = pd.DataFrame(
        {
            "subject_id": np.arange(offset, offset + rows),
            "hadm_id": np.arange(1000 + offset, 1000 + offset + rows),
            "target_hospital_expire_flag": [1 if i % 8 == 0 else 0 for i in range(rows)],
            "length_of_stay_days": rng.uniform(1, 8, size=rows),
            "anchor_age": rng.normal(65, 10, size=rows),
            "diagnosis_count": rng.integers(1, 12, size=rows),
            "gender_F": rng.integers(0, 2, size=rows),
        }
    )
    df.to_csv(path, index=False)


def test_modeling_helpers_train_on_small_matrices(tmp_path: Path) -> None:
    for split, rows, offset in [("train", 48, 0), ("val", 24, 100), ("test", 24, 200)]:
        write_matrix(tmp_path / f"{split}_model_matrix.csv", rows, offset)

    bundle, _ = load_model_matrices(tmp_path)
    _, _, xgb_metrics = train_xgboost_models(bundle)
    _, _, transformer_metrics = train_tabular_transformer_models(bundle, epochs=1, batch_size=16)

    assert set(xgb_metrics["model"]) == {"XGBoost"}
    assert set(transformer_metrics["model"]) == {"Tabular Transformer"}
    assert {"mortality_classification", "los_regression"} <= set(xgb_metrics["task"])
    assert {"mortality_classification", "los_regression"} <= set(transformer_metrics["task"])
