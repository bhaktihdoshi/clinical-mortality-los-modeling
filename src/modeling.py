"""Model training and evaluation helpers for the clinical ML notebook."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from src.preprocess_split_first import ID_COLUMNS, LOS_TARGET, MORTALITY_TARGET


FEATURE_EXCLUDE_COLUMNS = set(ID_COLUMNS + [MORTALITY_TARGET, LOS_TARGET])


@dataclass(frozen=True)
class DatasetBundle:
    x_train: pd.DataFrame
    x_val: pd.DataFrame
    x_test: pd.DataFrame
    y_train_mortality: pd.Series
    y_val_mortality: pd.Series
    y_test_mortality: pd.Series
    y_train_los: pd.Series
    y_val_los: pd.Series
    y_test_los: pd.Series


def load_model_matrices(processed_dir: Path) -> tuple[DatasetBundle, dict[str, pd.DataFrame]]:
    frames = {
        split: pd.read_csv(processed_dir / f"{split}_model_matrix.csv")
        for split in ["train", "val", "test"]
    }
    feature_columns = [
        col for col in frames["train"].columns if col not in FEATURE_EXCLUDE_COLUMNS
    ]
    bundle = DatasetBundle(
        x_train=frames["train"][feature_columns],
        x_val=frames["val"][feature_columns],
        x_test=frames["test"][feature_columns],
        y_train_mortality=frames["train"][MORTALITY_TARGET].astype(int),
        y_val_mortality=frames["val"][MORTALITY_TARGET].astype(int),
        y_test_mortality=frames["test"][MORTALITY_TARGET].astype(int),
        y_train_los=frames["train"][LOS_TARGET].astype(float),
        y_val_los=frames["val"][LOS_TARGET].astype(float),
        y_test_los=frames["test"][LOS_TARGET].astype(float),
    )
    return bundle, frames


def tune_threshold_for_f2(y_true: pd.Series, probabilities: np.ndarray) -> float:
    thresholds = np.linspace(0.01, 0.99, 99)
    best_threshold = 0.5
    best_score = -1.0
    beta_squared = 4.0
    for threshold in thresholds:
        y_pred = (probabilities >= threshold).astype(int)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        denominator = (beta_squared * precision) + recall
        score = 0.0 if denominator == 0 else (1 + beta_squared) * precision * recall / denominator
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def mortality_metrics(
    model_name: str,
    y_true: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float | str]:
    y_pred = (probabilities >= threshold).astype(int)
    return {
        "model": model_name,
        "task": "mortality_classification",
        "AUROC": float(roc_auc_score(y_true, probabilities)),
        "AUPRC": float(average_precision_score(y_true, probabilities)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "threshold": float(threshold),
    }


def los_metrics(model_name: str, y_true: pd.Series, predictions: np.ndarray) -> dict[str, float | str]:
    rmse = float(np.sqrt(mean_squared_error(y_true, predictions)))
    return {
        "model": model_name,
        "task": "los_regression",
        "RMSE": float(rmse),
        "MAE": float(mean_absolute_error(y_true, predictions)),
        "R2": float(r2_score(y_true, predictions)),
    }


def train_xgboost_models(bundle: DatasetBundle) -> tuple[object, object, pd.DataFrame]:
    from xgboost import XGBClassifier, XGBRegressor

    positive_count = max(int(bundle.y_train_mortality.sum()), 1)
    negative_count = max(int(len(bundle.y_train_mortality) - positive_count), 1)
    scale_pos_weight = negative_count / positive_count

    mortality_model = XGBClassifier(
        n_estimators=250,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
    )
    mortality_model.fit(bundle.x_train, bundle.y_train_mortality)

    val_prob = mortality_model.predict_proba(bundle.x_val)[:, 1]
    threshold = tune_threshold_for_f2(bundle.y_val_mortality, val_prob)
    test_prob = mortality_model.predict_proba(bundle.x_test)[:, 1]

    los_model = XGBRegressor(
        n_estimators=250,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )
    los_model.fit(bundle.x_train, bundle.y_train_los)
    los_pred = np.clip(los_model.predict(bundle.x_test), a_min=0, a_max=None)

    metrics = pd.DataFrame(
        [
            mortality_metrics("XGBoost", bundle.y_test_mortality, test_prob, threshold),
            los_metrics("XGBoost", bundle.y_test_los, los_pred),
        ]
    )
    return mortality_model, los_model, metrics


def train_tabular_transformer_models(
    bundle: DatasetBundle,
    epochs: int = 8,
    batch_size: int = 256,
) -> tuple[object, object, pd.DataFrame]:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_train = torch.tensor(bundle.x_train.to_numpy(dtype=np.float32), device=device)
    x_val = torch.tensor(bundle.x_val.to_numpy(dtype=np.float32), device=device)
    x_test = torch.tensor(bundle.x_test.to_numpy(dtype=np.float32), device=device)
    y_train_cls = torch.tensor(bundle.y_train_mortality.to_numpy(dtype=np.float32), device=device)
    y_train_reg = torch.tensor(bundle.y_train_los.to_numpy(dtype=np.float32), device=device)

    class TabularTransformer(nn.Module):
        def __init__(self, n_features: int, task: str):
            super().__init__()
            self.task = task
            d_model = 32
            self.feature_embedding = nn.Linear(1, d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=4,
                dim_feedforward=64,
                dropout=0.1,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
            self.head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            tokens = self.feature_embedding(x.unsqueeze(-1))
            encoded = self.encoder(tokens).mean(dim=1)
            return self.head(encoded).squeeze(-1)

    def fit_model(model: nn.Module, y_train: torch.Tensor, loss_fn: nn.Module) -> nn.Module:
        loader = DataLoader(
            TensorDataset(x_train, y_train),
            batch_size=batch_size,
            shuffle=True,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        model.train()
        for _ in range(epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                optimizer.step()
        return model.eval()

    pos_weight = torch.tensor(
        [(len(bundle.y_train_mortality) - bundle.y_train_mortality.sum()) / max(bundle.y_train_mortality.sum(), 1)],
        device=device,
        dtype=torch.float32,
    )
    mortality_model = TabularTransformer(bundle.x_train.shape[1], "classification").to(device)
    mortality_model = fit_model(
        mortality_model,
        y_train_cls,
        nn.BCEWithLogitsLoss(pos_weight=pos_weight),
    )

    with torch.no_grad():
        val_prob = torch.sigmoid(mortality_model(x_val)).detach().cpu().numpy()
        test_prob = torch.sigmoid(mortality_model(x_test)).detach().cpu().numpy()
    threshold = tune_threshold_for_f2(bundle.y_val_mortality, val_prob)

    los_model = TabularTransformer(bundle.x_train.shape[1], "regression").to(device)
    los_model = fit_model(los_model, y_train_reg, nn.MSELoss())
    with torch.no_grad():
        los_pred = np.clip(los_model(x_test).detach().cpu().numpy(), a_min=0, a_max=None)

    metrics = pd.DataFrame(
        [
            mortality_metrics("Tabular Transformer", bundle.y_test_mortality, test_prob, threshold),
            los_metrics("Tabular Transformer", bundle.y_test_los, los_pred),
        ]
    )
    return mortality_model, los_model, metrics
