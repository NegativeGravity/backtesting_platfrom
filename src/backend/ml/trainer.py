from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from backend.core.config import AppConfig
from backend.ml.artifacts import (
    ModelArtifact,
    create_model_artifact_dir,
    save_model_artifact,
)
from backend.ml.features import FEATURE_COLUMNS, FeatureConfig, build_feature_frame
from backend.ml.splits import time_based_train_validation_test_split


def train_ml_momentum_model(
    data: pd.DataFrame,
    config: AppConfig,
) -> ModelArtifact:
    feature_config = FeatureConfig(
        horizon_bars=config.ml.horizon_bars,
        min_return_threshold=config.ml.min_return_threshold,
        fee_rate=config.execution.fee_rate,
        slippage_bps=config.execution.slippage_bps,
    )

    feature_frame = build_feature_frame(data=data, config=feature_config)

    split = time_based_train_validation_test_split(
        data=feature_frame,
        train_ratio=config.ml.train_ratio,
        validation_ratio=config.ml.validation_ratio,
    )

    x_train = split.train[FEATURE_COLUMNS]
    y_train = split.train["target"]

    x_validation = split.validation[FEATURE_COLUMNS]
    y_validation = split.validation["target"]

    x_test = split.test[FEATURE_COLUMNS]
    y_test = split.test["target"]

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_validation_scaled = scaler.transform(x_validation)
    x_test_scaled = scaler.transform(x_test)

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(x_train_scaled, y_train)

    validation_probabilities = model.predict_proba(x_validation_scaled)[:, 1]
    selected_threshold = _select_threshold(
        probabilities=validation_probabilities,
        labels=y_validation.to_numpy(),
    )

    test_probabilities = model.predict_proba(x_test_scaled)[:, 1]

    validation_metrics = _classification_metrics(
        labels=y_validation.to_numpy(),
        probabilities=validation_probabilities,
        threshold=selected_threshold,
    )

    test_metrics = _classification_metrics(
        labels=y_test.to_numpy(),
        probabilities=test_probabilities,
        threshold=selected_threshold,
    )

    artifact_dir = create_model_artifact_dir(config.ml.model_artifact_dir)

    model_bundle = {
        "model": model,
        "scaler": scaler,
        "feature_columns": FEATURE_COLUMNS,
        "threshold": selected_threshold,
        "exit_probability": config.ml.exit_probability,
    }

    metadata = {
        "symbol": config.data.symbol,
        "timeframe": config.data.timeframe,
        "model_type": "LogisticRegression",
        "strategy": "ml_momentum",
        "no_data_leakage_controls": [
            "causal_features_only",
            "time_based_split",
            "scaler_fit_on_train_only",
            "threshold_selected_on_validation_only",
            "test_set_not_used_for_training_or_threshold_selection",
        ],
    }

    metrics = {
        "validation": validation_metrics,
        "test": test_metrics,
    }

    split_info = {
        "train_start": split.train.iloc[0]["timestamp"],
        "train_end": split.train.iloc[-1]["timestamp"],
        "validation_start": split.validation.iloc[0]["timestamp"],
        "validation_end": split.validation.iloc[-1]["timestamp"],
        "test_start": split.test.iloc[0]["timestamp"],
        "test_end": split.test.iloc[-1]["timestamp"],
        "train_rows": len(split.train),
        "validation_rows": len(split.validation),
        "test_rows": len(split.test),
    }

    return save_model_artifact(
        artifact_dir=artifact_dir,
        model_bundle=model_bundle,
        metadata=metadata,
        metrics=metrics,
        split_info=split_info,
        feature_config=asdict(feature_config),
    )


def _select_threshold(probabilities: np.ndarray, labels: np.ndarray) -> float:
    candidate_thresholds = np.linspace(0.50, 0.80, 31)

    best_threshold = 0.60
    best_score = -1.0

    for threshold in candidate_thresholds:
        predictions = (probabilities >= threshold).astype(int)
        score = f1_score(labels, predictions, zero_division=0)

        if score > best_score:
            best_score = score
            best_threshold = float(threshold)

    return best_threshold


def _classification_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    predictions = (probabilities >= threshold).astype(int)

    metrics: dict[str, Any] = {
        "threshold": threshold,
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "positive_label_ratio": float(np.mean(labels)),
        "positive_prediction_ratio": float(np.mean(predictions)),
        "average_predicted_probability": float(np.mean(probabilities)),
        "confusion_matrix": confusion_matrix(labels, predictions).tolist(),
    }

    if len(np.unique(labels)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(labels, probabilities))
    else:
        metrics["roc_auc"] = None

    return metrics