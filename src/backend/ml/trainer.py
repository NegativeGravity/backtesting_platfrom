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
from backend.ml.time_policy import (
    ModelTrainingMode,
    assert_no_2025,
    split_raw_for_model_training,
)

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

    raw_split = split_raw_for_model_training(
        data=data,
        mode=ModelTrainingMode.TUNED,
    )

    if raw_split.validation is None:
        raise ValueError("Tuned model requires validation data.")

    train_features = build_feature_frame(data=raw_split.train, config=feature_config)
    validation_features = build_feature_frame(data=raw_split.validation, config=feature_config)

    assert_no_2025(train_features, "ml_momentum_train_features")
    assert_no_2025(validation_features, "ml_momentum_validation_features")

    x_train = train_features[FEATURE_COLUMNS]
    y_train = train_features["target"]

    x_validation = validation_features[FEATURE_COLUMNS]
    y_validation = validation_features["target"]

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_validation_scaled = scaler.transform(x_validation)

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(x_train_scaled, y_train)

    validation_probabilities = model.predict_proba(x_validation_scaled)[:, 1]
    selected_threshold = _select_ev_threshold(
        probabilities=validation_probabilities,
        labels=y_validation.to_numpy(),
        net_returns=validation_features["net_future_return"].to_numpy(),
    )


    validation_metrics = _classification_metrics(
        labels=y_validation.to_numpy(),
        probabilities=validation_probabilities,
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
            "threshold_selected_by_validation_expected_value_after_cost",
            "model_never_loaded_2025_data",
            "train_2020_2023_validation_2024",
            "2025_reserved_for_backtest_platform",        ],
    }

    metrics = {
        "validation": validation_metrics,
    }

    split_info = {
        "policy": "calendar_no_2025_model_awareness",
        "mode": "tuned",
        "train_start": train_features.iloc[0]["timestamp"],
        "train_end": train_features.iloc[-1]["timestamp"],
        "validation_start": validation_features.iloc[0]["timestamp"],
        "validation_end": validation_features.iloc[-1]["timestamp"],
        "train_rows": len(train_features),
        "validation_rows": len(validation_features),
        "test_rows": 0,
        "test_policy": "reserved_for_backtest_platform_2025_only",
    }

    return save_model_artifact(
        artifact_dir=artifact_dir,
        model_bundle=model_bundle,
        metadata=metadata,
        metrics=metrics,
        split_info=split_info,
        feature_config=asdict(feature_config),
    )


def _select_ev_threshold(probabilities: np.ndarray, labels: np.ndarray, net_returns: np.ndarray) -> float:
    candidate_thresholds = np.linspace(0.50, 0.85, 36)
    min_trades = max(10, int(len(probabilities) * 0.01))
    best_threshold = _select_threshold(probabilities, labels)
    best_score = -np.inf

    for threshold in candidate_thresholds:
        selected = probabilities >= threshold
        trade_count = int(selected.sum())
        if trade_count < min_trades:
            continue
        selected_returns = net_returns[selected]
        if selected_returns.size == 0:
            continue
        score = float(np.nanmean(selected_returns)) * np.sqrt(trade_count)
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)

    return best_threshold


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