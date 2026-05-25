from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from backend.core.time import timestamp_for_run_id


@dataclass(frozen=True)
class ModelArtifact:
    artifact_id: str
    artifact_dir: Path
    model_path: Path
    metadata_path: Path
    metrics_path: Path
    split_info_path: Path
    feature_config_path: Path


def create_model_artifact_dir(base_dir: Path) -> Path:
    artifact_dir = base_dir / f"model_{timestamp_for_run_id()}"
    artifact_dir.mkdir(parents=True, exist_ok=False)
    return artifact_dir


def save_model_artifact(
    artifact_dir: Path,
    model_bundle: dict[str, Any],
    metadata: dict[str, Any],
    metrics: dict[str, Any],
    split_info: dict[str, Any],
    feature_config: dict[str, Any],
) -> ModelArtifact:
    model_path = artifact_dir / "model.joblib"
    metadata_path = artifact_dir / "metadata.json"
    metrics_path = artifact_dir / "metrics.json"
    split_info_path = artifact_dir / "split_info.json"
    feature_config_path = artifact_dir / "feature_config.json"

    joblib.dump(model_bundle, model_path)

    _write_json(metadata_path, metadata)
    _write_json(metrics_path, metrics)
    _write_json(split_info_path, split_info)
    _write_json(feature_config_path, feature_config)

    return ModelArtifact(
        artifact_id=artifact_dir.name,
        artifact_dir=artifact_dir,
        model_path=model_path,
        metadata_path=metadata_path,
        metrics_path=metrics_path,
        split_info_path=split_info_path,
        feature_config_path=feature_config_path,
    )


def load_model_bundle(artifact_dir: Path | str) -> dict[str, Any]:
    artifact_path = Path(artifact_dir)
    model_path = artifact_path / "model.joblib"

    if not model_path.exists():
        available_artifacts = []
        parent = artifact_path.parent

        if parent.exists():
            available_artifacts = [
                str(path)
                for path in parent.iterdir()
                if path.is_dir() and (path / "model.joblib").exists()
            ]

        hint = ""
        if available_artifacts:
            hint = (
                "\nAvailable model artifacts:\n"
                + "\n".join(f"- {artifact}" for artifact in available_artifacts)
            )

        raise FileNotFoundError(
            f"Model artifact not found: {model_path}"
            f"{hint}"
        )

    bundle = joblib.load(model_path)

    if not isinstance(bundle, dict):
        raise TypeError("Invalid model artifact format. Expected dictionary bundle.")

    return bundle


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, default=_json_default)


def _json_default(value: object) -> str:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if hasattr(value, "item"):
        return str(value.item())

    if hasattr(value, "__dict__"):
        return str(asdict(value))

    return str(value)