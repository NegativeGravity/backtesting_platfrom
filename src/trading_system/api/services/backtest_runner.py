from __future__ import annotations

import json
import multiprocessing as mp
import traceback
from pathlib import Path
from typing import Any

from trading_system.backtest.robot_engine import (
    BacktestRobotSpec,
    BacktestWorkerSpec,
    RobotBacktestEngine,
)
from trading_system.core.config import load_config
from trading_system.core.paths import resolve_project_path
from trading_system.data.csv_loader import load_ohlcv_csv
from trading_system.data.market_data import filter_date_range
from trading_system.data.validator import validate_ohlcv
from trading_system.strategy.factory import create_strategy


def run_backtest_from_request(
    config_path: str,
    strategy_name: str,
    model_artifact_path: str | None,
) -> tuple[Path, dict[str, Any]]:
    """Backward-compatible API: run one strategy as a one-worker robot.

    The actual engine runs in a separate process, so API latency and failures are isolated
    from the FastAPI process. The returned report format remains the same.
    """
    robot_payload = {
        "robot_id": f"{strategy_name}_robot",
        "display_name": f"{strategy_name} Robot",
        "strategy_workers": [
            {
                "worker_id": f"{strategy_name}_worker_1",
                "strategy": strategy_name,
                "model_artifact_path": model_artifact_path,
            }
        ],
    }
    return run_robot_backtest_from_request(config_path=config_path, robot=robot_payload)


def run_robot_backtest_from_request(
    config_path: str,
    robot: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    result_queue: mp.Queue = mp.Queue(maxsize=1)
    process = mp.Process(
        target=_run_robot_backtest_child,
        kwargs={
            "config_path": config_path,
            "robot": robot,
            "result_queue": result_queue,
        },
        daemon=False,
    )
    process.start()
    process.join()

    if result_queue.empty():
        raise RuntimeError(f"Backtest process exited without a result. exit_code={process.exitcode}")

    payload = result_queue.get()
    if not payload.get("ok"):
        raise RuntimeError(payload.get("error", "Backtest process failed."))

    run_dir = Path(payload["run_dir"])
    summary = _read_summary(run_dir)
    return run_dir, summary


def _run_robot_backtest_child(config_path: str, robot: dict[str, Any], result_queue: mp.Queue) -> None:
    try:
        config = load_config(config_path)
        data = load_ohlcv_csv(
            path=config.data.path,
            timestamp_column=config.data.timestamp_column,
        )
        data = filter_date_range(
            data=data,
            start_date=config.backtest.start_date,
            end_date=config.backtest.end_date,
        )
        validate_ohlcv(data)

        worker_specs: list[BacktestWorkerSpec] = []
        for raw_worker in robot.get("strategy_workers", []):
            strategy_name = str(raw_worker.get("strategy"))
            model_artifact_path = raw_worker.get("model_artifact_path")
            resolved_model_path = (
                resolve_project_path(model_artifact_path)
                if model_artifact_path is not None and str(model_artifact_path).strip()
                else None
            )
            strategy = create_strategy(
                config=config,
                strategy_name=strategy_name,
                model_artifact_path=resolved_model_path,
            )
            worker_specs.append(
                BacktestWorkerSpec(
                    worker_id=str(raw_worker.get("worker_id") or f"{strategy_name}_worker"),
                    strategy_name=strategy_name,
                    strategy=strategy,
                    model_artifact_path=str(resolved_model_path) if resolved_model_path else None,
                )
            )

        robot_spec = BacktestRobotSpec(
            robot_id=str(robot.get("robot_id") or "robot_1"),
            display_name=str(robot.get("display_name") or "Robot 1"),
            strategy_workers=worker_specs,
        )

        engine = RobotBacktestEngine(config=config, data=data, robot_spec=robot_spec)
        result = engine.run()
        result_queue.put({"ok": True, "run_dir": str(result.run_dir), "run_id": result.run_id})
    except Exception as exc:
        result_queue.put({"ok": False, "error": str(exc), "traceback": traceback.format_exc()})


def _read_summary(run_dir: Path) -> dict[str, Any]:
    summary_path = run_dir / "summary.json"
    with summary_path.open("r", encoding="utf-8") as file:
        return json.load(file)
