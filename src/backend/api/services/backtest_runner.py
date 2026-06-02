from __future__ import annotations

import json
import multiprocessing as mp
import queue
import threading
import traceback
from pathlib import Path
from typing import Any

from backend.backtest.robot_engine import (
    BacktestRobotSpec,
    BacktestWorkerSpec,
    RobotBacktestEngine,
)
from backend.core.config import load_config
from backend.core.paths import resolve_model_artifact_path
from backend.data.validator import validate_ohlcv
from backend.strategy.factory import create_strategy
from backend.data.market_store import MarketDataStore
from backend.ml.time_policy import (
    BACKTEST_START_STR,
    BACKTEST_END_STR,
    assert_backtest_2025_only,
)


def run_backtest_from_request(
    config_path: str,
    strategy_name: str,
    model_artifact_path: str | None,
    helformer_artifact_path: str | None = None,
    use_helformer_forecast: bool = False,
    cancel_event: threading.Event | None = None,
) -> tuple[Path, dict[str, Any]]:
    robot_payload = {
        "robot_id": f"{strategy_name}_robot",
        "display_name": f"{strategy_name} Robot",
        "strategy_workers": [
            {
                "worker_id": f"{strategy_name}_worker_1",
                "strategy": strategy_name,
                "model_artifact_path": model_artifact_path,
                "helformer_artifact_path": helformer_artifact_path,
                "use_helformer_forecast": use_helformer_forecast,
            }
        ],
    }
    return run_robot_backtest_from_request(config_path=config_path, robot=robot_payload, cancel_event=cancel_event)


def run_robot_backtest_from_request(
    config_path: str,
    robot: dict[str, Any],
    cancel_event: threading.Event | None = None,
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
    while process.is_alive():
        process.join(timeout=0.25)
        if cancel_event is not None and cancel_event.is_set():
            process.terminate()
            process.join(timeout=3)
            if process.is_alive():
                process.kill()
                process.join(timeout=3)
            raise RuntimeError("Backtest job cancelled.")

    try:
        payload = result_queue.get(timeout=5)
    except queue.Empty as exc:
        raise RuntimeError(f"Backtest process exited without a result. exit_code={process.exitcode}") from exc
    if not payload.get("ok"):
        raise RuntimeError(payload.get("error", "Backtest process failed."))

    run_dir = Path(payload["run_dir"])
    summary = _read_summary(run_dir)
    return run_dir, summary


def _run_robot_backtest_child(config_path: str, robot: dict[str, Any], result_queue: mp.Queue) -> None:
    try:
        config = load_config(config_path)
        data = MarketDataStore.from_yaml().load_ohlcv(
            start=BACKTEST_START_STR,
            end=BACKTEST_END_STR,
            symbol=config.data.symbol,
            interval=config.data.timeframe,
            use_cache=True,
        )

        assert_backtest_2025_only(data)
        validate_ohlcv(data)


        worker_specs: list[BacktestWorkerSpec] = []
        for raw_worker in robot.get("strategy_workers", []):
            strategy_name = str(raw_worker.get("strategy"))
            model_artifact_path = raw_worker.get("model_artifact_path")
            helformer_artifact_path = raw_worker.get("helformer_artifact_path")
            use_helformer_forecast = bool(raw_worker.get("use_helformer_forecast", False))
            resolved_model_path = (
                resolve_model_artifact_path(model_artifact_path)
                if model_artifact_path is not None and str(model_artifact_path).strip()
                else None
            )
            resolved_helformer_path = (
                resolve_model_artifact_path(helformer_artifact_path)
                if use_helformer_forecast and helformer_artifact_path is not None and str(helformer_artifact_path).strip()
                else None
            )
            strategy = create_strategy(
                config=config,
                strategy_name=strategy_name,
                model_artifact_path=resolved_model_path,
                helformer_artifact_path=resolved_helformer_path,
                use_helformer_forecast=use_helformer_forecast,
            )
            worker_specs.append(
                BacktestWorkerSpec(
                    worker_id=str(raw_worker.get("worker_id") or f"{strategy_name}_worker"),
                    strategy_name=strategy_name,
                    strategy=strategy,
                    model_artifact_path=str(resolved_model_path) if resolved_model_path else None,
                    helformer_artifact_path=str(resolved_helformer_path) if resolved_helformer_path else None,
                    use_helformer_forecast=use_helformer_forecast,
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
