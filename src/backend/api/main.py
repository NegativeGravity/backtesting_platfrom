from __future__ import annotations

import asyncio
import logging
from typing import Literal

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from backend.api.live_sessions import LiveSessionManager
from backend.api.schemas import (
    BacktestJobResponse,
    BacktestRunListItem,
    BacktestRunRequest,
    BacktestRunResponse,
    ChartDataResponse,
    DatasetInfo,
    HealthResponse,
    LiveReplayRequest,
    LiveReplayResponse,
    LiveSessionStatusResponse,
    ReportResponse,
    RobotBacktestRequest,
    StrategyInfo,
)
from backend.api.services.backtest_runner import (
    run_backtest_from_request,
    run_robot_backtest_from_request,
)
from backend.api.services.job_manager import BacktestJobManager
from backend.api.services.report_repository import ReportRepository
from backend.core.config import load_config
from backend.core.logging import setup_logging
from backend.engine.live_replay import LiveRobotSpec, LiveStrategyWorkerSpec
from backend.strategy.factory import STRATEGIES_REQUIRING_ARTIFACT, STRATEGIES_SUPPORTING_HELFORMER_FORECASTER, SUPPORTED_STRATEGIES
from backend.utils.ids import new_id

from pathlib import Path
from backend.core.paths import get_project_root

setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Trading Backtesting Platform API",
    version="0.4.0",
    description="Control API for robot-level backtesting, live replay, and report visualization.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[],
    allow_origin_regex=r".*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

repository = ReportRepository()
job_manager = BacktestJobManager()
live_sessions = LiveSessionManager()


def _job_response(job) -> BacktestJobResponse:
    return BacktestJobResponse(
        job_id=job.job_id,
        status=job.status,
        run_id=job.run_id,
        run_dir=job.run_dir,
        summary=job.summary,
        error=job.error,
        message=job.message,
        created_at=job.created_at,
        updated_at=job.updated_at,
        heartbeat_at=job.heartbeat_at,
        elapsed_seconds=job.elapsed_seconds,
    )


def _job_run_response(job) -> BacktestRunResponse:
    return BacktestRunResponse(
        job_id=job.job_id,
        status=job.status,
        run_id=job.run_id,
        run_dir=job.run_dir,
        summary=job.summary,
        error=job.error,
        message=job.message,
        created_at=job.created_at,
        updated_at=job.updated_at,
        heartbeat_at=job.heartbeat_at,
        elapsed_seconds=job.elapsed_seconds,
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", service="trading-backtesting-platform-api")


@app.get("/datasets", response_model=list[DatasetInfo])
def list_datasets() -> list[DatasetInfo]:
    config = load_config("configs/backtest.yaml")
    return [
        DatasetInfo(
            symbol=config.data.symbol,
            timeframe=config.data.timeframe,
            path=str(config.data.path),
        )
    ]


def _validate_strategy_artifacts(strategy: str, model_artifact_path: str | None, helformer_artifact_path: str | None, use_helformer_forecast: bool) -> None:
    if strategy in STRATEGIES_REQUIRING_ARTIFACT and not str(model_artifact_path or "").strip():
        raise HTTPException(status_code=400, detail=f"model_artifact_path is required for {strategy}.")
    if use_helformer_forecast and strategy not in STRATEGIES_SUPPORTING_HELFORMER_FORECASTER:
        raise HTTPException(status_code=400, detail=f"Helformer forecast projection is not supported for {strategy}.")
    if use_helformer_forecast and not str(helformer_artifact_path or "").strip():
        raise HTTPException(status_code=400, detail=f"helformer_artifact_path is required when Helformer forecast is enabled for {strategy}.")


@app.get("/strategies", response_model=list[StrategyInfo])
def list_strategies() -> list[StrategyInfo]:
    return [
        StrategyInfo(
            name=strategy_name,  # type: ignore[arg-type]
            requires_model_artifact=strategy_name in STRATEGIES_REQUIRING_ARTIFACT,
            requires_helformer_forecaster=False,
            supports_helformer_forecaster=strategy_name in STRATEGIES_SUPPORTING_HELFORMER_FORECASTER,
        )
        for strategy_name in sorted(SUPPORTED_STRATEGIES)
    ]


@app.post("/backtests/run", response_model=BacktestRunResponse)
def run_backtest(request: BacktestRunRequest) -> BacktestRunResponse:
    _validate_strategy_artifacts(request.strategy, request.model_artifact_path, request.helformer_artifact_path, request.use_helformer_forecast)
    job = job_manager.submit(
        lambda cancel_event: run_backtest_from_request(
            config_path=request.config_path,
            strategy_name=request.strategy,
            model_artifact_path=request.model_artifact_path,
            helformer_artifact_path=request.helformer_artifact_path,
            use_helformer_forecast=request.use_helformer_forecast,
            cancel_event=cancel_event,
        )
    )
    return _job_run_response(job)


@app.post("/robot-backtests/run", response_model=BacktestRunResponse)
def run_robot_backtest(request: RobotBacktestRequest) -> BacktestRunResponse:
    for worker in request.robot.strategy_workers:
        _validate_strategy_artifacts(worker.strategy, worker.model_artifact_path, worker.helformer_artifact_path, worker.use_helformer_forecast)
    robot_payload = {
        "robot_id": request.robot.robot_id,
        "display_name": request.robot.display_name,
        "strategy_workers": [
            {
                "worker_id": worker.worker_id or new_id(f"worker_{worker.strategy}"),
                "strategy": worker.strategy,
                "model_artifact_path": worker.model_artifact_path,
                "helformer_artifact_path": worker.helformer_artifact_path,
                "use_helformer_forecast": worker.use_helformer_forecast,
            }
            for worker in request.robot.strategy_workers
        ],
    }
    job = job_manager.submit(
        lambda cancel_event: run_robot_backtest_from_request(
            config_path=request.config_path,
            robot=robot_payload,
            cancel_event=cancel_event,
        )
    )
    return _job_run_response(job)


@app.get("/jobs/{job_id}", response_model=BacktestJobResponse)
def get_job(job_id: str) -> BacktestJobResponse:
    job = job_manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return _job_response(job)


@app.post("/jobs/{job_id}/cancel", response_model=BacktestJobResponse)
def cancel_job(job_id: str) -> BacktestJobResponse:
    job = job_manager.cancel(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return _job_response(job)


@app.get("/backtests", response_model=list[BacktestRunListItem])
def list_backtests() -> list[BacktestRunListItem]:
    return [BacktestRunListItem(**item) for item in repository.list_runs()]


@app.get("/backtests/{run_id}", response_model=ReportResponse)
def get_backtest_report(run_id: str) -> ReportResponse:
    try:
        report = repository.read_report(run_id)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return ReportResponse(**report)


@app.get("/backtests/{run_id}/summary")
def get_backtest_summary(run_id: str) -> dict:
    try:
        return repository.read_summary(run_id)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/backtests/{run_id}/chart-data", response_model=ChartDataResponse)
def get_backtest_chart_data(run_id: str) -> ChartDataResponse:
    try:
        chart_data = repository.read_chart_data(run_id)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return ChartDataResponse(**chart_data)


@app.get("/backtests/{run_id}/export")
def export_backtest(run_id: str, format: Literal["csv", "parquet", "html"] = "csv") -> FileResponse:
    try:
        path = repository.export_path(run_id, format)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    media_types = {
        "csv": "text/csv",
        "parquet": "application/octet-stream",
        "html": "text/html",
    }
    return FileResponse(
        path,
        media_type=media_types[format],
        filename=f"{run_id}.{format}",
    )


@app.post("/live-replay/start", response_model=LiveReplayResponse)
def start_live_replay(request: LiveReplayRequest) -> LiveReplayResponse:
    try:
        for robot in request.robots:
            for worker in robot.strategy_workers:
                _validate_strategy_artifacts(worker.strategy, worker.model_artifact_path, worker.helformer_artifact_path, worker.use_helformer_forecast)

        robot_specs = [
            LiveRobotSpec(
                robot_id=robot.robot_id,
                display_name=robot.display_name,
                strategy_workers=[
                    LiveStrategyWorkerSpec(
                        worker_id=worker.worker_id or new_id(f"worker_{worker.strategy}"),
                        strategy_name=worker.strategy,
                        model_artifact_path=worker.model_artifact_path,
                        helformer_artifact_path=worker.helformer_artifact_path,
                        use_helformer_forecast=worker.use_helformer_forecast,
                    )
                    for worker in robot.strategy_workers
                ],
            )
            for robot in request.robots
        ]

        session_id = live_sessions.create_session(
            config_path=request.config_path,
            robot_specs=robot_specs,
            replay_delay_seconds=request.replay_delay_seconds,
        )
    except Exception as exc:
        logger.exception("Failed to start live replay.")
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return LiveReplayResponse(session_id=session_id, websocket_url=f"/ws/live-replay/{session_id}")


@app.get("/live-replay/{session_id}/status", response_model=LiveSessionStatusResponse)
def get_live_replay_status(session_id: str) -> LiveSessionStatusResponse:
    session = live_sessions.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Live session not found.")
    return LiveSessionStatusResponse(
        session_id=session_id,
        is_running=session.is_running,
        error=session.error,
        event_count=len(session.events),
    )


@app.post("/live-replay/{session_id}/stop", response_model=LiveSessionStatusResponse)
def stop_live_replay(session_id: str) -> LiveSessionStatusResponse:
    if not live_sessions.stop_session(session_id):
        raise HTTPException(status_code=404, detail="Live session not found.")
    session = live_sessions.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Live session not found.")
    return LiveSessionStatusResponse(
        session_id=session_id,
        is_running=session.is_running,
        error=session.error,
        event_count=len(session.events),
    )


@app.websocket("/ws/live-replay/{session_id}")
async def live_replay_websocket(websocket: WebSocket, session_id: str) -> None:
    await websocket.accept()

    try:
        event_queue = await live_sessions.subscribe(session_id)
    except KeyError:
        await websocket.close(code=1008)
        return

    try:
        while True:
            event = await event_queue.get()
            await websocket.send_json(event)

            session = live_sessions.get_session(session_id)
            if session is not None and not session.is_running and event_queue.empty():
                await websocket.close(code=1000)
                break

            await asyncio.sleep(0)
    except WebSocketDisconnect:
        pass
    finally:
        live_sessions.unsubscribe(session_id, event_queue)


@app.get("/model-artifacts")
def list_model_artifacts() -> list[dict[str, str | int | None]]:
    roots = [
        get_project_root() / "outputs" / "models",
        get_project_root() / "models",
        get_project_root() / "artifacts",
    ]

    allowed_suffixes = {
        ".pkl",
        ".pickle",
        ".joblib",
        ".pt",
        ".pth",
        ".onnx",
        ".keras",
    }

    artifacts: list[dict[str, str | int | None]] = []
    seen: set[str] = set()

    for root in roots:
        if not root.exists() or not root.is_dir():
            continue

        for path in root.rglob("*"):
            if path.is_dir() and _looks_like_model_artifact_dir(path):
                relative_path = path.relative_to(get_project_root()).as_posix()
                if relative_path in seen:
                    continue
                seen.add(relative_path)
                metadata = _read_artifact_metadata(path)
                artifacts.append({
                    "path": relative_path,
                    "name": str(metadata.get("artifact_name") or metadata.get("artifact_role") or metadata.get("strategy") or path.name),
                    "size_bytes": _directory_size(path),
                    "modified_at": str(int(max((item.stat().st_mtime for item in path.rglob("*") if item.is_file()), default=path.stat().st_mtime))),
                    "strategy": None if metadata.get("strategy") is None else str(metadata.get("strategy")),
                    "artifact_role": None if metadata.get("artifact_role") is None else str(metadata.get("artifact_role")),
                    "model_type": None if metadata.get("model_type") is None else str(metadata.get("model_type")),
                })

        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in allowed_suffixes:
                continue
            if _is_inside_artifact_dir(path):
                continue
            relative_path = path.relative_to(get_project_root()).as_posix()
            if relative_path in seen:
                continue
            seen.add(relative_path)
            stat = path.stat()
            artifacts.append({
                "path": relative_path,
                "name": path.name,
                "size_bytes": stat.st_size,
                "modified_at": str(int(stat.st_mtime)),
                "strategy": None,
                "artifact_role": None,
                "model_type": None,
            })

    artifacts.sort(key=lambda item: int(item["modified_at"] or 0), reverse=True)
    return artifacts


def _looks_like_model_artifact_dir(path: Path) -> bool:
    return (path / "metadata.json").exists() and (
        (path / "model.joblib").exists()
        or (path / "model.keras").exists()
        or any(child.suffix.lower() in {".pt", ".pth", ".onnx", ".keras"} for child in path.iterdir() if child.is_file())
    )


def _is_inside_artifact_dir(path: Path) -> bool:
    return any(_looks_like_model_artifact_dir(parent) for parent in path.parents if parent != get_project_root())


def _directory_size(path: Path) -> int:
    return int(sum(item.stat().st_size for item in path.rglob("*") if item.is_file()))


def _read_artifact_metadata(path: Path) -> dict:
    metadata_path = path / "metadata.json"
    if not metadata_path.exists():
        return {}
    try:
        import json

        with metadata_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}
