from __future__ import annotations

import asyncio
import logging

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from trading_system.api.live_sessions import LiveSessionManager
from trading_system.api.schemas import (
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
from trading_system.api.services.backtest_runner import (
    run_backtest_from_request,
    run_robot_backtest_from_request,
)
from trading_system.api.services.report_repository import ReportRepository
from trading_system.core.config import load_config
from trading_system.core.logging import setup_logging
from trading_system.engine.live_replay import LiveRobotSpec, LiveStrategyWorkerSpec
from trading_system.strategy.factory import STRATEGIES_REQUIRING_ARTIFACT, SUPPORTED_STRATEGIES
from trading_system.utils.ids import new_id

setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Trading Backtesting Platform API",
    version="0.4.0",
    description="Control API for robot-level backtesting, live replay, and report visualization.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

repository = ReportRepository()
live_sessions = LiveSessionManager()


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


@app.get("/strategies", response_model=list[StrategyInfo])
def list_strategies() -> list[StrategyInfo]:
    return [
        StrategyInfo(
            name=strategy_name,  # type: ignore[arg-type]
            requires_model_artifact=strategy_name in STRATEGIES_REQUIRING_ARTIFACT,
        )
        for strategy_name in sorted(SUPPORTED_STRATEGIES)
    ]


@app.post("/backtests/run", response_model=BacktestRunResponse)
def run_backtest(request: BacktestRunRequest) -> BacktestRunResponse:
    try:
        run_dir, summary = run_backtest_from_request(
            config_path=request.config_path,
            strategy_name=request.strategy,
            model_artifact_path=request.model_artifact_path,
        )
    except Exception as exc:
        logger.exception("Backtest failed.")
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return BacktestRunResponse(run_id=run_dir.name, run_dir=str(run_dir), summary=summary)


@app.post("/robot-backtests/run", response_model=BacktestRunResponse)
def run_robot_backtest(request: RobotBacktestRequest) -> BacktestRunResponse:
    try:
        robot_payload = {
            "robot_id": request.robot.robot_id,
            "display_name": request.robot.display_name,
            "strategy_workers": [
                {
                    "worker_id": worker.worker_id or new_id(f"worker_{worker.strategy}"),
                    "strategy": worker.strategy,
                    "model_artifact_path": worker.model_artifact_path,
                }
                for worker in request.robot.strategy_workers
            ],
        }
        run_dir, summary = run_robot_backtest_from_request(
            config_path=request.config_path,
            robot=robot_payload,
        )
    except Exception as exc:
        logger.exception("Robot backtest failed.")
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return BacktestRunResponse(run_id=run_dir.name, run_dir=str(run_dir), summary=summary)


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


@app.post("/live-replay/start", response_model=LiveReplayResponse)
def start_live_replay(request: LiveReplayRequest) -> LiveReplayResponse:
    try:
        robot_specs = [
            LiveRobotSpec(
                robot_id=robot.robot_id,
                display_name=robot.display_name,
                strategy_workers=[
                    LiveStrategyWorkerSpec(
                        worker_id=worker.worker_id or new_id(f"worker_{worker.strategy}"),
                        strategy_name=worker.strategy,
                        model_artifact_path=worker.model_artifact_path,
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
