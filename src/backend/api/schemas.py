from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

StrategyName = Literal[
    "mean_reversion",
    "ml_momentum",
    "adaptive_trend_breakout",
    "liquidity_sweep_reversal",
    "ml_regime_meta_label",
    "dl_temporal_fusion_momentum",
]


class HealthResponse(BaseModel):
    status: str
    service: str


class BacktestRunRequest(BaseModel):
    strategy: StrategyName = "mean_reversion"
    config_path: str = "configs/backtest.yaml"
    model_artifact_path: str | None = None


class BacktestRunResponse(BaseModel):
    run_id: str
    run_dir: str
    summary: dict[str, Any]


class BacktestRunListItem(BaseModel):
    run_id: str
    strategy: str | None = None
    symbol: str | None = None
    final_equity: float | None = None
    total_return: float | None = None
    benchmark_return: float | None = None
    max_drawdown: float | None = None
    created_at: str | None = None


class DatasetInfo(BaseModel):
    symbol: str
    timeframe: str
    path: str


class StrategyInfo(BaseModel):
    name: StrategyName
    requires_model_artifact: bool


class CandlePoint(BaseModel):
    time: int
    open: float
    high: float
    low: float
    close: float


class LinePoint(BaseModel):
    time: int
    value: float


class TradeMarker(BaseModel):
    time: int
    position: Literal["aboveBar", "belowBar"]
    color: str
    shape: Literal["arrowUp", "arrowDown"]
    text: str


class ChartDataResponse(BaseModel):
    candles: list[CandlePoint] = Field(default_factory=list)
    equity_curve: list[LinePoint] = Field(default_factory=list)
    benchmark_curve: list[LinePoint] = Field(default_factory=list)
    markers: list[TradeMarker] = Field(default_factory=list)


class ReportResponse(BaseModel):
    run_id: str
    summary: dict[str, Any] = Field(default_factory=dict)
    trades: list[dict[str, Any]] = Field(default_factory=list)
    equity_curve: list[dict[str, Any]] = Field(default_factory=list)
    benchmark_curve: list[dict[str, Any]] = Field(default_factory=list)
    execution_log: list[dict[str, Any]] = Field(default_factory=list)


class LiveReplayResponse(BaseModel):
    session_id: str
    websocket_url: str


class LiveSessionStatusResponse(BaseModel):
    session_id: str
    is_running: bool
    error: str | None = None
    event_count: int


class LiveStrategyWorkerConfig(BaseModel):
    worker_id: str | None = None
    strategy: StrategyName
    model_artifact_path: str | None = None


class LiveRobotConfig(BaseModel):
    robot_id: str
    display_name: str
    strategy_workers: list[LiveStrategyWorkerConfig]


class LiveReplayRequest(BaseModel):
    config_path: str = "configs/backtest.yaml"
    robots: list[LiveRobotConfig]
    replay_delay_seconds: float = 0.02


class RobotBacktestRequest(BaseModel):
    config_path: str = "configs/backtest.yaml"
    robot: LiveRobotConfig
