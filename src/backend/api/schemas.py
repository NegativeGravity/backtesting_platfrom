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
    "helformer_momentum",
    "adaptive_trend_expansion_pro",
    "capitulation_reversal_pro",
    "volatility_squeeze_breakout",
    "meta_labeled_alpha_allocator_pro",
]

JobStatus = Literal["queued", "running", "completed", "failed", "cancelled"]


class HealthResponse(BaseModel):
    status: str
    service: str


class BacktestRunRequest(BaseModel):
    strategy: StrategyName = "mean_reversion"
    config_path: str = "configs/backtest.yaml"
    model_artifact_path: str | None = None
    helformer_artifact_path: str | None = None


class BacktestRunResponse(BaseModel):
    run_id: str | None = None
    run_dir: str | None = None
    summary: dict[str, Any] = Field(default_factory=dict)
    job_id: str | None = None
    status: JobStatus | None = None
    error: str | None = None
    message: str | None = None
    created_at: float | None = None
    updated_at: float | None = None
    heartbeat_at: float | None = None
    elapsed_seconds: float | None = None


class BacktestJobResponse(BaseModel):
    job_id: str
    status: JobStatus
    run_id: str | None = None
    run_dir: str | None = None
    summary: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    message: str | None = None
    created_at: float
    updated_at: float
    heartbeat_at: float | None = None
    elapsed_seconds: float


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
    requires_helformer_forecaster: bool = False


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
    closed_positions: list[dict[str, Any]] = Field(default_factory=list)
    equity_curve: list[dict[str, Any]] = Field(default_factory=list)
    benchmark_curve: list[dict[str, Any]] = Field(default_factory=list)
    execution_log: list[dict[str, Any]] = Field(default_factory=list)
    orders: list[dict[str, Any]] = Field(default_factory=list)
    config: dict[str, Any] = Field(default_factory=dict)
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    model: dict[str, Any] = Field(default_factory=dict)


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
    helformer_artifact_path: str | None = None


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
