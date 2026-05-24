from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from trading_system.core.exceptions import ConfigurationError
from trading_system.core.paths import resolve_project_path


@dataclass(frozen=True)
class ProjectConfig:
    name: str
    environment: str


@dataclass(frozen=True)
class DataConfig:
    symbol: str
    asset_class: str
    timeframe: str
    path: Path
    timestamp_column: str
    timezone: str


@dataclass(frozen=True)
class BacktestConfig:
    initial_capital: float
    base_currency: str
    start_date: str | None
    end_date: str | None
    periods_per_year: int


@dataclass(frozen=True)
class StrategyConfig:
    name: str
    lookback_window: int
    entry_z_score: float
    exit_z_score: float
    stop_z_score: float
    min_bars_required: int


@dataclass(frozen=True)
class MLConfig:
    horizon_bars: int
    train_ratio: float
    validation_ratio: float
    min_return_threshold: float
    probability_threshold: float
    exit_probability: float
    model_artifact_dir: Path
    model_artifact_path: Path | None


@dataclass(frozen=True)
class RiskConfig:
    risk_per_trade_pct: float
    max_position_notional_pct: float
    min_cash_pct: float
    stop_distance_pct: float
    min_order_notional: float


@dataclass(frozen=True)
class ExecutionConfig:
    execution_timing: str
    fee_rate: float
    slippage_bps: float
    allow_fractional_quantity: bool
    pessimistic_intrabar_policy: bool


@dataclass(frozen=True)
class ReportingConfig:
    output_dir: Path
    save_trades: bool
    save_equity_curve: bool
    save_benchmark_curve: bool
    save_execution_log: bool


@dataclass(frozen=True)
class AppConfig:
    project: ProjectConfig
    data: DataConfig
    backtest: BacktestConfig
    strategy: StrategyConfig
    ml: MLConfig
    risk: RiskConfig
    execution: ExecutionConfig
    reporting: ReportingConfig
    raw: dict[str, Any]


def _require(mapping: dict[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ConfigurationError(f"Missing required config key: {key}")
    return mapping[key]


def _optional_path(value: Any) -> Path | None:
    if value is None:
        return None
    return resolve_project_path(str(value))


def _validate_ratios(train_ratio: float, validation_ratio: float) -> None:
    if not 0 < train_ratio < 1:
        raise ConfigurationError("ml.train_ratio must be between 0 and 1.")

    if not 0 < validation_ratio < 1:
        raise ConfigurationError("ml.validation_ratio must be between 0 and 1.")

    if train_ratio + validation_ratio >= 1:
        raise ConfigurationError("ml.train_ratio + ml.validation_ratio must be less than 1.")


def load_config(config_path: str | Path) -> AppConfig:
    resolved_path = resolve_project_path(config_path)

    if not resolved_path.exists():
        raise ConfigurationError(f"Config file not found: {resolved_path}")

    with resolved_path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)

    if not isinstance(raw, dict):
        raise ConfigurationError("Config file must contain a YAML mapping.")

    project = _require(raw, "project")
    data = _require(raw, "data")
    backtest = _require(raw, "backtest")
    strategy = _require(raw, "strategy")
    ml = _require(raw, "ml")
    risk = _require(raw, "risk")
    execution = _require(raw, "execution")
    reporting = _require(raw, "reporting")

    train_ratio = float(_require(ml, "train_ratio"))
    validation_ratio = float(_require(ml, "validation_ratio"))
    _validate_ratios(train_ratio, validation_ratio)

    return AppConfig(
        project=ProjectConfig(
            name=str(_require(project, "name")),
            environment=str(_require(project, "environment")),
        ),
        data=DataConfig(
            symbol=str(_require(data, "symbol")),
            asset_class=str(_require(data, "asset_class")),
            timeframe=str(_require(data, "timeframe")),
            path=resolve_project_path(str(_require(data, "path"))),
            timestamp_column=str(_require(data, "timestamp_column")),
            timezone=str(_require(data, "timezone")),
        ),
        backtest=BacktestConfig(
            initial_capital=float(_require(backtest, "initial_capital")),
            base_currency=str(_require(backtest, "base_currency")),
            start_date=backtest.get("start_date"),
            end_date=backtest.get("end_date"),
            periods_per_year=int(_require(backtest, "periods_per_year")),
        ),
        strategy=StrategyConfig(
            name=str(_require(strategy, "name")),
            lookback_window=int(_require(strategy, "lookback_window")),
            entry_z_score=float(_require(strategy, "entry_z_score")),
            exit_z_score=float(_require(strategy, "exit_z_score")),
            stop_z_score=float(_require(strategy, "stop_z_score")),
            min_bars_required=int(_require(strategy, "min_bars_required")),
        ),
        ml=MLConfig(
            horizon_bars=int(_require(ml, "horizon_bars")),
            train_ratio=train_ratio,
            validation_ratio=validation_ratio,
            min_return_threshold=float(_require(ml, "min_return_threshold")),
            probability_threshold=float(_require(ml, "probability_threshold")),
            exit_probability=float(_require(ml, "exit_probability")),
            model_artifact_dir=resolve_project_path(str(_require(ml, "model_artifact_dir"))),
            model_artifact_path=_optional_path(ml.get("model_artifact_path")),
        ),
        risk=RiskConfig(
            risk_per_trade_pct=float(_require(risk, "risk_per_trade_pct")),
            max_position_notional_pct=float(_require(risk, "max_position_notional_pct")),
            min_cash_pct=float(_require(risk, "min_cash_pct")),
            stop_distance_pct=float(_require(risk, "stop_distance_pct")),
            min_order_notional=float(_require(risk, "min_order_notional")),
        ),
        execution=ExecutionConfig(
            execution_timing=str(_require(execution, "execution_timing")),
            fee_rate=float(_require(execution, "fee_rate")),
            slippage_bps=float(_require(execution, "slippage_bps")),
            allow_fractional_quantity=bool(_require(execution, "allow_fractional_quantity")),
            pessimistic_intrabar_policy=bool(_require(execution, "pessimistic_intrabar_policy")),
        ),
        reporting=ReportingConfig(
            output_dir=resolve_project_path(str(_require(reporting, "output_dir"))),
            save_trades=bool(_require(reporting, "save_trades")),
            save_equity_curve=bool(_require(reporting, "save_equity_curve")),
            save_benchmark_curve=bool(_require(reporting, "save_benchmark_curve")),
            save_execution_log=bool(_require(reporting, "save_execution_log")),
        ),
        raw=raw,
    )