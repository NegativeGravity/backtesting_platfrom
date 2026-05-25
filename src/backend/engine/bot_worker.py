from __future__ import annotations

import logging
import queue
from dataclasses import dataclass
from typing import Any

import pandas as pd

from backend.core.config import load_config
from backend.core.paths import resolve_project_path
from backend.engine.messages import BotCommand, BotResponse
from backend.strategy.factory import create_strategy
from backend.strategy.signals import Signal

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BotPortfolioView:
    symbol: str
    position_quantity: float
    equity: float
    cash: float
    position_side: str | None = None
    current_price: float | None = None
    unrealized_pnl: float | None = None


def run_bot_worker(
    bot_id: str,
    strategy_name: str,
    config_path: str,
    model_artifact_path: str | None,
    command_queue,
    response_queue,
) -> None:
    try:
        config = load_config(config_path)

        resolved_model_artifact_path = (
            None
            if model_artifact_path is None
            else resolve_project_path(model_artifact_path)
        )

        strategy = create_strategy(
            config=config,
            strategy_name=strategy_name,
            model_artifact_path=resolved_model_artifact_path,
        )

        response_queue.put(
            BotResponse(
                bot_id=bot_id,
                response_type="BOT_STARTED",
                payload={
                    "bot_id": bot_id,
                    "strategy": strategy_name,
                },
            )
        )

        while True:
            try:
                command: BotCommand = command_queue.get(timeout=1.0)
            except queue.Empty:
                response_queue.put(
                    BotResponse(
                        bot_id=bot_id,
                        response_type="BOT_HEARTBEAT",
                        payload={
                            "bot_id": bot_id,
                            "strategy": strategy_name,
                        },
                    )
                )
                continue

            if command.command_type == "STOP":
                response_queue.put(
                    BotResponse(
                        bot_id=bot_id,
                        response_type="BOT_STOPPED",
                        payload={
                            "bot_id": bot_id,
                            "strategy": strategy_name,
                            "reason": "stop_command",
                        },
                    )
                )
                return

            if command.command_type != "MARKET_WINDOW":
                continue

            payload = command.payload
            market_window = _market_window_from_payload(payload["market_window"])
            portfolio = _portfolio_from_payload(payload.get("portfolio", {}))

            signal: Signal = strategy.generate_signal(
                market_window=market_window,
                portfolio=portfolio,
            )

            response_queue.put(
                BotResponse(
                    bot_id=bot_id,
                    response_type="SIGNAL_GENERATED",
                    payload={
                        "bot_id": bot_id,
                        "strategy": strategy_name,
                        "timestamp": signal.timestamp.isoformat(),
                        "symbol": signal.symbol,
                        "signal_type": signal.signal_type.value,
                        "confidence": signal.confidence,
                        "reason": signal.reason,
                        "metadata": signal.metadata,
                    },
                )
            )

    except KeyboardInterrupt:
        logger.info("Bot worker interrupted: %s", bot_id)
        try:
            response_queue.put(
                BotResponse(
                    bot_id=bot_id,
                    response_type="BOT_STOPPED",
                    payload={
                        "bot_id": bot_id,
                        "strategy": strategy_name,
                        "reason": "keyboard_interrupt",
                    },
                )
            )
        except Exception:
            pass

    except Exception as exc:
        logger.exception("Bot worker failed: %s", bot_id)

        try:
            response_queue.put(
                BotResponse(
                    bot_id=bot_id,
                    response_type="BOT_ERROR",
                    payload={
                        "bot_id": bot_id,
                        "strategy": strategy_name,
                        "error": str(exc),
                    },
                )
            )
        except Exception:
            pass


def _market_window_from_payload(rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)

    if frame.empty:
        return frame

    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)

    for column in ("open", "high", "low", "close", "volume"):
        if column in frame.columns:
            frame[column] = frame[column].astype(float)

    return frame


def _portfolio_from_payload(payload: dict[str, Any]) -> BotPortfolioView:
    return BotPortfolioView(
        symbol=str(payload.get("symbol", "")),
        position_quantity=float(payload.get("position_quantity", 0.0)),
        equity=float(payload.get("equity", 0.0)),
        cash=float(payload.get("cash", 0.0)),
        position_side=(
            None
            if payload.get("position_side") is None
            else str(payload.get("position_side"))
        ),
        current_price=(
            None
            if payload.get("current_price") is None
            else float(payload.get("current_price"))
        ),
        unrealized_pnl=(
            None
            if payload.get("unrealized_pnl") is None
            else float(payload.get("unrealized_pnl"))
        ),
    )