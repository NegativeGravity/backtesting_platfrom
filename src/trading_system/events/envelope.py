from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import uuid4


class EventType(str, Enum):
    ENGINE_STARTED = "ENGINE_STARTED"
    ENGINE_STOPPED = "ENGINE_STOPPED"

    MARKET_BAR = "MARKET_BAR"

    SIGNAL_GENERATED = "SIGNAL_GENERATED"

    ORDER_SCHEDULED = "ORDER_SCHEDULED"
    ORDER_FILLED = "ORDER_FILLED"
    ORDER_REJECTED = "ORDER_REJECTED"
    ORDER_CANCELLED = "ORDER_CANCELLED"

    POSITION_OPENED = "POSITION_OPENED"
    POSITION_UPDATED = "POSITION_UPDATED"
    POSITION_CLOSED = "POSITION_CLOSED"

    PORTFOLIO_UPDATED = "PORTFOLIO_UPDATED"

    BOT_STARTED = "BOT_STARTED"
    BOT_STOPPED = "BOT_STOPPED"
    BOT_HEARTBEAT = "BOT_HEARTBEAT"
    BOT_ERROR = "BOT_ERROR"

    SYSTEM_ALERT = "SYSTEM_ALERT"


@dataclass(frozen=True)
class EventEnvelope:
    event_type: EventType
    source: str
    payload: dict[str, Any]
    correlation_id: str | None = None
    event_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "correlation_id": self.correlation_id,
            "payload": self.payload,
        }