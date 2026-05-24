from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import pandas as pd


class SignalType(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    EXIT = "EXIT"
    HOLD = "HOLD"


@dataclass(frozen=True)
class Signal:
    timestamp: pd.Timestamp
    symbol: str
    signal_type: SignalType
    confidence: float
    reason: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_actionable(self) -> bool:
        return self.signal_type in {
            SignalType.LONG,
            SignalType.SHORT,
            SignalType.EXIT,
        }