from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BotCommand:
    command_type: str
    payload: dict[str, Any]


@dataclass(frozen=True)
class BotResponse:
    bot_id: str
    response_type: str
    payload: dict[str, Any]