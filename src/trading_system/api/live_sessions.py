from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field
from typing import Any

from trading_system.engine.live_replay import LiveReplayEngine, LiveRobotSpec
from trading_system.utils.ids import new_id


@dataclass
class LiveSession:
    session_id: str
    events: list[dict[str, Any]] = field(default_factory=list)
    is_running: bool = False
    error: str | None = None


class LiveSessionManager:
    def __init__(self) -> None:
        self._sessions: dict[str, LiveSession] = {}
        self._subscribers: dict[str, set[asyncio.Queue]] = {}
        self._lock = threading.Lock()

    def create_session(
            self,
            config_path: str,
            robot_specs: list[LiveRobotSpec],
            replay_delay_seconds: float,
    ) -> str:

        session_id = new_id("session")
        session = LiveSession(session_id=session_id, is_running=True)

        with self._lock:
            self._sessions[session_id] = session
            self._subscribers[session_id] = set()

        thread = threading.Thread(
            target=self._run_engine_thread,
            kwargs={
                "session_id": session_id,
                "config_path": config_path,
                "robot_specs": robot_specs,
                "replay_delay_seconds": replay_delay_seconds,
            },
            daemon=True,
        )
        thread.start()

        return session_id

    def get_session(self, session_id: str) -> LiveSession | None:
        return self._sessions.get(session_id)

    async def subscribe(self, session_id: str) -> asyncio.Queue:
        event_queue: asyncio.Queue = asyncio.Queue(maxsize=1000)

        with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"Session not found: {session_id}")

            self._subscribers.setdefault(session_id, set()).add(event_queue)

            for event in self._sessions[session_id].events[-200:]:
                event_queue.put_nowait(event)

        return event_queue

    def unsubscribe(self, session_id: str, event_queue: asyncio.Queue) -> None:
        with self._lock:
            self._subscribers.get(session_id, set()).discard(event_queue)

    def _run_engine_thread(
            self,
            session_id: str,
            config_path: str,
            robot_specs: list[LiveRobotSpec],
            replay_delay_seconds: float,
    ) -> None:
        def publish(event: dict[str, Any]) -> None:
            self._publish_event(session_id, event)

        try:
            engine = LiveReplayEngine(
                config_path=config_path,
                robot_specs=robot_specs,
                replay_delay_seconds=replay_delay_seconds,
                event_callback=publish,
            )
            result = engine.run()
            publish(
                {
                    "event_type": "LIVE_SESSION_COMPLETED",
                    "source": "live_session_manager",
                    "payload": {
                        "session_id": result.session_id,
                        "output_dirs": result.output_dirs,
                    },
                }
            )
        except Exception as exc:
            with self._lock:
                session = self._sessions[session_id]
                session.error = str(exc)

            publish(
                {
                    "event_type": "LIVE_SESSION_ERROR",
                    "source": "live_session_manager",
                    "payload": {
                        "session_id": session_id,
                        "error": str(exc),
                    },
                }
            )
        finally:
            with self._lock:
                self._sessions[session_id].is_running = False

    def _publish_event(self, session_id: str, event: dict[str, Any]) -> None:
        with self._lock:
            session = self._sessions[session_id]
            session.events.append(event)
            subscribers = list(self._subscribers.get(session_id, set()))

        for subscriber in subscribers:
            try:
                subscriber.put_nowait(event)
            except asyncio.QueueFull:
                pass