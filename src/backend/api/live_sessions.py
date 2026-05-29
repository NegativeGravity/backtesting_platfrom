from __future__ import annotations

import asyncio
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque

from backend.engine.live_replay import LiveReplayEngine, LiveRobotSpec
from backend.utils.ids import new_id


@dataclass
class LiveSession:
    session_id: str
    events: Deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=10_000))
    is_running: bool = False
    error: str | None = None
    stop_event: threading.Event = field(default_factory=threading.Event)


class LiveSessionManager:
    def __init__(self) -> None:
        self._sessions: dict[str, LiveSession] = {}
        self._subscribers: dict[str, dict[asyncio.Queue, asyncio.AbstractEventLoop]] = {}
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
            self._subscribers[session_id] = {}

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
        with self._lock:
            return self._sessions.get(session_id)

    def stop_session(self, session_id: str) -> bool:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return False
            session.stop_event.set()
        self._publish_event(
            session_id,
            {
                "event_type": "LIVE_SESSION_STOP_REQUESTED",
                "source": "live_session_manager",
                "payload": {"session_id": session_id},
            },
        )
        return True

    async def subscribe(self, session_id: str) -> asyncio.Queue:
        event_queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        loop = asyncio.get_running_loop()

        with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"Session not found: {session_id}")

            self._subscribers.setdefault(session_id, {})[event_queue] = loop

            for event in list(self._sessions[session_id].events)[-200:]:
                event_queue.put_nowait(event)

        return event_queue

    def unsubscribe(self, session_id: str, event_queue: asyncio.Queue) -> None:
        with self._lock:
            self._subscribers.get(session_id, {}).pop(event_queue, None)

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
                stop_event=self._sessions[session_id].stop_event,
            )
            result = engine.run()
            event_type = "LIVE_SESSION_STOPPED" if self._sessions[session_id].stop_event.is_set() else "LIVE_SESSION_COMPLETED"
            publish(
                {
                    "event_type": event_type,
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
            subscribers = list(self._subscribers.get(session_id, {}).items())

        for subscriber, loop in subscribers:
            def enqueue(queue: asyncio.Queue = subscriber, item: dict[str, Any] = event) -> None:
                try:
                    queue.put_nowait(item)
                except asyncio.QueueFull:
                    pass

            try:
                loop.call_soon_threadsafe(enqueue)
            except RuntimeError:
                self.unsubscribe(session_id, subscriber)