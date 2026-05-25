from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.errors import EmptyDataError

from backend.core.paths import resolve_project_path


class ReportRepository:
    def __init__(self, output_dir: str | Path = "outputs/backtests") -> None:
        self._output_dir = resolve_project_path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def list_runs(self) -> list[dict[str, Any]]:
        return self._list_runs_impl()

    def read_runs(self) -> list[dict[str, Any]]:
        return self._list_runs_impl()

    def get_report(self, run_id: str) -> dict[str, Any]:
        return self._read_report_impl(run_id)

    def read_report(self, run_id: str) -> dict[str, Any]:
        return self._read_report_impl(run_id)

    def get_summary(self, run_id: str) -> dict[str, Any]:
        return self._read_summary_impl(run_id)

    def read_summary(self, run_id: str) -> dict[str, Any]:
        return self._read_summary_impl(run_id)

    def get_chart_data(self, run_id: str) -> dict[str, Any]:
        return self._read_chart_data_impl(run_id)

    def read_chart_data(self, run_id: str) -> dict[str, Any]:
        return self._read_chart_data_impl(run_id)

    def _list_runs_impl(self) -> list[dict[str, Any]]:
        runs: list[dict[str, Any]] = []

        if not self._output_dir.exists():
            return runs

        for run_dir in sorted(self._output_dir.iterdir(), reverse=True):
            if not run_dir.is_dir():
                continue

            summary_path = run_dir / "summary.json"
            if not summary_path.exists():
                continue

            try:
                summary = self._read_json(summary_path)
            except Exception:
                continue

            metrics = summary.get("metrics", {})
            runs.append(
                self._json_safe(
                    {
                        "run_id": run_dir.name,
                        "strategy": summary.get("strategy"),
                        "symbol": summary.get("symbol"),
                        "final_equity": metrics.get("final_equity"),
                        "total_return": metrics.get("total_return"),
                        "benchmark_return": metrics.get("benchmark_return"),
                        "max_drawdown": metrics.get("max_drawdown"),
                        "created_at": self._created_at_from_run_id(run_dir.name),
                    }
                )
            )

        return runs

    def _read_report_impl(self, run_id: str) -> dict[str, Any]:
        run_dir = self._get_run_dir(run_id)
        summary = self._read_json(run_dir / "summary.json")

        trades = self._read_csv_records(
            run_dir / "trades.csv",
            fallback_columns=self._trade_columns(),
        )
        closed_positions = self._read_csv_records(
            run_dir / "closed_positions.csv",
            fallback_columns=self._trade_columns(),
        )
        if not trades and closed_positions:
            trades = closed_positions

        execution_log = self._read_csv_records(
            run_dir / "execution_log.csv",
            fallback_columns=self._execution_log_columns(),
        )
        equity_curve = self._read_curve_records(
            run_dir / "equity_curve.csv",
            fallback_columns=[
                "timestamp",
                "cash",
                "position_quantity",
                "position_market_value",
                "equity",
                "drawdown",
            ],
        )
        benchmark_curve = self._read_curve_records(
            run_dir / "benchmark_curve.csv",
            fallback_columns=["timestamp", "equity", "benchmark_equity"],
        )

        return self._json_safe(
            {
                "run_id": summary.get("run_id", run_id),
                "summary": summary,
                "trades": trades,
                "closed_positions": closed_positions,
                "execution_log": execution_log,
                "equity_curve": equity_curve,
                "benchmark_curve": benchmark_curve,
            }
        )

    def _read_summary_impl(self, run_id: str) -> dict[str, Any]:
        run_dir = self._get_run_dir(run_id)
        return self._json_safe(self._read_json(run_dir / "summary.json"))

    def _read_chart_data_impl(self, run_id: str) -> dict[str, Any]:
        run_dir = self._get_run_dir(run_id)
        summary = self._read_json(run_dir / "summary.json")

        equity_curve = self._read_line_points(run_dir / "equity_curve.csv")
        benchmark_curve = self._read_line_points(run_dir / "benchmark_curve.csv")
        chart_time_range = self._line_time_range(equity_curve) or self._line_time_range(
            benchmark_curve
        )
        candles = self._read_candles(summary, time_range=chart_time_range)

        markers = self._read_trade_markers(run_dir / "trades.csv")
        if not markers:
            markers = self._read_trade_markers(run_dir / "closed_positions.csv")

        return self._json_safe(
            {
                "candles": candles,
                "equity_curve": equity_curve,
                "benchmark_curve": benchmark_curve,
                "markers": markers,
            }
        )

    def _read_candles(
        self,
        summary: dict[str, Any],
        time_range: tuple[int, int] | None = None,
    ) -> list[dict[str, Any]]:
        dataset_path = summary.get("dataset")
        if not dataset_path:
            return []

        dataset = self._safe_read_csv(
            resolve_project_path(dataset_path),
            fallback_columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        if dataset.empty:
            return []

        required_columns = {"timestamp", "open", "high", "low", "close"}
        if not required_columns.issubset(set(dataset.columns)):
            return []

        by_time: dict[int, dict[str, Any]] = {}
        start_time, end_time = time_range if time_range else (None, None)

        for _, row in dataset.iterrows():
            try:
                unix_time = self._to_unix_seconds(row["timestamp"])
                if start_time is not None and unix_time < start_time:
                    continue
                if end_time is not None and unix_time > end_time:
                    continue

                open_price = float(row["open"])
                high_price = float(row["high"])
                low_price = float(row["low"])
                close_price = float(row["close"])

                if not all(
                    math.isfinite(value)
                    for value in (open_price, high_price, low_price, close_price)
                ):
                    continue

                if high_price < low_price:
                    continue

                existing = by_time.get(unix_time)
                if existing is None:
                    by_time[unix_time] = {
                        "time": unix_time,
                        "open": open_price,
                        "high": max(high_price, open_price, close_price),
                        "low": min(low_price, open_price, close_price),
                        "close": close_price,
                    }
                else:
                    by_time[unix_time] = {
                        "time": unix_time,
                        "open": existing["open"],
                        "high": max(existing["high"], high_price, open_price, close_price),
                        "low": min(existing["low"], low_price, open_price, close_price),
                        "close": close_price,
                    }
            except Exception:
                continue

        return [by_time[key] for key in sorted(by_time)]

    def _read_line_points(self, path: Path) -> list[dict[str, Any]]:
        frame = self._safe_read_csv(
            path,
            fallback_columns=["timestamp", "equity", "value", "benchmark_equity"],
        )
        if frame.empty or "timestamp" not in frame.columns:
            return []

        value_column = None
        for candidate in ("equity", "value", "benchmark_equity"):
            if candidate in frame.columns:
                value_column = candidate
                break

        if value_column is None:
            return []

        by_time: dict[int, dict[str, Any]] = {}

        for _, row in frame.iterrows():
            try:
                value = row[value_column]
                if self._is_missing(value):
                    continue

                unix_time = self._to_unix_seconds(row["timestamp"])
                numeric_value = float(value)

                if not math.isfinite(numeric_value):
                    continue

                by_time[unix_time] = {"time": unix_time, "value": numeric_value}
            except Exception:
                continue

        return [by_time[key] for key in sorted(by_time)]

    def _read_trade_markers(self, path: Path) -> list[dict[str, Any]]:
        frame = self._safe_read_csv(
            path,
            fallback_columns=[
                "entry_time",
                "exit_time",
                "entry_price",
                "exit_price",
                "side",
                "net_pnl",
                "worker_id",
                "strategy",
            ],
        )
        if frame.empty:
            return []

        markers: list[dict[str, Any]] = []

        for _, row in frame.iterrows():
            side = str(row.get("side", "LONG") or "LONG").upper()
            is_short = "SHORT" in side
            worker_id = str(row.get("worker_id", "") or "")
            strategy = str(row.get("strategy", "") or "")
            label_prefix = " ".join(part for part in (worker_id, strategy) if part).strip()
            label_prefix = f"{label_prefix} " if label_prefix else ""

            entry_time = row.get("entry_time")
            exit_time = row.get("exit_time")

            if not self._is_missing(entry_time):
                try:
                    markers.append(
                        {
                            "time": self._to_unix_seconds(entry_time),
                            "position": "aboveBar" if is_short else "belowBar",
                            "color": "#f87171" if is_short else "#22c55e",
                            "shape": "arrowDown" if is_short else "arrowUp",
                            "text": f"{label_prefix}{'SHORT' if is_short else 'LONG'} OPEN".strip(),
                        }
                    )
                except Exception:
                    pass

            if not self._is_missing(exit_time):
                pnl_label = ""
                try:
                    net_pnl = row.get("net_pnl")
                    if not self._is_missing(net_pnl):
                        pnl = float(net_pnl)
                        pnl_label = f" {'+' if pnl >= 0 else ''}{pnl:.2f}"
                except Exception:
                    pnl_label = ""

                try:
                    markers.append(
                        {
                            "time": self._to_unix_seconds(exit_time),
                            "position": "belowBar" if is_short else "aboveBar",
                            "color": "#38bdf8" if pnl_label.startswith(" +") else "#ef4444",
                            "shape": "arrowUp" if is_short else "arrowDown",
                            "text": f"{label_prefix}CLOSE{pnl_label}".strip(),
                        }
                    )
                except Exception:
                    pass

        markers.sort(key=lambda item: item["time"])
        return markers

    def _get_run_dir(self, run_id: str) -> Path:
        if not run_id or run_id.strip() == "":
            raise FileNotFoundError("Run ID cannot be empty.")

        if "/" in run_id or "\\" in run_id or ".." in run_id:
            raise FileNotFoundError(f"Invalid run id: {run_id}")

        run_dir = self._output_dir / run_id

        if not run_dir.exists() or not run_dir.is_dir():
            raise FileNotFoundError(f"Run not found: {run_id}")

        return run_dir

    def _read_json(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}

        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        return data if isinstance(data, dict) else {}

    def _safe_read_csv(
        self,
        path: Path,
        fallback_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        columns = fallback_columns or []

        if not path.exists() or path.stat().st_size == 0:
            return pd.DataFrame(columns=columns)

        try:
            return pd.read_csv(path)
        except EmptyDataError:
            return pd.DataFrame(columns=columns)

    def _read_csv_records(
        self,
        path: Path,
        fallback_columns: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        frame = self._safe_read_csv(path, fallback_columns=fallback_columns)

        if frame.empty:
            return []

        frame = frame.where(pd.notnull(frame), None)
        return [self._json_safe(record) for record in frame.to_dict(orient="records")]

    def _read_curve_records(
        self,
        path: Path,
        fallback_columns: list[str],
    ) -> list[dict[str, Any]]:
        frame = self._safe_read_csv(path, fallback_columns=fallback_columns)

        if frame.empty:
            return []

        frame = frame.where(pd.notnull(frame), None)
        return [self._json_safe(record) for record in frame.to_dict(orient="records")]

    @staticmethod
    def _line_time_range(points: list[dict[str, Any]]) -> tuple[int, int] | None:
        times = [int(point["time"]) for point in points if "time" in point]

        if not times:
            return None

        return min(times), max(times)

    @staticmethod
    def _to_unix_seconds(value: Any) -> int:
        timestamp = pd.to_datetime(value, utc=True, errors="coerce")

        if pd.isna(timestamp):
            raise ValueError(f"Invalid timestamp: {value}")

        return int(timestamp.timestamp())

    @staticmethod
    def _is_missing(value: Any) -> bool:
        if value is None:
            return True

        try:
            return bool(pd.isna(value))
        except Exception:
            return False

    def _json_safe(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): self._json_safe(item) for key, item in value.items()}

        if isinstance(value, list):
            return [self._json_safe(item) for item in value]

        if isinstance(value, tuple):
            return [self._json_safe(item) for item in value]

        if isinstance(value, pd.Timestamp):
            return value.isoformat()

        if isinstance(value, float):
            if math.isnan(value) or math.isinf(value):
                return None
            return value

        if hasattr(value, "item"):
            try:
                return self._json_safe(value.item())
            except Exception:
                return value

        return value

    @staticmethod
    def _created_at_from_run_id(run_id: str) -> str | None:
        parts = run_id.split("_")

        if len(parts) < 3:
            return None

        date_part = parts[-2]
        time_part = parts[-1]

        if len(date_part) != 8 or len(time_part) != 6:
            return None

        return (
            f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:]}"
            f"T{time_part[:2]}:{time_part[2:4]}:{time_part[4:]}"
        )

    @staticmethod
    def _trade_columns() -> list[str]:
        return [
            "trade_id",
            "position_id",
            "symbol",
            "entry_time",
            "exit_time",
            "entry_price",
            "exit_price",
            "quantity",
            "gross_pnl",
            "net_pnl",
            "fees",
            "slippage_cost",
            "return_pct",
            "exit_reason",
            "worker_id",
            "strategy",
            "side",
        ]

    @staticmethod
    def _execution_log_columns() -> list[str]:
        return [
            "timestamp",
            "robot_id",
            "robot_name",
            "worker_id",
            "strategy",
            "symbol",
            "action",
            "side",
            "quantity",
            "requested_bar_time",
            "filled_bar_time",
            "fill_price",
            "fee",
            "slippage_cost",
            "status",
            "reason",
        ]