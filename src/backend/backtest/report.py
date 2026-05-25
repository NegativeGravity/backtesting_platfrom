from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from backend.core.paths import resolve_project_path


class BacktestReportWriter:
    def __init__(self, output_dir: str | Path) -> None:
        self._output_dir = resolve_project_path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        run_id: str,
        config_snapshot: dict[str, Any],
        summary: dict[str, Any],
        trades: list[Any],
        equity_curve: list[Any] | pd.DataFrame,
        benchmark_curve: list[Any] | pd.DataFrame,
        execution_log: list[Any],
    ) -> Path:
        run_dir = self._output_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        self._write_json(run_dir / "summary.json", summary)
        self._write_json(run_dir / "config_snapshot.json", config_snapshot)

        self._write_csv(
            path=run_dir / "trades.csv",
            records=self._records_from_items(trades),
            fallback_columns=[
                "trade_id",
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
            ],
        )

        self._write_csv(
            path=run_dir / "execution_log.csv",
            records=self._records_from_items(execution_log),
            fallback_columns=[
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
                "robot_equity",
                "available_capital_after_fill",
            ],
        )

        self._write_frame_or_records(
            path=run_dir / "equity_curve.csv",
            data=equity_curve,
            fallback_columns=[
                "timestamp",
                "cash",
                "position_quantity",
                "position_market_value",
                "equity",
                "drawdown",
            ],
        )

        self._write_frame_or_records(
            path=run_dir / "benchmark_curve.csv",
            data=benchmark_curve,
            fallback_columns=[
                "timestamp",
                "equity",
            ],
        )

        return run_dir

    def _write_json(self, path: Path, data: dict[str, Any]) -> None:
        with path.open("w", encoding="utf-8") as file:
            json.dump(
                self._json_safe(data),
                file,
                indent=2,
                ensure_ascii=False,
            )

    def _write_csv(
        self,
        path: Path,
        records: list[dict[str, Any]],
        fallback_columns: list[str],
    ) -> None:
        if records:
            frame = pd.DataFrame(records)
        else:
            frame = pd.DataFrame(columns=fallback_columns)

        frame = frame.reindex(columns=list(dict.fromkeys([*fallback_columns, *frame.columns])))
        frame.to_csv(path, index=False)

    def _write_frame_or_records(
        self,
        path: Path,
        data: list[Any] | pd.DataFrame,
        fallback_columns: list[str],
    ) -> None:
        if isinstance(data, pd.DataFrame):
            frame = data.copy()
        else:
            frame = pd.DataFrame(self._records_from_items(data))

        if frame.empty:
            frame = pd.DataFrame(columns=fallback_columns)

        frame = frame.reindex(columns=list(dict.fromkeys([*fallback_columns, *frame.columns])))
        frame.to_csv(path, index=False)

    def _records_from_items(self, items: list[Any] | pd.DataFrame) -> list[dict[str, Any]]:
        if isinstance(items, pd.DataFrame):
            return items.to_dict(orient="records")

        records: list[dict[str, Any]] = []

        for item in items:
            if is_dataclass(item):
                records.append(asdict(item))
            elif isinstance(item, dict):
                records.append(dict(item))
            else:
                records.append(dict(vars(item)))

        return records

    def _json_safe(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): self._json_safe(item) for key, item in value.items()}

        if isinstance(value, list):
            return [self._json_safe(item) for item in value]

        if isinstance(value, tuple):
            return [self._json_safe(item) for item in value]

        if hasattr(value, "isoformat"):
            return value.isoformat()

        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                pass

        return value