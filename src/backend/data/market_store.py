from __future__ import annotations

import hashlib
import io
import os
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd
import psycopg
import pyarrow as pa
import pyarrow.ipc as ipc
import redis
import yaml

from backend.core.paths import resolve_config_path, resolve_project_path

SplitName = Literal["train", "validation", "test", "full"]


@dataclass(frozen=True)
class QuestDBConfig:
    table: str = "btcusdt_klines_1h"
    pg_host: str = "questdb"
    pg_port: int = 8812
    pg_user: str = "admin"
    pg_password: str = "quest"
    pg_database: str = "qdb"


@dataclass(frozen=True)
class RedisConfig:
    host: str = "redis"
    port: int = 6379
    db: int = 0
    ttl_seconds: int = 604800
    key_prefix: str = "market:klines:v2"


@dataclass(frozen=True)
class MarketDataConfig:
    symbol: str = "BTCUSDT"
    interval: str = "1h"
    questdb: QuestDBConfig = QuestDBConfig()
    redis: RedisConfig = RedisConfig()

    @staticmethod
    def from_yaml(path: str | Path | None = None) -> "MarketDataConfig":
        config_path = resolve_config_path(path or os.getenv("MARKET_DATA_CONFIG") or "configs/market_data.yaml")

        with config_path.open("r", encoding="utf-8") as file:
            raw = yaml.safe_load(file)

        qdb = raw["questdb"]
        redis_config = raw["redis"]

        return MarketDataConfig(
            symbol=raw["symbol"],
            interval=raw["interval"],
            questdb=QuestDBConfig(
                table=qdb["table"],
                pg_host=(
                    os.getenv("QUESTDB_PG_HOST")
                    or os.getenv("QUESTDB_HOST")
                    or qdb["pg_host"]
                ),
                pg_port=int(
                    os.getenv("QUESTDB_PG_PORT")
                    or qdb["pg_port"]
                ),
                pg_user=qdb["pg_user"],
                pg_password=qdb["pg_password"],
                pg_database=qdb["pg_database"],
            ),
            redis=RedisConfig(
                host=os.getenv("REDIS_HOST") or redis_config["host"],
                port=int(os.getenv("REDIS_PORT") or redis_config["port"]),
                db=int(redis_config["db"]),
                ttl_seconds=int(redis_config["ttl_seconds"]),
                key_prefix=redis_config["key_prefix"],
            ),
        )


class MarketDataStore:

    def __init__(self, config: MarketDataConfig | None = None) -> None:
        self._config = config or MarketDataConfig()
        self._redis = redis.Redis(
            host=self._config.redis.host,
            port=self._config.redis.port,
            db=self._config.redis.db,
            socket_timeout=3,
            socket_connect_timeout=3,
        )

    @staticmethod
    def from_yaml(path: str | Path | None = None) -> "MarketDataStore":
        return MarketDataStore(MarketDataConfig.from_yaml(path))

    def load_ohlcv(
        self,
        start: str,
        end: str,
        symbol: str | None = None,
        interval: str | None = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        symbol = symbol or self._config.symbol
        interval = interval or self._config.interval
        cache_key = self._cache_key(symbol=symbol, interval=interval, start=start, end=end)

        if use_cache:
            cached = self._get_cache(cache_key)
            if cached is not None:
                return cached

        frame = self._query_questdb(symbol=symbol, interval=interval, start=start, end=end)
        frame = self._normalize_for_engine(frame)

        if use_cache:
            self._set_cache(cache_key, frame)

        return frame

    def load_split_from_yaml(
        self,
        split: SplitName,
        market_config_path: str | Path = "configs/market_data.yaml",
        split_policy: str = "recommended",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        with resolve_config_path(market_config_path).open("r", encoding="utf-8") as file:
            raw = yaml.safe_load(file)

        split_key = {
            "recommended": "splits_recommended",
            "user_requested_no_overlap": "splits_user_requested_no_overlap",
            "user_requested_with_leakage_warning": "splits_user_requested_with_leakage_warning",
        }[split_policy]
        split_config = raw[split_key]

        if split == "train":
            start, end = split_config["train_start"], split_config["train_end"]
        elif split == "validation":
            start, end = split_config["validation_start"], split_config["validation_end"]
        elif split == "test":
            start, end = split_config["test_start"], split_config["test_end"]
        else:
            start, end = "2020-01-01T00:00:00Z", "2026-01-01T00:00:00Z"

        return self.load_ohlcv(start=start, end=end, use_cache=use_cache)

    def _query_questdb(self, symbol: str, interval: str, start: str, end: str) -> pd.DataFrame:
        table = self._validated_table_name(self._config.questdb.table)
        sql = f"""
        SELECT
            ts AS timestamp,
            open,
            high,
            low,
            close,
            volume,
            quote_volume,
            trade_count,
            taker_buy_volume,
            taker_buy_quote_volume
        FROM {table}
        WHERE symbol = %(symbol)s
          AND interval = %(interval)s
          AND ts >= %(start)s
          AND ts < %(end)s
        ORDER BY ts ASC;
        """

        with psycopg.connect(self._pg_dsn()) as connection:
            return pd.read_sql(
                sql,
                connection,
                params={"symbol": symbol, "interval": interval, "start": start, "end": end},
            )

    @staticmethod
    def _validated_table_name(table: str) -> str:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)?", table):
            raise ValueError(f"Invalid QuestDB table identifier: {table}")
        return table

    @staticmethod
    def _normalize_for_engine(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame

        frame = frame.copy()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frame = frame.drop_duplicates(subset=["timestamp"], keep="last")
        frame = frame.sort_values("timestamp", kind="mergesort").reset_index(drop=True)

        base_columns = ["timestamp", "open", "high", "low", "close", "volume"]
        extra_columns = [column for column in frame.columns if column not in base_columns]
        return frame[base_columns + extra_columns]

    def _cache_key(self, symbol: str, interval: str, start: str, end: str) -> str:
        payload = json.dumps(
            {
                "symbol": symbol,
                "interval": interval,
                "start": start,
                "end": end,
                "table": self._config.questdb.table,
            },
            sort_keys=True,
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
        return f"{self._config.redis.key_prefix}:{symbol}:{interval}:{digest}"

    def _get_cache(self, key: str) -> pd.DataFrame | None:
        try:
            payload = self._redis.get(key)
        except Exception:
            return None

        if payload is None:
            return None

        return self._decode_arrow(payload)

    def _set_cache(self, key: str, frame: pd.DataFrame) -> None:
        try:
            self._redis.setex(key, self._config.redis.ttl_seconds, self._encode_arrow(frame))
        except Exception:
            return

    @staticmethod
    def _encode_arrow(frame: pd.DataFrame) -> bytes:
        table = pa.Table.from_pandas(frame, preserve_index=False)
        sink = io.BytesIO()
        options = ipc.IpcWriteOptions(compression="zstd")
        with ipc.new_file(sink, table.schema, options=options) as writer:
            writer.write(table)
        return sink.getvalue()

    @staticmethod
    def _decode_arrow(payload: bytes) -> pd.DataFrame:
        source = io.BytesIO(payload)
        with ipc.open_file(source) as reader:
            return reader.read_all().to_pandas()

    def _pg_dsn(self) -> str:
        qdb = self._config.questdb
        return (
            f"host={qdb.pg_host} "
            f"port={qdb.pg_port} "
            f"user={qdb.pg_user} "
            f"password={qdb.pg_password} "
            f"dbname={qdb.pg_database}"
        )
