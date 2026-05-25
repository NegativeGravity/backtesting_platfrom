from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import psycopg
import yaml
from questdb.ingress import Sender
from tqdm import tqdm


def pg_dsn(config: dict) -> str:
    qdb = config["questdb"]
    return (
        f"host={qdb['pg_host']} "
        f"port={qdb['pg_port']} "
        f"user={qdb['pg_user']} "
        f"password={qdb['pg_password']} "
        f"dbname={qdb['pg_database']}"
    )


def ensure_table(config: dict) -> None:
    table = config["questdb"]["table"]
    sql = f"""
    CREATE TABLE IF NOT EXISTS {table} (
        ts TIMESTAMP,
        symbol SYMBOL,
        interval SYMBOL,
        open DOUBLE,
        high DOUBLE,
        low DOUBLE,
        close DOUBLE,
        volume DOUBLE,
        quote_volume DOUBLE,
        trade_count LONG,
        taker_buy_volume DOUBLE,
        taker_buy_quote_volume DOUBLE,
        close_ts TIMESTAMP
    ) TIMESTAMP(ts) PARTITION BY MONTH WAL;
    """

    with psycopg.connect(pg_dsn(config), autocommit=True) as connection:
        with connection.cursor() as cursor:
            cursor.execute(sql)


def clear_table(config: dict) -> None:
    table = config["questdb"]["table"]
    with psycopg.connect(pg_dsn(config), autocommit=True) as connection:
        with connection.cursor() as cursor:
            cursor.execute(f"TRUNCATE TABLE {table};")


def ingest_dataframe(config: dict, frame: pd.DataFrame, batch_size: int) -> None:
    qdb = config["questdb"]
    conf = f"http::addr={qdb['ilp_http_host']}:{qdb['ilp_http_port']};"

    with Sender.from_conf(conf) as sender:
        for start in tqdm(range(0, len(frame), batch_size), desc="Ingesting to QuestDB"):
            batch = frame.iloc[start: start + batch_size].copy()

            batch['ts'] = batch['ts'].dt.tz_localize(None).astype('datetime64[ns]')

            if 'close_ts' in batch.columns:
                batch['close_ts'] = batch['close_ts'].dt.tz_localize(None).astype('datetime64[ns]')

            sender.dataframe(batch, table_name=qdb["table"], at="ts")

        sender.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/market_data.yaml")
    parser.add_argument("--parquet", default=None)
    parser.add_argument("--batch-size", type=int, default=10_000)
    parser.add_argument("--truncate", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    symbol = config["symbol"]
    interval = config["interval"]
    parquet_path = Path(
        args.parquet
        or Path(config["binance"]["processed_dir"]) / f"{symbol}-{interval}-2020_2025.parquet"
    )

    if not parquet_path.exists():
        raise FileNotFoundError(f"Combined parquet not found: {parquet_path}")

    frame = pd.read_parquet(parquet_path)
    frame["ts"] = pd.to_datetime(frame["ts"], utc=True)
    frame["close_ts"] = pd.to_datetime(frame["close_ts"], utc=True)
    frame = frame.drop_duplicates(subset=["ts"], keep="last")
    frame = frame.sort_values("ts", kind="mergesort").reset_index(drop=True)

    ensure_table(config)
    if args.truncate:
        clear_table(config)

    ingest_dataframe(config, frame, batch_size=args.batch_size)
    print(f"INGESTED_ROWS={len(frame)}")
    print(f"QUESTDB_TABLE={config['questdb']['table']}")


if __name__ == "__main__":
    main()
