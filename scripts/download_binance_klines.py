from __future__ import annotations

import argparse
import concurrent.futures as futures
import hashlib
import io
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests
import yaml
from tqdm import tqdm


KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trade_count",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "ignore",
]


@dataclass(frozen=True)
class MonthlyFile:
    year: int
    month: int
    url: str
    checksum_url: str
    zip_path: Path
    parquet_path: Path


def month_range(start_month: str, end_month: str) -> list[tuple[int, int]]:
    start = pd.Period(start_month, freq="M")
    end = pd.Period(end_month, freq="M")
    return [(period.year, period.month) for period in pd.period_range(start, end, freq="M")]


def build_monthly_files(config: dict) -> list[MonthlyFile]:
    symbol = config["symbol"]
    interval = config["interval"]
    base_url = config["binance"]["base_url"].rstrip("/")
    raw_dir = Path(config["binance"]["raw_dir"])
    processed_dir = Path(config["binance"]["processed_dir"])
    files: list[MonthlyFile] = []

    for year, month in month_range(config["binance"]["start_month"], config["binance"]["end_month"]):
        name = f"{symbol}-{interval}-{year}-{month:02d}.zip"
        url = f"{base_url}/data/spot/monthly/klines/{symbol}/{interval}/{name}"
        files.append(
            MonthlyFile(
                year=year,
                month=month,
                url=url,
                checksum_url=f"{url}.CHECKSUM",
                zip_path=raw_dir / name,
                parquet_path=processed_dir / f"{symbol}-{interval}-{year}-{month:02d}.parquet",
            )
        )

    return files


def download_bytes(url: str, timeout: int = 90) -> bytes:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.content


def remote_checksum(url: str) -> str | None:
    try:
        text = download_bytes(url, timeout=30).decode("utf-8", errors="replace").strip()
    except Exception:
        return None
    return text.split()[0].strip().lower() if text else None


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest().lower()


def normalize_timestamp_series(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    median = float(numeric.dropna().median())
    unit = "us" if median > 10_000_000_000_000 else "ms"
    return pd.to_datetime(numeric, unit=unit, utc=True)


def parse_zip_to_frame(payload: bytes, symbol: str, interval: str) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        csv_names = [name for name in archive.namelist() if name.endswith(".csv")]
        if not csv_names:
            raise ValueError("No CSV found inside zip.")
        with archive.open(csv_names[0]) as file:
            frame = pd.read_csv(file, header=None, names=KLINE_COLUMNS)

    frame["ts"] = normalize_timestamp_series(frame["open_time"])
    frame["close_ts"] = normalize_timestamp_series(frame["close_time"])
    frame["symbol"] = symbol
    frame["interval"] = interval

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "trade_count",
        "taker_buy_volume",
        "taker_buy_quote_volume",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame[
        [
            "ts",
            "symbol",
            "interval",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_volume",
            "trade_count",
            "taker_buy_volume",
            "taker_buy_quote_volume",
            "close_ts",
        ]
    ]

    frame = frame.dropna(subset=["ts", "open", "high", "low", "close"])
    frame = frame.drop_duplicates(subset=["ts"], keep="last")
    frame = frame.sort_values("ts", kind="mergesort").reset_index(drop=True)

    return frame


def fetch_and_process(month_file: MonthlyFile, symbol: str, interval: str, force: bool) -> tuple[str, int]:
    month_file.zip_path.parent.mkdir(parents=True, exist_ok=True)
    month_file.parquet_path.parent.mkdir(parents=True, exist_ok=True)

    if month_file.parquet_path.exists() and not force:
        frame = pd.read_parquet(month_file.parquet_path)
        return month_file.parquet_path.name, len(frame)

    if month_file.zip_path.exists() and not force:
        payload = month_file.zip_path.read_bytes()
    else:
        payload = download_bytes(month_file.url)
        checksum = remote_checksum(month_file.checksum_url)
        if checksum is not None:
            actual = sha256_bytes(payload)
            if actual != checksum:
                raise ValueError(
                    f"Checksum mismatch for {month_file.url}: expected {checksum}, got {actual}"
                )
        month_file.zip_path.write_bytes(payload)

    frame = parse_zip_to_frame(payload, symbol=symbol, interval=interval)
    frame.to_parquet(month_file.parquet_path, index=False)
    return month_file.parquet_path.name, len(frame)


def validate_dataset(frame: pd.DataFrame) -> None:
    if frame.empty:
        raise ValueError("Dataset is empty.")
    if frame["ts"].duplicated().any():
        raise ValueError(f"Duplicate timestamps: {int(frame['ts'].duplicated().sum())}")
    if not frame["ts"].is_monotonic_increasing:
        raise ValueError("Dataset is not sorted ascending.")
    bad_prices = frame[["open", "high", "low", "close"]].le(0).any(axis=1)
    if bad_prices.any():
        raise ValueError(f"Non-positive OHLC rows: {int(bad_prices.sum())}")


def combine_dataset(config: dict) -> Path:
    symbol = config["symbol"]
    interval = config["interval"]
    processed_dir = Path(config["binance"]["processed_dir"])
    output_path = processed_dir / f"{symbol}-{interval}-2020_2025.parquet"

    frames = []
    for path in sorted(processed_dir.glob(f"{symbol}-{interval}-*.parquet")):
        if path.name.endswith("2020_2025.parquet"):
            continue
        frames.append(pd.read_parquet(path))

    if not frames:
        raise RuntimeError("No processed monthly parquet files found.")

    dataset = pd.concat(frames, ignore_index=True)
    dataset = dataset.drop_duplicates(subset=["ts"], keep="last")
    dataset = dataset.sort_values("ts", kind="mergesort").reset_index(drop=True)
    validate_dataset(dataset)
    dataset.to_parquet(output_path, index=False)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/market_data.yaml")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    month_files = build_monthly_files(config)
    symbol = config["symbol"]
    interval = config["interval"]

    results: list[tuple[str, int]] = []
    with futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        jobs = [
            executor.submit(fetch_and_process, month_file, symbol, interval, args.force)
            for month_file in month_files
        ]
        for job in tqdm(futures.as_completed(jobs), total=len(jobs), desc=f"Downloading {symbol} {interval} monthly klines"):
            results.append(job.result())

    output_path = combine_dataset(config)
    print(f"MONTHLY_FILES={len(results)}")
    print(f"MONTHLY_ROWS={sum(row_count for _, row_count in results)}")
    print(f"COMBINED_PARQUET={output_path}")


if __name__ == "__main__":
    main()
