from __future__ import annotations

import argparse

import yaml

from backend.data.market_store import MarketDataStore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/market_data.yaml")
    parser.add_argument(
        "--split-policy",
        choices=["recommended", "user_requested_no_overlap", "user_requested_with_leakage_warning"],
        default="recommended",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    split_key = {
        "recommended": "splits_recommended",
        "user_requested_no_overlap": "splits_user_requested_no_overlap",
        "user_requested_with_leakage_warning": "splits_user_requested_with_leakage_warning",
    }[args.split_policy]
    splits = config[split_key]
    store = MarketDataStore.from_yaml(args.config)

    ranges: dict[str, tuple[str, str]] = {
        "train": (splits["train_start"], splits["train_end"]),
        "test": (splits["test_start"], splits["test_end"]),
    }
    if "validation_start" in splits:
        ranges["validation"] = (splits["validation_start"], splits["validation_end"])

    for name, (start, end) in ranges.items():
        frame = store.load_ohlcv(start=start, end=end, use_cache=True)
        print(f"CACHED_{name.upper()}_ROWS={len(frame)} START={start} END={end}")


if __name__ == "__main__":
    main()
