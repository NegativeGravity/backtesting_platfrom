from __future__ import annotations

import argparse

import yaml

from backend.data.market_store import MarketDataStore


def describe(name: str, frame) -> None:
    print(f"\n{name}")
    print("=" * len(name))
    print(f"rows={len(frame)}")
    if len(frame) > 0:
        print(f"start={frame['timestamp'].iloc[0]}")
        print(f"end={frame['timestamp'].iloc[-1]}")
        print(f"duplicates={int(frame['timestamp'].duplicated().sum())}")
        print(f"monotonic={bool(frame['timestamp'].is_monotonic_increasing)}")


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

    describe("TRAIN", store.load_ohlcv(splits["train_start"], splits["train_end"]))

    if "validation_start" in splits:
        describe("VALIDATION", store.load_ohlcv(splits["validation_start"], splits["validation_end"]))

    describe("TEST", store.load_ohlcv(splits["test_start"], splits["test_end"]))


if __name__ == "__main__":
    main()
