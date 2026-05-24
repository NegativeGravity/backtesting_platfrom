import pandas as pd

from trading_system.ml.features import FeatureConfig, build_feature_frame


def test_feature_frame_contains_target_without_future_features() -> None:
    data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2023-01-01", periods=100, freq="h", tz="UTC"),
            "open": [100.0 + i for i in range(100)],
            "high": [101.0 + i for i in range(100)],
            "low": [99.0 + i for i in range(100)],
            "close": [100.0 + i for i in range(100)],
            "volume": [1000.0 for _ in range(100)],
        }
    )

    config = FeatureConfig(
        horizon_bars=3,
        min_return_threshold=0.001,
        fee_rate=0.001,
        slippage_bps=5.0,
    )

    features = build_feature_frame(data=data, config=config)

    assert not features.empty
    assert "target" in features.columns
    assert "future_return" in features.columns
    assert "return_1" in features.columns