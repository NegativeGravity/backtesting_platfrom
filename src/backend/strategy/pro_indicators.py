from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from backend.strategy.market_view import MarketDataView


EPSILON = 1e-12


def safe_denominator(value: float, fallback: float = np.nan) -> float:
    value = float(value)
    if not np.isfinite(value) or abs(value) <= EPSILON:
        return fallback
    return value


def safe_denominator_array(values: np.ndarray) -> np.ndarray:
    safe = np.asarray(values, dtype=np.float64).copy()
    safe[~np.isfinite(safe) | (np.abs(safe) <= EPSILON)] = np.nan
    return safe


def finite_at(index: int, *arrays: np.ndarray | float) -> bool:
    for value in arrays:
        if isinstance(value, np.ndarray):
            item = float(value[index])
        else:
            item = float(value)
        if not np.isfinite(item):
            return False
    return True


def true_range(market: MarketDataView) -> np.ndarray:
    key = "pro:true_range"
    if key not in market._cache:
        high = market.arrays.high
        low = market.arrays.low
        close = market.arrays.close
        previous_close = np.empty_like(close, dtype=np.float64)
        previous_close[0] = np.nan
        previous_close[1:] = close[:-1]
        market._cache[key] = np.maximum.reduce([high - low, np.abs(high - previous_close), np.abs(low - previous_close)])
    return market._cache[key]


def kama(market: MarketDataView, window: int, fast: int = 2, slow: int = 30) -> np.ndarray:
    key = f"pro:kama:{window}:{fast}:{slow}"
    if key in market._cache:
        return market._cache[key]
    close = np.asarray(market.arrays.close, dtype=np.float64)
    output = np.full(close.shape, np.nan, dtype=np.float64)
    if close.size == 0:
        market._cache[key] = output
        return output
    output[0] = close[0]
    fast_sc = 2.0 / (fast + 1.0)
    slow_sc = 2.0 / (slow + 1.0)
    abs_diff = np.abs(np.diff(close, prepend=np.nan))
    for index in range(1, len(close)):
        if index < window:
            output[index] = close[index] if not np.isfinite(output[index - 1]) else output[index - 1] + slow_sc * slow_sc * (close[index] - output[index - 1])
            continue
        change = abs(close[index] - close[index - window])
        volatility = np.nansum(abs_diff[index - window + 1:index + 1])
        efficiency = 0.0 if not np.isfinite(volatility) or volatility <= EPSILON else change / volatility
        smoothing = (efficiency * (fast_sc - slow_sc) + slow_sc) ** 2.0
        previous = close[index - 1] if not np.isfinite(output[index - 1]) else output[index - 1]
        output[index] = previous + smoothing * (close[index] - previous)
    market._cache[key] = output
    return output


def adx_bundle(market: MarketDataView, window: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    key = f"pro:adx_bundle:{window}"
    if key in market._cache:
        packed = market._cache[key]
        return packed[:, 0], packed[:, 1], packed[:, 2]
    high = market.arrays.high
    low = market.arrays.low
    up_move = np.diff(high, prepend=np.nan)
    down_move = -np.diff(low, prepend=np.nan)
    plus_dm = np.where((up_move > down_move) & (up_move > 0.0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0.0), down_move, 0.0)
    atr = pd.Series(true_range(market)).ewm(alpha=1.0 / window, adjust=False).mean().to_numpy(dtype=np.float64)
    plus_smoothed = pd.Series(plus_dm).ewm(alpha=1.0 / window, adjust=False).mean().to_numpy(dtype=np.float64)
    minus_smoothed = pd.Series(minus_dm).ewm(alpha=1.0 / window, adjust=False).mean().to_numpy(dtype=np.float64)
    plus_di = 100.0 * plus_smoothed / safe_denominator_array(atr)
    minus_di = 100.0 * minus_smoothed / safe_denominator_array(atr)
    dx = 100.0 * np.abs(plus_di - minus_di) / safe_denominator_array(plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1.0 / window, adjust=False).mean().to_numpy(dtype=np.float64)
    packed = np.column_stack([adx, plus_di, minus_di])
    market._cache[key] = packed
    return adx, plus_di, minus_di


def vortex_bundle(market: MarketDataView, window: int) -> tuple[np.ndarray, np.ndarray]:
    key = f"pro:vortex:{window}"
    if key in market._cache:
        packed = market._cache[key]
        return packed[:, 0], packed[:, 1]
    high = market.arrays.high
    low = market.arrays.low
    previous_high = np.roll(high, 1)
    previous_low = np.roll(low, 1)
    previous_high[0] = np.nan
    previous_low[0] = np.nan
    vm_plus = np.abs(high - previous_low)
    vm_minus = np.abs(low - previous_high)
    tr_sum = pd.Series(true_range(market)).rolling(window).sum().to_numpy(dtype=np.float64)
    vi_plus = pd.Series(vm_plus).rolling(window).sum().to_numpy(dtype=np.float64) / safe_denominator_array(tr_sum)
    vi_minus = pd.Series(vm_minus).rolling(window).sum().to_numpy(dtype=np.float64) / safe_denominator_array(tr_sum)
    packed = np.column_stack([vi_plus, vi_minus])
    market._cache[key] = packed
    return vi_plus, vi_minus


def choppiness(market: MarketDataView, window: int) -> np.ndarray:
    key = f"pro:chop:{window}"
    if key in market._cache:
        return market._cache[key]
    tr_sum = pd.Series(true_range(market)).rolling(window).sum().to_numpy(dtype=np.float64)
    high_max = pd.Series(market.arrays.high).rolling(window).max().to_numpy(dtype=np.float64)
    low_min = pd.Series(market.arrays.low).rolling(window).min().to_numpy(dtype=np.float64)
    ratio = tr_sum / safe_denominator_array(high_max - low_min)
    output = 100.0 * np.log10(np.maximum(ratio, EPSILON)) / np.log10(float(window))
    market._cache[key] = output
    return output


def rolling_vwap(market: MarketDataView, window: int) -> np.ndarray:
    key = f"pro:vwap:{window}"
    if key in market._cache:
        return market._cache[key]
    close = market.arrays.close
    volume = market.arrays.volume
    pv = pd.Series(close * volume).rolling(window).sum().to_numpy(dtype=np.float64)
    vv = pd.Series(volume).rolling(window).sum().to_numpy(dtype=np.float64)
    market._cache[key] = pv / safe_denominator_array(vv)
    return market._cache[key]


def volume_z(market: MarketDataView, window: int) -> np.ndarray:
    key = f"pro:volume_z:{window}"
    if key in market._cache:
        return market._cache[key]
    volume = pd.Series(market.arrays.volume)
    mean = volume.rolling(window).mean().to_numpy(dtype=np.float64)
    std = volume.rolling(window).std(ddof=0).to_numpy(dtype=np.float64)
    output = (market.arrays.volume - mean) / safe_denominator_array(std)
    output = np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)
    market._cache[key] = output
    return output


def realized_vol_percentile(market: MarketDataView, vol_window: int = 24, percentile_window: int = 168) -> np.ndarray:
    key = f"pro:rv_percentile:{vol_window}:{percentile_window}"
    if key in market._cache:
        return market._cache[key]
    log_ret = market.log_diff("close", 1)
    rv = pd.Series(log_ret).rolling(vol_window).std().to_numpy(dtype=np.float64)
    series = pd.Series(rv)
    percentile = series.rolling(percentile_window).apply(_last_percent_rank, raw=True).to_numpy(dtype=np.float64)
    market._cache[key] = np.nan_to_num(percentile, nan=50.0, posinf=100.0, neginf=0.0)
    return market._cache[key]


def robust_return_z(market: MarketDataView, window: int) -> np.ndarray:
    key = f"pro:robust_return_z:{window}"
    if key in market._cache:
        return market._cache[key]
    log_ret = pd.Series(market.log_diff("close", 1))
    median = log_ret.rolling(window).median().to_numpy(dtype=np.float64)
    mad = log_ret.rolling(window).apply(_mad, raw=True).to_numpy(dtype=np.float64)
    output = (market.log_diff("close", 1) - median) / safe_denominator_array(1.4826 * mad)
    output = np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)
    market._cache[key] = output
    return output


def range_z(market: MarketDataView, window: int) -> np.ndarray:
    key = f"pro:range_z:{window}"
    if key in market._cache:
        return market._cache[key]
    candle_range = market.arrays.high - market.arrays.low
    median = pd.Series(candle_range).rolling(window).median().to_numpy(dtype=np.float64)
    output = candle_range / safe_denominator_array(median)
    output = np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)
    market._cache[key] = output
    return output


def relative_volume(market: MarketDataView, window: int) -> np.ndarray:
    key = f"pro:rvol:{window}"
    if key in market._cache:
        return market._cache[key]
    median = pd.Series(market.arrays.volume).rolling(window).median().to_numpy(dtype=np.float64)
    output = market.arrays.volume / safe_denominator_array(median)
    output = np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)
    market._cache[key] = output
    return output


def connors_rsi(market: MarketDataView, rsi_window: int = 3, streak_window: int = 2, rank_window: int = 100) -> np.ndarray:
    key = f"pro:connors_rsi:{rsi_window}:{streak_window}:{rank_window}"
    if key in market._cache:
        return market._cache[key]
    close = market.arrays.close
    close_rsi = _rsi(close, rsi_window)
    streak = np.zeros_like(close, dtype=np.float64)
    for index in range(1, len(close)):
        if close[index] > close[index - 1]:
            streak[index] = max(1.0, streak[index - 1] + 1.0)
        elif close[index] < close[index - 1]:
            streak[index] = min(-1.0, streak[index - 1] - 1.0)
        else:
            streak[index] = 0.0
    streak_rsi = _rsi(streak, streak_window)
    returns = pd.Series(np.diff(close, prepend=np.nan))
    rank = returns.rolling(rank_window).apply(_last_percent_rank, raw=True).to_numpy(dtype=np.float64)
    output = np.nanmean(np.column_stack([close_rsi, streak_rsi, rank]), axis=1)
    output = np.nan_to_num(output, nan=50.0, posinf=100.0, neginf=0.0)
    market._cache[key] = output
    return output


def bollinger_middle(market: MarketDataView, window: int) -> np.ndarray:
    return market.rolling_mean("close", window, min_periods=window)


def squeeze_release(market: MarketDataView, bb_window: int, kc_window: int, kc_atr_mult: float, min_squeeze_bars: int) -> np.ndarray:
    key = f"pro:squeeze_release:{bb_window}:{kc_window}:{kc_atr_mult}:{min_squeeze_bars}"
    if key in market._cache:
        return market._cache[key]
    close = market.arrays.close
    middle = market.rolling_mean("close", bb_window, min_periods=bb_window)
    std = market.rolling_std("close", bb_window, ddof=0, min_periods=bb_window)
    upper_bb = middle + 2.0 * std
    lower_bb = middle - 2.0 * std
    ema_mid = market.ema("close", kc_window)
    atr = market.atr(kc_window)
    upper_kc = ema_mid + kc_atr_mult * atr
    lower_kc = ema_mid - kc_atr_mult * atr
    in_squeeze = (upper_bb < upper_kc) & (lower_bb > lower_kc)
    prior_count = pd.Series(in_squeeze.astype(float)).rolling(min_squeeze_bars).sum().shift(1).to_numpy(dtype=np.float64)
    release = (~in_squeeze) & (prior_count >= float(min_squeeze_bars))
    release = np.where(np.isfinite(close), release, False)
    market._cache[key] = release.astype(np.float64)
    return market._cache[key]


def regime_snapshot(market: MarketDataView, index: int) -> dict[str, Any]:
    close = market.close_at(index)
    atr = market.atr(14)
    kama48 = kama(market, 48)
    kama144 = kama(market, 144)
    adx, _, _ = adx_bundle(market, 14)
    chop = choppiness(market, 48)
    ret_z = robust_return_z(market, 168)
    rz = range_z(market, 168)
    vz = volume_z(market, 72)
    vwap = rolling_vwap(market, 24)
    slope48 = _slope(kama48, index, 12)
    atr_pct = float(atr[index] / safe_denominator(close, np.nan))
    vwap_dev = float((close - vwap[index]) / safe_denominator(atr[index], np.nan))
    trend_up = kama48[index] > kama144[index]
    trend_down = kama48[index] < kama144[index]
    shock_strength = min(1.0, max(0.0, (abs(ret_z[index]) - 1.6) / 2.4)) * min(1.0, max(0.0, (rz[index] - 1.0) / 1.8))
    rapid_up = _clip01((ret_z[index] - 1.2) / 2.2) * _clip01((rz[index] - 1.1) / 1.7) * _clip01((vz[index] + 0.2) / 2.2)
    rapid_down = _clip01((-ret_z[index] - 1.2) / 2.2) * _clip01((rz[index] - 1.1) / 1.7) * _clip01((vz[index] + 0.2) / 2.2)
    steady_up = float(trend_up) * _clip01((adx[index] - 15.0) / 20.0) * _clip01((58.0 - chop[index]) / 28.0) * _clip01((slope48 + 0.0025) / 0.006)
    steady_down = float(trend_down) * _clip01((adx[index] - 15.0) / 20.0) * _clip01((58.0 - chop[index]) / 28.0) * _clip01((-slope48 + 0.0025) / 0.006)
    chop_score = _clip01((chop[index] - 45.0) / 25.0) * _clip01((24.0 - adx[index]) / 18.0) * _clip01((1.8 - abs(vwap_dev)) / 1.8)
    raw = {
        "steady_uptrend": steady_up,
        "rapid_uptrend": rapid_up,
        "steady_downtrend": steady_down,
        "rapid_downtrend": rapid_down,
        "chop": chop_score,
        "shock": shock_strength,
    }
    total = sum(max(0.0, float(value)) for value in raw.values())
    probabilities = {key: (max(0.0, float(value)) / total if total > EPSILON else 0.0) for key, value in raw.items()}
    label = max(probabilities, key=probabilities.get) if total > EPSILON else "chop"
    return {
        "regime": label,
        "probabilities": probabilities,
        "kama48": float(kama48[index]),
        "kama144": float(kama144[index]),
        "kama48_slope_12": float(slope48),
        "adx": float(adx[index]),
        "choppiness": float(chop[index]),
        "atr": float(atr[index]),
        "atr_pct": float(atr_pct),
        "return_z": float(ret_z[index]),
        "range_z": float(rz[index]),
        "volume_z": float(vz[index]),
        "vwap": float(vwap[index]),
        "vwap_dev_atr": float(vwap_dev),
    }


def bars_since_entry(portfolio: Any) -> int:
    try:
        return max(0, int(getattr(portfolio, "bars_since_entry", 0) or 0))
    except Exception:
        return 0


def position_side(portfolio: Any) -> str | None:
    side = getattr(portfolio, "position_side", None)
    if side in {"LONG", "SHORT"}:
        return str(side)
    quantity = float(getattr(portfolio, "position_quantity", 0.0) or 0.0)
    if quantity > 0.0:
        return "LONG"
    if quantity < 0.0:
        return "SHORT"
    return None


def target_fraction_from_risk(
    *,
    close: float,
    atr: float,
    stop_atr: float,
    risk_per_trade: float,
    max_notional_fraction: float,
    vol_percentile: float,
) -> tuple[float, float, float]:
    stop_distance = max(stop_atr * atr, close * 0.002)
    stop_pct = stop_distance / safe_denominator(close, 1.0)
    target_fraction = risk_per_trade / max(stop_pct, 1e-6)
    if vol_percentile > 95.0:
        target_fraction *= 0.30
    elif vol_percentile > 85.0:
        target_fraction *= 0.55
    target_fraction = min(max(target_fraction, 0.0), max_notional_fraction)
    return float(target_fraction), float(stop_pct), float(stop_pct * 2.0)


def _slope(values: np.ndarray, index: int, lookback: int) -> float:
    if index < lookback:
        return 0.0
    current = float(values[index])
    previous = float(values[index - lookback])
    if not np.isfinite(current) or not np.isfinite(previous) or abs(previous) <= EPSILON:
        return 0.0
    return current / previous - 1.0


def _rsi(values: np.ndarray, window: int) -> np.ndarray:
    delta = np.diff(values, prepend=np.nan)
    gain = np.where(delta > 0.0, delta, 0.0)
    loss = np.where(delta < 0.0, -delta, 0.0)
    average_gain = pd.Series(gain).ewm(alpha=1.0 / window, adjust=False).mean().to_numpy(dtype=np.float64)
    average_loss = pd.Series(loss).ewm(alpha=1.0 / window, adjust=False).mean().to_numpy(dtype=np.float64)
    rs = average_gain / safe_denominator_array(average_loss)
    output = 100.0 - (100.0 / (1.0 + rs))
    return np.nan_to_num(output, nan=50.0, posinf=100.0, neginf=0.0)


def _mad(values: np.ndarray) -> float:
    clean = values[np.isfinite(values)]
    if clean.size == 0:
        return np.nan
    median = float(np.median(clean))
    return float(np.median(np.abs(clean - median)))


def _last_percent_rank(values: np.ndarray) -> float:
    clean = values[np.isfinite(values)]
    if clean.size == 0:
        return np.nan
    last = clean[-1]
    if clean.size <= 1:
        return 50.0
    return float(100.0 * np.sum(clean[:-1] <= last) / (clean.size - 1))


def _clip01(value: float) -> float:
    value = float(value)
    if not np.isfinite(value):
        return 0.0
    return min(1.0, max(0.0, value))
