<div align="center">
  <img src="[assets/logo.png](https://chatgpt.com/s/m_6a1ea564ee808191b9c6e34f17411af1)" alt="Quant Research Terminal Logo" width="160" />
  <h1>Quant Research Terminal</h1>
  <p>
    A research-grade trading robot and backtesting platform for crypto and market assets, built with a clean Python backend, a modern React frontend, and multiple rule-based, machine-learning, deep-learning, and Helformer-assisted strategies.
  </p>
</div>

---

## Overview

Quant Research Terminal is an end-to-end trading research platform designed to evaluate automated trading strategies against a simple buy-and-hold benchmark. The system starts with an initial capital balance, runs a complete execution simulation, records trades, fees, slippage, equity curves, drawdowns, and strategy diagnostics, then compares the final result against holding the underlying asset.

The project supports both classic rule-based strategies and model-driven approaches. Strategies can run directly from closed market candles, or they can optionally use a Helformer next-bar forecast through a dedicated toggle. This makes it possible to compare the same strategy in two modes: with pure market-rule execution and with forecast-assisted execution.

## Core Features

| Area | Capability |
| --- | --- |
| Backtesting | Full portfolio simulation with capital, fees, slippage, positions, orders, trades, equity curve, benchmark curve, and drawdown |
| Strategies | 14 implemented strategies across classic rules, professional regime-aware rules, machine learning, deep learning, and Helformer |
| Forecast Toggle | Every supported strategy can be tested with or without Helformer forecast assistance |
| Execution Model | Signal on closed bar, fill on next bar open to reduce look-ahead bias |
| Risk Control | ATR-based stops, trailing stops, time stops, volatility targeting, cooldowns, and regime filters |
| Reporting | Summary cards, trade blotter, chart replay, diagnostics, configuration view, exportable reports |
| Frontend | React + Vite dashboard with strategy launcher, saved runs, live replay, charts, and run comparison |
| Backend | FastAPI-style service layer, strategy factory, robot engine, worker engine, and report generation |

## Final Strategy Universe

The platform currently supports 14 strategies:

| Strategy | Type | Helformer Forecast Support |
| --- | --- | --- |
| `mean_reversion` | Rule-based | Yes |
| `adaptive_trend_breakout` | Rule-based trend | Yes |
| `regime_adaptive_btc_trend_breakout` | BTC 1H regime-aware trend | Yes |
| `liquidity_sweep_reversal` | Rule-based reversal | Yes |
| `liquidation_shock_mean_reversion` | BTC 1H panic-reversal | Yes |
| `adaptive_trend_expansion_pro` | Professional trend expansion | Yes |
| `capitulation_reversal_pro` | Professional capitulation reversal | Yes |
| `volatility_squeeze_breakout` | Volatility compression breakout | Yes |
| `meta_labeled_ensemble_alpha` | Meta-gated ensemble | Yes |
| `meta_labeled_alpha_allocator_pro` | Professional alpha allocator | Yes |
| `ml_momentum` | Machine learning | Yes |
| `ml_regime_meta_label` | Machine learning meta-labeling | Yes |
| `dl_temporal_fusion_momentum` | Deep learning sequence model | Yes |
| `helformer_momentum` | Native Helformer strategy | No, because it is already Helformer-native |

## Architecture

The system is separated into clear layers. The backend owns data loading, strategy construction, execution, accounting, risk handling, job management, and report generation. The frontend owns strategy selection, model artifact selection, Helformer toggle control, result visualization, live replay, and export actions.

```text
Market Data
   ↓
Strategy Factory
   ↓
Strategy / Model / Helformer Wrapper
   ↓
Robot Engine
   ↓
Execution Engine
   ↓
Accounting and Risk Layer
   ↓
Backtest Report
   ↓
Frontend Dashboard
```

## Backtesting Design

The backtest engine follows a conservative event-driven design. Signals are generated only after a candle is closed, and orders are filled on the next bar. This avoids using information from the future candle and keeps the simulation closer to realistic trading behavior.

The engine tracks independent positions per strategy worker and symbol. It records entry and exit timestamps, side, quantity, fill price, fees, slippage cost, gross profit and loss, net profit and loss, return percentage, and exit reason. The accounting model keeps portfolio value based on free cash, reserved margin, and unrealized profit and loss.

## Strategy Design

The strategy layer contains both simple and advanced trading logic. The simpler strategies provide interpretable baselines such as mean reversion, trend breakout, and liquidity sweep reversal. The advanced strategies add regime awareness, volatility adaptation, robust shock detection, squeeze breakout logic, and meta-allocation.

The BTC-specific strategies were designed around the 1-hour BTCUSDT market structure. They focus on trend capture, panic reversal, and ensemble gating. The regime-aware trend strategy combines EMA trend filters, Donchian breakout, ADX, realized volatility, and volume confirmation. The liquidation shock strategy approximates forced-liquidation behavior with return shock, volume spike, RSI exhaustion, wick rejection, and close-location filters. The meta-labeled ensemble strategy combines trend, reversion, squeeze, and flat states through a higher-level decision gate.

## Model-Driven Components

The project includes machine-learning and deep-learning strategy paths. These models are designed around causal features computed from historical candles only. Typical features include returns over multiple horizons, volatility, ATR, RSI, MACD slope, EMA distance, Donchian position, volume z-score, wick ratios, candle body ratio, skew, kurtosis, and regime features.

The model training design uses time-aware validation rather than random splitting. This is important because financial series are ordered and highly non-stationary. The intended workflow is train on older periods, validate on later periods, and test on strictly unseen future periods. For classification-style models, the loss is binary cross entropy or calibrated log loss. For regression-style forecast models, the loss is typically mean squared error or Huber loss. For meta-labeling, the objective is not to predict every price move, but to filter weak trades and keep only trades with positive expected value after fees and slippage.

## Helformer Forecast Mode

Most strategies can optionally use a Helformer next-close forecaster. This is controlled through `use_helformer_forecast`. When disabled, the selected strategy trades only from its native rules or model. When enabled, the backend wraps the strategy with a forecast projection layer and requires a valid Helformer artifact path.

This design avoids duplicating strategies such as `strategy_with_helformer`. Instead, the same strategy can be compared in two clean modes:

```text
strategy = adaptive_trend_expansion_pro
use_helformer_forecast = false

strategy = adaptive_trend_expansion_pro
use_helformer_forecast = true
```

## Frontend

The frontend is built as a research dashboard rather than a generic admin panel. It contains a strategy launcher, artifact selectors, Helformer toggle, saved run archive, live replay page, chart view, trade table, execution blotter, diagnostics panel, configuration panel, and model metadata panel.

The strategy list is centralized in `strategyCatalog.ts`, which prevents mismatch between different strategy selectors. The user can launch a backtest, monitor background jobs, open completed runs automatically, and export results as HTML, CSV, or Parquet.

## Backend

The backend uses a clean separation between API schemas, service functions, strategy factory, execution engine, robot engine, and reporting. The strategy factory is responsible for creating the correct strategy instance and applying Helformer wrapping only when explicitly requested. This keeps the trading logic predictable and prevents hidden behavior.

The job monitor allows long-running backtests to continue server-side while the frontend remains responsive. Completed runs are persisted, listed, and opened through stable run identifiers.

## Research Basis

The strategy design is inspired by several research directions:

| Research Direction | Role in This Project |
| --- | --- |
| Strategy Allocation | Motivated the use of a higher-level allocator that chooses between local trading engines instead of relying on one fixed strategy |
| Logic-Q | Motivated regime-aware decision logic so the robot reacts differently to trend, chop, crash, and shock states |
| Quantformer | Motivated the use of temporal representation learning and transformer-style feature extraction for financial time series |
| FX Trading Signals with Walk-Forward Validation | Motivated realistic rolling validation, fee-aware signal conversion, and strict out-of-sample evaluation |
| Contextual Reinforcement Learning | Motivated the use of market context such as volatility, open interest proxy, liquidation proxy, and sentiment-like features |
| Incremental Financial Forecasting / IFF-DRL | Motivated model refresh, drift awareness, and periodic refitting rather than assuming a static market |

## Example Result

One of the best reported runs produced the following high-level result:

| Metric | Value |
| --- | ---: |
| Final Equity | `$10,256.42` |
| Total Return | `2.56%` |
| Benchmark Return | `-6.33%` |
| Excess Return | `8.90%` |
| Max Drawdown | `-2.93%` |
| Trades | `59` |
| Win Rate | `37.29%` |
| Profit Factor | `1.18` |
| Fees | `$311.64` |
| Slippage | `$155.82` |

The purpose of the project is not to claim guaranteed alpha. The goal is to demonstrate a complete implementation: strategy design, execution simulation, portfolio accounting, cost modeling, benchmark comparison, model integration, frontend visualization, and reproducible reporting.

## Running the Project

Install backend dependencies and start the API service:

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Install frontend dependencies and start the dashboard:

```bash
cd frontend
npm install
npm run dev
```

The default frontend expects the API at:

```text
http://127.0.0.1:8000
```

This can be changed with:

```bash
VITE_API_BASE_URL=http://127.0.0.1:8000
```

## Repository Structure

```text
backend/
  api/
  backtest/
  engine/
  strategy/
  models/
  reports/

frontend/
  src/
    api/
    components/
    hooks/
    stores/
    utils/
    strategyCatalog.ts
```

## Output Artifacts

A completed run contains the backtest summary, chart data, execution log, trades, equity curve, benchmark curve, diagnostics, configuration, model metadata, and export files. The dashboard supports direct run URLs and export actions for reporting and review.

## Disclaimer

This project is built for research, engineering evaluation, and educational backtesting. It is not financial advice and should not be used for live trading without additional validation, exchange integration checks, risk limits, monitoring, and compliance review.
