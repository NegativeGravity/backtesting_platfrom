class TradingSystemError(Exception):
    """Base exception for the trading system."""


class ConfigurationError(TradingSystemError):
    """Raised when configuration is invalid."""


class DataValidationError(TradingSystemError):
    """Raised when market data validation fails."""


class BacktestError(TradingSystemError):
    """Raised when backtest execution fails."""


class ExecutionError(TradingSystemError):
    """Raised when order execution simulation fails."""


class PortfolioError(TradingSystemError):
    """Raised when portfolio accounting fails."""
