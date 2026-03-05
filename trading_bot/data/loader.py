"""
Data Loader — OHLCV data loading from CSV and Yahoo Finance.

Provides a unified interface for loading price data into a standardized
pandas DataFrame with columns: [Open, High, Low, Close, Volume].
"""

import pandas as pd
import yfinance as yf


def load_csv(filepath: str) -> pd.DataFrame:
    """
    Load OHLCV data from a CSV file.

    Automatically detects the datetime column and standardizes column names.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        OHLCV DataFrame with DatetimeIndex.

    Raises
    ------
    ValueError
        If required OHLC columns are missing.
    """
    df = pd.read_csv(filepath)

    # Auto-detect datetime column
    date_col = None
    for col in df.columns:
        if col.lower().strip() in ("date", "datetime", "timestamp", "time"):
            date_col = col
            break
    if date_col is None:
        date_col = df.columns[0]

    df[date_col] = pd.to_datetime(df[date_col])
    df.set_index(date_col, inplace=True)
    df.index.name = "Datetime"

    # Normalize column names
    col_map = {}
    for col in df.columns:
        cl = col.lower().strip()
        if cl == "open":
            col_map[col] = "Open"
        elif cl == "high":
            col_map[col] = "High"
        elif cl == "low":
            col_map[col] = "Low"
        elif cl == "close":
            col_map[col] = "Close"
        elif cl in ("volume", "vol"):
            col_map[col] = "Volume"
    df.rename(columns=col_map, inplace=True)

    required = ["Open", "High", "Low", "Close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    if "Volume" not in df.columns:
        df["Volume"] = 0

    df.sort_index(inplace=True)
    return df[["Open", "High", "Low", "Close", "Volume"]].astype(float)


def fetch_live(
    ticker: str = "BTC-USD",
    period: str = "5d",
    interval: str = "5m",
) -> pd.DataFrame:
    """
    Fetch OHLCV data from Yahoo Finance.

    Parameters
    ----------
    ticker : str
        Ticker symbol (e.g. "BTC-USD", "AAPL").
    period : str
        Lookback period ("1d", "5d", "1mo", "3mo").
        Note: yfinance limits 5m data to ~60 days.
    interval : str
        Candle interval (e.g. "5m", "15m", "1h").

    Returns
    -------
    pd.DataFrame
        OHLCV DataFrame with DatetimeIndex.
    """
    print(f"  Downloading {ticker} | period={period} | interval={interval} ...")
    data = yf.download(
        ticker,
        period=period,
        interval=interval,
        progress=False,
        auto_adjust=True,
    )

    if data.empty:
        raise ValueError(f"No data for {ticker} (period={period}, interval={interval})")

    # Flatten MultiIndex columns if present (yfinance quirk for single tickers)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data.index.name = "Datetime"
    data.sort_index(inplace=True)

    return data[["Open", "High", "Low", "Close", "Volume"]].astype(float)


def generate_sample_data(n_candles: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic OHLCV data for testing.

    Creates realistic-looking price data with consolidation zones,
    breakouts, and trending sections.

    Parameters
    ----------
    n_candles : int
        Number of candles to generate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Synthetic OHLCV data.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n_candles, freq="5min")

    # Base price with mean-reverting behavior and occasional trends
    price = 100.0
    prices = []
    for i in range(n_candles):
        # Mean-revert towards 100, with occasional momentum
        drift = -0.001 * (price - 100)
        shock = rng.normal(0, 0.15)
        price += drift + shock
        prices.append(max(price, 1.0))  # Floor at 1.0

    closes = np.array(prices)
    noise = rng.uniform(0.05, 0.3, n_candles)
    highs = closes + rng.uniform(0.01, 0.5, n_candles) * noise * 2
    lows = closes - rng.uniform(0.01, 0.5, n_candles) * noise * 2
    opens = closes + rng.uniform(-0.2, 0.2, n_candles) * noise
    volumes = rng.integers(100, 10000, n_candles).astype(float)

    return pd.DataFrame(
        {"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": volumes},
        index=idx,
    )
