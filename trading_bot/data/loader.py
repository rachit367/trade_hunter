"""
Data Loader — OHLCV data loading from CSV and Delta Exchange API.

Provides a unified interface for loading price data into a standardized
pandas DataFrame with columns: [Open, High, Low, Close, Volume].
"""

import pandas as pd


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


def fetch_delta(
    symbol: str = "BTCUSD",
    resolution: str = "5m",
    lookback_hours: int = 4,
) -> pd.DataFrame:
    """
    Fetch OHLCV data from Delta Exchange API.

    Parameters
    ----------
    symbol : str
        Delta Exchange product symbol (e.g. "BTCUSD", "ETHUSD").
    resolution : str
        Candle resolution ("1m", "5m", "15m", "1h", "1d").
    lookback_hours : int
        How many hours of history to fetch.

    Returns
    -------
    pd.DataFrame
        OHLCV DataFrame with DatetimeIndex.
    """
    from trading_bot.exchange.delta_connector import DeltaConnector

    print(f"  Fetching {symbol} from Delta Exchange | resolution={resolution} | lookback={lookback_hours}h ...")
    connector = DeltaConnector()
    return connector.fetch_candles(
        symbol=symbol,
        resolution=resolution,
        lookback_hours=lookback_hours,
    )


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
