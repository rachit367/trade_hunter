"""
Data Loader — OHLCV data loading from CSV and Delta Exchange API.

Provides a unified interface for loading price data into a standardized
pandas DataFrame with columns: [Open, High, Low, Close, Volume].
"""

import pandas as pd


from typing import Optional, Union, Tuple

def load_csv(filepath: str, correlated_csv: Optional[str] = None) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Load OHLCV data from a CSV file.

    Parameters
    ----------
    filepath : str
        Path to the primary CSV file.
    correlated_csv : str, optional
        Path to a correlated asset CSV for SMT divergence (e.g. ETHUSD).

    Returns
    -------
    pd.DataFrame or Tuple[pd.DataFrame, pd.DataFrame]
        The formatted primary DataFrame, or a tuple of (primary, correlated) DataFrames.
    """
    def _load_single(path: str) -> pd.DataFrame:
        df = pd.read_csv(path)

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
            raise ValueError(f"CSV {path} missing required columns: {missing}")

        if "Volume" not in df.columns:
            df["Volume"] = 0

        df.sort_index(inplace=True)
        return df[["Open", "High", "Low", "Close", "Volume"]].astype(float)

    primary_df = _load_single(filepath)
    if correlated_csv:
        correlated_df = _load_single(correlated_csv)
        return primary_df, correlated_df
        
    return primary_df


def fetch_delta(
    symbol: str = "BTCUSD",
    resolution: str = "5m",
    lookback_hours: int = 4,
    correlated_symbol: Optional[str] = None,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Fetch OHLCV data from Delta Exchange API.

    Parameters
    ----------
    symbol : str
        Delta Exchange product symbol (e.g. "BTCUSD").
    resolution : str
        Candle resolution ("1m", "5m", "15m", "1h", "1d").
    lookback_hours : int
        How many hours of history to fetch.
    correlated_symbol : str, optional
        Correlated symbol to fetch for SMT Divergence (e.g. "ETHUSD").

    Returns
    -------
    pd.DataFrame or Tuple[pd.DataFrame, pd.DataFrame]
        OHLCV DataFrame with DatetimeIndex, or tuple if correlated_symbol provided.
    """
    from trading_bot.exchange.delta_connector import DeltaConnector

    print(f"  Fetching {symbol} from Delta Exchange | resolution={resolution} | lookback={lookback_hours}h ...")
    connector = DeltaConnector()
    primary_df = connector.fetch_candles(
        symbol=symbol,
        resolution=resolution,
        lookback_hours=lookback_hours,
    )
    
    if correlated_symbol:
        print(f"  Fetching {correlated_symbol} from Delta Exchange | resolution={resolution} | lookback={lookback_hours}h ...")
        correlated_df = connector.fetch_candles(
            symbol=correlated_symbol,
            resolution=resolution,
            lookback_hours=lookback_hours,
        )
        return primary_df, correlated_df
        
    return primary_df


def fetch_multi_tf(
    symbol: str = "BTCUSD",
    timeframes: list = None,
    lookback_hours: int = 24,
) -> dict:
    """
    Fetch OHLCV data for multiple timeframes from Delta Exchange.

    Parameters
    ----------
    symbol : str
        Delta Exchange product symbol.
    timeframes : list of str
        Timeframes to fetch (e.g. ["5m", "15m", "1h"]).
    lookback_hours : int
        Hours of candle history to fetch per timeframe.

    Returns
    -------
    dict
        Mapping of timeframe -> pd.DataFrame.
    """
    from trading_bot.exchange.delta_connector import DeltaConnector

    if timeframes is None:
        timeframes = ["5m"]

    connector = DeltaConnector()
    result = {}

    for tf in timeframes:
        print(f"  Fetching {symbol} @ {tf} | lookback={lookback_hours}h ...")
        result[tf] = connector.fetch_candles(
            symbol=symbol,
            resolution=tf,
            lookback_hours=lookback_hours,
        )
        print(f"  [OK] {len(result[tf])} candles loaded for {tf}")

    return result


def generate_sample_data(n_candles: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic OHLCV data for testing the AMD strategy.

    Creates realistic-looking price data with:
      - Consolidation zones (accumulation)
      - Sweep wicks (manipulation)
      - Trending sections (distribution)
      - BTC-like price levels for realistic position sizing

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

    # Start at BTC-like price for realistic contract sizing
    price = 95000.0
    prices = []
    
    # Create phases: consolidation -> sweep -> trend -> consolidation -> ...
    phase_lengths = []
    phase_types = []
    remaining = n_candles
    while remaining > 0:
        phase_type = rng.choice(["consolidation", "trend_up", "trend_down", "sweep_up", "sweep_down"])
        if phase_type.startswith("consolidation"):
            length = min(rng.integers(15, 35), remaining)
        elif phase_type.startswith("trend"):
            length = min(rng.integers(10, 25), remaining)
        else:  # sweep
            length = min(rng.integers(3, 8), remaining)
        phase_lengths.append(length)
        phase_types.append(phase_type)
        remaining -= length

    candle_idx = 0
    for phase_type, phase_len in zip(phase_types, phase_lengths):
        for j in range(phase_len):
            if phase_type == "consolidation":
                # Tight range with small moves
                drift = -0.0005 * (price - 95000)
                shock = rng.normal(0, price * 0.0008)
            elif phase_type == "trend_up":
                drift = price * 0.0015
                shock = rng.normal(0, price * 0.001)
            elif phase_type == "trend_down":
                drift = -price * 0.0015
                shock = rng.normal(0, price * 0.001)
            elif phase_type == "sweep_up":
                if j < phase_len // 2:
                    drift = price * 0.003  # Sharp spike up
                    shock = rng.normal(0, price * 0.0005)
                else:
                    drift = -price * 0.004  # Reversal down
                    shock = rng.normal(0, price * 0.0008)
            elif phase_type == "sweep_down":
                if j < phase_len // 2:
                    drift = -price * 0.003  # Sharp spike down
                    shock = rng.normal(0, price * 0.0005)
                else:
                    drift = price * 0.004  # Reversal up
                    shock = rng.normal(0, price * 0.0008)
            else:
                drift = 0
                shock = rng.normal(0, price * 0.001)

            price += drift + shock
            price = max(price, 1000.0)
            prices.append(price)
            candle_idx += 1

    closes = np.array(prices[:n_candles])
    
    # Generate OHLC from closes
    noise_factor = closes * 0.001
    highs = closes + rng.uniform(0.1, 1.0, n_candles) * noise_factor
    lows = closes - rng.uniform(0.1, 1.0, n_candles) * noise_factor
    opens = closes + rng.uniform(-0.5, 0.5, n_candles) * noise_factor * 0.5
    
    # Ensure OHLC consistency
    highs = np.maximum(highs, np.maximum(opens, closes))
    lows = np.minimum(lows, np.minimum(opens, closes))
    
    volumes = rng.integers(100, 10000, n_candles).astype(float)

    return pd.DataFrame(
        {"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": volumes},
        index=idx,
    )
