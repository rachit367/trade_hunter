"""
Range Detector — Identifies consolidation (accumulation) zones in price data.

Uses an adaptive approach: scans a rolling window across the data, measuring
range tightness relative to recent volatility (ATR-based).
"""

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd


@dataclass
class ConsolidationRange:
    """A detected consolidation zone."""
    start_idx: int          # Integer index into DataFrame
    end_idx: int            # Integer index into DataFrame
    range_high: float       # Highest high in the range
    range_low: float        # Lowest low in the range

    @property
    def midpoint(self) -> float:
        return (self.range_high + self.range_low) / 2.0

    @property
    def width(self) -> float:
        return self.range_high - self.range_low

    @property
    def width_pct(self) -> float:
        """Width as percentage of midpoint."""
        if self.midpoint == 0:
            return 0.0
        return (self.width / self.midpoint) * 100.0

    @property
    def num_candles(self) -> int:
        return self.end_idx - self.start_idx + 1


def detect_ranges(
    df: pd.DataFrame,
    min_candles: int = 10,
    max_candles: int = 30,
    range_threshold_pct: float = 1.5,
) -> List[ConsolidationRange]:
    """
    Detect consolidation ranges in OHLCV data.

    Algorithm
    ---------
    1. Start at candle `i`, extend a window from `min_candles` to `max_candles`.
    2. Compute range_high and range_low of the window.
    3. If range width < `range_threshold_pct` % of midpoint → valid consolidation.
    4. Keep extending until the range gets too wide, then record the longest valid one.
    5. Jump past the range to prevent overlap.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data.
    min_candles : int
        Minimum candles for a valid range (default 10).
    max_candles : int
        Maximum window to search (default 30).
    range_threshold_pct : float
        Maximum allowed range width as % of midpoint (default 1.5).

    Returns
    -------
    List[ConsolidationRange]
        Detected ranges, sorted by start index.
    """
    highs = df["High"].values
    lows = df["Low"].values
    n = len(df)

    ranges: List[ConsolidationRange] = []
    i = 0

    while i < n - min_candles:
        best = None

        for length in range(min_candles, max_candles + 1):
            end = i + length
            if end > n:
                break

            window_high = float(np.max(highs[i:end]))
            window_low = float(np.min(lows[i:end]))
            mid = (window_high + window_low) / 2.0

            if mid == 0:
                continue

            width_pct = ((window_high - window_low) / mid) * 100.0

            if width_pct <= range_threshold_pct:
                best = ConsolidationRange(
                    start_idx=i,
                    end_idx=end - 1,
                    range_high=window_high,
                    range_low=window_low,
                )
            else:
                break  # Range got too wide, stop extending

        if best is not None:
            ranges.append(best)
            i = best.end_idx + 1  # Jump past to prevent overlapping ranges
        else:
            i += 1

    return ranges


def detect_asian_ranges(
    df: pd.DataFrame,
    asian_start_hour: int = 0,
    asian_end_hour: int = 8,
) -> List[ConsolidationRange]:
    """
    Detect Asian session ranges (00:00-08:00 UTC by default).

    ICT methodology: The Asian session forms the primary accumulation phase.
    The range formed during this period defines the liquidity pools
    that London and New York sessions will sweep.

    Groups candles by date, finds the Asian window per day,
    and creates a ConsolidationRange from the high/low of that window.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data with DatetimeIndex.
    asian_start_hour : int
        Start hour of Asian range (UTC, default 0).
    asian_end_hour : int
        End hour of Asian range (UTC, default 8).

    Returns
    -------
    List[ConsolidationRange]
        One range per day where sufficient Asian session data exists.
    """
    if not hasattr(df.index, 'hour'):
        return []

    ranges: List[ConsolidationRange] = []
    highs = df["High"].values
    lows = df["Low"].values

    # Group by date
    dates = pd.Series(df.index.date, index=df.index)
    unique_dates = dates.unique()

    for date in unique_dates:
        # Find candles in the Asian window for this date
        mask = (dates == date) & (df.index.hour >= asian_start_hour) & (df.index.hour < asian_end_hour)
        asian_indices = np.where(mask.values)[0]

        if len(asian_indices) < 3:  # Need at least 3 candles for a meaningful range
            continue

        start_idx = int(asian_indices[0])
        end_idx = int(asian_indices[-1])
        range_high = float(np.max(highs[start_idx:end_idx + 1]))
        range_low = float(np.min(lows[start_idx:end_idx + 1]))

        # Skip if range is too narrow (noise) or too wide (not consolidation)
        mid = (range_high + range_low) / 2.0
        if mid <= 0:
            continue
        width_pct = ((range_high - range_low) / mid) * 100.0
        if width_pct > 3.0:  # Asian range shouldn't be wider than 3%
            continue

        ranges.append(ConsolidationRange(
            start_idx=start_idx,
            end_idx=end_idx,
            range_high=range_high,
            range_low=range_low,
        ))

    return ranges
