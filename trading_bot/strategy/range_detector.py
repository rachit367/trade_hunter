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
