"""
Divergence Detector — Detects RSI divergences for confirming manipulation.

Bearish Divergence: Price ↑ higher high, RSI ↓ lower high → sell signal.
Bullish Divergence: Price ↓ lower low,  RSI ↑ higher low → buy signal.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from trading_bot.indicators.rsi import find_swing_highs, find_swing_lows


@dataclass
class DivergenceResult:
    """Details of a detected divergence."""
    divergence_type: str     # "bearish" or "bullish"
    current_idx: int         # Index of the current candle
    prior_idx: int           # Index of the prior swing point
    current_price: float     # Price at current candle
    prior_price: float       # Price at prior swing
    current_rsi: float       # RSI at current candle
    prior_rsi: float         # RSI at prior swing


def detect_bearish_divergence(
    highs: np.ndarray,
    rsi: np.ndarray,
    current_idx: int,
    lookback: int = 10,
    swing_order: int = 2,
) -> Optional[DivergenceResult]:
    """
    Check for bearish divergence at `current_idx`.

    Bearish = price makes a HIGHER HIGH while RSI makes a LOWER HIGH.

    Parameters
    ----------
    highs : np.ndarray
        Array of high prices.
    rsi : np.ndarray
        Array of RSI values.
    current_idx : int
        Current candle index (the breakout candle).
    lookback : int
        How many candles back to search for the prior swing high.
    swing_order : int
        Order parameter for swing detection.

    Returns
    -------
    DivergenceResult or None
        Divergence details if found, else None.
    """
    if current_idx < lookback + swing_order:
        return None

    current_rsi = rsi[current_idx]
    current_high = highs[current_idx]

    if np.isnan(current_rsi):
        return None

    # Find swing highs in the lookback window BEFORE current candle
    search_start = max(0, current_idx - lookback)
    search_end = current_idx  # exclusive — don't include current

    swing_high_indices = find_swing_highs(highs, search_start, search_end, order=swing_order)

    if not swing_high_indices:
        # Fallback: use the max high in the lookback window
        window = highs[search_start:search_end]
        if len(window) == 0:
            return None
        max_idx = search_start + int(np.argmax(window))
        if np.isnan(rsi[max_idx]):
            return None
        swing_high_indices = [max_idx]

    # Check each prior swing high for divergence
    for prior_idx in reversed(swing_high_indices):  # Most recent first
        prior_high = highs[prior_idx]
        prior_rsi = rsi[prior_idx]

        if np.isnan(prior_rsi):
            continue

        # Bearish divergence: price HH + RSI LH
        if current_high > prior_high and current_rsi < prior_rsi:
            return DivergenceResult(
                divergence_type="bearish",
                current_idx=current_idx,
                prior_idx=prior_idx,
                current_price=current_high,
                prior_price=prior_high,
                current_rsi=current_rsi,
                prior_rsi=prior_rsi,
            )

    return None


def detect_bullish_divergence(
    lows: np.ndarray,
    rsi: np.ndarray,
    current_idx: int,
    lookback: int = 10,
    swing_order: int = 2,
) -> Optional[DivergenceResult]:
    """
    Check for bullish divergence at `current_idx`.

    Bullish = price makes a LOWER LOW while RSI makes a HIGHER LOW.

    Parameters
    ----------
    lows : np.ndarray
        Array of low prices.
    rsi : np.ndarray
        Array of RSI values.
    current_idx : int
        Current candle index (the breakdown candle).
    lookback : int
        How many candles back to search for the prior swing low.
    swing_order : int
        Order parameter for swing detection.

    Returns
    -------
    DivergenceResult or None
        Divergence details if found, else None.
    """
    if current_idx < lookback + swing_order:
        return None

    current_rsi = rsi[current_idx]
    current_low = lows[current_idx]

    if np.isnan(current_rsi):
        return None

    search_start = max(0, current_idx - lookback)
    search_end = current_idx

    swing_low_indices = find_swing_lows(lows, search_start, search_end, order=swing_order)

    if not swing_low_indices:
        window = lows[search_start:search_end]
        if len(window) == 0:
            return None
        min_idx = search_start + int(np.argmin(window))
        if np.isnan(rsi[min_idx]):
            return None
        swing_low_indices = [min_idx]

    for prior_idx in reversed(swing_low_indices):
        prior_low = lows[prior_idx]
        prior_rsi = rsi[prior_idx]

        if np.isnan(prior_rsi):
            continue

        # Bullish divergence: price LL + RSI HL
        if current_low < prior_low and current_rsi > prior_rsi:
            return DivergenceResult(
                divergence_type="bullish",
                current_idx=current_idx,
                prior_idx=prior_idx,
                current_price=current_low,
                prior_price=prior_low,
                current_rsi=current_rsi,
                prior_rsi=prior_rsi,
            )

    return None
