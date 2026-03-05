"""
RSI Indicator — Calculates the Relative Strength Index using the `ta` library.

Also provides helper functions for swing-point detection on RSI values.
"""

import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator


def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate RSI using the `ta` library (Wilder's smoothing).

    Parameters
    ----------
    close : pd.Series
        Close prices.
    period : int
        RSI lookback window (default 14).

    Returns
    -------
    pd.Series
        RSI values (0–100). First `period` values will be NaN.
    """
    rsi_indicator = RSIIndicator(close=close, window=period)
    return rsi_indicator.rsi()


def find_swing_highs(
    series: np.ndarray,
    start: int,
    end: int,
    order: int = 2,
) -> list:
    """
    Find local maxima (swing highs) in a numpy array within [start, end).

    A swing high at index i means:
        series[i] > all neighbors within `order` distance on both sides.

    Parameters
    ----------
    series : np.ndarray
        Array of values (e.g. prices or RSI).
    start : int
        Start index (inclusive).
    end : int
        End index (exclusive).
    order : int
        Number of neighbors on each side to compare.

    Returns
    -------
    list of int
        Indices of swing highs.
    """
    swings = []
    for i in range(max(start, order), min(end, len(series) - order)):
        is_high = True
        for j in range(1, order + 1):
            if series[i] <= series[i - j] or series[i] <= series[i + j]:
                is_high = False
                break
        if is_high:
            swings.append(i)
    return swings


def find_swing_lows(
    series: np.ndarray,
    start: int,
    end: int,
    order: int = 2,
) -> list:
    """
    Find local minima (swing lows) in a numpy array within [start, end).

    Parameters
    ----------
    series : np.ndarray
        Array of values.
    start : int
        Start index (inclusive).
    end : int
        End index (exclusive).
    order : int
        Number of neighbors on each side to compare.

    Returns
    -------
    list of int
        Indices of swing lows.
    """
    swings = []
    for i in range(max(start, order), min(end, len(series) - order)):
        is_low = True
        for j in range(1, order + 1):
            if series[i] >= series[i - j] or series[i] >= series[i + j]:
                is_low = False
                break
        if is_low:
            swings.append(i)
    return swings
