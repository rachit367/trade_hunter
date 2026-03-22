"""Tests for displacement detection — strong vs weak candles after sweep."""

import pytest
import pandas as pd
import numpy as np

from trading_bot.strategy.amd_strategy import detect_displacement


def _make_df(candles):
    """Build a DataFrame from a list of (open, high, low, close) tuples."""
    idx = pd.date_range("2025-01-01", periods=len(candles), freq="5min")
    data = {
        "Open": [c[0] for c in candles],
        "High": [c[1] for c in candles],
        "Low": [c[2] for c in candles],
        "Close": [c[3] for c in candles],
        "Volume": [1000.0] * len(candles),
    }
    return pd.DataFrame(data, index=idx)


class TestDisplacement:
    def test_bearish_displacement_detected(self):
        """A strong bearish candle after a sweep should be displacement."""
        # Candles: neutral, neutral, strong bearish
        candles = [
            (100, 101, 99, 100),   # neutral
            (100, 101, 99, 100),   # neutral
            (100, 101, 99, 100),   # sweep candle (index=2)
            (100, 100.5, 96, 96.5),  # strong bearish displacement
            (96.5, 97, 95, 95.5),  # continuation
        ]
        df = _make_df(candles)
        atr = np.array([2.0] * len(candles))  # ATR = 2.0

        result = detect_displacement(
            df, sweep_idx=2, direction="bearish", atr=atr,
            body_ratio_min=0.6, body_atr_min=0.5, lookforward=2,
        )
        assert result is True

    def test_bullish_displacement_detected(self):
        """A strong bullish candle after a sweep should be displacement."""
        candles = [
            (100, 101, 99, 100),
            (100, 101, 99, 100),
            (100, 101, 99, 100),   # sweep candle (index=2)
            (100, 104, 99.5, 103.5),  # strong bullish displacement
            (103.5, 105, 103, 104.5),
        ]
        df = _make_df(candles)
        atr = np.array([2.0] * len(candles))

        result = detect_displacement(
            df, sweep_idx=2, direction="bullish", atr=atr,
            body_ratio_min=0.6, body_atr_min=0.5, lookforward=2,
        )
        assert result is True

    def test_weak_candle_not_displacement(self):
        """A small-body candle (doji-like) should NOT be displacement."""
        candles = [
            (100, 101, 99, 100),
            (100, 101, 99, 100),
            (100, 101, 99, 100),   # sweep
            (100, 100.3, 99.7, 100.1),  # doji — no displacement
            (100.1, 100.4, 99.8, 100.2),
        ]
        df = _make_df(candles)
        atr = np.array([2.0] * len(candles))

        result = detect_displacement(
            df, sweep_idx=2, direction="bearish", atr=atr,
            body_ratio_min=0.6, body_atr_min=0.5, lookforward=2,
        )
        assert result is False

    def test_displacement_at_end_of_data(self):
        """If sweep is at the last candle, no room for displacement."""
        candles = [
            (100, 101, 99, 100),
            (100, 101, 99, 100),
        ]
        df = _make_df(candles)
        atr = np.array([2.0] * len(candles))

        result = detect_displacement(
            df, sweep_idx=1, direction="bearish", atr=atr,
            body_ratio_min=0.6, body_atr_min=0.5, lookforward=2,
        )
        assert result is False
