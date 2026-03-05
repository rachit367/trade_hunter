"""
AMD Strategy — Full ICT Accumulation → Manipulation → Distribution orchestration.

Combines range detection, breakout detection, RSI divergence confirmation,
and risk management into a unified signal generator.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum

import pandas as pd
from ta.volatility import AverageTrueRange

from trading_bot.indicators.rsi import calculate_rsi
from trading_bot.strategy.range_detector import detect_ranges, ConsolidationRange
from trading_bot.strategy.divergence_detector import (
    detect_bearish_divergence,
    detect_bullish_divergence,
    DivergenceResult,
)


class Direction(Enum):
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass
class TradeSignal:
    """A fully defined trade signal."""
    direction: Direction
    entry_price: float
    stop_loss: float
    take_profit: float
    entry_idx: int
    entry_time: object
    range_high: float
    range_low: float
    range_start_idx: int
    range_end_idx: int
    divergence: Optional[DivergenceResult] = None

    @property
    def risk(self) -> float:
        return abs(self.entry_price - self.stop_loss)

    @property
    def reward(self) -> float:
        return abs(self.take_profit - self.entry_price)

    @property
    def rr_ratio(self) -> float:
        r = self.risk
        return self.reward / r if r > 0 else 0.0

    def __repr__(self) -> str:
        return (
            f"TradeSignal({self.direction.value} @ {self.entry_price:.2f}, "
            f"SL={self.stop_loss:.2f}, TP={self.take_profit:.2f}, "
            f"R:R={self.rr_ratio:.1f}, time={self.entry_time})"
        )


@dataclass
class StrategyConfig:
    """All tunable parameters for the AMD strategy."""
    # RSI
    rsi_period: int = 14

    # Range detection
    min_range_candles: int = 10
    max_range_candles: int = 30
    range_threshold_pct: float = 1.5

    # Breakout
    breakout_pct: float = 0.10   # Min % beyond range to confirm breakout

    # Divergence
    divergence_lookback: int = 10
    swing_order: int = 2
    min_rsi_diff: float = 6.0        # Minimum RSI difference to consider valid
    divergence_atr_multiplier: float = 0.3 # Minimum price move in terms of ATR -> divergence valid

    # Risk management
    risk_reward_ratio: float = 2.0
    sl_buffer_pct: float = 0.05      # Buffer above/below manipulation wick
    risk_per_trade_pct: float = 1.0  # % of account risked per trade

    # Post-range scan window
    max_scan_after_range: int = 20   # Max candles after range to look for breakout


def generate_signals(
    df: pd.DataFrame,
    config: StrategyConfig = None,
) -> List[TradeSignal]:
    """
    Execute the full AMD pipeline and produce trade signals.

    Pipeline
    --------
    1. Calculate RSI on close prices.
    2. Detect consolidation ranges (Accumulation).
    3. For each range, scan subsequent candles for a breakout (Manipulation).
    4. On breakout, check RSI divergence for confirmation.
    5. If confirmed → generate trade signal with SL / TP (Distribution target).

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data with DatetimeIndex.
    config : StrategyConfig, optional
        Strategy parameters.

    Returns
    -------
    List[TradeSignal]
        All generated signals.
    """
    if config is None:
        config = StrategyConfig()

    # Step 1: Calculate RSI & ATR
    rsi_series = calculate_rsi(df["Close"], config.rsi_period)
    rsi = rsi_series.values
    atr_series = AverageTrueRange(df["High"], df["Low"], df["Close"], window=14).average_true_range()
    atr = atr_series.values
    highs = df["High"].values
    lows = df["Low"].values
    closes = df["Close"].values
    n = len(df)

    # Step 2: Detect consolidation ranges
    ranges = detect_ranges(
        df,
        min_candles=config.min_range_candles,
        max_candles=config.max_range_candles,
        range_threshold_pct=config.range_threshold_pct,
    )

    signals: List[TradeSignal] = []
    seen_times = set()

    # Step 3–5: For each range, detect breakout + divergence
    for rng in ranges:
        scan_start = rng.end_idx + 1
        scan_end = min(scan_start + config.max_scan_after_range, n)
        signal_found = False

        for i in range(scan_start, scan_end):
            if signal_found:
                break

            # --- Breakout ABOVE range high ---
            breakout_level = rng.range_high * (1 + config.breakout_pct / 100.0)
            if highs[i] > breakout_level:
                div = detect_bearish_divergence(
                    highs, rsi, atr, i,
                    lookback=config.divergence_lookback,
                    swing_order=config.swing_order,
                    min_rsi_diff=config.min_rsi_diff,
                    atr_multiplier=config.divergence_atr_multiplier,
                )
                if div is not None:
                    if df.index[i] in seen_times:
                        signal_found = True
                        continue

                    # SHORT signal — manipulation of buy-side liquidity
                    manipulation_high = highs[i]
                    entry = closes[i]
                    sl = manipulation_high * (1 + config.sl_buffer_pct / 100.0)
                    # TP = range low (opposite side of the range)
                    tp = rng.range_low

                    risk = abs(entry - sl)
                    reward = entry - tp
                    if risk > 0:
                        rr = reward / risk
                        rr = max(1.5, min(rr, 5.0))
                        tp = entry - (rr * risk)

                    signals.append(TradeSignal(
                        direction=Direction.SHORT,
                        entry_price=entry,
                        stop_loss=sl,
                        take_profit=tp,
                        entry_idx=i,
                        entry_time=df.index[i],
                        range_high=rng.range_high,
                        range_low=rng.range_low,
                        range_start_idx=rng.start_idx,
                        range_end_idx=rng.end_idx,
                        divergence=div,
                    ))
                    seen_times.add(df.index[i])
                    signal_found = True
                    continue

            # --- Breakdown BELOW range low ---
            breakdown_level = rng.range_low * (1 - config.breakout_pct / 100.0)
            if lows[i] < breakdown_level:
                div = detect_bullish_divergence(
                    lows, rsi, atr, i,
                    lookback=config.divergence_lookback,
                    swing_order=config.swing_order,
                    min_rsi_diff=config.min_rsi_diff,
                    atr_multiplier=config.divergence_atr_multiplier,
                )
                if div is not None:
                    if df.index[i] in seen_times:
                        signal_found = True
                        continue

                    # LONG signal — manipulation of sell-side liquidity
                    manipulation_low = lows[i]
                    entry = closes[i]
                    sl = manipulation_low * (1 - config.sl_buffer_pct / 100.0)
                    # TP = range high (opposite side of the range)
                    tp = rng.range_high

                    risk = abs(entry - sl)
                    reward = tp - entry
                    if risk > 0:
                        rr = reward / risk
                        rr = max(1.5, min(rr, 5.0))
                        tp = entry + (rr * risk)

                    signals.append(TradeSignal(
                        direction=Direction.LONG,
                        entry_price=entry,
                        stop_loss=sl,
                        take_profit=tp,
                        entry_idx=i,
                        entry_time=df.index[i],
                        range_high=rng.range_high,
                        range_low=rng.range_low,
                        range_start_idx=rng.start_idx,
                        range_end_idx=rng.end_idx,
                        divergence=div,
                    ))
                    seen_times.add(df.index[i])
                    signal_found = True

    return signals


def calculate_position_size(
    account_balance: float,
    risk_pct: float,
    entry_price: float,
    stop_loss: float,
) -> float:
    """
    Calculate position size based on fixed-percentage risk.

    Parameters
    ----------
    account_balance : float
        Total account balance.
    risk_pct : float
        Percentage of account to risk (e.g. 1.0 = 1%).
    entry_price : float
        Entry price.
    stop_loss : float
        Stop-loss price.

    Returns
    -------
    float
        Position size (number of units/contracts).
    """
    risk_amount = account_balance * (risk_pct / 100.0)
    risk_per_unit = abs(entry_price - stop_loss)
    if risk_per_unit == 0:
        return 0.0
    return risk_amount / risk_per_unit
