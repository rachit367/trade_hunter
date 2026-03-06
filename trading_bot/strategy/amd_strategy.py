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

    strict_divergence: bool = False  # Require divergence to take a trade

    # HTF Alignment Filter
    htf_alignment_mode: str = "ema_50"            # "market_structure", "liquidity_draw", "ema_50", or "none"
    htf_lookback: int = 12                        # Candles for MS (12 * 5m = 1H)
    pdl_pdh_lookback: int = 288                   # Candles for PDH/PDL (288 * 5m = 24H)

    # Divergence
    divergence_lookback: int = 15
    swing_order: int = 2
    min_divergence_swings: int = 2   # Allow standard 2-point divergence
    max_divergence_swings: int = 5   # Discard if trend is too exhausted (>= 6)
    divergence_atr_multiplier: float = 0.5 # Relaxed ATR

    # FVG / Entry Timing
    require_fvg: bool = True
    fvg_lookforward: int = 20        # Max candles to wait for an FVG to form after the sweep
    retrace_lookforward: int = 20    # Max candles to wait for retracement into FVG

    # Risk management
    risk_reward_ratio: float = 2.0
    sl_buffer_pct: float = 0.05      # Buffer above/below manipulation wick
    risk_per_trade_pct: float = 1.0  # % of account risked per trade

    # Post-range scan window
    max_scan_after_range: int = 20   # Max candles after range to look for breakout


def detect_liquidity_sweep(df: pd.DataFrame, i: int, lookback: int = 10) -> Optional[str]:
    """
    Check if the current candle forms a liquidity sweep.
    A sweep is when price wicks beyond a local high/low but closes back inside.
    """
    if i < lookback:
        return None
        
    prev_high = df["High"].iloc[i-lookback:i].max()
    prev_low = df["Low"].iloc[i-lookback:i].min()
    
    candle = df.iloc[i]
    
    # Relaxed Bearish sweep: Wicks above local max. We don't mandate the exact same 5m 
    # candle closes below, because the FVG requirement + Divergence handles the reversal confirmation.
    # We just require the sweep wick to take liquidity but close underneath the absolute wick high.
    if candle["High"] > prev_high and candle["Close"] < candle["High"]:
        return "bearish"
        
    # Relaxed Bullish sweep: Wicks below local min, but closes above the absolute wick low.
    if candle["Low"] < prev_low and candle["Close"] > candle["Low"]:
        return "bullish"
        
    return None


def get_htf_bias(df: pd.DataFrame, i: int, config: StrategyConfig) -> str:
    """
    Determine the Higher Timeframe Bias based on the configured mode.
    Outputs: "bullish", "bearish", or "neutral"
    """
    mode = config.htf_alignment_mode
    if mode == "none" or mode is None:
        return "neutral"
        
    if mode == "market_structure":
        if i < config.htf_lookback:
            return "neutral"
        
        # Check for Higher Highs / Lower Lows
        # A simple approach: compare current high/low to max/min of lookback period
        current_high = df["High"].iloc[i]
        prev_period_high = df["High"].iloc[i - config.htf_lookback : i].max()
        
        current_low = df["Low"].iloc[i]
        prev_period_low = df["Low"].iloc[i - config.htf_lookback : i].min()
        
        if current_high > prev_period_high:
            return "bullish" # Breaking above previous highs
        elif current_low < prev_period_low:
            return "bearish" # Breaking below previous lows
        return "neutral"
        
    elif mode == "liquidity_draw":
        if i < config.pdl_pdh_lookback:
            return "neutral"
            
        pdh = df["High"].iloc[i - config.pdl_pdh_lookback : i].max()
        pdl = df["Low"].iloc[i - config.pdl_pdh_lookback : i].min()
        current_price = df["Close"].iloc[i]
        
        # If price is closer to PDH, bias is bullish (drawn to buy-side liquidity)
        if abs(current_price - pdh) < abs(current_price - pdl):
            return "bullish"
        else:
            return "bearish"
            
    elif mode == "ema_50":
        # This will be handled in generate_signals directly to save Pandas Series calls
        return "neutral" 
        
    return "neutral"


def detect_fvg(df: pd.DataFrame, start_idx: int, end_idx: int, direction: str) -> Optional[tuple]:
    """
    Scan for the first Fair Value Gap (FVG) in the given direction.
    Bearish FVG: Low of candle 1 > High of candle 3
    Bullish FVG: High of candle 1 < Low of candle 3
    
    Returns: (fvg_top, fvg_bottom, fvg_idx) or None
    """
    for j in range(start_idx, end_idx):
        if j < 2:
            continue
            
        if direction == "bearish":
            # Candle 1 = j-2, Candle 3 = j
            c1_low = df["Low"].iloc[j-2]
            c3_high = df["High"].iloc[j]
            # Must also be a bearish displacement candle (c2 close < c2 open)
            c2_open = df["Open"].iloc[j-1]
            c2_close = df["Close"].iloc[j-1]
            
            if c1_low > c3_high and c2_close < c2_open:
                fvg_top = c1_low
                fvg_bottom = c3_high
                return (fvg_top, fvg_bottom, j)
                
        elif direction == "bullish":
            # Candle 1 = j-2, Candle 3 = j
            c1_high = df["High"].iloc[j-2]
            c3_low = df["Low"].iloc[j]
            # Must also be a bullish displacement candle (c2 close > c2 open)
            c2_open = df["Open"].iloc[j-1]
            c2_close = df["Close"].iloc[j-1]
            
            if c1_high < c3_low and c2_close > c2_open:
                fvg_top = c3_low
                fvg_bottom = c1_high
                return (fvg_top, fvg_bottom, j)
                
    return None


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

    # Step 1: Calculate RSI & ATR & EMA
    rsi_series = calculate_rsi(df["Close"], config.rsi_period)
    rsi = rsi_series.values
    atr_series = AverageTrueRange(df["High"], df["Low"], df["Close"], window=14).average_true_range()
    atr = atr_series.values
    
    # Pre-calculate 50 EMA if using that mode
    ema50 = None
    if config.htf_alignment_mode == "ema_50":
        ema50 = df["Close"].ewm(span=50, adjust=False).mean().values
        
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
    seen_sweep_indices = set()
    seen_entry_indices = set()
    last_trade_range_idx = -1

    # Step 3–5: For each range, detect breakout + divergence
    for rng in ranges:
        if rng.start_idx == last_trade_range_idx:
            continue
            
        scan_start = rng.end_idx + 1
        scan_end = min(scan_start + config.max_scan_after_range, n)
        signal_found = False

        for i in range(scan_start, scan_end):
            if signal_found:
                break

            # --- Breakout ABOVE range high ---
            breakout_level = rng.range_high * (1 + config.breakout_pct / 100.0)
            if highs[i] > breakout_level:
                sweep = detect_liquidity_sweep(df, i, lookback=10)
                if sweep == "bearish":
                    sweep_idx = i
                    if sweep_idx in seen_sweep_indices:
                        continue
                        
                    div = detect_bearish_divergence(
                        highs, rsi, atr, i,
                        lookback=config.divergence_lookback,
                        swing_order=config.swing_order,
                        atr_multiplier=config.divergence_atr_multiplier,
                        min_swings=config.min_divergence_swings,
                    )
                    
                    if config.strict_divergence and div is None:
                        continue # Skip if divergence is required but not found
                        
                    if div is not None and div.swings > config.max_divergence_swings:
                        continue # Too exhausted

                    # SHORT signal — manipulation of buy-side liquidity
                    # Check Trend at the START of the range (before manipulation)
                    bias = get_htf_bias(df, rng.start_idx, config)
                    if config.htf_alignment_mode == "ema_50":
                        bias = "bullish" if closes[rng.start_idx] > ema50[rng.start_idx] else "bearish"
                        
                    if bias == "bullish":
                        continue # Only take SHORTs if bias is bearish or neutral

                    manipulation_high = highs[i]
                    original_sl = max(manipulation_high, rng.range_high)
                    sl = original_sl * (1 + config.sl_buffer_pct / 100.0)
                    
                    if not config.require_fvg:
                        entry = closes[i]
                        sl = original_sl
                        tp = rng.range_low
                        entry_idx = i
                    else:
                        # Scan forward for Bearish FVG
                        fvg_search_end = min(i + config.fvg_lookforward, n)
                        fvg_result = detect_fvg(df, i+2, fvg_search_end, "bearish")
                        
                        if fvg_result is None:
                            continue # Setup failed, no displacement FVG
                            
                        fvg_top, fvg_bottom, fvg_idx = fvg_result
                        
                        # Scan forward from FVG to wait for retracement entry
                        retrace_search_start = fvg_idx + 1
                        retrace_search_end = min(retrace_search_start + config.retrace_lookforward, n)
                        
                        entry_idx = -1
                        entry_price = 0.0
                        for k in range(retrace_search_start, retrace_search_end):
                            if highs[k] >= fvg_bottom:
                                entry_idx = k
                                # Entry triggers at the bottom of the FVG as price returns up into it
                                entry_price = fvg_bottom 
                                break
                                
                        if entry_idx == -1:
                            continue # Setup valid, but price never retraced
                            
                        entry = entry_price
                        if sl <= entry:
                            sl = entry * 1.0015
                        tp = rng.range_low
                        i = entry_idx # Fast-forward the outer index for timeline consistency

                    if entry_idx in seen_entry_indices:
                        continue

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
                        entry_idx=entry_idx,
                        entry_time=df.index[entry_idx],
                        range_high=rng.range_high,
                        range_low=rng.range_low,
                        range_start_idx=rng.start_idx,
                        range_end_idx=rng.end_idx,
                        divergence=div,
                    ))
                    seen_sweep_indices.add(sweep_idx)
                    seen_entry_indices.add(entry_idx)
                    last_trade_range_idx = rng.start_idx
                    signal_found = True
                    break

            # --- Breakdown BELOW range low ---
            breakdown_level = rng.range_low * (1 - config.breakout_pct / 100.0)
            if lows[i] < breakdown_level:
                sweep = detect_liquidity_sweep(df, i, lookback=10)
                if sweep == "bullish":
                    sweep_idx = i
                    if sweep_idx in seen_sweep_indices:
                        continue
                        
                    div = detect_bullish_divergence(
                        lows, rsi, atr, i,
                        lookback=config.divergence_lookback,
                        swing_order=config.swing_order,
                        atr_multiplier=config.divergence_atr_multiplier,
                        min_swings=config.min_divergence_swings,
                    )
                    
                    if config.strict_divergence and div is None:
                        continue # Skip if divergence is required but not found
                        
                    if div is not None and div.swings > config.max_divergence_swings:
                        continue # Too exhausted

                    # LONG signal — manipulation of sell-side liquidity
                    # Check Trend at the START of the range (before manipulation)
                    bias = get_htf_bias(df, rng.start_idx, config)
                    if config.htf_alignment_mode == "ema_50":
                        bias = "bullish" if closes[rng.start_idx] > ema50[rng.start_idx] else "bearish"
                        
                    if bias == "bearish":
                        continue # Only take LONGs if bias is bullish or neutral

                    manipulation_low = lows[i]
                    original_sl = min(manipulation_low, rng.range_low)
                    sl = original_sl * (1 - config.sl_buffer_pct / 100.0)
                    
                    if not config.require_fvg:
                        entry = closes[i]
                        sl = original_sl
                        tp = rng.range_high
                        entry_idx = i
                    else:
                        # Scan forward for Bullish FVG
                        fvg_search_end = min(i + config.fvg_lookforward, n)
                        fvg_result = detect_fvg(df, i+2, fvg_search_end, "bullish")
                        
                        if fvg_result is None:
                            continue # Setup failed, no displacement FVG
                            
                        fvg_top, fvg_bottom, fvg_idx = fvg_result
                        
                        # Scan forward from FVG to wait for retracement entry
                        retrace_search_start = fvg_idx + 1
                        retrace_search_end = min(retrace_search_start + config.retrace_lookforward, n)
                        
                        entry_idx = -1
                        entry_price = 0.0
                        for k in range(retrace_search_start, retrace_search_end):
                            if lows[k] <= fvg_top:
                                entry_idx = k
                                # Entry triggers at the top of the FVG as price drops into it
                                entry_price = fvg_top 
                                break
                                
                        if entry_idx == -1:
                            continue # Setup valid, but price never retraced
                            
                        entry = entry_price
                        if sl >= entry:
                            sl = entry * 0.9985
                        tp = rng.range_high
                        i = entry_idx # Fast-forward the outer index

                    if entry_idx in seen_entry_indices:
                        continue

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
                        entry_idx=entry_idx,
                        entry_time=df.index[entry_idx],
                        range_high=rng.range_high,
                        range_low=rng.range_low,
                        range_start_idx=rng.start_idx,
                        range_end_idx=rng.end_idx,
                        divergence=div,
                    ))
                    seen_sweep_indices.add(sweep_idx)
                    seen_entry_indices.add(entry_idx)
                    last_trade_range_idx = rng.start_idx
                    signal_found = True
                    break

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
