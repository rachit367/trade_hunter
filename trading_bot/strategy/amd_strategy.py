"""
AMD Strategy — Full ICT Accumulation → Manipulation → Distribution orchestration.

Combines range detection, breakout detection, RSI divergence confirmation,
higher timeframe trend alignment, session filtering (killzones),
market structure (BOS/CHoCH), order block detection,
and risk management into a unified signal generator.

ICT/SMC Concepts Implemented:
  - Accumulation: Consolidation range detection
  - Manipulation: Liquidity sweep beyond range highs/lows
  - Distribution: Entry via FVG or Order Block after sweep, targeting opposite range boundary
  - HTF Alignment: 1H structure from aggregated 5m candles + EMA 200/50
  - Session Filtering: London (02:00-05:00 UTC) and NY (12:00-15:00 UTC) killzones
  - Market Structure: BOS (Break of Structure) and CHoCH (Change of Character)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from enum import Enum

import pandas as pd
import numpy as np
from ta.volatility import AverageTrueRange

from trading_bot.indicators.rsi import calculate_rsi
from trading_bot.strategy.range_detector import detect_ranges, ConsolidationRange
from trading_bot.strategy.divergence_detector import (
    detect_bearish_divergence,
    detect_bullish_divergence,
    DivergenceResult,
)
from trading_bot.strategy.smt_detector import detect_smt_divergence, SMTResult
from trading_bot.strategy.range_detector import detect_asian_ranges


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
    smt_divergence: Optional[SMTResult] = None
    htf_bias: str = "neutral"
    session: str = "unknown"
    entry_type: str = "fvg"  # "bpr", "fvg", "order_block", or "close"
    confidence: float = 0.5  # 0.0-1.0 signal quality score for dynamic sizing

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
            f"R:R={self.rr_ratio:.1f}, time={self.entry_time}, "
            f"bias={self.htf_bias}, session={self.session}, entry={self.entry_type})"
        )


@dataclass
class StrategyConfig:
    """All tunable parameters for the AMD strategy."""
    # RSI
    rsi_period: int = 14

    # Range detection
    min_range_candles: int = 8   # Lowered from 10, tighter than 6 which caught noise
    max_range_candles: int = 30
    range_threshold_pct: float = 1.5

    # Breakout
    breakout_pct: float = 0.20   # Min % beyond range to confirm breakout (raised from 0.10)
    breakout_mode: str = "pct"   # "pct" or "atr" (ATR-based breakout threshold)
    breakout_atr_multiplier: float = 0.3  # Breakout must exceed 0.3 * ATR

    strict_divergence: bool = False  # Divergence boosts confidence but not required

    # HTF Alignment Filter
    htf_alignment_mode: str = "multi_tf"  # "multi_tf", "ema_50", "market_structure", "liquidity_draw", or "none"
    htf_lookback: int = 12               # Candles for MS (12 * 5m = 1H)
    pdl_pdh_lookback: int = 288          # Candles for PDH/PDL (288 * 5m = 24H)
    htf_ema_fast: int = 50               # Fast EMA period for 1H HTF
    htf_ema_slow: int = 200              # Slow EMA period for 1H HTF
    htf_ema_slope_confirm: bool = True   # Require EMA fast to be RISING for LONG, FALLING for SHORT
    htf_ema_slope_periods: int = 3       # How many HTF bars to measure EMA slope over

    # Session Filtering (ICT Killzones)
    use_session_filter: bool = True
    # London Killzone: 02:00 - 05:00 UTC
    london_start_hour: int = 2
    london_end_hour: int = 5
    # New York Killzone: 12:00 - 15:00 UTC
    ny_start_hour: int = 12
    ny_end_hour: int = 17  # Extended to capture full NY session (Change 10)
    # Asian session: 00:00 - 02:00 UTC (lower probability, optional)
    allow_asian_session: bool = False
    asian_start_hour: int = 0
    asian_end_hour: int = 2

    # Divergence
    divergence_lookback: int = 15
    swing_order: int = 2
    min_divergence_swings: int = 2
    max_divergence_swings: int = 5
    divergence_atr_multiplier: float = 0.5

    # SMT Divergence (BTC vs ETH correlation)
    use_smt_divergence: bool = False
    smt_lookback: int = 15

    # FVG / BPR / Entry Timing
    require_fvg: bool = True
    use_bpr_priority: bool = True  # Prioritize Balanced Price Ranges
    use_order_blocks: bool = True   # OBs kept but gated by min_signal_confidence (OB+divergence = 0.65, OB alone = 0.40)
    fvg_lookforward: int = 20
    retrace_lookforward: int = 20

    # Risk management
    risk_reward_ratio: float = 2.0
    sl_buffer_pct: float = 0.15
    sl_buffer_mode: str = "atr"
    sl_buffer_atr_mult: float = 0.3  # Wider ATR buffer — prevents candle-1/2 SL hits
    risk_per_trade_pct: float = 1.0
    min_rr_ratio: float = 2.0  # Raised from 1.5 — marginal RR setups not worth fees
    max_rr_ratio: float = 5.0

    # Signal deduplication — prevents clustering of signals in same price zone
    signal_dedup_hours: float = 4.0   # Min hours between signals in same direction + zone
    signal_dedup_pct: float = 1.5     # Consider signals in same zone if within X% price

    # Entry momentum confirmation (disabled — FVG/BPR entries naturally retrace, close is counter-momentum by design)
    require_entry_momentum: bool = False

    # Minimum confidence gate: filters low-quality signals (e.g. OB without divergence scores 0.40, rejected)
    min_signal_confidence: float = 0.45

    # ADX ranging-market filter — in choppy/ranging conditions, penalise non-FVG entries
    adx_period: int = 14
    adx_ranging_threshold: float = 20.0    # ADX below this = ranging market
    adx_range_extra_penalty: float = 0.15  # Extra confidence deduction in ranging market

    # ATR volatility spike guard — skip entries when short-term ATR >> baseline ATR
    max_atr_spike_ratio: float = 2.0       # 0 = disabled; skip if 5-candle ATR > N × 50-candle ATR

    # Order Block candle quality filter — OB candle body must be meaningful
    ob_min_body_atr_ratio: float = 0.3     # body < 0.3 × ATR → skip (doji/micro candle)

    # Dynamic position sizing based on signal confidence
    use_confidence_sizing: bool = True    # Scale position size by signal confidence
    min_confidence_multiplier: float = 0.5  # Weakest signals get 50% of normal size
    max_confidence_multiplier: float = 1.2  # Strongest signals get 120%

    # Post-range scan window
    max_scan_after_range: int = 15

    # Allow multiple signals from the same range's scan window (increases frequency)
    max_signals_per_range: int = 1  # 1 = current behaviour; 2 = allow a second entry from same range

    # HTF alignment strictness
    require_htf_alignment: bool = True

    # Allow neutral HTF bias when divergence is confirmed — divergence is the strongest ICT
    # confirmation and a neutral structure with divergence is a valid setup
    allow_neutral_with_divergence: bool = False

    # Entry confirmation
    require_entry_candle_close: bool = False

    # Minimum volatility filter
    min_atr_pct: float = 0.15

    # Allow out-of-killzone trades with reduced confidence
    allow_off_session_trades: bool = False  # Off-session trades have very poor win rate
    off_session_confidence_penalty: float = 0.3  # Reduce confidence by 0.3 for off-session

    # Market Structure
    use_market_structure: bool = True
    ms_swing_order: int = 3        # Swing order for detecting HH/HL/LH/LL
    ms_lookback: int = 60          # Candles to look back for market structure

    # Displacement detection (Phase 2A)
    require_displacement: bool = True
    displacement_body_ratio: float = 0.6   # Body must be > 60% of candle range
    displacement_atr_ratio: float = 0.5    # Body must be > 0.5 * ATR
    displacement_lookforward: int = 5      # Candles after sweep to find displacement

    # Volume spike confirmation (Phase 2B)
    require_volume_confirmation: bool = True
    volume_spike_multiplier: float = 1.5   # Volume > 1.5x 20-candle average
    volume_lookback: int = 20

    # Asian session range (Phase 2C)
    use_asian_range: bool = True
    asian_range_start_hour: int = 0
    asian_range_end_hour: int = 8

    # PDH/PDL/PWH/PWL liquidity levels (Phase 2D)
    use_liquidity_levels: bool = True

    # FVG unfilled validation (Phase 2E)
    validate_fvg_unfilled: bool = True

    # Dynamic time exit (Phase 2G) — extended (Change 6)
    time_exit_candles: int = 48    # Default for 5m (4 hours, was 2h)

    # Trailing stop (Phase 2F) — relaxed (Change 11)
    trailing_swing_lookback: int = 5  # Was 3 — gives more room

    # Execution timeframe (for multi-TF support)
    execution_tf_minutes: int = 5

    @classmethod
    def for_timeframe(cls, resolution: str) -> "StrategyConfig":
        """Create a StrategyConfig with parameters scaled for the given timeframe."""
        presets = {
            "1m": dict(min_range_candles=30, max_range_candles=90,
                       divergence_lookback=30, time_exit_candles=240,
                       fvg_lookforward=40, retrace_lookforward=40,
                       max_scan_after_range=20, execution_tf_minutes=1),
            "5m": dict(min_range_candles=10, max_range_candles=30,
                       divergence_lookback=15, time_exit_candles=48,
                       fvg_lookforward=20, retrace_lookforward=20,
                       max_scan_after_range=25,       # expanded: 15 → 25 candles (125 min window)
                       execution_tf_minutes=5,
                       min_signal_confidence=0.50,
                       signal_dedup_hours=4.0,
                       adx_ranging_threshold=20.0,
                       max_atr_spike_ratio=2.0,
                       max_signals_per_range=2,       # allow 2nd entry from same range
                       allow_neutral_with_divergence=True),
            "15m": dict(min_range_candles=6, max_range_candles=20,
                        divergence_lookback=10, time_exit_candles=32,
                        fvg_lookforward=12, retrace_lookforward=12,
                        max_scan_after_range=18,      # expanded: 10 → 18 candles (270 min window)
                        execution_tf_minutes=15,
                        min_rr_ratio=2.5,
                        min_signal_confidence=0.55,
                        signal_dedup_hours=5.0,       # reduced: 8h → 5h (less restrictive)
                        adx_ranging_threshold=22.0,
                        max_atr_spike_ratio=1.8,
                        max_signals_per_range=2,
                        allow_neutral_with_divergence=True),
            "1h": dict(min_range_candles=4, max_range_candles=12,
                       divergence_lookback=8, time_exit_candles=16,
                       fvg_lookforward=8, retrace_lookforward=8,
                       max_scan_after_range=10,       # expanded: 6 → 10 candles
                       execution_tf_minutes=60,
                       strict_divergence=True,
                       min_rr_ratio=2.5,
                       adx_ranging_threshold=18.0,
                       max_atr_spike_ratio=0,         # disable on 1h — volatility expected
                       max_signals_per_range=1,       # keep 1h strict (fewer signals expected)
                       allow_neutral_with_divergence=True),
            "4h": dict(min_range_candles=3, max_range_candles=8,
                       divergence_lookback=6, time_exit_candles=8,
                       fvg_lookforward=6, retrace_lookforward=6,
                       max_scan_after_range=4, execution_tf_minutes=240,
                       strict_divergence=True),
        }
        params = presets.get(resolution, presets["5m"])
        return cls(**params)


# =====================================================================
# ICT Killzone / Session Detection
# =====================================================================

def get_session(dt, config: StrategyConfig) -> Optional[str]:
    """
    Determine which ICT session/killzone the given datetime falls in.
    
    Returns: "london", "new_york", "asian", or None (outside killzones)
    """
    if not hasattr(dt, 'hour'):
        return None
    
    hour = dt.hour
    
    if config.london_start_hour <= hour < config.london_end_hour:
        return "london"
    if config.ny_start_hour <= hour < config.ny_end_hour:
        return "new_york"
    if config.allow_asian_session and config.asian_start_hour <= hour < config.asian_end_hour:
        return "asian"
    
    return None


def is_in_killzone(dt, config: StrategyConfig) -> bool:
    """Check if the given datetime is within an ICT killzone."""
    return get_session(dt, config) is not None


# =====================================================================
# Market Structure Detection (BOS / CHoCH)
# =====================================================================

def detect_market_structure(highs: np.ndarray, lows: np.ndarray, 
                            end_idx: int, lookback: int = 60,
                            swing_order: int = 3) -> str:
    """
    Detect market structure bias via swing point analysis.
    
    Looks for:
    - Bullish structure: Higher Highs (HH) + Higher Lows (HL) → BOS upward
    - Bearish structure: Lower Highs (LH) + Lower Lows (LL) → BOS downward
    - CHoCH: A break in the existing structure (e.g., first LL after HH/HL sequence)
    
    Returns: "bullish", "bearish", or "neutral"
    """
    start_idx = max(0, end_idx - lookback)
    if end_idx - start_idx < swing_order * 4:
        return "neutral"
    
    # Find swing highs and lows
    swing_highs = []
    swing_lows = []
    
    for i in range(start_idx + swing_order, end_idx - swing_order):
        # Swing high
        is_sh = True
        for j in range(1, swing_order + 1):
            if highs[i] <= highs[i - j] or highs[i] <= highs[i + j]:
                is_sh = False
                break
        if is_sh:
            swing_highs.append((i, highs[i]))
        
        # Swing low
        is_sl = True
        for j in range(1, swing_order + 1):
            if lows[i] >= lows[i - j] or lows[i] >= lows[i + j]:
                is_sl = False
                break
        if is_sl:
            swing_lows.append((i, lows[i]))
    
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return "neutral"
    
    # Analyze the last 3-4 swing points for structure
    recent_highs = swing_highs[-3:]
    recent_lows = swing_lows[-3:]
    
    # Count HH/LH and HL/LL
    hh_count = 0
    lh_count = 0
    for i in range(1, len(recent_highs)):
        if recent_highs[i][1] > recent_highs[i-1][1]:
            hh_count += 1
        elif recent_highs[i][1] < recent_highs[i-1][1]:
            lh_count += 1
    
    hl_count = 0
    ll_count = 0
    for i in range(1, len(recent_lows)):
        if recent_lows[i][1] > recent_lows[i-1][1]:
            hl_count += 1
        elif recent_lows[i][1] < recent_lows[i-1][1]:
            ll_count += 1
    
    # Determine structure
    bullish_score = hh_count + hl_count
    bearish_score = lh_count + ll_count
    
    if bullish_score >= 2 and bullish_score > bearish_score:
        return "bullish"
    elif bearish_score >= 2 and bearish_score > bullish_score:
        return "bearish"
    
    return "neutral"


# =====================================================================
# Higher Timeframe Bias (Multi-TF from aggregated candles)
# =====================================================================

def get_htf_minutes(execution_tf_minutes: int) -> int:
    """
    Map execution timeframe to higher timeframe for bias analysis.

    5m  -> 1H  (60 min)
    15m -> 4H  (240 min)
    1H  -> 1D  (1440 min)
    4H  -> 1W  (10080 min)
    """
    mapping = {
        1: 15,      # 1m -> 15m
        5: 60,      # 5m -> 1H
        15: 240,    # 15m -> 4H
        60: 1440,   # 1H -> Daily
        240: 10080, # 4H -> Weekly
    }
    return mapping.get(execution_tf_minutes, 60)


def aggregate_to_htf(df: pd.DataFrame, tf_minutes: int = 60) -> pd.DataFrame:
    """
    Aggregate candles into a higher timeframe.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data with DatetimeIndex.
    tf_minutes : int
        Target timeframe in minutes (default: 60 = 1H).

    Returns
    -------
    pd.DataFrame
        Aggregated OHLCV data.
    """
    rule = f"{tf_minutes}min"
    htf = df.resample(rule).agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }).dropna()
    return htf


def _ema_bias_on_htf_df(htf_df: pd.DataFrame, config: StrategyConfig) -> str:
    """
    Internal: compute EMA-based bias on a pre-aggregated HTF DataFrame.

    Returns "bullish", "bearish", or "neutral".

    Logic (fast=EMA20, slow=EMA50 by default on the HTF candles):
      - Bullish: close > fast > slow  AND  fast EMA slope > 0 (still rising)
      - Bearish: close < fast < slow  AND  fast EMA slope < 0 (still falling)
      - Neutral: anything in between (including topping / bottoming out)
    """
    if len(htf_df) < 10:
        return "neutral"

    ema_fast = htf_df["Close"].ewm(span=min(config.htf_ema_fast, len(htf_df)), adjust=False).mean()
    ema_slow = htf_df["Close"].ewm(span=min(config.htf_ema_slow, len(htf_df)), adjust=False).mean()

    last_close = htf_df["Close"].iloc[-1]
    last_ef = ema_fast.iloc[-1]
    last_es = ema_slow.iloc[-1]

    # EMA slope over N HTF bars — detects momentum turning points before crossover
    n = min(config.htf_ema_slope_periods, len(ema_fast) - 1)
    slope = (ema_fast.iloc[-1] - ema_fast.iloc[-(n + 1)]) if n > 0 else 0.0

    use_slope = config.htf_ema_slope_confirm

    if last_close > last_ef and last_ef > last_es:
        if use_slope and slope <= 0:
            return "neutral"   # Trend topping — EMA fast already rolling over
        return "bullish"

    if last_close < last_ef and last_ef < last_es:
        if use_slope and slope >= 0:
            return "neutral"   # Trend bottoming — EMA fast still ticking up
        return "bearish"

    return "neutral"


def get_htf_bias_multi_tf(df: pd.DataFrame, i: int, config: StrategyConfig) -> str:
    """
    Determine HTF bias using aggregated HTF candles with responsive EMA20/50.

    5m  execution → 1H HTF  (EMA20 and EMA50 on 1H bars)
    15m execution → 4H HTF  (EMA20 and EMA50 on 4H bars)

    Logic:
      - Bullish: close > EMA20 > EMA50  AND  EMA20 slope is rising
      - Bearish: close < EMA20 < EMA50  AND  EMA20 slope is falling
      - Neutral: everything else (including trend reversals / topping / bottoming)

    Using EMA20/50 instead of EMA50/200:
      - EMA20 reacts in ~20 1H bars (~20h) vs EMA50 in ~50 1H bars (~2d)
      - Much faster at detecting tops/bottoms before the old 50/200 cross
      - Slope confirmation catches the roll-over even before the crossover
    """
    if i < 200:  # Need at least ~16h of 5m data for meaningful 1H EMAs
        return "neutral"

    sub_df = df.iloc[:i + 1]
    htf_tf = get_htf_minutes(config.execution_tf_minutes)
    try:
        htf_df = aggregate_to_htf(sub_df, tf_minutes=htf_tf)
    except Exception:
        return "neutral"

    if len(htf_df) < config.htf_ema_slow // 3:  # Need reasonable amount of HTF candles
        return "neutral"

    return _ema_bias_on_htf_df(htf_df, config)


def get_htf_bias(df: pd.DataFrame, i: int, config: StrategyConfig, 
                 ema50: np.ndarray = None, closes: np.ndarray = None,
                 highs: np.ndarray = None, lows: np.ndarray = None) -> str:
    """
    Determine the Higher Timeframe Bias based on the configured mode.
    
    IMPORTANT: Evaluates bias at candle index `i` (the breakout candle),
    NOT at the range start.
    
    Returns: "bullish", "bearish", or "neutral"
    """
    mode = config.htf_alignment_mode
    if mode == "none" or mode is None:
        return "neutral"
    
    if mode == "multi_tf":
        # Primary: use aggregated 1H structure
        htf_bias = get_htf_bias_multi_tf(df, i, config)
        
        # If multi-TF is neutral, check market structure as secondary
        if htf_bias == "neutral" and config.use_market_structure and highs is not None and lows is not None:
            htf_bias = detect_market_structure(
                highs, lows, i,
                lookback=config.ms_lookback,
                swing_order=config.ms_swing_order
            )
        
        return htf_bias
        
    elif mode == "ema_50":
        if ema50 is not None and closes is not None:
            return "bullish" if closes[i] > ema50[i] else "bearish"
        return "neutral"
        
    elif mode == "market_structure":
        if highs is not None and lows is not None:
            return detect_market_structure(
                highs, lows, i,
                lookback=config.ms_lookback,
                swing_order=config.ms_swing_order
            )
        if i < config.htf_lookback:
            return "neutral"
        current_high = df["High"].iloc[i]
        prev_period_high = df["High"].iloc[i - config.htf_lookback : i].max()
        current_low = df["Low"].iloc[i]
        prev_period_low = df["Low"].iloc[i - config.htf_lookback : i].min()
        if current_high > prev_period_high:
            return "bullish"
        elif current_low < prev_period_low:
            return "bearish"
        return "neutral"
        
    elif mode == "liquidity_draw":
        if i < config.pdl_pdh_lookback:
            return "neutral"
        pdh = df["High"].iloc[i - config.pdl_pdh_lookback : i].max()
        pdl = df["Low"].iloc[i - config.pdl_pdh_lookback : i].min()
        current_price = df["Close"].iloc[i]
        if abs(current_price - pdh) < abs(current_price - pdl):
            return "bullish"
        else:
            return "bearish"
    
    return "neutral"


# =====================================================================
# Liquidity Sweep Detection
# =====================================================================

def detect_liquidity_sweep(df: pd.DataFrame, i: int, lookback: int = 10) -> Optional[str]:
    """
    Check if the current candle forms a liquidity sweep.
    
    A sweep is when price wicks beyond a local high/low but closes back inside,
    showing rejection (wick at least 40% of total candle range on sweep side).
    """
    if i < lookback:
        return None
        
    prev_high = df["High"].iloc[i-lookback:i].max()
    prev_low = df["Low"].iloc[i-lookback:i].min()
    
    candle = df.iloc[i]
    candle_range = candle["High"] - candle["Low"]
    if candle_range <= 0:
        return None
    
    # Bearish sweep: Wicks above local max but closes below.
    # Require meaningful wick (rejection) above the previous high.
    if candle["High"] > prev_high and candle["Close"] < candle["High"]:
        upper_wick = candle["High"] - max(candle["Open"], candle["Close"])
        wick_ratio = upper_wick / candle_range
        # At least some rejection wick (30% of candle range)
        if wick_ratio >= 0.25:
            return "bearish"
    
    # Bullish sweep: Wicks below local min but closes above.
    if candle["Low"] < prev_low and candle["Close"] > candle["Low"]:
        lower_wick = min(candle["Open"], candle["Close"]) - candle["Low"]
        wick_ratio = lower_wick / candle_range
        if wick_ratio >= 0.25:
            return "bullish"
    
    return None


# =====================================================================
# Displacement Detection (Phase 2A)
# =====================================================================

def detect_displacement(df: pd.DataFrame, sweep_idx: int, direction: str,
                        atr: np.ndarray, lookforward: int = 5,
                        body_ratio_min: float = 0.6,
                        body_atr_min: float = 0.5) -> bool:
    """
    Check for a displacement candle after a liquidity sweep.

    A displacement is a strong momentum candle in the expected reversal direction:
    - Body > 60% of the candle's total range
    - Body > 0.5 * ATR (significant move)
    - Direction matches the expected reversal

    Parameters
    ----------
    df : pd.DataFrame
    sweep_idx : int - Index of the sweep candle
    direction : str - "bearish" (expect bearish displacement) or "bullish"
    atr : np.ndarray
    lookforward : int - How many candles after sweep to check
    body_ratio_min : float - Minimum body/range ratio
    body_atr_min : float - Minimum body/ATR ratio

    Returns
    -------
    bool - True if displacement confirmed
    """
    n = len(df)
    for j in range(sweep_idx + 1, min(sweep_idx + lookforward + 1, n)):
        o = df["Open"].iloc[j]
        c = df["Close"].iloc[j]
        h = df["High"].iloc[j]
        l = df["Low"].iloc[j]

        candle_range = h - l
        if candle_range <= 0:
            continue

        body = abs(c - o)
        body_ratio = body / candle_range

        # Check ATR threshold
        current_atr = atr[j] if j < len(atr) and not np.isnan(atr[j]) else 0
        if current_atr <= 0:
            continue

        body_atr_ratio = body / current_atr

        if body_ratio < body_ratio_min or body_atr_ratio < body_atr_min:
            continue

        # Check direction
        if direction == "bearish" and c < o:  # Bearish candle
            return True
        if direction == "bullish" and c > o:  # Bullish candle
            return True

    return False


# =====================================================================
# Volume Spike Confirmation (Phase 2B)
# =====================================================================

def check_volume_spike(volumes: np.ndarray, idx: int,
                       lookback: int = 20, multiplier: float = 1.5) -> bool:
    """
    Check if the candle at idx has above-average volume.
    Volume spike confirms institutional participation in the sweep.

    Returns True if volume[idx] > multiplier * mean(volume[idx-lookback:idx])
    Returns True if volume data is all zeros (skip check for missing data).
    """
    if idx < lookback:
        return True  # Not enough data, allow trade

    recent_vol = volumes[idx - lookback:idx]

    # If volume data is all zeros/nan, skip the check (soft filter)
    if np.all(recent_vol == 0) or np.all(np.isnan(recent_vol)):
        return True

    avg_vol = np.nanmean(recent_vol)
    if avg_vol <= 0:
        return True

    return volumes[idx] > multiplier * avg_vol


# =====================================================================
# PDH/PDL/PWH/PWL Liquidity Levels (Phase 2D)
# =====================================================================

@dataclass
class LiquidityLevels:
    """Key liquidity levels from higher timeframes."""
    pdh: float  # Previous day high
    pdl: float  # Previous day low
    pwh: float  # Previous week high
    pwl: float  # Previous week low


def calculate_liquidity_levels(df: pd.DataFrame, current_idx: int) -> Optional[LiquidityLevels]:
    """
    Calculate PDH/PDL/PWH/PWL relative to the current candle.

    Uses the datetime index to find the previous day's and previous week's
    high/low values. Returns None if insufficient data.
    """
    if not hasattr(df.index, 'date'):
        return None

    current_time = df.index[current_idx]
    current_date = current_time.date() if hasattr(current_time, 'date') else None
    if current_date is None:
        return None

    # Get all candles before today
    prev_day_mask = df.index[:current_idx + 1].date < current_date
    if not np.any(prev_day_mask):
        return None

    prev_day_df = df.iloc[:current_idx + 1][prev_day_mask]
    if len(prev_day_df) == 0:
        return None

    # Previous day = the last complete day before current_date
    prev_dates = pd.Series(prev_day_df.index.date).unique()
    if len(prev_dates) == 0:
        return None

    last_prev_date = prev_dates[-1]
    prev_day_candles = prev_day_df[prev_day_df.index.date == last_prev_date]

    pdh = prev_day_candles["High"].max() if len(prev_day_candles) > 0 else None
    pdl = prev_day_candles["Low"].min() if len(prev_day_candles) > 0 else None

    if pdh is None or pdl is None:
        return None

    # Previous week (last 5 trading days before today, excluding current day)
    if len(prev_dates) >= 2:
        week_dates = prev_dates[-5:] if len(prev_dates) >= 5 else prev_dates
        week_mask = np.isin(prev_day_df.index.date, week_dates)
        week_candles = prev_day_df[week_mask]
        pwh = week_candles["High"].max()
        pwl = week_candles["Low"].min()
    else:
        pwh = pdh
        pwl = pdl

    return LiquidityLevels(pdh=pdh, pdl=pdl, pwh=pwh, pwl=pwl)


# =====================================================================
# Fair Value Gap (FVG) Detection
# =====================================================================

def detect_fvg(df: pd.DataFrame, start_idx: int, end_idx: int, direction: str,
               validate_unfilled: bool = True) -> Optional[tuple]:
    """
    Scan for the first unfilled Fair Value Gap (FVG) in the given direction.

    Bearish FVG: Low of candle 1 > High of candle 3 (gap down)
    Bullish FVG: High of candle 1 < Low of candle 3 (gap up)

    When validate_unfilled=True, checks that subsequent candles have NOT
    already traded through (mitigated) the FVG before returning it.

    Returns: (fvg_top, fvg_bottom, fvg_idx) or None
    """
    n = len(df)
    for j in range(start_idx, end_idx):
        if j < 2:
            continue

        if direction == "bearish":
            c1_low = df["Low"].iloc[j-2]
            c3_high = df["High"].iloc[j]
            c2_open = df["Open"].iloc[j-1]
            c2_close = df["Close"].iloc[j-1]

            if c1_low > c3_high and c2_close < c2_open:
                fvg_top = c1_low
                fvg_bottom = c3_high

                # Check if FVG has been mitigated (price rallied back above fvg_top)
                if validate_unfilled:
                    mitigated = False
                    for m in range(j + 1, min(j + 20, n)):
                        if df["High"].iloc[m] > fvg_top:
                            mitigated = True
                            break
                    if mitigated:
                        continue  # Skip this FVG, find next one

                return (fvg_top, fvg_bottom, j)

        elif direction == "bullish":
            c1_high = df["High"].iloc[j-2]
            c3_low = df["Low"].iloc[j]
            c2_open = df["Open"].iloc[j-1]
            c2_close = df["Close"].iloc[j-1]

            if c1_high < c3_low and c2_close > c2_open:
                fvg_top = c3_low
                fvg_bottom = c1_high

                # Check if FVG has been mitigated (price dropped below fvg_bottom)
                if validate_unfilled:
                    mitigated = False
                    for m in range(j + 1, min(j + 20, n)):
                        if df["Low"].iloc[m] < fvg_bottom:
                            mitigated = True
                            break
                    if mitigated:
                        continue  # Skip this FVG, find next one

                return (fvg_top, fvg_bottom, j)

    return None


# =====================================================================
# Order Block Detection (ICT)
# =====================================================================

def detect_order_block(df: pd.DataFrame, sweep_idx: int, direction: str,
                       lookback: int = 10, atr_series=None,
                       min_body_atr_ratio: float = 0.3) -> Optional[Tuple[float, float, int]]:
    """
    Detect the most recent Order Block before the displacement move.

    Bearish OB: The last bullish (up-close) candle before a strong bearish move.
                Use the high/low of that candle as the OB zone.

    Bullish OB: The last bearish (down-close) candle before a strong bullish move.
                Use the high/low of that candle as the OB zone.

    Filters doji/micro-body candles using min_body_atr_ratio × ATR when atr_series provided.

    Returns: (ob_top, ob_bottom, ob_idx) or None
    """
    search_start = max(0, sweep_idx - lookback)

    if direction == "bearish":
        # Find the last meaningful bullish candle before the sweep
        for j in range(sweep_idx - 1, search_start - 1, -1):
            candle_open = df["Open"].iloc[j]
            candle_close = df["Close"].iloc[j]
            if candle_close > candle_open:  # Bullish candle
                # Body size filter — skip doji/micro candles
                if atr_series is not None and min_body_atr_ratio > 0:
                    body = abs(candle_close - candle_open)
                    if j < len(atr_series) and not np.isnan(atr_series.iloc[j]):
                        if body < atr_series.iloc[j] * min_body_atr_ratio:
                            continue  # Too small — not a meaningful OB
                ob_top = df["High"].iloc[j]
                ob_bottom = df["Low"].iloc[j]
                return (ob_top, ob_bottom, j)

    elif direction == "bullish":
        # Find the last meaningful bearish candle before the sweep
        for j in range(sweep_idx - 1, search_start - 1, -1):
            candle_open = df["Open"].iloc[j]
            candle_close = df["Close"].iloc[j]
            if candle_close < candle_open:  # Bearish candle
                # Body size filter — skip doji/micro candles
                if atr_series is not None and min_body_atr_ratio > 0:
                    body = abs(candle_close - candle_open)
                    if j < len(atr_series) and not np.isnan(atr_series.iloc[j]):
                        if body < atr_series.iloc[j] * min_body_atr_ratio:
                            continue  # Too small — not a meaningful OB
                ob_top = df["High"].iloc[j]
                ob_bottom = df["Low"].iloc[j]
                return (ob_top, ob_bottom, j)

    return None


# =====================================================================
# BPR (Balanced Price Range) Detection
# =====================================================================

def detect_bpr(df: pd.DataFrame, start_idx: int, end_idx: int, direction: str) -> Optional[tuple]:
    """
    Scan for a Balanced Price Range (BPR).
    A BPR occurs when an upward FVG and a downward FVG overlap in a specific price zone
    within a short time frame, indicating aggressive, algorithmic repricing.
    
    Returns: (bpr_top, bpr_bottom, bpr_idx) or None
    """
    for j in range(start_idx, end_idx):
        if j < 4:
            continue
            
        # We need a massive opposing move that leaves an FVG, immediately followed/preceded
        # by an FVG in the opposite direction.
        
        if direction == "bearish":
            # We are looking to go SHORT. We want to see an old BULLISH FVG being
            # aggressively traded through by a new BEARISH FVG.
            
            # 1. Find recent bullish FVG (gap up)
            bullish_fvg_top = None
            bullish_fvg_bottom = None
            for k in range(j-10, j):
                if k < 2: continue
                c1_high = df["High"].iloc[k-2]
                c3_low = df["Low"].iloc[k]
                if c1_high < c3_low:
                    bullish_fvg_top = c3_low
                    bullish_fvg_bottom = c1_high
                    break
                    
            if bullish_fvg_top is not None:
                # 2. Check if current formation is a bearish FVG that overlaps it
                c1_low = df["Low"].iloc[j-2]
                c3_high = df["High"].iloc[j]
                if c1_low > c3_high:
                    bearish_fvg_top = c1_low
                    bearish_fvg_bottom = c3_high
                    
                    # 3. Check for overlap
                    overlap_top = min(bullish_fvg_top, bearish_fvg_top)
                    overlap_bottom = max(bullish_fvg_bottom, bearish_fvg_bottom)
                    
                    if overlap_top > overlap_bottom:
                        return (overlap_top, overlap_bottom, j)
                        
        elif direction == "bullish":
            # We are looking to go LONG. We want to see an old BEARISH FVG being
            # aggressively traded through by a new BULLISH FVG.
            
            # 1. Find recent bearish FVG (gap down)
            bearish_fvg_top = None
            bearish_fvg_bottom = None
            for k in range(j-10, j):
                if k < 2: continue
                c1_low = df["Low"].iloc[k-2]
                c3_high = df["High"].iloc[k]
                if c1_low > c3_high:
                    bearish_fvg_top = c1_low
                    bearish_fvg_bottom = c3_high
                    break
                    
            if bearish_fvg_top is not None:
                # 2. Check if current formation is a bullish FVG that overlaps it
                c1_high = df["High"].iloc[j-2]
                c3_low = df["Low"].iloc[j]
                if c1_high < c3_low:
                    bullish_fvg_top = c3_low
                    bullish_fvg_bottom = c1_high
                    
                    # 3. Check for overlap
                    overlap_top = min(bullish_fvg_top, bearish_fvg_top)
                    overlap_bottom = max(bullish_fvg_bottom, bearish_fvg_bottom)
                    
                    if overlap_top > overlap_bottom:
                        return (overlap_top, overlap_bottom, j)
                        
    return None


# =====================================================================
# Signal Confidence Scoring
# =====================================================================

def calculate_signal_confidence(
    has_divergence: bool,
    has_volume_spike: bool,
    session: str,
    entry_type: str,
    rr_ratio: float,
    has_displacement: bool = True,
    is_ranging_market: bool = False,
    ranging_penalty: float = 0.15,
) -> float:
    """
    Calculate a 0.0-1.0 confidence score for a trade signal.

    Weights:
      - Divergence present:   +0.25
      - Volume spike:         +0.15
      - Killzone session:     +0.20 (London/NY), +0.05 (Asian/off-session)
      - Entry type quality:   +0.20 (BPR best), +0.15 (FVG), +0.10 (OB)
      - R:R ratio bonus:      +0.10 if RR >= 2.5, +0.05 if RR >= 2.0
      - Displacement:         +0.10
    Penalties:
      - OB without divergence: -0.20 (historical ~20% WR, must require divergence)
      - Ranging market (ADX<threshold) + non-FVG + no divergence: -ranging_penalty

    Base score starts at 0.0, max possible ~1.0.
    """
    score = 0.0

    # Divergence confirmation
    if has_divergence:
        score += 0.25

    # Volume spike
    if has_volume_spike:
        score += 0.15

    # Session quality
    if session in ("london", "new_york"):
        score += 0.20
    elif session == "asian":
        score += 0.05
    # off-session gets 0

    # Entry type quality (BPR > FVG > OB > close)
    entry_scores = {"bpr": 0.20, "fvg": 0.15, "order_block": 0.10, "close": 0.05}
    score += entry_scores.get(entry_type, 0.05)

    # R:R bonus
    if rr_ratio >= 2.5:
        score += 0.10
    elif rr_ratio >= 2.0:
        score += 0.05

    # Displacement
    if has_displacement:
        score += 0.10

    # OB entries without divergence have ~20% WR in backtests — apply hard penalty
    # so that only OB+divergence combos pass the 0.50 gate
    if entry_type == "order_block" and not has_divergence:
        score -= 0.20

    # Ranging market: non-FVG entries without divergence have lower WR in choppy conditions
    if is_ranging_market and entry_type != "fvg" and not has_divergence:
        score -= ranging_penalty

    return min(score, 1.0)


# =====================================================================
# Main Signal Generator
# =====================================================================

def generate_signals(
    df: pd.DataFrame,
    config: StrategyConfig = None,
    correlated_df: pd.DataFrame = None,
) -> List[TradeSignal]:
    """
    Execute the full AMD pipeline and produce trade signals.

    Pipeline (ICT/SMC Flow)
    -----------------------
    1. Calculate RSI, ATR, EMAs on close prices.
    2. Detect consolidation ranges (Accumulation phase).
    3. For each range, scan subsequent candles for a breakout (Manipulation/Sweep).
    4. On sweep, check:
       a. Session filter (must be in London/NY killzone)
       b. HTF alignment (trade with the higher timeframe trend)
       c. SMT Divergence (optional)
       d. RSI divergence for confirmation
       e. BPR, FVG, or Order Block for precise entry
    5. If confirmed → generate trade signal with SL / TP (Distribution target).

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data with DatetimeIndex.
    config : StrategyConfig, optional
        Strategy parameters.
    correlated_df : pd.DataFrame, optional
        Correlated asset data for SMT Divergence.

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

    # ADX for ranging-market detection (computed once, used per-signal)
    adx_series = _compute_adx(df, config.adx_period) if config.adx_ranging_threshold > 0 else None

    # Pre-calculate 50 EMA if using that mode
    ema50 = None
    if config.htf_alignment_mode == "ema_50":
        ema50 = df["Close"].ewm(span=50, adjust=False).mean().values

    highs = df["High"].values
    lows = df["Low"].values
    closes = df["Close"].values
    volumes = df["Volume"].values if "Volume" in df.columns else np.zeros(len(df))
    n = len(df)

    # Step 2: Detect consolidation ranges (technical + Asian session)
    ranges = detect_ranges(
        df,
        min_candles=config.min_range_candles,
        max_candles=config.max_range_candles,
        range_threshold_pct=config.range_threshold_pct,
    )

    # Merge Asian session ranges if enabled
    if config.use_asian_range:
        asian_ranges = detect_asian_ranges(
            df,
            asian_start_hour=config.asian_range_start_hour,
            asian_end_hour=config.asian_range_end_hour,
        )
        # Combine and sort by start index, Asian ranges get priority
        all_ranges = asian_ranges + ranges
        all_ranges.sort(key=lambda r: r.start_idx)
        # De-duplicate overlapping ranges (keep first encountered)
        deduped = []
        last_end = -1
        for r in all_ranges:
            if r.start_idx > last_end:
                deduped.append(r)
                last_end = r.end_idx
        ranges = deduped

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
        signals_from_range = 0

        for i in range(scan_start, scan_end):
            if signals_from_range >= config.max_signals_per_range:
                break

            # === SESSION FILTER ===
            if config.use_session_filter:
                candle_time = df.index[i]
                session = get_session(candle_time, config)
                if session is None:
                    if config.allow_off_session_trades:
                        session = "off_session"  # Allow but penalize confidence
                    else:
                        continue  # Outside killzones — skip
            else:
                session = "unfiltered"

            # --- Breakout ABOVE range high (potential SHORT setup) ---
            if config.breakout_mode == "atr" and i < len(atr) and not np.isnan(atr[i]):
                breakout_level = rng.range_high + (atr[i] * config.breakout_atr_multiplier)
            else:
                breakout_level = rng.range_high * (1 + config.breakout_pct / 100.0)
            if highs[i] > breakout_level:
                sweep = detect_liquidity_sweep(df, i, lookback=10)
                if sweep == "bearish":
                    sweep_idx = i
                    if sweep_idx in seen_sweep_indices:
                        continue

                    # Displacement check (hard filter)
                    if config.require_displacement:
                        if not detect_displacement(
                            df, i, "bearish", atr,
                            lookforward=config.displacement_lookforward,
                            body_ratio_min=config.displacement_body_ratio,
                            body_atr_min=config.displacement_atr_ratio,
                        ):
                            continue

                    # Volume spike check (soft — tracked for confidence, not hard filter)
                    has_volume_spike = check_volume_spike(
                        volumes, i,
                        lookback=config.volume_lookback,
                        multiplier=config.volume_spike_multiplier,
                    )

                    # SMT Divergence Check
                    smt_div = None
                    if config.use_smt_divergence and correlated_df is not None:
                        smt_div = detect_smt_divergence(
                            df, correlated_df, i, "bearish", lookback=config.smt_lookback
                        )
                        if smt_div is None:
                            continue  # Strict SMT required but not found

                    div = detect_bearish_divergence(
                        highs, rsi, atr, i,
                        lookback=config.divergence_lookback,
                        swing_order=config.swing_order,
                        atr_multiplier=config.divergence_atr_multiplier,
                        min_swings=config.min_divergence_swings,
                    )

                    if config.strict_divergence and div is None:
                        continue

                    if div is not None and div.swings > config.max_divergence_swings:
                        continue

                    # HTF Bias — check at the BREAKOUT candle (not range start)
                    bias = get_htf_bias(
                        df, i, config, ema50=ema50, closes=closes,
                        highs=highs, lows=lows
                    )

                    if bias == "bullish":
                        continue  # Don't short against bullish bias
                    if config.require_htf_alignment and bias != "bearish":
                        # Allow neutral bias if divergence is confirmed (strong ICT signal)
                        if not (config.allow_neutral_with_divergence and div is not None):
                            continue  # Require bearish alignment for shorts

                    # Entry via BPR, FVG, Order Block, or Close
                    manipulation_high = highs[i]
                    original_sl = max(manipulation_high, rng.range_high)
                    if config.sl_buffer_mode == "atr" and i < len(atr) and not np.isnan(atr[i]):
                        sl = original_sl + (atr[i] * config.sl_buffer_atr_mult)
                    else:
                        sl = original_sl * (1 + config.sl_buffer_pct / 100.0)
                    entry_type = "close"
                    entry_idx = -1
                    entry = 0.0
                    
                    if config.require_fvg or config.use_bpr_priority:
                        fvg_search_end = min(i + config.fvg_lookforward, n)
                        
                        # Priority 1: BPR
                        bpr_result = detect_bpr(df, i+2, fvg_search_end, "bearish") if config.use_bpr_priority else None
                        
                        # Priority 2: FVG (with unfilled validation)
                        fvg_result = detect_fvg(
                            df, i+2, fvg_search_end, "bearish",
                            validate_unfilled=config.validate_fvg_unfilled,
                        ) if bpr_result is None and config.require_fvg else None

                        if bpr_result is not None:
                            overlap_top, overlap_bottom, bpr_idx = bpr_result
                            entry_type = "bpr"
                            retrace_search_start = bpr_idx + 1
                            retrace_search_end = min(retrace_search_start + config.retrace_lookforward, n)
                            for k in range(retrace_search_start, retrace_search_end):
                                if config.require_entry_candle_close:
                                    if closes[k] >= overlap_bottom:
                                        entry_idx = k
                                        entry = overlap_bottom
                                        break
                                else:
                                    if highs[k] >= overlap_bottom:
                                        entry_idx = k
                                        entry = overlap_bottom
                                        break

                        elif fvg_result is not None:
                            fvg_top, fvg_bottom, fvg_idx = fvg_result
                            entry_type = "fvg"
                            retrace_search_start = fvg_idx + 1
                            retrace_search_end = min(retrace_search_start + config.retrace_lookforward, n)
                            for k in range(retrace_search_start, retrace_search_end):
                                if config.require_entry_candle_close:
                                    if closes[k] >= fvg_bottom:
                                        entry_idx = k
                                        entry = fvg_bottom
                                        break
                                else:
                                    if highs[k] >= fvg_bottom:
                                        entry_idx = k
                                        entry = fvg_bottom
                                        break

                        # Priority 3: Order Block Fallback
                        if entry_idx == -1 and config.use_order_blocks:
                            ob_result = detect_order_block(df, i, "bearish", lookback=10,
                                                           atr_series=atr_series,
                                                           min_body_atr_ratio=config.ob_min_body_atr_ratio)
                            if ob_result is not None:
                                ob_top, ob_bottom, ob_idx = ob_result
                                entry_type = "order_block"
                                for k in range(i + 1, min(i + config.retrace_lookforward, n)):
                                    if config.require_entry_candle_close:
                                        if closes[k] >= ob_bottom:
                                            entry_idx = k
                                            entry = ob_bottom
                                            break
                                    else:
                                        if highs[k] >= ob_bottom:
                                            entry_idx = k
                                            entry = ob_bottom
                                            break

                        if entry_idx == -1:
                            continue  # No entry condition met

                        if sl <= entry:
                            sl = entry * 1.0015

                        # TP target: use liquidity levels if available and better
                        tp = rng.range_low
                        if config.use_liquidity_levels:
                            levels = calculate_liquidity_levels(df, entry_idx)
                            if levels is not None and levels.pdl < tp:
                                tp = levels.pdl  # PDL offers further target for shorts
                    else:
                        # No FVG required — enter at close
                        entry = closes[i]
                        sl = original_sl
                        tp = rng.range_low
                        entry_idx = i
                        if config.use_liquidity_levels:
                            levels = calculate_liquidity_levels(df, entry_idx)
                            if levels is not None and levels.pdl < tp:
                                tp = levels.pdl

                    if entry_idx in seen_entry_indices:
                        continue

                    # Min ATR filter
                    if config.min_atr_pct > 0 and i < len(atr) and not np.isnan(atr[i]):
                        if (atr[i] / closes[i]) * 100.0 < config.min_atr_pct:
                            continue

                    # Min SL distance: at least 0.25% of price to avoid oversized positions/fees
                    risk = abs(entry - sl)
                    if risk / entry < 0.0025:
                        continue  # SL too tight — position would be too large

                    # R:R filter — skip if below minimum
                    reward = entry - tp
                    if risk > 0:
                        rr = reward / risk
                        if rr < config.min_rr_ratio:
                            continue  # Skip low R:R setups
                        rr = min(rr, config.max_rr_ratio)
                        tp = entry - (rr * risk)
                    else:
                        rr = 0

                    # ATR volatility spike guard — skip if market is in expansion (candle-1 SL risk)
                    if config.max_atr_spike_ratio > 0 and entry_idx < len(atr_series):
                        short_atr = atr_series.iloc[max(0, entry_idx-4):entry_idx+1].mean()
                        base_atr = atr_series.iloc[max(0, entry_idx-50):entry_idx].mean()
                        if base_atr > 0 and (short_atr / base_atr) > config.max_atr_spike_ratio:
                            continue  # Volatility expansion — high candle-1 SL risk

                    # Ranging-market flag for confidence scoring
                    is_ranging = (
                        adx_series is not None
                        and entry_idx < len(adx_series)
                        and not np.isnan(adx_series.iloc[entry_idx])
                        and adx_series.iloc[entry_idx] < config.adx_ranging_threshold
                    )

                    # Calculate signal confidence
                    conf = calculate_signal_confidence(
                        has_divergence=div is not None,
                        has_volume_spike=has_volume_spike,
                        session=session if isinstance(session, str) else "off_session",
                        entry_type=entry_type,
                        rr_ratio=rr,
                        is_ranging_market=is_ranging,
                        ranging_penalty=config.adx_range_extra_penalty,
                    )
                    if session == "off_session":
                        conf = max(0.1, conf - config.off_session_confidence_penalty)

                    # Minimum confidence gate — filters OBs without divergence etc.
                    if conf < config.min_signal_confidence:
                        continue

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
                        smt_divergence=smt_div,
                        htf_bias=bias,
                        session=session if isinstance(session, str) else "unknown",
                        entry_type=entry_type,
                        confidence=conf,
                    ))
                    seen_sweep_indices.add(sweep_idx)
                    seen_entry_indices.add(entry_idx)
                    last_trade_range_idx = rng.start_idx
                    signals_from_range += 1
                    continue  # Check next candle (up to max_signals_per_range)

            # --- Breakdown BELOW range low (potential LONG setup) ---
            if config.breakout_mode == "atr" and i < len(atr) and not np.isnan(atr[i]):
                breakdown_level = rng.range_low - (atr[i] * config.breakout_atr_multiplier)
            else:
                breakdown_level = rng.range_low * (1 - config.breakout_pct / 100.0)
            if lows[i] < breakdown_level:
                sweep = detect_liquidity_sweep(df, i, lookback=10)
                if sweep == "bullish":
                    sweep_idx = i
                    if sweep_idx in seen_sweep_indices:
                        continue

                    # Displacement check (hard filter)
                    if config.require_displacement:
                        if not detect_displacement(
                            df, i, "bullish", atr,
                            lookforward=config.displacement_lookforward,
                            body_ratio_min=config.displacement_body_ratio,
                            body_atr_min=config.displacement_atr_ratio,
                        ):
                            continue

                    # Volume spike check (soft — tracked for confidence, not hard filter)
                    has_volume_spike = check_volume_spike(
                        volumes, i,
                        lookback=config.volume_lookback,
                        multiplier=config.volume_spike_multiplier,
                    )

                    # SMT Divergence Check
                    smt_div = None
                    if config.use_smt_divergence and correlated_df is not None:
                        smt_div = detect_smt_divergence(
                            df, correlated_df, i, "bullish", lookback=config.smt_lookback
                        )
                        if smt_div is None:
                            continue  # Strict SMT required but not found

                    div = detect_bullish_divergence(
                        lows, rsi, atr, i,
                        lookback=config.divergence_lookback,
                        swing_order=config.swing_order,
                        atr_multiplier=config.divergence_atr_multiplier,
                        min_swings=config.min_divergence_swings,
                    )
                    
                    if config.strict_divergence and div is None:
                        continue
                        
                    if div is not None and div.swings > config.max_divergence_swings:
                        continue

                    # HTF Bias — check at the BREAKOUT candle (not range start)
                    bias = get_htf_bias(
                        df, i, config, ema50=ema50, closes=closes,
                        highs=highs, lows=lows
                    )
                        
                    if bias == "bearish":
                        continue  # Don't go long against bearish bias
                    if config.require_htf_alignment and bias != "bullish":
                        # Allow neutral bias if divergence is confirmed (strong ICT signal)
                        if not (config.allow_neutral_with_divergence and div is not None):
                            continue  # Require bullish alignment for longs

                    # Entry via BPR, FVG, Order Block, or Close
                    manipulation_low = lows[i]
                    original_sl = min(manipulation_low, rng.range_low)
                    if config.sl_buffer_mode == "atr" and i < len(atr) and not np.isnan(atr[i]):
                        sl = original_sl - (atr[i] * config.sl_buffer_atr_mult)
                    else:
                        sl = original_sl * (1 - config.sl_buffer_pct / 100.0)
                    entry_type = "close"
                    entry_idx = -1
                    entry = 0.0
                    
                    if config.require_fvg or config.use_bpr_priority:
                        fvg_search_end = min(i + config.fvg_lookforward, n)
                        
                        # Priority 1: BPR
                        bpr_result = detect_bpr(df, i+2, fvg_search_end, "bullish") if config.use_bpr_priority else None
                        
                        # Priority 2: FVG (with unfilled validation)
                        fvg_result = detect_fvg(
                            df, i+2, fvg_search_end, "bullish",
                            validate_unfilled=config.validate_fvg_unfilled,
                        ) if bpr_result is None and config.require_fvg else None

                        if bpr_result is not None:
                            overlap_top, overlap_bottom, bpr_idx = bpr_result
                            entry_type = "bpr"
                            retrace_search_start = bpr_idx + 1
                            retrace_search_end = min(retrace_search_start + config.retrace_lookforward, n)
                            for k in range(retrace_search_start, retrace_search_end):
                                if config.require_entry_candle_close:
                                    if closes[k] <= overlap_top:
                                        entry_idx = k
                                        entry = overlap_top
                                        break
                                else:
                                    if lows[k] <= overlap_top:
                                        entry_idx = k
                                        entry = overlap_top
                                        break

                        elif fvg_result is not None:
                            fvg_top, fvg_bottom, fvg_idx = fvg_result
                            entry_type = "fvg"
                            retrace_search_start = fvg_idx + 1
                            retrace_search_end = min(retrace_search_start + config.retrace_lookforward, n)
                            for k in range(retrace_search_start, retrace_search_end):
                                if config.require_entry_candle_close:
                                    if closes[k] <= fvg_top:
                                        entry_idx = k
                                        entry = fvg_top
                                        break
                                else:
                                    if lows[k] <= fvg_top:
                                        entry_idx = k
                                        entry = fvg_top
                                        break

                        # Priority 3: Order Block Fallback
                        if entry_idx == -1 and config.use_order_blocks:
                            ob_result = detect_order_block(df, i, "bullish", lookback=10,
                                                           atr_series=atr_series,
                                                           min_body_atr_ratio=config.ob_min_body_atr_ratio)
                            if ob_result is not None:
                                ob_top, ob_bottom, ob_idx = ob_result
                                entry_type = "order_block"
                                for k in range(i + 1, min(i + config.retrace_lookforward, n)):
                                    if config.require_entry_candle_close:
                                        if closes[k] <= ob_top:
                                            entry_idx = k
                                            entry = ob_top
                                            break
                                    else:
                                        if lows[k] <= ob_top:
                                            entry_idx = k
                                            entry = ob_top
                                            break

                        if entry_idx == -1:
                            continue  # No entry condition met

                        if sl >= entry:
                            sl = entry * 0.9985

                        # TP target: use liquidity levels if available and better
                        tp = rng.range_high
                        if config.use_liquidity_levels:
                            levels = calculate_liquidity_levels(df, entry_idx)
                            if levels is not None and levels.pdh > tp:
                                tp = levels.pdh  # PDH offers further target for longs
                    else:
                        entry = closes[i]
                        sl = original_sl
                        tp = rng.range_high
                        entry_idx = i
                        if config.use_liquidity_levels:
                            levels = calculate_liquidity_levels(df, entry_idx)
                            if levels is not None and levels.pdh > tp:
                                tp = levels.pdh

                    if entry_idx in seen_entry_indices:
                        continue

                    # Min ATR filter
                    if config.min_atr_pct > 0 and i < len(atr) and not np.isnan(atr[i]):
                        if (atr[i] / closes[i]) * 100.0 < config.min_atr_pct:
                            continue

                    # Min SL distance: at least 0.25% of price to avoid oversized positions/fees
                    risk = abs(entry - sl)
                    if risk / entry < 0.0025:
                        continue  # SL too tight — position would be too large

                    # R:R filter — skip if below minimum
                    reward = tp - entry
                    if risk > 0:
                        rr = reward / risk
                        if rr < config.min_rr_ratio:
                            continue  # Skip low R:R setups
                        rr = min(rr, config.max_rr_ratio)
                        tp = entry + (rr * risk)
                    else:
                        rr = 0

                    # ATR volatility spike guard — skip if market is in expansion (candle-1 SL risk)
                    if config.max_atr_spike_ratio > 0 and entry_idx < len(atr_series):
                        short_atr = atr_series.iloc[max(0, entry_idx-4):entry_idx+1].mean()
                        base_atr = atr_series.iloc[max(0, entry_idx-50):entry_idx].mean()
                        if base_atr > 0 and (short_atr / base_atr) > config.max_atr_spike_ratio:
                            continue  # Volatility expansion — high candle-1 SL risk

                    # Ranging-market flag for confidence scoring
                    is_ranging = (
                        adx_series is not None
                        and entry_idx < len(adx_series)
                        and not np.isnan(adx_series.iloc[entry_idx])
                        and adx_series.iloc[entry_idx] < config.adx_ranging_threshold
                    )

                    # Calculate signal confidence
                    conf = calculate_signal_confidence(
                        has_divergence=div is not None,
                        has_volume_spike=has_volume_spike,
                        session=session if isinstance(session, str) else "off_session",
                        entry_type=entry_type,
                        rr_ratio=rr,
                        is_ranging_market=is_ranging,
                        ranging_penalty=config.adx_range_extra_penalty,
                    )
                    if session == "off_session":
                        conf = max(0.1, conf - config.off_session_confidence_penalty)

                    # Minimum confidence gate — filters OBs without divergence etc.
                    if conf < config.min_signal_confidence:
                        continue

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
                        smt_divergence=smt_div,
                        htf_bias=bias,
                        session=session if isinstance(session, str) else "unknown",
                        entry_type=entry_type,
                        confidence=conf,
                    ))
                    seen_sweep_indices.add(sweep_idx)
                    seen_entry_indices.add(entry_idx)
                    last_trade_range_idx = rng.start_idx
                    signals_from_range += 1
                    # Continue scanning for a 2nd entry from the same range setup

    # Post-generation: deduplicate signals in same price zone and direction
    if config.signal_dedup_hours > 0 and len(signals) > 1:
        signals = _dedup_signals(signals, config.signal_dedup_hours, config.signal_dedup_pct)

    return signals


def _compute_adx(df: pd.DataFrame, period: int) -> pd.Series:
    """Compute ADX (Average Directional Index) as a pd.Series aligned to df.index."""
    from ta.trend import ADXIndicator
    adx_ind = ADXIndicator(df["High"], df["Low"], df["Close"], window=period, fillna=False)
    return adx_ind.adx()


def _dedup_signals(
    signals: list,
    dedup_hours: float,
    dedup_pct: float,
) -> list:
    """
    Remove duplicate signals that are in the same direction and price zone
    within dedup_hours of each other.

    Keep the higher-confidence signal when two are in conflict.
    This prevents the 'cluster' failure mode where multiple signals
    fire from the same price structure in a single session.
    """
    if not signals:
        return signals

    # Sort by confidence (desc) so we keep the best signal when deduplying
    sorted_sigs = sorted(signals, key=lambda s: s.confidence, reverse=True)
    kept = []
    dedup_seconds = dedup_hours * 3600.0

    for sig in sorted_sigs:
        conflict = False
        for kept_sig in kept:
            # Same direction?
            if sig.direction != kept_sig.direction:
                continue
            # Within time window?
            try:
                time_diff = abs((sig.entry_time - kept_sig.entry_time).total_seconds())
            except Exception:
                continue
            if time_diff > dedup_seconds:
                continue
            # Within price proximity?
            price_diff_pct = abs(sig.entry_price - kept_sig.entry_price) / kept_sig.entry_price * 100.0
            if price_diff_pct <= dedup_pct:
                conflict = True
                break
        if not conflict:
            kept.append(sig)

    # Restore chronological order
    kept.sort(key=lambda s: s.entry_time)
    return kept


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
    # Cap position size: max notional = 20x account balance (prevents runaway sizing on tiny SL)
    pos_size = risk_amount / risk_per_unit
    max_notional = account_balance * 20.0
    max_pos_size = max_notional / entry_price if entry_price > 0 else pos_size
    return min(pos_size, max_pos_size)
