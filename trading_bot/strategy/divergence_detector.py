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
    swings: int              # Number of consecutive swings holding the divergence


def detect_bearish_divergence(
    highs: np.ndarray,
    rsi: np.ndarray,
    atr: np.ndarray,
    current_idx: int,
    lookback: int = 10,
    swing_order: int = 2,
    min_rsi_diff: float = 0.0,
    atr_multiplier: float = 0.7,
    min_swings: int = 2,
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
    min_rsi_diff : float
        Minimum absolute difference in RSI to be considered a strong divergence (Deprecated in multi-swing, defaults 0.0).
    atr_multiplier : float
        Initial price move must exceed atr * multiplier.
    min_swings : int
        Minimum number of consecutive divergence swings to consider valid.

    Returns
    -------
    DivergenceResult or None
        Divergence details if found, else None.
    """
    if current_idx < lookback + swing_order:
        return None

    current_rsi = rsi[current_idx]
    current_high = highs[current_idx]
    current_atr = atr[current_idx]

    if np.isnan(current_rsi) or np.isnan(current_atr):
        return None

    # Check if current_idx correctly forms a swing high
    left_start = max(0, current_idx - swing_order)
    right_end = min(len(highs), current_idx + swing_order + 1)
    
    # Must have enough historical candles to form a clear swing
    if current_idx < left_start + swing_order:
        return None
        
    is_swing = True
    for j in range(left_start, right_end):
        if j != current_idx and highs[j] >= current_high:
            is_swing = False
            break
            
    if not is_swing:
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
    best_divergence = None
    
    for prior_idx in reversed(swing_high_indices):  # Most recent first
        prior_high = highs[prior_idx]
        prior_rsi = rsi[prior_idx]

        if np.isnan(prior_rsi):
            continue

        # Bearish divergence: price HH + RSI LH
        if current_high > prior_high and current_rsi < prior_rsi:
            
            # Check initial price move vs ATR
            price_move = current_high - prior_high
            if price_move < (current_atr * atr_multiplier):
                continue
                
            # If we optionally enforce RSI diff
            if abs(prior_rsi - current_rsi) < min_rsi_diff:
                continue

            # We found at least a 2-point divergence. 
            # Now trace further back to see how many swings hold this pattern.
            swings_count = 2 
            
            # The reference points for the next loop backward
            ref_high = prior_high
            ref_rsi = prior_rsi
            
            # Continue checking older swings in the list
            older_swing_indices = [idx for idx in reversed(swing_high_indices) if idx < prior_idx]
            
            for older_idx in older_swing_indices:
                older_high = highs[older_idx]
                older_rsi = rsi[older_idx]
                
                if np.isnan(older_rsi):
                    continue
                    
                # To extend the bearish divergence, older high must be lower, and older RSI must be higher
                if ref_high > older_high and ref_rsi < older_rsi:
                    swings_count += 1
                    ref_high = older_high
                    ref_rsi = older_rsi
                else:
                    break   # Sequence broken
            
            if swings_count >= min_swings:
                # We overwrite with the strongest divergence we find natively
                if best_divergence is None or swings_count > best_divergence.swings:
                    best_divergence = DivergenceResult(
                        divergence_type="bearish",
                        current_idx=current_idx,
                        prior_idx=prior_idx,
                        current_price=current_high,
                        prior_price=prior_high,
                        current_rsi=current_rsi,
                        prior_rsi=prior_rsi,
                        swings=swings_count,
                    )

    return best_divergence


def detect_bullish_divergence(
    lows: np.ndarray,
    rsi: np.ndarray,
    atr: np.ndarray,
    current_idx: int,
    lookback: int = 10,
    swing_order: int = 2,
    min_rsi_diff: float = 0.0,
    atr_multiplier: float = 0.7,
    min_swings: int = 3,
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
    min_rsi_diff : float
        Minimum absolute difference in RSI to be considered a strong divergence.
    atr_multiplier : float
        Initial price move must exceed atr * multiplier.
    min_swings : int
        Minimum number of consecutive divergence swings to consider valid.

    Returns
    -------
    DivergenceResult or None
        Divergence details if found, else None.
    """
    if current_idx < lookback + swing_order:
        return None

    current_rsi = rsi[current_idx]
    current_low = lows[current_idx]
    current_atr = atr[current_idx]

    if np.isnan(current_rsi) or np.isnan(current_atr):
        return None

    # Check if current_idx correctly forms a swing low
    left_start = max(0, current_idx - swing_order)
    right_end = min(len(lows), current_idx + swing_order + 1)
    
    if current_idx < left_start + swing_order:
        return None
        
    is_swing = True
    for j in range(left_start, right_end):
        if j != current_idx and lows[j] <= current_low:
            is_swing = False
            break
            
    if not is_swing:
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

    best_divergence = None
    
    for prior_idx in reversed(swing_low_indices):
        prior_low = lows[prior_idx]
        prior_rsi = rsi[prior_idx]

        if np.isnan(prior_rsi):
            continue

        # Bullish divergence: price LL + RSI HL
        if current_low < prior_low and current_rsi > prior_rsi:
            
            # Check price move vs ATR
            price_move = prior_low - current_low
            if price_move < (current_atr * atr_multiplier):
                continue
                
            if abs(current_rsi - prior_rsi) < min_rsi_diff:
                continue

            # We found at least a 2-point divergence. 
            # Now trace further back to see how many swings hold this pattern.
            swings_count = 2 
            
            ref_low = prior_low
            ref_rsi = prior_rsi
            
            older_swing_indices = [idx for idx in reversed(swing_low_indices) if idx < prior_idx]
            
            for older_idx in older_swing_indices:
                older_low = lows[older_idx]
                older_rsi = rsi[older_idx]
                
                if np.isnan(older_rsi):
                    continue
                    
                # To extend bullish divergence, older low must be higher, older RSI must be lower
                if ref_low < older_low and ref_rsi > older_rsi:
                    swings_count += 1
                    ref_low = older_low
                    ref_rsi = older_rsi
                else:
                    break   # Sequence broken
            
            if swings_count >= min_swings:
                if best_divergence is None or swings_count > best_divergence.swings:
                    best_divergence = DivergenceResult(
                        divergence_type="bullish",
                        current_idx=current_idx,
                        prior_idx=prior_idx,
                        current_price=current_low,
                        prior_price=prior_low,
                        current_rsi=current_rsi,
                        prior_rsi=prior_rsi,
                        swings=swings_count,
                    )

    return best_divergence
