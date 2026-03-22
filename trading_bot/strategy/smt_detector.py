"""
Smart Money Tool (SMT) Divergence Detection.

SMT Divergence compares highly correlated assets (e.g., BTC and ETH).
If the primary asset (BTC) sweeps a high (Higher High) but the correlated
asset (ETH) fails to make a Higher High, this is a structural divergence.
It indicates weakness in the primary asset's move and strongly confirms
a manipulative liquidity sweep.
"""

from dataclasses import dataclass
from typing import Optional
import pandas as pd


@dataclass
class SMTResult:
    """Result of an SMT Divergence check."""
    divergence_type: str  # "bearish" (HH vs LH) or "bullish" (LL vs HL)
    primary_price: float
    correlated_price: float
    is_confirmed: bool

    def __repr__(self):
        return (
            f"SMTDivergence({self.divergence_type}, "
            f"Primary={self.primary_price:,.2f}, Correlated={self.correlated_price:,.2f})"
        )


def detect_smt_divergence(
    primary_df: pd.DataFrame,
    correlated_df: pd.DataFrame,
    current_idx: int,
    direction: str,
    lookback: int = 20
) -> Optional[SMTResult]:
    """
    Check for SMT divergence between a primary and correlated asset at a specific index.
    
    Parameters
    ----------
    primary_df : pd.DataFrame
        OHLCV data for the primary asset (e.g., BTC).
    correlated_df : pd.DataFrame
        OHLCV data for the correlated asset (e.g., ETH), ideally aligned by datetime index.
    current_idx : int
        The row index in primary_df where we are checking for a sweep/divergence.
    direction : str
        "bearish" (checking highs/HH) or "bullish" (checking lows/LL).
    lookback : int
        Number of candles to look back for the previous local extreme.
        
    Returns
    -------
    SMTResult or None
        Returns SMTResult if divergence is found, else None.
    """
    if current_idx < lookback:
        return None
        
    start_idx = current_idx - lookback
    primary_time = primary_df.index[current_idx]
    
    # Try to find the exact same time in the correlated dataframe
    try:
        corr_idx = correlated_df.index.get_loc(primary_time)
        if isinstance(corr_idx, slice) or isinstance(corr_idx, pd.Series):
             return None # Ambiguous index
    except KeyError:
        # If the exact timestamp doesn't exist, we can't reliably check SMT
        return None
        
    if corr_idx < lookback:
        return None

    if direction == "bearish":
        # Primary asset must be making a local Higher High
        primary_current_high = primary_df["High"].iloc[current_idx]
        primary_prev_high = primary_df["High"].iloc[start_idx:current_idx].max()
        
        if primary_current_high > primary_prev_high:
            # Correlated asset should make a Lower High (or equal high) to diverge
            corr_current_high = correlated_df["High"].iloc[corr_idx]
            corr_prev_high = correlated_df["High"].iloc[corr_idx - lookback : corr_idx].max()
            
            if corr_current_high <= corr_prev_high:
                return SMTResult(
                    divergence_type="bearish",
                    primary_price=primary_current_high,
                    correlated_price=corr_current_high,
                    is_confirmed=True
                )
                
    elif direction == "bullish":
        # Primary asset must be making a local Lower Low
        primary_current_low = primary_df["Low"].iloc[current_idx]
        primary_prev_low = primary_df["Low"].iloc[start_idx:current_idx].min()
        
        if primary_current_low < primary_prev_low:
             # Correlated asset should make a Higher Low (or equal low) to diverge
            corr_current_low = correlated_df["Low"].iloc[corr_idx]
            corr_prev_low = correlated_df["Low"].iloc[corr_idx - lookback : corr_idx].min()
            
            if corr_current_low >= corr_prev_low:
                return SMTResult(
                    divergence_type="bullish",
                    primary_price=primary_current_low,
                    correlated_price=corr_current_low,
                    is_confirmed=True
                )
                
    return None
