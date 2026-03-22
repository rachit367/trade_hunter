"""
Backtest Engine — Simulates the AMD strategy on historical data.

Walks forward through candles, executing trade signals and tracking
position lifecycle from entry to SL/TP exit.

Includes:
  - Realistic fee model (maker/taker fees + slippage)
  - Same-candle event priority (SL checked first, worst-case)
  - Break-even trailing, partial profits, and swing-based trailing stop
  - Dynamic time exit scaled by timeframe
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from enum import Enum

import pandas as pd
import numpy as np

from trading_bot.strategy.amd_strategy import (
    generate_signals,
    calculate_position_size,
    TradeSignal,
    Direction,
    StrategyConfig,
)
from trading_bot.strategy.risk_manager import RiskManager, RiskConfig


class TradeOutcome(Enum):
    WIN = "WIN"
    LOSS = "LOSS"
    OPEN = "OPEN"
    TIME_EXIT = "TIME_EXIT"


@dataclass
class BacktestConfig:
    """Backtest simulation parameters for realistic cost modeling."""
    taker_fee_pct: float = 0.075    # 0.075% per fill (Delta Exchange taker)
    maker_fee_pct: float = 0.05     # 0.05% per fill (Delta Exchange maker)
    slippage_pct: float = 0.02      # 0.02% per execution (market order fill)
    use_maker_for_limit: bool = True  # FVG/OB entries are limit orders


@dataclass
class CompletedTrade:
    """A resolved trade with entry and exit details."""
    signal: TradeSignal
    outcome: TradeOutcome
    exit_price: float
    exit_idx: int
    exit_time: object
    pnl: float               # Absolute P&L per unit
    pnl_pct: float            # P&L as % of entry
    position_size: float      # Units traded
    pnl_dollar: float         # Dollar P&L (after fees)
    holding_candles: int      # Number of candles held
    entry_fee: float = 0.0
    exit_fee: float = 0.0
    slippage_cost: float = 0.0
    gross_pnl_dollar: float = 0.0   # P&L before fees
    net_pnl_dollar: float = 0.0     # P&L after fees

    def __repr__(self) -> str:
        return (
            f"Trade({self.signal.direction.value} "
            f"entry={self.signal.entry_price:.2f} -> "
            f"exit={self.exit_price:.2f} | "
            f"{self.outcome.value} | "
            f"PnL=${self.net_pnl_dollar:+.2f} ({self.pnl_pct:+.2f}%) | "
            f"fees=${self.entry_fee + self.exit_fee:.2f} | "
            f"held={self.holding_candles} candles)"
        )


@dataclass
class BacktestResult:
    """Full backtest results with equity curve and trade log."""
    trades: List[CompletedTrade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    initial_balance: float = 10000.0

    @property
    def total_trades(self) -> int:
        return len(self.trades)

    @property
    def closed_trades(self) -> List[CompletedTrade]:
        return [t for t in self.trades if t.outcome != TradeOutcome.OPEN]

    @property
    def wins(self) -> int:
        return sum(1 for t in self.trades if t.outcome == TradeOutcome.WIN)

    @property
    def losses(self) -> int:
        return sum(1 for t in self.trades if t.outcome == TradeOutcome.LOSS)

    @property
    def open_count(self) -> int:
        return sum(1 for t in self.trades if t.outcome == TradeOutcome.OPEN)

    @property
    def final_balance(self) -> float:
        return self.equity_curve[-1] if self.equity_curve else self.initial_balance

    @property
    def total_fees(self) -> float:
        return sum(t.entry_fee + t.exit_fee for t in self.trades)

    @property
    def total_slippage(self) -> float:
        return sum(t.slippage_cost for t in self.trades)


def _calc_entry_fee(entry_price: float, pos_size: float, entry_type: str,
                    bt_config: BacktestConfig) -> Tuple[float, float]:
    """
    Calculate entry fee and slippage cost.

    Returns (fee_dollar, slippage_dollar).
    FVG/OB/BPR entries use maker fee (limit order).
    Close entries use taker fee (market order).
    """
    notional = entry_price * pos_size

    if bt_config.use_maker_for_limit and entry_type in ("fvg", "bpr", "order_block"):
        fee = notional * (bt_config.maker_fee_pct / 100.0)
        slippage = 0.0  # Limit orders have no slippage
    else:
        fee = notional * (bt_config.taker_fee_pct / 100.0)
        slippage = notional * (bt_config.slippage_pct / 100.0)

    return fee, slippage


def _calc_exit_fee(exit_price: float, pos_size: float, is_tp: bool,
                   bt_config: BacktestConfig) -> Tuple[float, float]:
    """
    Calculate exit fee and slippage cost.

    TP exits use maker fee (limit order). SL/time exits use taker (market).
    """
    notional = exit_price * pos_size

    if is_tp and bt_config.use_maker_for_limit:
        fee = notional * (bt_config.maker_fee_pct / 100.0)
        slippage = 0.0
    else:
        fee = notional * (bt_config.taker_fee_pct / 100.0)
        slippage = notional * (bt_config.slippage_pct / 100.0)

    return fee, slippage


def run_backtest(
    df: pd.DataFrame,
    config: StrategyConfig = None,
    initial_balance: float = 10000.0,
    signals: Optional[List[TradeSignal]] = None,
    bt_config: Optional[BacktestConfig] = None,
    risk_config: Optional[RiskConfig] = None,
) -> BacktestResult:
    """
    Run a full backtest simulation with realistic costs.

    For each signal:
    1. Calculate position size based on risk % (accounting for fees).
    2. Walk forward candle-by-candle from entry.
    3. On each candle, check SL FIRST (worst-case assumption).
       Only process break-even/partial if SL was NOT hit on that candle.
    4. After break-even, trail stop behind recent swing points.
    5. Record the trade outcome and update equity.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data.
    config : StrategyConfig, optional
        Strategy parameters.
    initial_balance : float
        Starting account balance.
    signals : List[TradeSignal], optional
        Pre-generated signals. If None, will generate them.
    bt_config : BacktestConfig, optional
        Fee/slippage configuration. Defaults to Delta Exchange rates.

    Returns
    -------
    BacktestResult
        Complete results with equity curve and trade log.
    """
    if config is None:
        config = StrategyConfig()
    if bt_config is None:
        bt_config = BacktestConfig()

    if signals is None:
        signals = generate_signals(df, config)

    # Initialize risk manager
    risk_mgr = RiskManager(
        config=risk_config or RiskConfig(),
        initial_balance=initial_balance,
    )

    highs = df["High"].values
    lows = df["Low"].values
    closes = df["Close"].values
    opens = df["Open"].values
    n = len(df)

    # Pre-compute ATR for trailing stop buffer
    from ta.volatility import AverageTrueRange
    atr_indicator = AverageTrueRange(
        high=df["High"], low=df["Low"], close=df["Close"], window=14
    )
    atr_values = atr_indicator.average_true_range().values

    balance = initial_balance
    equity_curve = [balance]
    completed: List[CompletedTrade] = []

    # Config-driven trade management parameters
    time_exit_candles = getattr(config, 'time_exit_candles', 48)
    trailing_lookback = getattr(config, 'trailing_swing_lookback', 5)
    trailing_atr_buffer = 0.3  # Increased from 0.2 for more breathing room
    use_confidence_sizing = getattr(config, 'use_confidence_sizing', True)
    min_conf_mult = getattr(config, 'min_confidence_multiplier', 0.5)
    max_conf_mult = getattr(config, 'max_confidence_multiplier', 1.2)

    for sig in signals:
        # Risk manager gate check
        allowed, reason = risk_mgr.can_trade(sig.entry_time)
        if not allowed:
            continue

        risk_mgr.on_trade_open()

        # Estimate round-trip fees for position sizing
        est_fee_pct = (bt_config.taker_fee_pct + bt_config.slippage_pct) * 2 / 100.0
        adjusted_balance = balance * (1.0 - est_fee_pct)

        # Dynamic position sizing based on signal confidence
        risk_pct = config.risk_per_trade_pct
        if use_confidence_sizing:
            # Map confidence (0.0-1.0) to multiplier range
            conf_mult = min_conf_mult + (sig.confidence * (max_conf_mult - min_conf_mult))
            risk_pct *= conf_mult

        original_pos_size = calculate_position_size(
            account_balance=adjusted_balance,
            risk_pct=risk_pct,
            entry_price=sig.entry_price,
            stop_loss=sig.stop_loss,
        )

        if original_pos_size <= 0:
            continue

        # Deduct entry fee from balance
        entry_fee, entry_slippage = _calc_entry_fee(
            sig.entry_price, original_pos_size, sig.entry_type, bt_config
        )
        balance -= (entry_fee + entry_slippage)

        pos_size = original_pos_size  # Working size (reduced by partials)

        # Walk forward from entry
        outcome = TradeOutcome.OPEN
        exit_price = sig.entry_price
        exit_idx = sig.entry_idx
        exit_time = sig.entry_time

        # Trade Management State
        break_even_moved = False
        partial_1_taken = False  # 25% at 1.5R
        partial_2_taken = False  # 35% at 2.0R
        initial_sl = sig.stop_loss
        current_sl = initial_sl
        accumulated_partial_pnl = 0.0
        total_exit_fee = 0.0
        total_exit_slippage = 0.0
        peak_r_reached = 0.0  # Track highest R multiple reached

        # R:R targets
        risk_per_unit = abs(sig.entry_price - initial_sl)
        if risk_per_unit <= 0:
            risk_mgr.on_trade_close(0, sig.entry_time)
            continue

        if sig.direction == Direction.SHORT:
            target_1_5r = sig.entry_price - (1.5 * risk_per_unit)
            target_2_0r = sig.entry_price - (2.0 * risk_per_unit)
        else:
            target_1_5r = sig.entry_price + (1.5 * risk_per_unit)
            target_2_0r = sig.entry_price + (2.0 * risk_per_unit)

        for j in range(sig.entry_idx + 1, n):
            candles_held = j - sig.entry_idx

            # --- Smart Time-Based Exit ---
            if candles_held >= time_exit_candles:
                # Near-breakeven optimization: if within 0.3R of entry, exit at entry
                if sig.direction == Direction.LONG:
                    unrealized_r = (closes[j] - sig.entry_price) / risk_per_unit
                else:
                    unrealized_r = (sig.entry_price - closes[j]) / risk_per_unit

                if break_even_moved and abs(unrealized_r) < 0.3:
                    # Close at entry to avoid fee bleed on flat trades
                    exit_price = sig.entry_price
                else:
                    exit_price = closes[j]

                outcome = TradeOutcome.TIME_EXIT
                exit_idx = j
                exit_time = df.index[j]
                fee, slip = _calc_exit_fee(exit_price, pos_size, False, bt_config)
                total_exit_fee += fee
                total_exit_slippage += slip
                break

            if sig.direction == Direction.SHORT:
                # Track peak favorable excursion
                current_r = (sig.entry_price - lows[j]) / risk_per_unit
                peak_r_reached = max(peak_r_reached, current_r)

                # === SL CHECK FIRST (worst-case assumption) ===
                if highs[j] >= current_sl:
                    outcome = TradeOutcome.LOSS if current_sl == initial_sl else TradeOutcome.WIN
                    exit_price = current_sl
                    exit_idx = j
                    exit_time = df.index[j]
                    fee, slip = _calc_exit_fee(exit_price, pos_size, False, bt_config)
                    total_exit_fee += fee
                    total_exit_slippage += slip
                    break

                # === TP CHECK (remaining position) ===
                if lows[j] <= sig.take_profit:
                    outcome = TradeOutcome.WIN
                    exit_price = sig.take_profit
                    exit_idx = j
                    exit_time = df.index[j]
                    fee, slip = _calc_exit_fee(exit_price, pos_size, True, bt_config)
                    total_exit_fee += fee
                    total_exit_slippage += slip
                    break

                # === STATE CHANGES ===
                # Tiered Break-Even: at 1.5R, move SL to entry + 0.1R (not exact entry)
                if not break_even_moved and lows[j] <= target_1_5r:
                    current_sl = sig.entry_price - (0.1 * risk_per_unit)  # Give 0.1R cushion
                    break_even_moved = True

                # Partial 1: Take 25% at 1.5R
                if not partial_1_taken and lows[j] <= target_1_5r:
                    partial_1_taken = True
                    partial_size = pos_size * 0.25
                    partial_pnl = (sig.entry_price - target_1_5r) * partial_size
                    fee, slip = _calc_exit_fee(target_1_5r, partial_size, True, bt_config)
                    total_exit_fee += fee
                    total_exit_slippage += slip
                    accumulated_partial_pnl += partial_pnl - fee - slip
                    balance += (partial_pnl - fee - slip)
                    pos_size *= 0.75

                # Partial 2: Take 35% at 2.0R, tighten SL to exact entry
                if not partial_2_taken and lows[j] <= target_2_0r:
                    partial_2_taken = True
                    # Tighten SL to exact entry after 2R hit
                    current_sl = sig.entry_price
                    partial_size = pos_size * (0.35 / 0.75)  # 35% of original
                    partial_pnl = (sig.entry_price - target_2_0r) * partial_size
                    fee, slip = _calc_exit_fee(target_2_0r, partial_size, True, bt_config)
                    total_exit_fee += fee
                    total_exit_slippage += slip
                    accumulated_partial_pnl += partial_pnl - fee - slip
                    balance += (partial_pnl - fee - slip)
                    pos_size -= partial_size  # Remaining ~40%

                # Trailing Stop (after break-even, trail behind recent highs + ATR buffer)
                if break_even_moved:
                    trail_start = max(sig.entry_idx + 1, j - trailing_lookback)
                    recent_high = max(highs[trail_start:j + 1])
                    atr_buf = atr_values[j] * trailing_atr_buffer if j < len(atr_values) and not np.isnan(atr_values[j]) else 0
                    trail_level = recent_high + atr_buf
                    if trail_level < current_sl:
                        current_sl = trail_level

            elif sig.direction == Direction.LONG:
                # Track peak favorable excursion
                current_r = (highs[j] - sig.entry_price) / risk_per_unit
                peak_r_reached = max(peak_r_reached, current_r)

                # === SL CHECK FIRST (worst-case assumption) ===
                if lows[j] <= current_sl:
                    outcome = TradeOutcome.LOSS if current_sl == initial_sl else TradeOutcome.WIN
                    exit_price = current_sl
                    exit_idx = j
                    exit_time = df.index[j]
                    fee, slip = _calc_exit_fee(exit_price, pos_size, False, bt_config)
                    total_exit_fee += fee
                    total_exit_slippage += slip
                    break

                # === TP CHECK ===
                if highs[j] >= sig.take_profit:
                    outcome = TradeOutcome.WIN
                    exit_price = sig.take_profit
                    exit_idx = j
                    exit_time = df.index[j]
                    fee, slip = _calc_exit_fee(exit_price, pos_size, True, bt_config)
                    total_exit_fee += fee
                    total_exit_slippage += slip
                    break

                # === STATE CHANGES ===
                # Tiered Break-Even: at 1.5R, move SL to entry + 0.1R
                if not break_even_moved and highs[j] >= target_1_5r:
                    current_sl = sig.entry_price + (0.1 * risk_per_unit)  # Give 0.1R cushion
                    break_even_moved = True

                # Partial 1: Take 25% at 1.5R
                if not partial_1_taken and highs[j] >= target_1_5r:
                    partial_1_taken = True
                    partial_size = pos_size * 0.25
                    partial_pnl = (target_1_5r - sig.entry_price) * partial_size
                    fee, slip = _calc_exit_fee(target_1_5r, partial_size, True, bt_config)
                    total_exit_fee += fee
                    total_exit_slippage += slip
                    accumulated_partial_pnl += partial_pnl - fee - slip
                    balance += (partial_pnl - fee - slip)
                    pos_size *= 0.75

                # Partial 2: Take 35% at 2.0R, tighten SL to exact entry
                if not partial_2_taken and highs[j] >= target_2_0r:
                    partial_2_taken = True
                    current_sl = sig.entry_price  # Tighten to exact entry
                    partial_size = pos_size * (0.35 / 0.75)
                    partial_pnl = (target_2_0r - sig.entry_price) * partial_size
                    fee, slip = _calc_exit_fee(target_2_0r, partial_size, True, bt_config)
                    total_exit_fee += fee
                    total_exit_slippage += slip
                    accumulated_partial_pnl += partial_pnl - fee - slip
                    balance += (partial_pnl - fee - slip)
                    pos_size -= partial_size

                # Trailing Stop (after break-even, trail behind recent lows - ATR buffer)
                if break_even_moved:
                    trail_start = max(sig.entry_idx + 1, j - trailing_lookback)
                    recent_low = min(lows[trail_start:j + 1])
                    atr_buf = atr_values[j] * trailing_atr_buffer if j < len(atr_values) and not np.isnan(atr_values[j]) else 0
                    trail_level = recent_low - atr_buf
                    if trail_level > current_sl:
                        current_sl = trail_level

        # Calculate P&L for REMAINING position only (partials already added)
        if sig.direction == Direction.LONG:
            remaining_pnl_per_unit = exit_price - sig.entry_price
        else:
            remaining_pnl_per_unit = sig.entry_price - exit_price

        remaining_pnl_dollar = remaining_pnl_per_unit * pos_size
        gross_pnl_dollar = accumulated_partial_pnl + remaining_pnl_dollar + (entry_fee + entry_slippage) + total_exit_fee + total_exit_slippage

        # Net P&L = partials + remaining
        net_pnl_dollar = accumulated_partial_pnl + remaining_pnl_dollar

        # P&L % based on total trade (using original position size for accuracy)
        pnl_pct = (net_pnl_dollar / (sig.entry_price * original_pos_size)) * 100.0 if original_pos_size > 0 else 0.0

        # Update balance with remaining position P&L
        balance += remaining_pnl_dollar
        equity_curve.append(balance)

        # Update risk manager
        risk_mgr.on_trade_close(net_pnl_dollar, exit_time)
        risk_mgr.balance = balance

        completed.append(CompletedTrade(
            signal=sig,
            outcome=outcome,
            exit_price=exit_price,
            exit_idx=exit_idx,
            exit_time=exit_time,
            pnl=remaining_pnl_per_unit,
            pnl_pct=pnl_pct,
            position_size=original_pos_size,
            pnl_dollar=net_pnl_dollar,
            holding_candles=exit_idx - sig.entry_idx,
            entry_fee=entry_fee + entry_slippage,
            exit_fee=total_exit_fee + total_exit_slippage,
            slippage_cost=entry_slippage + total_exit_slippage,
            gross_pnl_dollar=gross_pnl_dollar,
            net_pnl_dollar=net_pnl_dollar,
        ))

    return BacktestResult(
        trades=completed,
        equity_curve=equity_curve,
        initial_balance=initial_balance,
    )
