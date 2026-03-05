"""
Backtest Engine — Simulates the AMD strategy on historical data.

Walks forward through candles, executing trade signals and tracking
position lifecycle from entry to SL/TP exit.
"""

from dataclasses import dataclass, field
from typing import List, Optional
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


class TradeOutcome(Enum):
    WIN = "WIN"
    LOSS = "LOSS"
    OPEN = "OPEN"


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
    pnl_dollar: float         # Dollar P&L (pnl × position_size)
    holding_candles: int      # Number of candles held

    def __repr__(self) -> str:
        return (
            f"Trade({self.signal.direction.value} "
            f"entry={self.signal.entry_price:.2f} -> "
            f"exit={self.exit_price:.2f} | "
            f"{self.outcome.value} | "
            f"PnL=${self.pnl_dollar:+.2f} ({self.pnl_pct:+.2f}%) | "
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


def run_backtest(
    df: pd.DataFrame,
    config: StrategyConfig = None,
    initial_balance: float = 10000.0,
    signals: Optional[List[TradeSignal]] = None,
) -> BacktestResult:
    """
    Run a full backtest simulation.

    For each signal:
    1. Calculate position size based on risk %.
    2. Walk forward candle-by-candle from entry.
    3. Check if SL or TP is hit (SL checked first — worst-case assumption).
    4. Record the trade outcome and update equity.

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

    Returns
    -------
    BacktestResult
        Complete results with equity curve and trade log.
    """
    if config is None:
        config = StrategyConfig()

    if signals is None:
        signals = generate_signals(df, config)

    highs = df["High"].values
    lows = df["Low"].values
    n = len(df)

    balance = initial_balance
    equity_curve = [balance]
    completed: List[CompletedTrade] = []

    for sig in signals:
        # Position sizing
        pos_size = calculate_position_size(
            account_balance=balance,
            risk_pct=config.risk_per_trade_pct,
            entry_price=sig.entry_price,
            stop_loss=sig.stop_loss,
        )

        if pos_size <= 0:
            continue

        # Walk forward from entry
        outcome = TradeOutcome.OPEN
        exit_price = sig.entry_price
        exit_idx = sig.entry_idx
        exit_time = sig.entry_time

        for j in range(sig.entry_idx + 1, n):
            if sig.direction == Direction.SHORT:
                # SL hit? (price goes up to SL)
                if highs[j] >= sig.stop_loss:
                    outcome = TradeOutcome.LOSS
                    exit_price = sig.stop_loss
                    exit_idx = j
                    exit_time = df.index[j]
                    break
                # TP hit? (price drops to TP)
                if lows[j] <= sig.take_profit:
                    outcome = TradeOutcome.WIN
                    exit_price = sig.take_profit
                    exit_idx = j
                    exit_time = df.index[j]
                    break

            elif sig.direction == Direction.LONG:
                # SL hit? (price drops to SL)
                if lows[j] <= sig.stop_loss:
                    outcome = TradeOutcome.LOSS
                    exit_price = sig.stop_loss
                    exit_idx = j
                    exit_time = df.index[j]
                    break
                # TP hit? (price rises to TP)
                if highs[j] >= sig.take_profit:
                    outcome = TradeOutcome.WIN
                    exit_price = sig.take_profit
                    exit_idx = j
                    exit_time = df.index[j]
                    break

        # Calculate P&L
        if sig.direction == Direction.LONG:
            pnl_per_unit = exit_price - sig.entry_price
        else:
            pnl_per_unit = sig.entry_price - exit_price

        pnl_pct = (pnl_per_unit / sig.entry_price) * 100.0
        pnl_dollar = pnl_per_unit * pos_size

        # Update balance
        balance += pnl_dollar
        equity_curve.append(balance)

        completed.append(CompletedTrade(
            signal=sig,
            outcome=outcome,
            exit_price=exit_price,
            exit_idx=exit_idx,
            exit_time=exit_time,
            pnl=pnl_per_unit,
            pnl_pct=pnl_pct,
            position_size=pos_size,
            pnl_dollar=pnl_dollar,
            holding_candles=exit_idx - sig.entry_idx,
        ))

    return BacktestResult(
        trades=completed,
        equity_curve=equity_curve,
        initial_balance=initial_balance,
    )
