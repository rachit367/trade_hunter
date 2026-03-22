"""
Performance Metrics — Compute trading performance statistics.

Calculates:
  - Win rate
  - Profit factor
  - Sharpe ratio (annualized)
  - Max drawdown
  - Equity curve analysis
  - Average win / loss
  - Expectancy
"""

from dataclasses import dataclass
from typing import List

import numpy as np

from trading_bot.backtest.engine import BacktestResult, TradeOutcome


@dataclass
class PerformanceMetrics:
    """Complete performance statistics."""
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    open_trades: int = 0
    win_rate: float = 0.0

    total_pnl: float = 0.0
    avg_pnl: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    best_trade: float = 0.0
    worst_trade: float = 0.0
    expectancy: float = 0.0

    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    total_return_pct: float = 0.0

    avg_holding_candles: float = 0.0

    # Fee & cost tracking
    total_fees: float = 0.0
    total_slippage: float = 0.0
    gross_pnl: float = 0.0
    net_pnl: float = 0.0

    # Streak tracking
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0

    # Calmar ratio (annualized return / max drawdown)
    calmar_ratio: float = 0.0

    # Time exits
    time_exits: int = 0

    def summary(self) -> str:
        """Formatted performance report."""
        lines = [
            "",
            "+" + "=" * 55 + "+",
            "|" + "  BACKTEST PERFORMANCE REPORT".center(55) + "|",
            "+" + "=" * 55 + "+",
            f"|  {'Total Trades':<25} {self.total_trades:>25}  |",
            f"|  {'Wins':<25} {self.wins:>25}  |",
            f"|  {'Losses':<25} {self.losses:>25}  |",
            f"|  {'Still Open':<25} {self.open_trades:>25}  |",
            f"|  {'Win Rate':<25} {self.win_rate:>24.1f}%  |",
            "+" + "-" * 55 + "+",
            f"|  {'Total P&L':<25} ${self.total_pnl:>23.2f}  |",
            f"|  {'Total Return':<25} {self.total_return_pct:>24.2f}%  |",
            f"|  {'Average P&L / Trade':<25} ${self.avg_pnl:>23.2f}  |",
            f"|  {'Average Win':<25} ${self.avg_win:>23.2f}  |",
            f"|  {'Average Loss':<25} ${self.avg_loss:>23.2f}  |",
            f"|  {'Best Trade':<25} ${self.best_trade:>23.2f}  |",
            f"|  {'Worst Trade':<25} ${self.worst_trade:>23.2f}  |",
            f"|  {'Expectancy':<25} ${self.expectancy:>23.2f}  |",
            "+" + "-" * 55 + "+",
            f"|  {'Profit Factor':<25} {self.profit_factor:>25.2f}  |",
            f"|  {'Sharpe Ratio':<25} {self.sharpe_ratio:>25.2f}  |",
            f"|  {'Max Drawdown':<25} ${self.max_drawdown:>23.2f}  |",
            f"|  {'Max Drawdown %':<25} {self.max_drawdown_pct:>24.2f}%  |",
            f"|  {'Calmar Ratio':<25} {self.calmar_ratio:>25.2f}  |",
            "+" + "-" * 55 + "+",
            f"|  {'Gross P&L':<25} ${self.gross_pnl:>23.2f}  |",
            f"|  {'Net P&L':<25} ${self.net_pnl:>23.2f}  |",
            f"|  {'Total Fees':<25} ${self.total_fees:>23.2f}  |",
            f"|  {'Total Slippage':<25} ${self.total_slippage:>23.2f}  |",
            "+" + "-" * 55 + "+",
            f"|  {'Avg Holding (candles)':<25} {self.avg_holding_candles:>25.1f}  |",
            f"|  {'Time Exits':<25} {self.time_exits:>25}  |",
            f"|  {'Max Consec. Wins':<25} {self.max_consecutive_wins:>25}  |",
            f"|  {'Max Consec. Losses':<25} {self.max_consecutive_losses:>25}  |",
            "+" + "=" * 55 + "+",
            "",
        ]
        return "\n".join(lines)


def calculate_metrics(result: BacktestResult) -> PerformanceMetrics:
    """
    Compute comprehensive performance metrics from a BacktestResult.

    Parameters
    ----------
    result : BacktestResult
        Output from the backtesting engine.

    Returns
    -------
    PerformanceMetrics
        All computed metrics.
    """
    metrics = PerformanceMetrics()
    trades = result.trades

    if not trades:
        return metrics

    metrics.total_trades = len(trades)
    metrics.wins = sum(1 for t in trades if t.outcome == TradeOutcome.WIN)
    metrics.losses = sum(1 for t in trades if t.outcome == TradeOutcome.LOSS)
    metrics.open_trades = sum(1 for t in trades if t.outcome == TradeOutcome.OPEN)

    closed = metrics.wins + metrics.losses
    metrics.win_rate = (metrics.wins / closed * 100.0) if closed > 0 else 0.0

    # P&L stats
    pnls = [t.pnl_dollar for t in trades]
    metrics.total_pnl = sum(pnls)
    metrics.avg_pnl = np.mean(pnls)
    metrics.best_trade = max(pnls)
    metrics.worst_trade = min(pnls)

    win_pnls = [t.pnl_dollar for t in trades if t.outcome == TradeOutcome.WIN]
    loss_pnls = [t.pnl_dollar for t in trades if t.outcome == TradeOutcome.LOSS]

    metrics.avg_win = np.mean(win_pnls) if win_pnls else 0.0
    metrics.avg_loss = np.mean(loss_pnls) if loss_pnls else 0.0

    # Expectancy = (win_rate × avg_win) + ((1 - win_rate) × avg_loss)
    if closed > 0:
        wr = metrics.wins / closed
        metrics.expectancy = (wr * metrics.avg_win) + ((1 - wr) * metrics.avg_loss)

    # Profit Factor = gross_wins / |gross_losses|
    gross_wins = sum(p for p in pnls if p > 0)
    gross_losses = abs(sum(p for p in pnls if p < 0))
    metrics.profit_factor = (gross_wins / gross_losses) if gross_losses > 0 else float("inf")

    # Sharpe Ratio (annualized, assuming 5-min candles → ~252 trading days × 78 candles/day)
    if len(pnls) > 1:
        pnl_returns = np.array(pnls) / result.initial_balance
        mean_return = np.mean(pnl_returns)
        std_return = np.std(pnl_returns, ddof=1)
        if std_return > 0:
            # Annualize: sqrt(252 × 78) ≈ sqrt(19656) ≈ 140
            annualization = np.sqrt(252 * 78)
            metrics.sharpe_ratio = (mean_return / std_return) * annualization

    # Max Drawdown from equity curve
    equity = np.array(result.equity_curve)
    if len(equity) > 1:
        running_max = np.maximum.accumulate(equity)
        drawdowns = equity - running_max
        metrics.max_drawdown = float(np.min(drawdowns))
        # Drawdown as % of peak
        dd_pct = drawdowns / running_max * 100.0
        metrics.max_drawdown_pct = float(np.min(dd_pct))

    # Total return
    metrics.total_return_pct = (
        (result.final_balance - result.initial_balance) / result.initial_balance * 100.0
    )

    # Average holding period
    holding = [t.holding_candles for t in trades]
    metrics.avg_holding_candles = np.mean(holding) if holding else 0.0

    # Fee & cost tracking
    metrics.total_fees = sum(t.entry_fee + t.exit_fee for t in trades)
    metrics.total_slippage = sum(t.slippage_cost for t in trades)
    metrics.gross_pnl = sum(t.gross_pnl_dollar for t in trades)
    metrics.net_pnl = sum(t.net_pnl_dollar for t in trades)

    # Time exits
    metrics.time_exits = sum(1 for t in trades if t.outcome == TradeOutcome.TIME_EXIT)

    # Consecutive wins/losses
    max_cw = 0
    max_cl = 0
    current_cw = 0
    current_cl = 0
    for t in trades:
        if t.outcome == TradeOutcome.WIN:
            current_cw += 1
            current_cl = 0
            max_cw = max(max_cw, current_cw)
        elif t.outcome == TradeOutcome.LOSS:
            current_cl += 1
            current_cw = 0
            max_cl = max(max_cl, current_cl)
        else:
            current_cw = 0
            current_cl = 0
    metrics.max_consecutive_wins = max_cw
    metrics.max_consecutive_losses = max_cl

    # Calmar ratio (annualized return / |max drawdown|)
    if metrics.max_drawdown < 0:
        metrics.calmar_ratio = metrics.total_return_pct / abs(metrics.max_drawdown_pct)
    else:
        metrics.calmar_ratio = float("inf") if metrics.total_return_pct > 0 else 0.0

    return metrics
