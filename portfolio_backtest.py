"""
Portfolio Backtest — Run multiple symbol/timeframe combinations together
with a shared balance and shared RiskManager, exactly as the live system would.

Usage:
    python portfolio_backtest.py
    python portfolio_backtest.py --lookback 720 --balance 10000

The shared RiskManager enforces:
  - Max 3% daily drawdown across ALL pairs
  - Max 2 concurrent open positions
  - Cooldown after 3 consecutive losses
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np

from trading_bot.data.loader import fetch_delta
from trading_bot.strategy.amd_strategy import generate_signals, StrategyConfig
from trading_bot.backtest.engine import run_backtest, BacktestConfig, BacktestResult, CompletedTrade, TradeOutcome
from trading_bot.backtest.performance import calculate_metrics
from trading_bot.strategy.risk_manager import RiskManager, RiskConfig


PORTFOLIO = [
    # (symbol, resolution)  — 30-day validated profitable configs
    # BTCUSD excluded: PF 0.40 on 15m, PF 0.97 on 5m — net negative
    # AVAXUSD excluded: PF 0.16, -$359 — terrible
    # BNBUSD excluded: 0% WR — terrible
    ("ETHUSD", "5m"),   # PF 1.60, 33% WR, +$233 / 30d
    ("SOLUSD", "5m"),   # PF 1.81, 56% WR, +$309 / 30d — best performer
    ("XRPUSD", "5m"),   # PF 3.08, 67% WR, +$198 / 30d — high quality, low count
    ("DOTUSD", "5m"),   # PF 1.38, 43% WR, +$135 / 30d
]

BANNER = """
================================================================
     ICT AMD PORTFOLIO BACKTEST
     Multi-Pair | Shared Risk | Delta Exchange
================================================================
"""


def run_portfolio(lookback_hours: int = 720, initial_balance: float = 10000.0):
    print(BANNER)

    bt_config = BacktestConfig()
    risk_config = RiskConfig(
        max_daily_loss_pct=3.0,
        max_concurrent_trades=2,
        consecutive_loss_cooldown=3,
        cooldown_minutes=30,
    )

    # Shared risk manager
    risk_mgr = RiskManager(config=risk_config, initial_balance=initial_balance)

    balance = initial_balance
    all_trades: list[CompletedTrade] = []
    equity_snapshots: list[float] = [balance]
    pair_results = {}

    print(f"  Portfolio: {', '.join(f'{s} {tf}' for s, tf in PORTFOLIO)}")
    print(f"  Lookback : {lookback_hours}h")
    print(f"  Balance  : ${initial_balance:,.2f}")
    print()

    # --- Fetch data for all pairs ---
    datasets = {}
    for symbol, resolution in PORTFOLIO:
        print(f"  Fetching {symbol} {resolution} ({lookback_hours}h)...", end=" ")
        try:
            df = fetch_delta(symbol=symbol, resolution=resolution, lookback_hours=lookback_hours)
            print(f"{len(df)} candles OK")
            datasets[(symbol, resolution)] = df
        except Exception as e:
            print(f"FAILED: {e}")

    print()

    # --- Generate signals for all pairs ---
    all_signals_by_pair = {}
    for (symbol, resolution), df in datasets.items():
        config = StrategyConfig.for_timeframe(resolution)
        sigs = generate_signals(df, config)
        print(f"  {symbol} {resolution}: {len(sigs)} signals")
        all_signals_by_pair[(symbol, resolution)] = (df, config, sigs)

    print()

    # --- Merge all signals chronologically and simulate ---
    # Build a flat list of (entry_time, symbol, resolution, signal, df, config)
    flat = []
    for (symbol, resolution), (df, config, sigs) in all_signals_by_pair.items():
        for sig in sigs:
            flat.append((sig.entry_time, symbol, resolution, sig, df, config))

    flat.sort(key=lambda x: x[0])  # Sort by entry time

    print(f"  Total signals across portfolio: {len(flat)}")
    print()
    print("-" * 60)

    # Direction-streak filter: block a direction after 2 consecutive losses on that
    # pair+direction combo. Resets on any win or a direction switch.
    # Key: f"{symbol}_{resolution}_{direction}" → consecutive loss count
    DIR_BAN_THRESHOLD = 2
    direction_streaks: dict = {}

    for (entry_time, symbol, resolution, sig, df, config) in flat:
        pair_key = f"{symbol}_{resolution}"
        dir_key = f"{pair_key}_{sig.direction.value}"

        # Direction-streak gate — skip if we've hit N consecutive losses on this direction
        streak = direction_streaks.get(dir_key, 0)
        if streak >= DIR_BAN_THRESHOLD:
            print(f"  [SKIP] [{symbol} {resolution}] {sig.direction.value} @ {sig.entry_price:.2f}"
                  f" - direction banned after {streak} consecutive losses")
            continue

        # Shared risk gate — per-pair cooldown, global daily DD + concurrent limit
        allowed, reason = risk_mgr.can_trade(sig.entry_time, pair=pair_key)
        if not allowed:
            print(f"  [{symbol}] SKIPPED {sig.direction.value} @ {sig.entry_price:.2f} - {reason}")
            continue

        # Run single-trade backtest with current balance
        result = run_backtest(
            df=df,
            config=config,
            initial_balance=balance,
            signals=[sig],
            bt_config=bt_config,
        )

        if not result.trades:
            continue

        trade = result.trades[0]
        all_trades.append(trade)
        balance = result.final_balance
        equity_snapshots.append(balance)

        # Update risk manager — pass pair_key so cooldown is tracked per pair
        risk_mgr.on_trade_open()
        risk_mgr.on_trade_close(trade.net_pnl_dollar, trade.exit_time, pair=pair_key)
        risk_mgr.balance = balance

        # Update direction streak
        if trade.net_pnl_dollar < 0:
            direction_streaks[dir_key] = streak + 1
        else:
            direction_streaks[dir_key] = 0  # Any non-loss resets the streak

        pnl_sign = "+" if trade.net_pnl_dollar >= 0 else ""
        outcome_icon = "[WIN]" if trade.outcome == TradeOutcome.WIN else ("[LOSS]" if trade.outcome == TradeOutcome.LOSS else "[TIME]")
        print(f"  {outcome_icon} [{symbol} {resolution}] {sig.direction.value} @ {sig.entry_price:.2f} "
              f"-> {trade.outcome.value} | {pnl_sign}${trade.net_pnl_dollar:.2f} | Bal: ${balance:.2f}")

        # Track per-pair
        key = f"{symbol} {resolution}"
        if key not in pair_results:
            pair_results[key] = {"trades": 0, "wins": 0, "losses": 0, "pnl": 0.0}
        pair_results[key]["trades"] += 1
        pair_results[key]["pnl"] += trade.net_pnl_dollar
        if trade.outcome == TradeOutcome.WIN:
            pair_results[key]["wins"] += 1
        elif trade.outcome == TradeOutcome.LOSS:
            pair_results[key]["losses"] += 1

    # --- Portfolio Summary ---
    print()
    print("=" * 60)
    print("  PORTFOLIO RESULTS")
    print("=" * 60)

    total_trades = len(all_trades)
    wins = sum(1 for t in all_trades if t.outcome == TradeOutcome.WIN)
    losses = sum(1 for t in all_trades if t.outcome == TradeOutcome.LOSS)
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    total_pnl = balance - initial_balance
    total_return = (total_pnl / initial_balance) * 100

    # Max drawdown from equity curve
    eq = np.array(equity_snapshots)
    peak = np.maximum.accumulate(eq)
    drawdown = eq - peak
    max_dd = float(np.min(drawdown))
    max_dd_pct = (max_dd / initial_balance) * 100

    # Profit factor
    gross_wins = sum(t.net_pnl_dollar for t in all_trades if t.net_pnl_dollar > 0)
    gross_losses = abs(sum(t.net_pnl_dollar for t in all_trades if t.net_pnl_dollar < 0))
    pf = (gross_wins / gross_losses) if gross_losses > 0 else float('inf')

    # Total fees
    total_fees = sum(t.entry_fee + t.exit_fee for t in all_trades)

    print(f"  Pairs Traded    : {', '.join(pair_results.keys())}")
    print(f"  Total Trades    : {total_trades}  (W:{wins} L:{losses})")
    print(f"  Win Rate        : {win_rate:.1f}%")
    print(f"  Total P&L       : ${total_pnl:+,.2f}  ({total_return:+.2f}%)")
    print(f"  Profit Factor   : {pf:.2f}")
    print(f"  Max Drawdown    : ${max_dd:,.2f}  ({max_dd_pct:.2f}%)")
    print(f"  Total Fees Paid : ${total_fees:.2f}")
    print(f"  Final Balance   : ${balance:,.2f}")
    print()
    print("  --- Per-Pair Breakdown ---")
    for key, stats in pair_results.items():
        wr = (stats["wins"] / stats["trades"] * 100) if stats["trades"] > 0 else 0
        print(f"  {key:15s} | {stats['trades']} trades | WR {wr:.0f}% | P&L ${stats['pnl']:+.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ICT AMD Portfolio Backtest")
    parser.add_argument("--lookback", type=int, default=720, help="Hours of history (default 720 = 30d)")
    parser.add_argument("--balance", type=float, default=10000.0, help="Starting balance")
    args = parser.parse_args()
    run_portfolio(lookback_hours=args.lookback, initial_balance=args.balance)
