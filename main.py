"""
Main Entry Point -- CLI for the ICT AMD Trading System.

Usage:
    python main.py --mode backtest --symbol BTCUSD --lookback 24
    python main.py --mode backtest --csv data.csv
    python main.py --mode signals  --symbol ETHUSD --lookback 12
    python main.py --mode backtest --sample
    python main.py --mode live --symbol BTCUSD --dry-run
    python main.py --mode live --symbol BTCUSD --no-dry-run
"""

import argparse
import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trading_bot.data.loader import load_csv, fetch_delta, generate_sample_data
from trading_bot.strategy.amd_strategy import generate_signals, StrategyConfig
from trading_bot.backtest.engine import run_backtest
from trading_bot.backtest.performance import calculate_metrics
from trading_bot.visualization.charts import create_trading_chart, create_matplotlib_chart


BANNER = """
================================================================
     ICT AMD TRADING SYSTEM
     Accumulation -> Manipulation -> Distribution
     5-Minute Timeframe | Delta Exchange
================================================================
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ICT AMD Trading System -- Algorithmic Crypto Trading Bot (Delta Exchange)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode backtest --symbol BTCUSD --lookback 24
  python main.py --mode backtest --csv my_data.csv
  python main.py --mode signals  --symbol ETHUSD --lookback 12
  python main.py --mode backtest --sample
  python main.py --mode live --symbol BTCUSD --dry-run
  python main.py --mode live --symbol BTCUSD --no-dry-run
        """,
    )

    # Data source (for backtest/signals modes)
    data_group = parser.add_mutually_exclusive_group()
    data_group.add_argument("--csv", type=str, help="Path to CSV file with OHLCV data")
    data_group.add_argument("--sample", action="store_true", help="Use generated sample data")

    # Mode
    parser.add_argument("--mode", choices=["backtest", "signals", "live"], default="backtest",
                        help="'backtest', 'signals', or 'live' (Delta Exchange)")

    # Symbol (shared across all modes)
    parser.add_argument("--symbol", type=str, default="BTCUSD",
                        help="Delta Exchange product symbol (default: BTCUSD)")

    # Data options
    parser.add_argument("--lookback", type=int, default=24,
                        help="Hours of candle history to fetch from Delta Exchange (default: 24)")
    parser.add_argument("--resolution", type=str, default="5m",
                        help="Candle resolution: 1m, 5m, 15m, 1h, 1d (default: 5m)")

    # Output
    parser.add_argument("--chart", type=str, default="trading_chart.html",
                        help="Output chart path (.html for Plotly, .png for matplotlib)")
    parser.add_argument("--no-chart", action="store_true", help="Skip chart generation")
    parser.add_argument("--balance", type=float, default=10000.0,
                        help="Initial account balance for backtesting")

    # Strategy overrides
    parser.add_argument("--rsi-period", type=int, default=None)
    parser.add_argument("--min-range", type=int, default=None)
    parser.add_argument("--max-range", type=int, default=None)
    parser.add_argument("--range-pct", type=float, default=None)
    parser.add_argument("--breakout-pct", type=float, default=None)
    parser.add_argument("--rr-ratio", type=float, default=None)
    parser.add_argument("--risk-pct", type=float, default=None)

    # Live trading options (Delta Exchange)
    parser.add_argument("--dry-run", dest="dry_run", action="store_true", default=True,
                        help="Dry-run mode: detect signals but don't place orders (default)")
    parser.add_argument("--no-dry-run", dest="dry_run", action="store_false",
                        help="LIVE mode: actually place orders on Delta Exchange")
    parser.add_argument("--loop-interval", type=int, default=300,
                        help="Seconds between live trading cycles (default: 300)")

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> StrategyConfig:
    """Build StrategyConfig from CLI args with defaults."""
    cfg = StrategyConfig()
    if args.rsi_period is not None:
        cfg.rsi_period = args.rsi_period
    if args.min_range is not None:
        cfg.min_range_candles = args.min_range
    if args.max_range is not None:
        cfg.max_range_candles = args.max_range
    if args.range_pct is not None:
        cfg.range_threshold_pct = args.range_pct
    if args.breakout_pct is not None:
        cfg.breakout_pct = args.breakout_pct
    if args.rr_ratio is not None:
        cfg.risk_reward_ratio = args.rr_ratio
    if args.risk_pct is not None:
        cfg.risk_per_trade_pct = args.risk_pct
    return cfg


def main():
    args = parse_args()
    config = build_config(args)

    print(BANNER)

    # ==============================================
    # LIVE MODE (Delta Exchange)
    # ==============================================
    if args.mode == "live":
        _run_live(args, config)
        return

    # ==============================================
    # BACKTEST / SIGNALS MODES
    # ==============================================

    # 1. Load Data
    print("-" * 60)
    print("  STEP 1: Loading Data")
    print("-" * 60)

    if args.csv:
        print(f"  Source: CSV file -> {args.csv}")
        df = load_csv(args.csv)
    elif args.sample:
        print("  Source: Generated sample data (500 candles)")
        df = generate_sample_data(n_candles=500)
    else:
        print(f"  Source: Delta Exchange -> {args.symbol}")
        df = fetch_delta(
            symbol=args.symbol,
            resolution=args.resolution,
            lookback_hours=args.lookback,
        )

    print(f"  [OK] Loaded {len(df)} candles")
    print(f"      From : {df.index[0]}")
    print(f"      To   : {df.index[-1]}")
    print(f"      Price: {df['Close'].iloc[-1]:.2f}")

    # 2. Generate signals
    print()
    print("-" * 60)
    print("  STEP 2: Signal Generation (AMD Pipeline)")
    print("-" * 60)

    signals = generate_signals(df, config)
    print(f"  [OK] Detected {len(signals)} trade signal(s)")

    if signals:
        print()
        for i, sig in enumerate(signals, 1):
            print(f"  Signal {i}: {sig}")
            if sig.divergence:
                d = sig.divergence
                print(f"           Divergence: {d.divergence_type} | "
                      f"Price {d.prior_price:.2f}->{d.current_price:.2f} | "
                      f"RSI {d.prior_rsi:.1f}->{d.current_rsi:.1f}")

    # 3. Signals-only mode
    if args.mode == "signals":
        if not signals:
            print("\n  [!!] No active signals at this time.")
        if not args.no_chart and signals:
            _generate_chart(df, signals, config, None, args.chart)
        print("\n  [OK] Done.")
        return

    # 4. Backtest
    print()
    print("-" * 60)
    print("  STEP 3: Backtesting")
    print("-" * 60)

    result = run_backtest(df, config, initial_balance=args.balance, signals=signals)
    metrics = calculate_metrics(result)
    print(metrics.summary())

    if result.trades:
        print("  -- Trade Log --")
        for i, trade in enumerate(result.trades, 1):
            print(f"  {i:>3}. {trade}")

    # 5. Chart
    if not args.no_chart:
        _generate_chart(df, signals, config, result, args.chart)

    print(f"\n  [OK] Done. Final balance: ${result.final_balance:,.2f}")


def _run_live(args, config):
    """Start the Delta Exchange live trading loop."""
    from trading_bot.exchange.live_trader import LiveTrader, setup_logging

    log_filename = f"trading_{args.symbol.lower()}.log"
    setup_logging(log_filename)

    mode_label = "DRY RUN" if args.dry_run else "** LIVE TRADING **"
    print("-" * 60)
    print(f"  Mode: {mode_label}")
    print(f"  Symbol: {args.symbol}")
    print(f"  Lookback: {args.lookback}h | Loop: {args.loop_interval}s")
    print("-" * 60)

    if not args.dry_run:
        print()
        print("  !! WARNING: LIVE TRADING MODE !!")
        print("  Real orders WILL be placed on Delta Exchange.")
        print("  Press Ctrl+C within 5 seconds to abort...")
        try:
            import time
            time.sleep(5)
        except KeyboardInterrupt:
            print("\n  Aborted.")
            return

    trader = LiveTrader(
        symbol=args.symbol,
        config=config,
        dry_run=args.dry_run,
        lookback_hours=args.lookback,
        loop_interval=args.loop_interval,
    )
    trader.run_loop()


def _generate_chart(df, signals, config, result, chart_path):
    """Generate the appropriate chart type based on file extension."""
    print()
    print("-" * 60)
    print("  STEP 4: Chart Generation")
    print("-" * 60)

    if chart_path.endswith(".html"):
        create_trading_chart(df, signals, config, result, save_path=chart_path)
    else:
        create_matplotlib_chart(df, signals, config, save_path=chart_path)


if __name__ == "__main__":
    main()
