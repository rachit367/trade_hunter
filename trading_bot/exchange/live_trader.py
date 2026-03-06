"""
Live Trader -- Automated AMD strategy execution loop on Delta Exchange.

Continuously:
  1. Fetches latest 5m candles from Delta Exchange
  2. Runs the AMD signal generator
  3. If a new signal is detected and no position is open -> executes trade
  4. Logs everything
  5. Sleeps until the next candle close

Supports dry-run mode (default) where signals are logged but no orders placed.
"""

import time
import logging
import sys
from datetime import datetime, timezone
from typing import Optional

from trading_bot.exchange.delta_connector import DeltaConnector
from trading_bot.strategy.amd_strategy import (
    generate_signals,
    StrategyConfig,
    TradeSignal,
    Direction,
    calculate_position_size,
)
from trading_bot.visualization.charts import create_trading_chart

logger = logging.getLogger("live_trader")


class LiveTrader:
    """
    Automated AMD strategy execution on Delta Exchange.

    Parameters
    ----------
    symbol : str
        Delta Exchange product symbol (e.g. "BTCUSD").
    config : StrategyConfig
        Strategy configuration.
    dry_run : bool
        If True (default), detect signals but don't place orders.
    lookback_hours : int
        Hours of candle history to fetch each cycle.
    loop_interval : int
        Seconds between each cycle (default: 300 = 5 minutes).
    """

    def __init__(
        self,
        symbol: str = "BTCUSD",
        config: StrategyConfig = None,
        dry_run: bool = True,
        lookback_hours: int = 4,
        loop_interval: int = 300,
    ):
        self.symbol = symbol
        self.config = config or StrategyConfig()
        self.dry_run = dry_run
        self.lookback_hours = lookback_hours
        self.loop_interval = loop_interval
        self.leverage = 10  # Default to 10x leverage as requested

        # Initialize connector
        self.connector = DeltaConnector()

        # Lookup product
        product_info = self.connector.get_product_info(symbol)
        self.product_id = product_info["id"]
        self.tick_size = float(product_info.get("tick_size", "0.5"))

        # Set default leverage (10x)
        if not self.dry_run:
            self.connector.set_leverage(self.product_id, self.leverage)

        # Track last signal to avoid duplicates
        self._last_signal_time = None

        mode_label = "DRY RUN" if dry_run else "** LIVE **"
        logger.info("=" * 50)
        logger.info("  LiveTrader initialized")
        logger.info("  Symbol     : %s (product_id=%d)", symbol, self.product_id)
        logger.info("  Mode       : %s", mode_label)
        logger.info("  Tick Size  : %s", self.tick_size)
        logger.info("  Lookback   : %d hours", lookback_hours)
        logger.info("  Interval   : %d seconds", loop_interval)
        logger.info("=" * 50)

    def run_once(self) -> Optional[TradeSignal]:
        """
        Execute one cycle of the strategy.

        Returns
        -------
        TradeSignal or None
            The signal that was acted upon, or None.
        """
        logger.info("-" * 40)
        logger.info("Cycle start: %s", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"))

        # 1. Fetch candles
        try:
            df = self.connector.fetch_candles(
                symbol=self.symbol,
                resolution="5m",
                lookback_hours=self.lookback_hours,
            )
            logger.info("Fetched %d candles | Last close: %.2f",
                        len(df), df["Close"].iloc[-1])
        except Exception as e:
            logger.error("Failed to fetch candles: %s", e)
            return None

        # 2. Generate signals
        signals = generate_signals(df, self.config)
        logger.info("Signals detected: %d", len(signals))

        if not signals:
            logger.info("No signals -- waiting for next cycle")
            return None

        # Take the most recent signal
        latest_signal = signals[-1]
        logger.info("Latest signal: %s", latest_signal)

        # 3. Check for duplicate (same timestamp as last executed)
        if self._last_signal_time == latest_signal.entry_time:
            logger.info("Signal already processed -- skipping")
            return None

        # 4. Check for existing position
        position = self.connector.get_open_position(self.product_id)
        if position is not None:
            logger.info("Position already open (size=%s) -- skipping new signal",
                       position.get("size"))
            return None

        # 5. Execute or log the signal
        if self.dry_run:
            self._log_dry_run(latest_signal)
        else:
            self._execute_signal(latest_signal)

        self._last_signal_time = latest_signal.entry_time
        return latest_signal

    def _log_dry_run(self, signal: TradeSignal):
        """Log a signal without placing any orders."""
        logger.info("=" * 40)
        logger.info("[DRY RUN] Would execute:")
        logger.info("  Direction  : %s", signal.direction.value)
        logger.info("  Entry      : %.2f", signal.entry_price)
        logger.info("  Stop Loss  : %.2f", signal.stop_loss)
        logger.info("  Take Profit: %.2f", signal.take_profit)
        logger.info("  R:R Ratio  : %.1f", signal.rr_ratio)
        if signal.divergence:
            d = signal.divergence
            logger.info("  Divergence : %s (Price %.2f->%.2f, RSI %.1f->%.1f)",
                       d.divergence_type, d.prior_price, d.current_price,
                       d.prior_rsi, d.current_rsi)
        logger.info("=" * 40)

        # Print to console too
        print()
        print("  " + "=" * 50)
        print(f"  [DRY RUN] SIGNAL DETECTED")
        print(f"  Direction  : {signal.direction.value}")
        print(f"  Entry      : {signal.entry_price:.2f}")
        print(f"  Stop Loss  : {signal.stop_loss:.2f}")
        print(f"  Take Profit: {signal.take_profit:.2f}")
        print(f"  R:R Ratio  : {signal.rr_ratio:.1f}")
        print("  " + "=" * 50)

    def _execute_signal(self, signal: TradeSignal):
        """Place a real bracket order on Delta Exchange."""
        logger.info("EXECUTING LIVE TRADE")

        # Determine side
        if signal.direction == Direction.LONG:
            side = "buy"
        else:
            side = "sell"

        # Get balance and calculate position size
        wallet = self.connector.get_balance()
        if wallet is None:
            logger.error("Cannot get wallet balance -- aborting trade")
            return

        available = float(wallet.get("available_balance", 0))
        logger.info("Available balance: %.2f", available)

        # Calculate size (contracts)
        # For perpetual futures, size is in contracts (typically 1 contract = $1)
        risk_amount = available * (self.config.risk_per_trade_pct / 100.0)
        risk_per_contract = abs(signal.entry_price - signal.stop_loss)

        if risk_per_contract <= 0:
            logger.error("Invalid risk per contract -- aborting")
            return

        size = max(1, int(risk_amount / risk_per_contract))
        logger.info("Position size: %d contracts (risk=$%.2f)", size, risk_amount)

        try:
            # Place bracket order with SL and TP
            result = self.connector.place_bracket_order(
                product_id=self.product_id,
                size=size,
                side=side,
                stop_loss_price=signal.stop_loss,
                take_profit_price=signal.take_profit,
                tick_size=self.tick_size,
            )

            logger.info("Trade executed successfully!")
            logger.info("Order result: %s", result)

            print()
            print("  " + "=" * 50)
            print(f"  [LIVE] ORDER PLACED")
            print(f"  Direction  : {signal.direction.value}")
            print(f"  Size       : {size} contracts")
            print(f"  Stop Loss  : {signal.stop_loss:.2f}")
            print(f"  Take Profit: {signal.take_profit:.2f}")
            print("  " + "=" * 50)

        except Exception as e:
            logger.error("Failed to execute trade: %s", e)
            print(f"  [ERROR] Trade execution failed: {e}")

    def run_loop(self):
        """
        Run the strategy in a continuous loop.

        Fetches candles, generates signals, and optionally executes trades
        every `loop_interval` seconds. Press Ctrl+C to stop.
        """
        mode_label = "DRY RUN" if self.dry_run else "** LIVE TRADING **"
        print()
        print("=" * 60)
        print(f"  AMD LiveTrader -- {mode_label}")
        print(f"  Symbol: {self.symbol} | Interval: {self.loop_interval}s")
        print(f"  Press Ctrl+C to stop")
        print("=" * 60)

        cycle = 0
        while True:
            try:
                cycle += 1
                logger.info("=== Cycle %d ===", cycle)
                print(f"\n  [Cycle {cycle}] {datetime.now().strftime('%H:%M:%S')} -- Scanning...", end="")

                signal = self.run_once()

                if signal:
                    print(f" SIGNAL: {signal.direction.value} @ {signal.entry_price:.2f}")
                else:
                    print(" No signal")

                # Sleep until next cycle
                logger.info("Sleeping %d seconds...", self.loop_interval)
                time.sleep(self.loop_interval)

            except KeyboardInterrupt:
                print("\n\n  [STOPPED] LiveTrader stopped by user.")
                logger.info("LiveTrader stopped by KeyboardInterrupt")
                break
            except Exception as e:
                logger.error("Unexpected error in cycle %d: %s", cycle, e)
                print(f"\n  [FATAL ERROR] {e}")
                print("  Exiting with status code 1 so the host (Render) can auto-restart the service.")
                sys.exit(1)


def setup_logging(log_file: str = "trading_bot.log"):
    """Configure logging for the trading bot."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    # Suppress noisy libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
