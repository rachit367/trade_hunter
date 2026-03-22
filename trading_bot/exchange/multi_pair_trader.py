"""
Multi-Pair Trader — Orchestrates LiveTrader instances across multiple
symbol/timeframe combinations with a shared RiskManager.

The shared RiskManager enforces global limits:
  - Max 3% daily drawdown across ALL pairs
  - Max 2 concurrent open trades across ALL pairs
  - Cooldown after consecutive losses applies globally
"""

import logging
import time
from datetime import datetime, timezone
from typing import List, Optional

from trading_bot.strategy.amd_strategy import StrategyConfig
from trading_bot.strategy.risk_manager import RiskManager, RiskConfig
from trading_bot.exchange.live_trader import LiveTrader
from trading_bot.exchange.trade_store import TradeStore

logger = logging.getLogger("multi_pair_trader")


class MultiPairTrader:
    """
    Manages multiple LiveTrader instances sharing a single RiskManager.

    Parameters
    ----------
    symbols : list of str
        Symbols to trade (e.g. ["BTCUSD", "ETHUSD"]).
    timeframes : list of str
        Timeframes to run (e.g. ["5m", "15m"]).
    dry_run : bool
        Dry-run mode for all traders.
    risk_config : RiskConfig
        Global risk configuration.
    loop_interval : int
        Seconds between cycles.
    lookback_hours : int
        Hours of candle history per fetch.
    """

    def __init__(
        self,
        symbols: List[str],
        timeframes: List[str] = None,
        dry_run: bool = True,
        risk_config: RiskConfig = None,
        loop_interval: int = 300,
        lookback_hours: int = 4,
        initial_balance: float = 10000.0,
    ):
        self.symbols = symbols
        self.timeframes = timeframes or ["5m"]
        self.dry_run = dry_run
        self.loop_interval = loop_interval

        # Shared risk manager across all pairs
        self.risk_config = risk_config or RiskConfig()
        self.risk_mgr = RiskManager(
            config=self.risk_config,
            initial_balance=initial_balance,
        )

        # Global trade store
        import pathlib
        pathlib.Path("logs").mkdir(exist_ok=True)
        self.trade_store = TradeStore(filepath="logs/trade_state_global.json")
        saved = self.trade_store.load()
        if saved:
            self.risk_mgr.from_dict(saved)

        # Create a LiveTrader per symbol/timeframe combo
        # Auto-scale loop_interval to the SHORTEST timeframe (fastest cycle)
        _tf_seconds = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "4h": 14400}
        if loop_interval == 300:  # Default — auto-scale
            min_tf_secs = min(_tf_seconds.get(tf, 300) for tf in self.timeframes)
            self.loop_interval = min_tf_secs
        else:
            self.loop_interval = loop_interval

        self.traders: List[LiveTrader] = []
        for symbol in symbols:
            for tf in self.timeframes:
                config = StrategyConfig.for_timeframe(tf)
                trader = LiveTrader(
                    symbol=symbol,
                    config=config,
                    dry_run=dry_run,
                    lookback_hours=lookback_hours,
                    resolution=tf,          # ← Fixed: pass per-trader timeframe
                    risk_manager=self.risk_mgr,  # Shared!
                )
                # Override the per-trader store with global store
                trader.trade_store = self.trade_store
                self.traders.append(trader)
                logger.info("Added trader: %s @ %s", symbol, tf)

    def run_loop(self):
        """
        Round-robin through all traders in a continuous loop.

        Each cycle iterates through every symbol/TF combo, running one
        strategy check per trader. The shared RiskManager gates all entries.
        """
        import signal

        def handle_sigterm(signum, frame):
            raise KeyboardInterrupt()

        try:
            signal.signal(signal.SIGTERM, handle_sigterm)
        except (ValueError, AttributeError):
            pass

        mode_label = "DRY RUN" if self.dry_run else "** LIVE TRADING **"
        pairs_label = ", ".join(self.symbols)
        tf_label = ", ".join(self.timeframes)

        print()
        print("=" * 60)
        print(f"  MultiPairTrader -- {mode_label}")
        print(f"  Symbols    : {pairs_label}")
        print(f"  Timeframes : {tf_label}")
        print(f"  Traders    : {len(self.traders)}")
        print(f"  Risk Limits: {self.risk_config.max_daily_loss_pct}% daily | "
              f"{self.risk_config.max_concurrent_trades} concurrent")
        print(f"  Press Ctrl+C to stop")
        print("=" * 60)

        cycle = 0
        while True:
            try:
                cycle += 1
                logger.info("=== Multi-Pair Cycle %d ===", cycle)
                print(f"\n--- Cycle {cycle} | {datetime.now().strftime('%H:%M:%S')} ---")

                for trader in self.traders:
                    # Check global risk gate — per-pair cooldown so one pair's
                    # losing streak doesn't block the other pair's signals
                    _pair_key = f"{trader.symbol}_{trader.resolution}"
                    allowed, reason = self.risk_mgr.can_trade(
                        datetime.now(timezone.utc), pair=_pair_key
                    )
                    if not allowed:
                        logger.info("[%s] Skipped: %s", trader.symbol, reason)
                        print(f"  [{trader.symbol}] Risk gate: {reason}")
                        continue

                    try:
                        result = trader.run_once()
                        if result:
                            # Persist state after any trade action
                            self.trade_store.save(self.risk_mgr.to_dict())
                    except Exception as e:
                        logger.error("[%s] Error: %s", trader.symbol, e)
                        print(f"  [{trader.symbol}] Error: {e}")

                # Persist state at end of each cycle
                self.trade_store.save(self.risk_mgr.to_dict())

                logger.info("Sleeping %d seconds...", self.loop_interval)
                time.sleep(self.loop_interval)

            except KeyboardInterrupt:
                print(f"\n\n  [STOPPED] MultiPairTrader stopped by user.")
                self.trade_store.save(self.risk_mgr.to_dict())
                logger.info("MultiPairTrader stopped. State saved.")
                break
