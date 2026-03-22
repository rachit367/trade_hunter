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
from datetime import datetime, timezone, timedelta
from typing import Optional

from trading_bot.exchange.delta_connector import DeltaConnector
from trading_bot.strategy.amd_strategy import (
    generate_signals,
    StrategyConfig,
    TradeSignal,
    Direction,
    calculate_position_size,
)
from trading_bot.strategy.risk_manager import RiskManager, RiskConfig
from trading_bot.exchange.trade_store import TradeStore
from trading_bot.visualization.charts import create_trading_chart
from trading_bot.notifications import telegram_notifier as _tg

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
        loop_interval: int = None,
        resolution: str = "5m",
        risk_config: RiskConfig = None,
        risk_manager: RiskManager = None,
    ):
        self.symbol = symbol
        self.resolution = resolution
        self.config = config or StrategyConfig.for_timeframe(resolution)
        self.dry_run = dry_run
        self.lookback_hours = lookback_hours
        self.leverage = 10  # Default to 10x leverage as requested

        # Auto-scale loop interval to match candle close frequency
        _tf_seconds = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "4h": 14400}
        if loop_interval is None:
            self.loop_interval = _tf_seconds.get(resolution, 300)
        else:
            self.loop_interval = loop_interval

        # Initialize connector
        self.connector = DeltaConnector()

        # Risk management (can be shared across pairs via risk_manager param)
        import pathlib
        pathlib.Path("logs").mkdir(exist_ok=True)
        self.trade_store = TradeStore(filepath=f"logs/trade_state_{symbol}.json")
        if risk_manager is not None:
            self.risk_mgr = risk_manager
        else:
            initial_bal = 10000.0
            if not dry_run:
                wallet = self.connector.get_balance()
                if wallet:
                    initial_bal = float(wallet.get("available_balance", 10000.0))
            else:
                initial_bal = self.connector._dry_run_balance
            self.risk_mgr = RiskManager(
                config=risk_config or RiskConfig(),
                initial_balance=initial_bal,
            )
            # Restore persisted state
            saved = self.trade_store.load()
            if saved:
                self.risk_mgr.from_dict(saved)

        # Lookup product
        product_info = self.connector.get_product_info(symbol)
        self.product_id = product_info["id"]
        self.tick_size = float(product_info.get("tick_size", "0.5"))

        # Set default leverage (10x)
        if not self.dry_run:
            self.connector.set_leverage(self.product_id, self.leverage)

        # Track last signal to avoid duplicates (use entry_price + direction as key)
        self._last_signal_key = None

        # Daily stats for summary notification
        self._daily_trades = 0
        self._daily_wins = 0
        self._daily_losses = 0
        self._daily_pnl = 0.0
        self._last_summary_date = None

        # Track active virtual dry-run position
        self._virtual_position: Optional[TradeSignal] = None
        self._virtual_entry_time: Optional[datetime] = None
        self._virtual_pos_size: float = 0.0
        self._virtual_partial_pnl: float = 0.0
        self._virtual_break_even_moved: bool = False
        self._virtual_partial_1_taken: bool = False   # 25% at 1.5R
        self._virtual_partial_2_taken: bool = False   # 35% at 2.0R
        # Keep legacy flag for compatibility
        self._virtual_partial_taken: bool = False

        # Signal freshness: how many minutes old a signal can be to still be "actionable"
        # Scale to 3 candles worth of the configured resolution
        _tf_minutes = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240}
        _candle_minutes = _tf_minutes.get(resolution, 5)
        self._signal_freshness_minutes = _candle_minutes * 3  # 3 candles grace period
        self._time_exit_candles = getattr(self.config, 'time_exit_candles', 48)

        # Direction-streak filter: after N consecutive losses on the same direction,
        # block that direction until a win resets the streak.
        # Key: direction string ("LONG"/"SHORT") → consecutive loss count
        self._dir_streak: dict = {}  # {"LONG": 0, "SHORT": 0}
        self._dir_ban_threshold: int = 2

        mode_label = "DRY RUN" if dry_run else "** LIVE **"
        logger.info("=" * 50)
        logger.info("  LiveTrader initialized")
        logger.info("  Symbol     : %s (product_id=%d)", symbol, self.product_id)
        logger.info("  Resolution : %s", self.resolution)
        logger.info("  Mode       : %s", mode_label)
        logger.info("  Tick Size  : %s", self.tick_size)
        logger.info("  Lookback   : %d hours", lookback_hours)
        logger.info("  Interval   : %d seconds", self.loop_interval)
        logger.info("  Freshness  : %d min (%s x 3)", self._signal_freshness_minutes, self.resolution)
        if dry_run:
            logger.info("  Balance    : $%.2f (Simulated)", self.connector._dry_run_balance)
        logger.info("=" * 50)

    def _get_signal_key(self, signal: TradeSignal) -> str:
        """Generate a unique key for a signal to detect duplicates."""
        return f"{signal.direction.value}_{signal.entry_price:.2f}_{signal.range_start_idx}_{signal.range_end_idx}"

    def _is_signal_fresh(self, signal: TradeSignal, df) -> bool:
        """
        Check if a signal is fresh enough to act upon.
        
        Uses TIME-BASED freshness: the signal's entry_time must be within
        the last N minutes of the latest candle in the data.
        """
        latest_candle_time = df.index[-1]
        signal_time = signal.entry_time
        
        # Convert both to timezone-aware or timezone-naive for comparison
        if hasattr(latest_candle_time, 'tzinfo') and latest_candle_time.tzinfo is not None:
            if hasattr(signal_time, 'tzinfo') and signal_time.tzinfo is None:
                signal_time = signal_time.replace(tzinfo=timezone.utc)
        elif hasattr(signal_time, 'tzinfo') and signal_time.tzinfo is not None:
            signal_time = signal_time.replace(tzinfo=None)
        
        age = latest_candle_time - signal_time
        age_minutes = age.total_seconds() / 60.0
        
        is_fresh = age_minutes <= self._signal_freshness_minutes
        
        logger.info("Signal freshness: entry_time=%s, latest_candle=%s, age=%.1f min, fresh=%s",
                    signal_time, latest_candle_time, age_minutes, is_fresh)
        
        return is_fresh

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
                resolution=self.resolution,
                lookback_hours=self.lookback_hours,
            )
            logger.info("Fetched %d candles | Last close: %.2f",
                        len(df), df["Close"].iloc[-1])
        except Exception as e:
            logger.error("Failed to fetch candles: %s", e)
            return None

        # Check virtual position outcome before finding new signals
        if self.dry_run and self._virtual_position:
            self._check_virtual_position(df)
            if self._virtual_position:
                logger.info("Virtual position still open. Waiting for resolution...")
                print(f"  [{self.symbol}] Virtual trade still active. Waiting...")
                return None
                
        # Manage real live positions (Trailing stops / Break-even)
        if not self.dry_run:
            self._manage_live_positions(df)

        # 2. Generate signals
        signals = generate_signals(df, self.config)
        logger.info("Signals detected: %d", len(signals))

        if not signals:
            logger.info("No signals -- waiting for next cycle")
            return None

        # Take the most recent signal
        latest_signal = signals[-1]
        logger.info("Latest signal: %s", latest_signal)

        # 3. TIME-BASED freshness check (replaces broken index-based check)
        if not self._is_signal_fresh(latest_signal, df):
            logger.info("Signal is STALE (too old) -- skipping")
            return None

        # 4. Check for duplicate (same setup as last executed)
        signal_key = self._get_signal_key(latest_signal)
        if self._last_signal_key == signal_key:
            logger.info("Signal already processed (same setup) -- skipping")
            return None

        # 5a. Direction-streak gate — block direction after N consecutive losses
        _dir = latest_signal.direction.value
        _streak = self._dir_streak.get(_dir, 0)
        if _streak >= self._dir_ban_threshold:
            logger.info("Direction ban: %d consecutive %s losses", _streak, _dir)
            print(f"  [{self.symbol}] Direction ban: {_streak} consecutive {_dir} losses — skipping")
            return None

        # 5b. Risk manager gate check — pass pair key for per-pair cooldown isolation
        _pair_key = f"{self.symbol}_{self.resolution}"
        allowed, reason = self.risk_mgr.can_trade(datetime.now(timezone.utc), pair=_pair_key)
        if not allowed:
            logger.info("Risk gate blocked: %s", reason)
            print(f"  [{self.symbol}] Risk gate: {reason}")
            return None

        # 6. Check for existing position (live mode only)
        if not self.dry_run:
            position = self.connector.get_open_position(self.product_id)
            if position is not None:
                current_size = abs(int(float(position.get("size", 0))))
                if current_size > 0:
                    logger.info("Position already open (size=%s) -- skipping new signal", current_size)
                    return None

        # 7. Execute or log the signal
        self.risk_mgr.on_trade_open()
        if self.dry_run:
            self._log_dry_run(latest_signal)
        else:
            self._execute_signal(latest_signal)

        self._last_signal_key = signal_key
        return latest_signal
        
    def _manage_live_positions(self, df):
        """
        Monitor open live positions:
          - Move SL to break-even when price reaches 1.5R
          - After break-even, trail SL using swing low/high + ATR buffer (matches backtest engine)
        """
        position = self.connector.get_open_position(self.product_id)
        if not position:
            return

        current_size = abs(int(float(position.get("size", 0))))
        if current_size == 0:
            return

        entry_price = float(position.get("entry_price", 0))
        if entry_price == 0:
            return

        side = position.get("side", "").lower()
        active_orders = self.connector.get_active_orders(self.product_id)

        # Find all active stop-loss orders
        stop_orders = [o for o in active_orders
                       if o.get("order_type") == "stop_order" and o.get("state") == "open"]
        if not stop_orders:
            return

        current_price = df["Close"].iloc[-1]

        # Pre-compute ATR and swing levels from latest candles for trailing
        from ta.volatility import AverageTrueRange as _ATR
        _atr_series = _ATR(df["High"], df["Low"], df["Close"], window=14).average_true_range()
        atr_now = float(_atr_series.iloc[-1]) if not _atr_series.empty else 0.0
        trail_lookback = getattr(self.config, "trailing_swing_lookback", 5)
        recent_lows  = df["Low"].iloc[-trail_lookback:].min()
        recent_highs = df["High"].iloc[-trail_lookback:].max()
        atr_buf = atr_now * getattr(self.config, "sl_buffer_atr_mult", 0.3)

        for stop_order in stop_orders:
            stop_price = float(stop_order.get("stop_price", 0))
            order_id   = stop_order.get("id")
            if stop_price == 0 or order_id is None:
                continue

            risk_per_unit = abs(entry_price - stop_price)

            if side == "buy":
                # --- Phase 1: Move to B/E when price hits 1.5R ---
                if stop_price < entry_price:
                    if risk_per_unit == 0:
                        continue
                    target_1_5r = entry_price + (1.5 * risk_per_unit)
                    if current_price >= target_1_5r:
                        new_sl = round(entry_price, 8)
                        logger.info("Live LONG: SL → Break-Even %.4f", new_sl)
                        self.connector.modify_order(order_id, self.product_id, new_sl, self.tick_size)

                # --- Phase 2: Trail after B/E using swing low − ATR buffer ---
                else:
                    if atr_buf > 0:
                        trail_sl = recent_lows - atr_buf
                        if trail_sl > stop_price:   # Only ever tighten
                            logger.info("Live LONG: Trailing SL %.4f → %.4f", stop_price, trail_sl)
                            self.connector.modify_order(order_id, self.product_id, trail_sl, self.tick_size)

            else:  # sell / SHORT
                # --- Phase 1: Move to B/E when price hits 1.5R ---
                if stop_price > entry_price:
                    if risk_per_unit == 0:
                        continue
                    target_1_5r = entry_price - (1.5 * risk_per_unit)
                    if current_price <= target_1_5r:
                        new_sl = round(entry_price, 8)
                        logger.info("Live SHORT: SL → Break-Even %.4f", new_sl)
                        self.connector.modify_order(order_id, self.product_id, new_sl, self.tick_size)

                # --- Phase 2: Trail after B/E using swing high + ATR buffer ---
                else:
                    if atr_buf > 0:
                        trail_sl = recent_highs + atr_buf
                        if trail_sl < stop_price:   # Only ever tighten
                            logger.info("Live SHORT: Trailing SL %.4f → %.4f", stop_price, trail_sl)
                            self.connector.modify_order(order_id, self.product_id, trail_sl, self.tick_size)

    def _check_virtual_position(self, df):
        """
        Check if an open virtual position hit SL or TP.

        Mirrors the backtest engine exactly:
          - Partial 1: 25% at 1.5R, SL moves to entry + 0.1R (cushion)
          - Partial 2: 35% at 2.0R, SL tightens to exact entry
          - Remaining 40% trails behind recent swing ± ATR buffer
          - Time exit after config.time_exit_candles candles
        """
        pos = self._virtual_position
        if not pos:
            return

        risk_per_unit = abs(pos.entry_price - pos.stop_loss)
        if risk_per_unit <= 0:
            self._reset_virtual_position()
            return

        # Confidence-based position sizing (matches backtest engine)
        wallet_balance = self.connector._dry_run_balance
        risk_pct = self.config.risk_per_trade_pct
        if getattr(self.config, 'use_confidence_sizing', True):
            min_m = getattr(self.config, 'min_confidence_multiplier', 0.5)
            max_m = getattr(self.config, 'max_confidence_multiplier', 1.2)
            conf_mult = min_m + (getattr(pos, 'confidence', 0.5) * (max_m - min_m))
            risk_pct *= conf_mult
        risk_dollar = wallet_balance * (risk_pct / 100.0)

        if self._virtual_pos_size <= 0:
            raw = risk_dollar / risk_per_unit
            # Cap: max notional = 20x balance
            max_size = (wallet_balance * 20.0) / pos.entry_price
            self._virtual_pos_size = min(raw, max_size)

        # R targets
        if pos.direction == Direction.SHORT:
            target_1_5r = pos.entry_price - (1.5 * risk_per_unit)
            target_2_0r = pos.entry_price - (2.0 * risk_per_unit)
        else:
            target_1_5r = pos.entry_price + (1.5 * risk_per_unit)
            target_2_0r = pos.entry_price + (2.0 * risk_per_unit)

        # Restore current SL from state
        if self._virtual_partial_2_taken:
            current_sl = pos.entry_price          # Tightened to exact entry after 2R
        elif self._virtual_break_even_moved:
            if pos.direction == Direction.LONG:
                current_sl = pos.entry_price + (0.1 * risk_per_unit)
            else:
                current_sl = pos.entry_price - (0.1 * risk_per_unit)
        else:
            current_sl = pos.stop_loss

        # Check candles formed AFTER entry
        mask = df.index > pos.entry_time
        recent_df = df[mask]

        for dt, row in recent_df.iterrows():
            high = row["High"]
            low  = row["Low"]
            close = row["Close"]

            if pos.direction == Direction.LONG:
                # --- SL FIRST (worst-case) ---
                if low <= current_sl:
                    remaining_pnl = (current_sl - pos.entry_price) * self._virtual_pos_size
                    self.connector.update_dry_run_balance(remaining_pnl)
                    outcome = "BREAK-EVEN" if self._virtual_break_even_moved else "STOP LOSS"
                    total_pnl = self._virtual_partial_pnl + remaining_pnl
                    self._print_virtual_close(outcome, total_pnl, pos)
                    self._reset_virtual_position()
                    return

                # --- TP ---
                if high >= pos.take_profit:
                    remaining_pnl = (pos.take_profit - pos.entry_price) * self._virtual_pos_size
                    self.connector.update_dry_run_balance(remaining_pnl)
                    total_pnl = self._virtual_partial_pnl + remaining_pnl
                    self._print_virtual_close("TAKE PROFIT", total_pnl, pos)
                    self._reset_virtual_position()
                    return

                # --- Partial 1: 25% at 1.5R ---
                if not self._virtual_partial_1_taken and high >= target_1_5r:
                    self._virtual_partial_1_taken = True
                    self._virtual_break_even_moved = True
                    current_sl = pos.entry_price + (0.1 * risk_per_unit)
                    partial_size = self._virtual_pos_size * 0.25
                    partial_pnl = (target_1_5r - pos.entry_price) * partial_size
                    self._virtual_partial_pnl += partial_pnl
                    self.connector.update_dry_run_balance(partial_pnl)
                    self._virtual_pos_size *= 0.75
                    logger.info("Virtual LONG: Partial 1 (25%%) at 1.5R +$%.2f", partial_pnl)

                # --- Partial 2: 35% at 2.0R ---
                if not self._virtual_partial_2_taken and high >= target_2_0r:
                    self._virtual_partial_2_taken = True
                    self._virtual_partial_taken = True
                    current_sl = pos.entry_price  # Tighten to exact entry
                    partial_size = self._virtual_pos_size * (0.35 / 0.75)
                    partial_pnl = (target_2_0r - pos.entry_price) * partial_size
                    self._virtual_partial_pnl += partial_pnl
                    self.connector.update_dry_run_balance(partial_pnl)
                    self._virtual_pos_size -= partial_size
                    logger.info("Virtual LONG: Partial 2 (35%%) at 2.0R +$%.2f", partial_pnl)

            else:  # SHORT
                # --- SL FIRST ---
                if high >= current_sl:
                    remaining_pnl = (pos.entry_price - current_sl) * self._virtual_pos_size
                    self.connector.update_dry_run_balance(remaining_pnl)
                    outcome = "BREAK-EVEN" if self._virtual_break_even_moved else "STOP LOSS"
                    total_pnl = self._virtual_partial_pnl + remaining_pnl
                    self._print_virtual_close(outcome, total_pnl, pos)
                    self._reset_virtual_position()
                    return

                # --- TP ---
                if low <= pos.take_profit:
                    remaining_pnl = (pos.entry_price - pos.take_profit) * self._virtual_pos_size
                    self.connector.update_dry_run_balance(remaining_pnl)
                    total_pnl = self._virtual_partial_pnl + remaining_pnl
                    self._print_virtual_close("TAKE PROFIT", total_pnl, pos)
                    self._reset_virtual_position()
                    return

                # --- Partial 1: 25% at 1.5R ---
                if not self._virtual_partial_1_taken and low <= target_1_5r:
                    self._virtual_partial_1_taken = True
                    self._virtual_break_even_moved = True
                    current_sl = pos.entry_price - (0.1 * risk_per_unit)
                    partial_size = self._virtual_pos_size * 0.25
                    partial_pnl = (pos.entry_price - target_1_5r) * partial_size
                    self._virtual_partial_pnl += partial_pnl
                    self.connector.update_dry_run_balance(partial_pnl)
                    self._virtual_pos_size *= 0.75
                    logger.info("Virtual SHORT: Partial 1 (25%%) at 1.5R +$%.2f", partial_pnl)

                # --- Partial 2: 35% at 2.0R ---
                if not self._virtual_partial_2_taken and low <= target_2_0r:
                    self._virtual_partial_2_taken = True
                    self._virtual_partial_taken = True
                    current_sl = pos.entry_price  # Tighten to exact entry
                    partial_size = self._virtual_pos_size * (0.35 / 0.75)
                    partial_pnl = (pos.entry_price - target_2_0r) * partial_size
                    self._virtual_partial_pnl += partial_pnl
                    self.connector.update_dry_run_balance(partial_pnl)
                    self._virtual_pos_size -= partial_size
                    logger.info("Virtual SHORT: Partial 2 (35%%) at 2.0R +$%.2f", partial_pnl)

        # --- Smart Time Exit (matches backtest engine) ---
        if len(recent_df) >= self._time_exit_candles:
            exit_price = df["Close"].iloc[-1]
            # Near-breakeven: if within 0.3R of entry after BE moved, exit at entry
            if self._virtual_break_even_moved:
                if pos.direction == Direction.LONG:
                    unrealized_r = (exit_price - pos.entry_price) / risk_per_unit
                else:
                    unrealized_r = (pos.entry_price - exit_price) / risk_per_unit
                if abs(unrealized_r) < 0.3:
                    exit_price = pos.entry_price
            if pos.direction == Direction.LONG:
                remaining_pnl = (exit_price - pos.entry_price) * self._virtual_pos_size
            else:
                remaining_pnl = (pos.entry_price - exit_price) * self._virtual_pos_size
            self.connector.update_dry_run_balance(remaining_pnl)
            total_pnl = self._virtual_partial_pnl + remaining_pnl
            self._print_virtual_close("TIME EXIT", total_pnl, pos)
            self._reset_virtual_position()

    def _print_virtual_close(self, outcome: str, total_pnl: float, pos=None):
        """Print virtual trade closure details and write to CSV journal."""
        if pos is None:
            pos = self._virtual_position
        pnl_sign = "+" if total_pnl >= 0 else ""
        new_balance = self.connector._dry_run_balance
        now = datetime.now(timezone.utc)

        print()
        print("  " + "=" * 50)
        print(f"  [{self.symbol}] [DRY RUN] TRADE CLOSED — {outcome}!")
        print(f"  [{self.symbol}] Direction : {pos.direction.value}")
        print(f"  [{self.symbol}] Entry     : {pos.entry_price:.2f}")
        print(f"  [{self.symbol}] SL        : {pos.stop_loss:.2f} | TP: {pos.take_profit:.2f}")
        print(f"  [{self.symbol}] R:R       : {pos.rr_ratio:.1f} | Confidence: {getattr(pos, 'confidence', 0):.2f}")
        print(f"  [{self.symbol}] P&L       : {pnl_sign}${total_pnl:.2f} | Balance: ${new_balance:.2f}")
        logger.info("Virtual %s closed: %s | %s$%.2f | Bal: $%.2f",
                    pos.direction.value, outcome, pnl_sign, total_pnl, new_balance)
        print("  " + "=" * 50)

        # Write to CSV journal
        self._write_trade_journal(pos, outcome, total_pnl, new_balance, now)

        # Update risk manager and persist state — per-pair cooldown
        _pair_key = f"{self.symbol}_{self.resolution}"
        self.risk_mgr.on_trade_close(total_pnl, now, pair=_pair_key)
        self.risk_mgr.balance = new_balance
        self.trade_store.save(self.risk_mgr.to_dict())

        # Update direction-streak filter
        _dir = pos.direction.value
        if total_pnl < 0:
            self._dir_streak[_dir] = self._dir_streak.get(_dir, 0) + 1
            if self._dir_streak[_dir] >= self._dir_ban_threshold:
                logger.warning("Direction ban activated: %d consecutive %s losses on %s",
                               self._dir_streak[_dir], _dir, self.symbol)
        else:
            self._dir_streak[_dir] = 0  # Any non-loss resets the streak

        # Track daily stats
        self._daily_trades += 1
        self._daily_pnl += total_pnl
        if total_pnl > 0:
            self._daily_wins += 1
        elif total_pnl < 0:
            self._daily_losses += 1

        # Telegram: trade close notification
        _tg.send_trade_close(self.symbol, self.resolution, outcome, total_pnl, new_balance, pos)

    def _write_trade_journal(self, pos, outcome: str, pnl: float, balance: float, closed_at: datetime):
        """Append completed trade to CSV journal for performance tracking."""
        import csv, os
        journal_path = f"logs/trade_journal_{self.symbol}_{self.resolution}.csv"
        file_exists = os.path.isfile(journal_path)
        row = {
            "closed_at": closed_at.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "symbol": self.symbol,
            "resolution": self.resolution,
            "direction": pos.direction.value,
            "entry_price": round(pos.entry_price, 4),
            "stop_loss": round(pos.stop_loss, 4),
            "take_profit": round(pos.take_profit, 4),
            "rr_ratio": round(pos.rr_ratio, 2),
            "confidence": round(getattr(pos, 'confidence', 0), 3),
            "htf_bias": pos.htf_bias,
            "session": pos.session,
            "entry_type": pos.entry_type,
            "outcome": outcome,
            "pnl_dollar": round(pnl, 2),
            "balance_after": round(balance, 2),
            "partial_pnl": round(self._virtual_partial_pnl, 2),
        }
        with open(journal_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
        logger.info("Trade journal updated: %s", journal_path)

    def _reset_virtual_position(self):
        """Reset virtual position state."""
        self._virtual_position = None
        self._virtual_entry_time = None
        self._virtual_pos_size = 0.0
        self._virtual_partial_pnl = 0.0
        self._virtual_break_even_moved = False
        self._virtual_partial_1_taken = False
        self._virtual_partial_2_taken = False
        self._virtual_partial_taken = False

    def _notify(self, message: str):
        """Send Telegram notification (silently skipped if not configured)."""
        _tg.send(message)

    def _maybe_send_daily_summary(self):
        """Send end-of-day summary if the UTC date has rolled over."""
        today = datetime.now(timezone.utc).date()
        if self._last_summary_date and self._last_summary_date != today:
            _tg.send_daily_summary(self.symbol, self.resolution, {
                "trades": self._daily_trades,
                "wins": self._daily_wins,
                "losses": self._daily_losses,
                "daily_pnl": self._daily_pnl,
                "balance": self.connector._dry_run_balance,
            })
            self._daily_trades = 0
            self._daily_wins = 0
            self._daily_losses = 0
            self._daily_pnl = 0.0
            # Reset direction streaks on new day — fresh start each session
            self._dir_streak = {}
            logger.info("New day: direction streaks reset for %s %s", self.symbol, self.resolution)
        self._last_summary_date = today

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
        print(f"  [{self.symbol}] [DRY RUN] SIGNAL DETECTED")
        print(f"  [{self.symbol}] Direction  : {signal.direction.value}")
        print(f"  Entry      : {signal.entry_price:.2f}")
        print(f"  Stop Loss  : {signal.stop_loss:.2f}")
        print(f"  Take Profit: {signal.take_profit:.2f}")
        print(f"  R:R Ratio  : {signal.rr_ratio:.1f}")
        if signal.divergence:
            print(f"  Divergence : {signal.divergence.divergence_type}")
        print("  " + "=" * 50)

        # Save as active virtual position
        self._virtual_position = signal
        self._virtual_entry_time = datetime.now(timezone.utc)

        # Telegram: new signal notification
        _tg.send_signal(self.symbol, self.resolution, signal)

        # Calculate initial position size
        wallet_balance = self.connector._dry_run_balance
        risk_amount = wallet_balance * (self.config.risk_per_trade_pct / 100.0)
        risk_per_contract = abs(signal.entry_price - signal.stop_loss)
        if risk_per_contract > 0:
            self._virtual_pos_size = max(1, int(risk_amount / risk_per_contract))
        else:
            self._virtual_pos_size = 1

    def _execute_signal(self, signal: TradeSignal):
        """
        Place 3 bracket orders matching the backtest's tiered partial-exit model:
          Order 1 — 25% of position, TP at 1.5R  (first partial)
          Order 2 — 35% of position, TP at 2.0R  (second partial)
          Order 3 — 40% of position, TP at range target (let it run)

        All three share the same initial SL.  After Order 1 fills (price hits 1.5R),
        _manage_live_positions() detects the 1.5R level and moves the SLs of Orders
        2 & 3 to break-even, then trails Order 3 using swing low/high + ATR.
        """
        logger.info("EXECUTING LIVE TRADE")

        side = "buy" if signal.direction == Direction.LONG else "sell"

        # Get live balance
        wallet = self.connector.get_balance()
        if wallet is None:
            logger.error("Cannot get wallet balance -- aborting trade")
            return

        available = float(wallet.get("available_balance", 0))
        logger.info("Available balance: %.2f", available)

        risk_amount = available * (self.config.risk_per_trade_pct / 100.0)
        risk_per_contract = abs(signal.entry_price - signal.stop_loss)
        if risk_per_contract <= 0:
            logger.error("Invalid risk per contract -- aborting")
            return

        # Confidence-based size scaling (matches backtest)
        if getattr(self.config, "use_confidence_sizing", True):
            min_m = getattr(self.config, "min_confidence_multiplier", 0.5)
            max_m = getattr(self.config, "max_confidence_multiplier", 1.2)
            conf_mult = min_m + (getattr(signal, "confidence", 0.5) * (max_m - min_m))
            risk_amount *= conf_mult

        total_size = max(3, int(risk_amount / risk_per_contract))  # At least 3 for 3-way split

        # 25 / 35 / 40 split — must sum to total_size
        size_1 = max(1, round(total_size * 0.25))
        size_2 = max(1, round(total_size * 0.35))
        size_3 = max(1, total_size - size_1 - size_2)

        # TP levels: 1.5R, 2.0R, range target
        r = risk_per_contract
        if signal.direction == Direction.LONG:
            tp1 = signal.entry_price + (1.5 * r)
            tp2 = signal.entry_price + (2.0 * r)
            tp3 = signal.take_profit  # range high / liquidity target
        else:
            tp1 = signal.entry_price - (1.5 * r)
            tp2 = signal.entry_price - (2.0 * r)
            tp3 = signal.take_profit

        logger.info("Live trade: total=%d contracts | 25%%=%d @TP1=%.4f | 35%%=%d @TP2=%.4f | 40%%=%d @TP3=%.4f",
                    total_size, size_1, tp1, size_2, tp2, size_3, tp3)

        errors = []
        try:
            self.connector.place_bracket_order(
                product_id=self.product_id, size=size_1, side=side,
                stop_loss_price=signal.stop_loss, take_profit_price=tp1,
                tick_size=self.tick_size,
            )
        except Exception as e:
            errors.append(f"Order1(25%@1.5R): {e}")

        try:
            self.connector.place_bracket_order(
                product_id=self.product_id, size=size_2, side=side,
                stop_loss_price=signal.stop_loss, take_profit_price=tp2,
                tick_size=self.tick_size,
            )
        except Exception as e:
            errors.append(f"Order2(35%@2.0R): {e}")

        try:
            self.connector.place_bracket_order(
                product_id=self.product_id, size=size_3, side=side,
                stop_loss_price=signal.stop_loss, take_profit_price=tp3,
                tick_size=self.tick_size,
            )
        except Exception as e:
            errors.append(f"Order3(40%@TP): {e}")

        print()
        print("  " + "=" * 55)
        if errors:
            print(f"  [{self.symbol}] [LIVE] PARTIAL ORDER FAILURE")
            for err in errors:
                print(f"  [ERROR] {err}")
        else:
            print(f"  [{self.symbol}] [LIVE] 3-TIER ORDERS PLACED")
        print(f"  Direction  : {signal.direction.value}")
        print(f"  Total Size : {total_size} contracts")
        print(f"  Order 1 (25%) : {size_1}x | TP: {tp1:.4f}  [1.5R partial]")
        print(f"  Order 2 (35%) : {size_2}x | TP: {tp2:.4f}  [2.0R partial]")
        print(f"  Order 3 (40%) : {size_3}x | TP: {tp3:.4f}  [run to target]")
        print(f"  Stop Loss      : {signal.stop_loss:.4f}  → B/E at 1.5R → trailing")
        print("  " + "=" * 55)

    def run_loop(self):
        """
        Run the strategy in a continuous loop.

        Fetches candles, generates signals, and optionally executes trades
        every `loop_interval` seconds. Press Ctrl+C to stop.
        """
        import signal
        def handle_sigterm(signum, frame):
            raise KeyboardInterrupt()

        try:
            signal.signal(signal.SIGTERM, handle_sigterm)
        except (ValueError, AttributeError):
            pass

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

                balance_str = ""
                if self.dry_run:
                    balance_str = f" [Bal: ${self.connector._dry_run_balance:.2f}]"

                vpos_str = ""
                if self._virtual_position:
                    vpos_str = f" [POS: {self._virtual_position.direction.value}]"

                print(f"\n  [{self.symbol}] [Cycle {cycle}]{balance_str}{vpos_str} {datetime.now().strftime('%H:%M:%S')} -- Scanning...", end="")

                signal = self.run_once()
                self._maybe_send_daily_summary()

                if signal:
                    print(f" SIGNAL: {signal.direction.value} @ {signal.entry_price:.2f}")
                else:
                    print(" No signal")

                # Sleep until next cycle
                logger.info("Sleeping %d seconds...", self.loop_interval)
                time.sleep(self.loop_interval)

            except KeyboardInterrupt:
                print(f"\n\n  [{self.symbol}] [STOPPED] LiveTrader stopped by user.")
                if self.dry_run:
                    print(f"  [{self.symbol}] Final Session Balance [{mode_label}]: ${self.connector._dry_run_balance:.2f}")
                logger.info("LiveTrader stopped by KeyboardInterrupt")
                break
            except Exception as e:
                logger.error("Unexpected error in cycle %d: %s", cycle, e)
                print(f"\n  [{self.symbol}] [FATAL ERROR] {e}")
                print(f"  [{self.symbol}] Exiting with status code 1 so the host (Render) can auto-restart the service.")
                sys.exit(1)


def setup_logging(log_file: str = "logs/trading_bot.log"):
    """Configure logging for the trading bot."""
    import pathlib
    pathlib.Path(log_file).parent.mkdir(parents=True, exist_ok=True)
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
