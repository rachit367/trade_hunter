"""
Risk Manager — Centralized risk management for both backtest and live trading.

Enforces:
  - Max risk per trade (default 1%)
  - Max daily drawdown (default 3%) — GLOBAL across all pairs
  - Max concurrent open trades (default 2) — GLOBAL across all pairs
  - Cooldown after consecutive losses — PER-PAIR by default (pair= param)
    so one pair's losing streak doesn't block the other pair
  - Automatic daily counter reset
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger("risk_manager")


@dataclass
class RiskConfig:
    """Risk management parameters."""
    max_risk_per_trade_pct: float = 1.0       # Max 1% of balance per trade
    max_daily_loss_pct: float = 3.0           # Max 3% daily drawdown (global)
    max_concurrent_trades: int = 2             # Max open positions at once (global)
    consecutive_loss_cooldown: int = 3         # After N consecutive losses
    cooldown_minutes: int = 30                 # Pause for 30 minutes
    per_pair_cooldown: bool = True             # If True, cooldown is isolated per pair


class RiskManager:
    """
    Centralized risk management for both backtest and live trading.

    Global limits (shared across all pairs):
      - Daily loss limit
      - Max concurrent open trades

    Per-pair limits (isolated per symbol+resolution key):
      - Consecutive loss cooldown (when per_pair_cooldown=True)

    This means a losing streak on ETHUSD won't block SOLUSD signals.
    Pass pair="ETHUSD_5m" (or any string key) to can_trade / on_trade_close
    to enable per-pair tracking.
    """

    def __init__(self, config: RiskConfig = None, initial_balance: float = 10000.0):
        self.config = config or RiskConfig()
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.daily_start_balance = initial_balance
        self.daily_pnl = 0.0
        self.open_trades = 0
        self.current_date = None
        self.total_trades_today = 0

        # Global (legacy / single-pair) streak tracking
        self.consecutive_losses = 0
        self.cooldown_until: Optional[datetime] = None

        # Per-pair streak tracking: { pair_key: {"streak": int, "cooldown": datetime|None} }
        self._pair_state: Dict[str, dict] = {}

    def _get_pair_state(self, pair: str) -> dict:
        """Get or create per-pair streak/cooldown state."""
        if pair not in self._pair_state:
            self._pair_state[pair] = {"streak": 0, "cooldown": None}
        return self._pair_state[pair]

    def _check_cooldown(self, cooldown_until: Optional[datetime],
                        streak: int, current_time: datetime) -> Tuple[bool, str]:
        """Return (blocked, reason) for a cooldown entry."""
        if cooldown_until is None:
            return False, ""

        # Normalize timezone
        cd = cooldown_until
        ct = current_time
        if cd.tzinfo is not None and ct.tzinfo is None:
            ct = ct.replace(tzinfo=timezone.utc)
        elif cd.tzinfo is None and ct.tzinfo is not None:
            cd = cd.replace(tzinfo=timezone.utc)

        if ct < cd:
            remaining = (cd - ct).total_seconds() / 60.0
            reason = (f"Cooldown active: {streak} consecutive losses. "
                      f"Resuming in {remaining:.0f} min")
            return True, reason
        return False, ""

    def can_trade(self, current_time: datetime = None,
                  pair: str = None) -> Tuple[bool, str]:
        """
        Check all risk gates. Returns (allowed, reason).

        Parameters
        ----------
        current_time : datetime
            Current timestamp (UTC). Defaults to now.
        pair : str, optional
            Pair key (e.g. "ETHUSD_5m"). When provided and per_pair_cooldown
            is enabled, cooldown is checked per-pair instead of globally.

        Checks in order:
        1. Daily loss limit (global)
        2. Max concurrent trades (global)
        3. Cooldown after consecutive losses (per-pair or global)
        """
        if current_time is None:
            current_time = datetime.now(timezone.utc)

        # Check for day rollover
        current_date = current_time.date() if hasattr(current_time, 'date') else None
        if current_date and current_date != self.current_date:
            self.on_new_day(current_date, self.balance)

        # 1. Daily loss limit (global)
        max_daily_loss = self.daily_start_balance * (self.config.max_daily_loss_pct / 100.0)
        if self.daily_pnl <= -max_daily_loss:
            reason = (f"Daily loss limit hit: ${self.daily_pnl:.2f} "
                      f"(max -${max_daily_loss:.2f})")
            logger.warning("RISK GATE: %s", reason)
            return False, reason

        # 2. Max concurrent trades (global)
        if self.open_trades >= self.config.max_concurrent_trades:
            reason = (f"Max concurrent trades reached: {self.open_trades}/"
                      f"{self.config.max_concurrent_trades}")
            logger.info("RISK GATE: %s", reason)
            return False, reason

        # 3. Cooldown after consecutive losses
        if pair and self.config.per_pair_cooldown:
            # Per-pair cooldown
            ps = self._get_pair_state(pair)
            blocked, reason = self._check_cooldown(
                ps["cooldown"], ps["streak"], current_time
            )
            if blocked:
                # Cooldown still active
                logger.info("RISK GATE [%s]: %s", pair, reason)
                return False, reason
            elif ps["cooldown"] is not None:
                ps["cooldown"] = None  # Expired — clear it
        else:
            # Global (legacy) cooldown
            blocked, reason = self._check_cooldown(
                self.cooldown_until, self.consecutive_losses, current_time
            )
            if blocked:
                logger.info("RISK GATE: %s", reason)
                return False, reason
            elif self.cooldown_until is not None:
                self.cooldown_until = None  # Expired — clear it

        return True, "OK"

    def on_trade_open(self):
        """Increment open trade counter."""
        self.open_trades += 1
        self.total_trades_today += 1
        logger.info("Trade opened. Open positions: %d/%d",
                     self.open_trades, self.config.max_concurrent_trades)

    def on_trade_close(self, pnl: float, close_time: datetime = None,
                       pair: str = None):
        """
        Update state after a trade closes.

        Parameters
        ----------
        pnl : float
            Dollar P&L of the closed trade.
        close_time : datetime
            Time the trade was closed.
        pair : str, optional
            Pair key (e.g. "ETHUSD_5m"). When provided and per_pair_cooldown
            is enabled, streak is tracked per-pair.
        """
        if close_time is None:
            close_time = datetime.now(timezone.utc)

        self.open_trades = max(0, self.open_trades - 1)
        self.daily_pnl += pnl
        self.balance += pnl

        if pnl < 0:
            # Update streak — per-pair or global
            if pair and self.config.per_pair_cooldown:
                ps = self._get_pair_state(pair)
                ps["streak"] += 1
                streak = ps["streak"]
                logger.info("Loss recorded [%s]: $%.2f (streak: %d)", pair, pnl, streak)

                if streak >= self.config.consecutive_loss_cooldown:
                    ps["cooldown"] = close_time + timedelta(
                        minutes=self.config.cooldown_minutes
                    )
                    logger.warning(
                        "COOLDOWN ACTIVATED [%s]: %d consecutive losses. "
                        "Pausing until %s UTC",
                        pair, streak,
                        ps["cooldown"].strftime("%H:%M:%S")
                    )
                    print(f"COOLDOWN ACTIVATED [{pair}]: {streak} consecutive losses. "
                          f"Pausing until {ps['cooldown'].strftime('%H:%M:%S')} UTC")
            else:
                self.consecutive_losses += 1
                logger.info("Loss recorded: $%.2f (consecutive: %d)",
                            pnl, self.consecutive_losses)

                if self.consecutive_losses >= self.config.consecutive_loss_cooldown:
                    self.cooldown_until = close_time + timedelta(
                        minutes=self.config.cooldown_minutes
                    )
                    logger.warning(
                        "COOLDOWN ACTIVATED: %d consecutive losses. "
                        "Pausing until %s",
                        self.consecutive_losses,
                        self.cooldown_until.strftime("%H:%M:%S UTC")
                    )
                    print(f"COOLDOWN ACTIVATED: {self.consecutive_losses} consecutive losses. "
                          f"Pausing until {self.cooldown_until.strftime('%H:%M:%S')} UTC")
        else:
            # Win — reset streak
            if pair and self.config.per_pair_cooldown:
                ps = self._get_pair_state(pair)
                ps["streak"] = 0
                logger.info("Win recorded [%s]: +$%.2f", pair, pnl)
            else:
                self.consecutive_losses = 0
                logger.info("Win recorded: +$%.2f", pnl)

        logger.info("Daily P&L: $%.2f | Balance: $%.2f | Open: %d",
                     self.daily_pnl, self.balance, self.open_trades)

    def on_new_day(self, date, balance: float):
        """Reset daily counters for a new trading day."""
        self.current_date = date
        self.daily_start_balance = balance
        self.daily_pnl = 0.0
        self.total_trades_today = 0
        # Reset global cooldown on new day
        self.cooldown_until = None
        # Reset per-pair cooldowns on new day (but NOT streaks — carry across days)
        for ps in self._pair_state.values():
            ps["cooldown"] = None
        logger.info("New trading day: %s | Start balance: $%.2f", date, balance)

    def get_position_size(self, entry: float, sl: float) -> float:
        """
        Calculate position size respecting max risk per trade.

        Returns 0 if the trade would exceed risk limits.
        """
        risk_per_unit = abs(entry - sl)
        if risk_per_unit <= 0:
            return 0.0

        max_risk = self.balance * (self.config.max_risk_per_trade_pct / 100.0)
        return max_risk / risk_per_unit

    def to_dict(self) -> dict:
        """Serialize state for persistence."""
        # Serialize per-pair state
        pair_state_serial = {}
        for key, ps in self._pair_state.items():
            pair_state_serial[key] = {
                "streak": ps["streak"],
                "cooldown": ps["cooldown"].isoformat() if ps["cooldown"] else None,
            }

        return {
            "balance": self.balance,
            "daily_start_balance": self.daily_start_balance,
            "daily_pnl": self.daily_pnl,
            "open_trades": self.open_trades,
            "consecutive_losses": self.consecutive_losses,
            "cooldown_until": self.cooldown_until.isoformat() if self.cooldown_until else None,
            "current_date": self.current_date.isoformat() if self.current_date else None,
            "total_trades_today": self.total_trades_today,
            "pair_state": pair_state_serial,
        }

    def from_dict(self, data: dict):
        """Restore state from persistence."""
        self.balance = data.get("balance", self.initial_balance)
        self.daily_start_balance = data.get("daily_start_balance", self.initial_balance)
        self.daily_pnl = data.get("daily_pnl", 0.0)
        self.open_trades = data.get("open_trades", 0)
        self.consecutive_losses = data.get("consecutive_losses", 0)
        self.total_trades_today = data.get("total_trades_today", 0)

        cooldown_str = data.get("cooldown_until")
        self.cooldown_until = datetime.fromisoformat(cooldown_str) if cooldown_str else None

        date_str = data.get("current_date")
        if date_str:
            from datetime import date as date_cls
            self.current_date = date_cls.fromisoformat(date_str)
        else:
            self.current_date = None

        # Restore per-pair state
        self._pair_state = {}
        for key, ps in data.get("pair_state", {}).items():
            cd_str = ps.get("cooldown")
            self._pair_state[key] = {
                "streak": ps.get("streak", 0),
                "cooldown": datetime.fromisoformat(cd_str) if cd_str else None,
            }
