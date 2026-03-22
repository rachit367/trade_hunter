"""Tests for RiskManager — daily loss limits, concurrent trades, cooldown, day rollover."""

import pytest
from datetime import datetime, timedelta, timezone

from trading_bot.strategy.risk_manager import RiskManager, RiskConfig


@pytest.fixture
def rm():
    """Default risk manager with $10k balance."""
    return RiskManager(config=RiskConfig(), initial_balance=10000.0)


class TestCanTrade:
    def test_allows_trade_initially(self, rm):
        allowed, reason = rm.can_trade(datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc))
        assert allowed is True
        assert reason == "OK"

    def test_blocks_after_daily_loss_limit(self, rm):
        """3% of $10k = $300 max daily loss."""
        rm.on_new_day(datetime(2025, 1, 1).date(), 10000.0)
        rm.on_trade_close(-150.0, datetime(2025, 1, 1, 10, 0, tzinfo=timezone.utc))
        rm.on_trade_close(-160.0, datetime(2025, 1, 1, 11, 0, tzinfo=timezone.utc))

        allowed, reason = rm.can_trade(datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc))
        assert allowed is False
        assert "Daily loss limit" in reason

    def test_blocks_max_concurrent_trades(self, rm):
        rm.on_trade_open()
        rm.on_trade_open()
        allowed, reason = rm.can_trade(datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc))
        assert allowed is False
        assert "concurrent" in reason.lower()

    def test_allows_after_trade_close(self, rm):
        rm.on_trade_open()
        rm.on_trade_open()
        rm.on_trade_close(50.0)
        allowed, _ = rm.can_trade(datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc))
        assert allowed is True

    def test_cooldown_after_consecutive_losses(self, rm):
        t = datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)
        rm.on_new_day(t.date(), 10000.0)
        # 3 consecutive losses (default cooldown threshold)
        for i in range(3):
            rm.on_trade_open()
            rm.on_trade_close(-10.0, t + timedelta(minutes=i * 5))

        # Should be in cooldown
        allowed, reason = rm.can_trade(t + timedelta(minutes=15))
        assert allowed is False
        assert "Cooldown" in reason

    def test_cooldown_expires(self, rm):
        t = datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)
        rm.on_new_day(t.date(), 10000.0)
        for i in range(3):
            rm.on_trade_open()
            rm.on_trade_close(-10.0, t + timedelta(minutes=i))

        # After cooldown (30 min default)
        allowed, _ = rm.can_trade(t + timedelta(minutes=35))
        assert allowed is True


class TestDayRollover:
    def test_resets_daily_pnl(self, rm):
        day1 = datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)
        rm.on_new_day(day1.date(), 10000.0)
        rm.on_trade_close(-200.0, day1)

        # New day
        day2 = datetime(2025, 1, 2, 12, 0, tzinfo=timezone.utc)
        rm.can_trade(day2)  # Triggers day rollover

        assert rm.daily_pnl == 0.0
        assert rm.total_trades_today == 0

    def test_consecutive_losses_carry_across_days(self, rm):
        """Consecutive losses should NOT reset on day boundary."""
        t = datetime(2025, 1, 1, 23, 0, tzinfo=timezone.utc)
        rm.on_new_day(t.date(), 10000.0)
        rm.on_trade_open()
        rm.on_trade_close(-10.0, t)
        rm.on_trade_open()
        rm.on_trade_close(-10.0, t + timedelta(minutes=5))

        # Day rolls over
        t2 = datetime(2025, 1, 2, 1, 0, tzinfo=timezone.utc)
        rm.can_trade(t2)

        assert rm.consecutive_losses == 2  # Not reset


class TestPositionSizing:
    def test_basic_sizing(self, rm):
        size = rm.get_position_size(entry=100.0, sl=99.0)
        # 1% of $10k = $100 risk, $1 per unit = 100 units
        assert size == pytest.approx(100.0)

    def test_zero_risk_returns_zero(self, rm):
        assert rm.get_position_size(entry=100.0, sl=100.0) == 0.0


class TestSerialization:
    def test_round_trip(self, rm):
        rm.on_new_day(datetime(2025, 1, 1).date(), 10000.0)
        rm.on_trade_open()
        rm.on_trade_close(-50.0, datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc))

        data = rm.to_dict()

        rm2 = RiskManager(initial_balance=10000.0)
        rm2.from_dict(data)

        assert rm2.balance == rm.balance
        assert rm2.daily_pnl == rm.daily_pnl
        assert rm2.consecutive_losses == rm.consecutive_losses
        assert rm2.open_trades == rm.open_trades
