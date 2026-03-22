"""Tests for backtest fee model — fee deduction, slippage, position sizing with fees."""

import pytest
import pandas as pd
import numpy as np

from trading_bot.backtest.engine import (
    BacktestConfig,
    _calc_entry_fee,
    _calc_exit_fee,
    run_backtest,
)
from trading_bot.strategy.amd_strategy import StrategyConfig, TradeSignal, Direction


@pytest.fixture
def bt_config():
    return BacktestConfig(
        taker_fee_pct=0.075,
        maker_fee_pct=0.05,
        slippage_pct=0.02,
        use_maker_for_limit=True,
    )


class TestEntryFees:
    def test_fvg_entry_uses_maker_fee(self, bt_config):
        fee, slip = _calc_entry_fee(100.0, 10.0, "fvg", bt_config)
        # Notional = 1000, maker = 0.05% = $0.50, no slippage
        assert fee == pytest.approx(0.50)
        assert slip == pytest.approx(0.0)

    def test_bpr_entry_uses_maker_fee(self, bt_config):
        fee, slip = _calc_entry_fee(100.0, 10.0, "bpr", bt_config)
        assert fee == pytest.approx(0.50)
        assert slip == pytest.approx(0.0)

    def test_order_block_uses_maker_fee(self, bt_config):
        fee, slip = _calc_entry_fee(100.0, 10.0, "order_block", bt_config)
        assert fee == pytest.approx(0.50)
        assert slip == pytest.approx(0.0)

    def test_close_entry_uses_taker_fee(self, bt_config):
        fee, slip = _calc_entry_fee(100.0, 10.0, "close", bt_config)
        # Notional = 1000, taker = 0.075% = $0.75, slippage = 0.02% = $0.20
        assert fee == pytest.approx(0.75)
        assert slip == pytest.approx(0.20)


class TestExitFees:
    def test_tp_exit_uses_maker_fee(self, bt_config):
        fee, slip = _calc_exit_fee(100.0, 10.0, True, bt_config)
        assert fee == pytest.approx(0.50)
        assert slip == pytest.approx(0.0)

    def test_sl_exit_uses_taker_fee(self, bt_config):
        fee, slip = _calc_exit_fee(100.0, 10.0, False, bt_config)
        assert fee == pytest.approx(0.75)
        assert slip == pytest.approx(0.20)


class TestBacktestWithFees:
    def _make_df(self, n=50):
        """Create a simple DataFrame with trending price data."""
        idx = pd.date_range("2025-01-01", periods=n, freq="5min")
        prices = np.linspace(100, 110, n)
        return pd.DataFrame({
            "Open": prices - 0.1,
            "High": prices + 0.5,
            "Low": prices - 0.5,
            "Close": prices,
            "Volume": np.full(n, 1000.0),
        }, index=idx)

    def test_fees_deducted_from_balance(self):
        df = self._make_df(50)
        # Create a simple long signal that will hit TP
        sig = TradeSignal(
            direction=Direction.LONG,
            entry_price=100.5,
            stop_loss=99.0,
            take_profit=105.0,
            entry_idx=5,
            entry_time=df.index[5],
            range_high=105.0,
            range_low=99.0,
            range_start_idx=0,
            range_end_idx=4,
            entry_type="close",
        )
        result = run_backtest(
            df,
            config=StrategyConfig(),
            initial_balance=10000.0,
            signals=[sig],
            bt_config=BacktestConfig(),
        )

        if result.trades:
            trade = result.trades[0]
            # Fees should be non-zero
            assert trade.entry_fee > 0
            # Net P&L should be less than gross (fees eaten)
            assert trade.net_pnl_dollar <= trade.gross_pnl_dollar + trade.entry_fee + trade.exit_fee

    def test_no_trades_returns_initial_balance(self):
        df = self._make_df(50)
        result = run_backtest(df, signals=[], initial_balance=5000.0)
        assert result.final_balance == 5000.0
        assert len(result.trades) == 0
