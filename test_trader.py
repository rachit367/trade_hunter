import time
import sys
sys.path.insert(0, '.')
from trading_bot.exchange.live_trader import LiveTrader, setup_logging

setup_logging('test.log')
trader = LiveTrader(symbol='ETHUSD', dry_run=True, lookback_hours=24, loop_interval=10)

try:
    print('Running one cycle...')
    trader.run_once()
    print('Raising KeyboardInterrupt now to simulate Ctrl+C...')
    raise KeyboardInterrupt
except KeyboardInterrupt:
    print(f'\n\n  [{trader.symbol}] [STOPPED] LiveTrader stopped by user.')
    if trader.dry_run:
        print(f'  [{trader.symbol}] Final Session Balance [DRY RUN]: ${trader.connector._dry_run_balance:.2f}')
