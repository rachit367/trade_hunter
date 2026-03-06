import sys
import pandas as pd
from datetime import datetime
sys.path.insert(0, ".")
from trading_bot.data.loader import fetch_delta
from trading_bot.strategy.amd_strategy import generate_signals, StrategyConfig
from trading_bot.backtest.engine import run_backtest

def main():
    print("Fetching data from Delta Exchange for BTCUSD...", flush=True)
    df = fetch_delta("BTCUSD", "5m", 24)
    config = StrategyConfig()
    signals = generate_signals(df, config)
    result = run_backtest(df, config, initial_balance=10000.0, signals=signals)
    
    print("\n" + "="*50)
    print("                 SIGNAL DATA")
    print("="*50)
    if not signals:
        print("No signals found in the last 24h.")
        return
        
    for i, sig in enumerate(signals, 1):
        print(f"\n--- Signal {i} ---")
        sig_time = sig.entry_time
        print(f"Time: {sig_time} | Type: {sig.direction} | @ {sig.entry_price}")
        
        # Get the row from the dataframe
        try:
            # If sig_time is not exactly in index, we might need nearest
            idx = df.index.get_loc(sig_time)
            candle = df.iloc[idx]
            print("\nCandle OHLCV generating the signal:")
            print(f"Open:   {candle['Open']:.2f}")
            print(f"High:   {candle['High']:.2f}")
            print(f"Low:    {candle['Low']:.2f}")
            print(f"Close:  {candle['Close']:.2f}")
            print(f"Volume: {candle['Volume']:.2f}")
        except KeyError:
            print(f"Could not find exact timestamp {sig_time} in dataframe index.")
            
    print("\n" + "="*50)
    print("              FINAL BALANCE")
    print("="*50)
    
    current_balance = 10000.0
    for i, trade in enumerate(result.trades, 1):
        current_balance += trade.pnl_dollar
        print(f"\nTrade {i}: {trade.signal.direction.value} from {trade.signal.entry_price} -> {trade.exit_price}")
        print(f"Exit Time: {trade.exit_time}")
        print(f"PnL:       ${trade.pnl_dollar:.2f}")
        print(f"Balance after trade:  ${current_balance:.2f}")
        
    print(f"\nTotal Final Balance: ${result.final_balance:.2f}")
    
if __name__ == "__main__":
    main()
