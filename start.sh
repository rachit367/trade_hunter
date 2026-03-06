#!/usr/bin/env bash

# Auto-restart wrapper script for Render
# This ensures that if the bot crashes, it waits a few seconds and restarts.

echo "Starting Trade Hunter Bot..."

while true; do
    # Read configuration from env vars (set in render.yaml or dashboard)
    MODE=${BOT_MODE:-live}
    SYMBOL=${TRADE_SYMBOL:-BTCUSD}
    INTERVAL=${TRADE_INTERVAL:-300}

    echo "[$(date)] Launching bot in $MODE mode for $SYMBOL..."
    
    if [ "$MODE" = "live" ]; then
        # Real trading
        python main.py --mode live --symbol $SYMBOL --loop-interval $INTERVAL --no-dry-run
    elif [ "$MODE" = "dry-run" ]; then
        # Live signals but no real orders placed
        python main.py --mode live --symbol $SYMBOL --loop-interval $INTERVAL --dry-run
    elif [ "$MODE" = "backtest" ]; then
        # Backtest latest 24 hours
        python main.py --mode backtest --symbol $SYMBOL --lookback 24
    elif [ "$MODE" = "signals" ]; then
        # Scan for current signals
        python main.py --mode signals --symbol $SYMBOL --lookback 12
    else
        echo "Unknown BOT_MODE: $MODE. Exiting."
        exit 1
    fi
    
    EXIT_CODE=$?
    
    echo "[$(date)] Bot process exited with code $EXIT_CODE"
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "Bot exited cleanly. Stopping."
        break
    else
        echo "Bot crashed with exit code $EXIT_CODE! (Possible network or API issue)"
        echo "Waiting 60 seconds for network to recover before restarting..."
        sleep 60
    fi
done
