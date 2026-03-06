#!/usr/bin/env bash

# Auto-restart wrapper script for Render
# This ensures that if the bot crashes, it waits a few seconds and restarts.

echo "Starting Trade Hunter Bot..."

while true; do
    # Read symbol and interval from env vars (set in render.yaml or dashboard)
    SYMBOL=${TRADE_SYMBOL:-BTCUSD}
    INTERVAL=${TRADE_INTERVAL:-300}

    echo "[$(date)] Launching bot for $SYMBOL with interval $INTERVAL seconds..."
    
    # Run in live mode. Remove --dry-run if you want to place real trades!
    python main.py --mode live --symbol $SYMBOL --loop-interval $INTERVAL --no-dry-run
    
    EXIT_CODE=$?
    
    echo "[$(date)] Bot process exited with code $EXIT_CODE"
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "Bot exited cleanly. Stopping."
        break
    else
        echo "Bot crashed! Restarting in 10 seconds..."
        sleep 10
    fi
done
