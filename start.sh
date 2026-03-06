#!/usr/bin/env bash

# Auto-restart wrapper script for Render
# This ensures that if the bot crashes, it waits a few seconds and restarts.

# Read configuration from env vars (set in render.yaml or dashboard)
MODE=${BOT_MODE:-live}
# Fallback to comma-separated symbols if multiple are wanted, e.g., "BTCUSD,ETHUSD"
SYMBOLS=${TRADE_SYMBOL:-BTCUSD}
INTERVAL=${TRADE_INTERVAL:-300}

# Convert comma-separated string to an array
IFS=',' read -r -a SYMBOL_ARRAY <<< "$SYMBOLS"

echo "================================================="
echo " Starting Trade Hunter Bot (Multi-Instance Mode)"
echo " Mode: $MODE"
echo " Symbols: ${SYMBOL_ARRAY[*]}"
echo " Interval: $INTERVAL seconds"
echo "================================================="

# Function to run a single bot instance
run_bot() {
    local sym=$1
    while true; do
        echo "[$(date)] Launching bot in $MODE mode for $sym..."
        
        if [ "$MODE" = "live" ]; then
            python main.py --mode live --symbol "$sym" --loop-interval "$INTERVAL" --no-dry-run
        elif [ "$MODE" = "dry-run" ]; then
            python main.py --mode live --symbol "$sym" --loop-interval "$INTERVAL" --dry-run
        elif [ "$MODE" = "backtest" ]; then
            python main.py --mode backtest --symbol "$sym" --lookback 24
            # Backtest finishes immediately, no need to loop
            break 
        elif [ "$MODE" = "signals" ]; then
            python main.py --mode signals --symbol "$sym" --lookback 12
            break
        else
            echo "Unknown BOT_MODE: $MODE. Exiting."
            exit 1
        fi
        
        EXIT_CODE=$?
        echo "[$(date)] Bot $sym exited with code $EXIT_CODE"
        
        if [ $EXIT_CODE -eq 0 ]; then
            echo "Bot $sym exited cleanly. Stopping."
            break
        else
            echo "Bot $sym crashed with exit code $EXIT_CODE! (Possible network or API issue)"
            echo "Waiting 60 seconds before restarting $sym..."
            sleep 60
        fi
    done
}

# Launch all instances in the background
PIDS=""
for sym in "${SYMBOL_ARRAY[@]}"; do
    run_bot "$sym" &
    PIDS="$PIDS $!"
done

# Wait for all background processes to finish
wait $PIDS
echo "All bot instances have stopped."
