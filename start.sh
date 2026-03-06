#!/usr/bin/env bash

# Auto-restart wrapper script for Render
# This ensures that if the bot crashes, it waits a few seconds and restarts.

# If a local .env file exists, load it directly so local runs mimic server runs
if [ -f ".env" ]; then
    export $(cat .env | tr -d '\r' | grep -v '^#' | xargs)
fi
MODE=${BOT_MODE:-dry-run}
MODE=$(echo "$MODE" | tr -d '\r')
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
            venv/Scripts/python.exe main.py --mode live --symbol "$sym" --loop-interval "$INTERVAL" --no-dry-run
        elif [ "$MODE" = "dry-run" ]; then
            venv/Scripts/python.exe main.py --mode live --symbol "$sym" --loop-interval "$INTERVAL" --dry-run
        elif [ "$MODE" = "backtest" ]; then
            venv/Scripts/python.exe main.py --mode backtest --symbol "$sym" --lookback 24
            # Backtest finishes immediately, no need to loop
            break 
        elif [ "$MODE" = "signals" ]; then
            venv/Scripts/python.exe main.py --mode signals --symbol "$sym" --lookback 12
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

# Cleanup function to gracefully stop all background bots
cleanup() {
    echo ""
    echo "[$(date)] Stopping all background bots..."
    # Send SIGTERM to python processes so they print final balance
    kill -TERM $PIDS 2>/dev/null
    wait $PIDS 2>/dev/null
    echo "All bots gracefully stopped."
    exit 0
}

# Trap SIGINT (Ctrl+C) and SIGTERM (Render shutdown) so we can gracefully exit
trap cleanup SIGINT SIGTERM

# Wait for all background processes to finish
wait $PIDS
echo "All bot instances have stopped."
