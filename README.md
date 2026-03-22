# Trade Hunter

Algorithmic crypto trading bot implementing the **ICT AMD (Accumulation → Manipulation → Distribution)** smart money strategy on **Delta Exchange India**.

Trades BTC/ETH/SOL/AVAX perpetual futures using liquidity sweeps, displacement candles, Fair Value Gaps, order blocks, RSI divergence, SMT divergence, and multi-timeframe bias alignment.

---

## Quick Start (Windows)

1. **Configure API keys** — copy `.env.example` to `.env` and fill in your Delta Exchange credentials
2. **Double-click `start.bat`** — creates venv, installs all dependencies, opens dashboard at `http://localhost:8501`

> Requires Python 3.10+ installed globally. Everything else installs into the local `venv/`.

---

## Configure API Keys

Copy the template and add your [Delta Exchange](https://www.delta.exchange) credentials:

```
cp .env.example .env
```

Edit `.env`:

```env
DELTA_API_KEY=your_api_key_here
DELTA_API_SECRET=your_api_secret_here
DELTA_BASE_URL=https://api.india.delta.exchange

# Optional: Telegram push alerts
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
```

API keys need **read + trade** permissions from your Delta Exchange account settings.

---

## Modes

### Dashboard (recommended)

```bat
start.bat
```

Opens Streamlit UI at `http://localhost:8501` with tabs for:
- **Backtest** — run and visualize backtests inline
- **Portfolio** — multi-pair backtest with shared risk
- **Signals** — check live signals for any pair/timeframe
- **Live Trading** — start/stop dry-run or live trader from the browser
- **Trade Journal** — view all past trades from CSV logs

### CLI — Backtest

```bash
# ETH 5m last 30 days (recommended)
python main.py --mode backtest --symbol ETHUSD --lookback 720 --resolution 5m

# Multi-pair portfolio backtest
python portfolio_backtest.py --lookback 720 --balance 10000

# Full scan: 13 pairs × 3 timeframes × 3 lookbacks
python run_full_analysis.py

# No API key needed
python main.py --mode backtest --sample
```

### CLI — Live Trading

```bash
# Dry-run (no real orders) — recommended to start
python main.py --mode live --symbols ETHUSD,AVAXUSD --timeframes 5m,15m --dry-run

# Live trading (real orders)
python main.py --mode live --symbols ETHUSD,AVAXUSD --timeframes 5m,15m --no-dry-run
```

> `--no-dry-run` places real orders on Delta Exchange. Bot gives 5 seconds to abort on startup.

---

## Strategy

### ICT AMD Pipeline

1. **Accumulation** — detect consolidation ranges (tight price ranges + Asian session 00:00–08:00 UTC)
2. **Manipulation** — identify liquidity sweeps beyond range highs/lows
3. **Confirmation** — validate with displacement candles, RSI divergence, FVG/BPR presence, volume
4. **Distribution** — enter on FVG / Balanced Price Range / Order Block, target opposite side

### Entry Filters

| Filter | Type |
|---|---|
| Consolidation range (8–30 candles) | Required |
| Liquidity sweep beyond range | Required |
| Displacement candle after sweep | Hard filter |
| RSI divergence | Required |
| Min SL distance ≥ 0.15% | Hard filter |
| HTF bias alignment (5m→1H, 15m→4H) | Alignment |
| London (02–05 UTC) / NY (12–15 UTC) session | Priority |

### Signal Confidence Scoring

| Factor | Score |
|---|---|
| RSI divergence | +0.25 |
| London or NY session | +0.20 |
| BPR entry | +0.20 |
| FVG entry | +0.15 |
| Volume spike | +0.15 |
| Displacement candle | +0.10 |
| R:R ≥ 2.5 | +0.10 |
| Order block entry | +0.10 |
| Order block without divergence | −0.20 |
| Ranging market (ADX < 20) without divergence | −0.15 |

Minimum confidence to trade: **0.50** (5m), **0.55** (15m). Position size scales with confidence.

### Trade Management

| Stage | Trigger | Action |
|---|---|---|
| Partial profit 1 | Price at 1.5R | Close 25% of position |
| Break-even | Price at 1.5R | Move SL to entry + 0.1R |
| Partial profit 2 | Price at 2.0R | Close 35% of position |
| Trailing stop | After break-even | Trail swing highs/lows with 0.3× ATR buffer |
| Time exit | 48/32/16 candles (5m/15m/1h) | Exit at entry if within 0.3R, else market |

### Risk Management

| Rule | Default |
|---|---|
| Max risk per trade | 1% (scaled by confidence) |
| Max daily loss | 3% — stops all trading |
| Max concurrent trades | 2 (global across all pairs) |
| Cooldown after 3 consecutive losses | 30 minutes |

### Recommended Pairs (30-day backtest results)

| Pair | Timeframe | Win Rate | Profit Factor | Return |
|---|---|---|---|---|
| ETHUSD | 5m | ~67% | ~4.1 | **+8.8%** |
| AVAXUSD | 15m | ~100% | ~19.9 | **+6.1%** |
| SOLUSD | 15m | ~83% | ~8.4 | **+7.0%** |
| BTCUSD | 5m | ~42% | ~1.1 | ~−0.6% |

**Recommended deployment:** `ETHUSD 5m + AVAXUSD 15m`

---

## Project Structure

```
trade_hunter/
  start.bat                    # One-click launcher — start here
  main.py                      # CLI entry point (backtest / signals / live)
  dashboard.py                 # Streamlit web UI
  portfolio_backtest.py        # Multi-pair portfolio backtest
  run_full_analysis.py         # Full scan: all pairs × timeframes × lookbacks
  requirements.txt             # Python dependencies
  .env.example                 # API key template
  logs/                        # Runtime output (auto-created, gitignored)
    trading_*.log              #   Live trading logs
    trade_journal_*.csv        #   Per-trade CSV journal
    trade_state_*.json         #   Bot state (survives restarts)
  trading_bot/
    strategy/
      amd_strategy.py          # Core AMD signal generator + confidence scoring
      range_detector.py        # Consolidation + Asian session range detection
      divergence_detector.py   # RSI divergence detection
      smt_detector.py          # SMT divergence (cross-asset)
      risk_manager.py          # Daily DD limit, cooldown, concurrency cap
    backtest/
      engine.py                # Backtest simulator with tiered partials
      performance.py           # Win rate, Sharpe, Calmar, drawdown metrics
    exchange/
      delta_connector.py       # Delta Exchange REST API wrapper
      live_trader.py           # Single-pair live trading loop
      multi_pair_trader.py     # Multi-pair orchestrator (shared risk)
      trade_store.py           # JSON state persistence (survives restarts)
    data/
      loader.py                # CSV + API data loading
    indicators/
      rsi.py                   # RSI + swing high/low detection
    notifications/
      telegram_notifier.py     # Telegram push alerts (optional)
    visualization/
      charts.py                # Plotly + matplotlib charting
  tests/
    test_risk_manager.py
    test_backtest_fees.py
    test_displacement.py
```

---

## CLI Reference

| Argument | Default | Description |
|---|---|---|
| `--mode` | `backtest` | `backtest`, `signals`, or `live` |
| `--symbol` | `BTCUSD` | Single pair |
| `--symbols` | — | Comma-separated pairs (e.g. `ETHUSD,AVAXUSD`) |
| `--timeframes` | — | Comma-separated TFs (e.g. `5m,15m`) |
| `--lookback` | `24` | Hours of candle history |
| `--resolution` | `5m` | Candle TF: `1m`, `5m`, `15m`, `1h` |
| `--balance` | `10000` | Starting balance for backtest |
| `--dry-run` | `true` | Simulate — no real orders |
| `--no-dry-run` | — | Live trading — real orders |

---

## Running Tests

```bash
venv\Scripts\python.exe -m pytest tests/ -v
```

---

## Disclaimer

Algorithmic trading involves significant financial risk. This software is for educational purposes only. Past performance does not guarantee future results. Use at your own risk.
