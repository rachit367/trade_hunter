# 📈 ICT AMD Trading Bot

An algorithmic crypto trading bot implementing the **ICT Accumulation → Manipulation → Distribution (AMD)** smart money concept on **Delta Exchange**, optimized for the 5-minute timeframe.

![Language](https://img.shields.io/badge/language-python-blue)
![Exchange](https://img.shields.io/badge/exchange-Delta_Exchange-orange)
![Strategy](https://img.shields.io/badge/strategy-ICT_AMD-purple)

---

## 🚀 Overview

This bot automates the detection and execution of ICT AMD patterns on crypto perpetual futures via [Delta Exchange India](https://www.delta.exchange). It identifies periods of price consolidation (Accumulation), looks for liquidity grabs beyond the range (Manipulation) confirmed by RSI divergence, and targets the subsequent expansion move (Distribution).

### Strategy Pipeline
1.  **Accumulation** — Detects tight consolidation ranges based on price volatility and candle count.
2.  **Manipulation** — Monitors for breakouts/breakdowns beyond the range high/low (stop hunts).
3.  **Confirmation** — Validates breakouts using RSI Bullish/Bearish divergence to filter fakeouts.
4.  **Distribution** — Generates trade signals with SL (above/below manipulation wick) and TP (based on R:R ratio).

---

## ✨ Features

-   **Backtesting Engine** — Simulate the strategy on historical candle data from Delta Exchange or custom CSV files.
-   **Signal Detection** — Scan for active AMD setups on any Delta Exchange crypto pair.
-   **Live Trading** — Automated order execution on **Delta Exchange India** with bracket orders (SL + TP).
-   **Dry Run Mode** — Test live signal detection without placing real orders.
-   **Interactive Charts** — Rich Plotly HTML charts and static Matplotlib images for trade analysis.
-   **Risk Management** — Fixed-percentage risk-based position sizing with configurable R:R ratios.

---

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd "Algo Trader"
```

### 2. Setup Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

**Key Dependencies:**
| Package | Purpose |
| :--- | :--- |
| `delta-rest-client` | Delta Exchange API client |
| `pandas` / `numpy` | Data manipulation |
| `ta` | Technical indicators (RSI) |
| `plotly` / `matplotlib` | Charting and visualization |
| `python-dotenv` | Environment variable management |

---

## ⚙️ Configuration

Create a `.env` file in the root directory with your [Delta Exchange API credentials](https://www.delta.exchange):

```env
DELTA_API_KEY=your_api_key_here
DELTA_API_SECRET=your_api_secret_here
DELTA_BASE_URL=https://api.india.delta.exchange
```

> **Note:** API keys can be generated from your Delta Exchange account settings. The bot requires **read + trade** permissions.

---

## 📋 Usage

The bot is controlled via a command-line interface through `main.py`.

### Backtesting
Run a backtest on BTCUSD with the last 24 hours of 5m candles:
```bash
python main.py --mode backtest --symbol BTCUSD --lookback 24
```

Backtest ETHUSD with 1-hour candles over the last 72 hours:
```bash
python main.py --mode backtest --symbol ETHUSD --resolution 1h --lookback 72
```

Use generated sample data (no API key required):
```bash
python main.py --mode backtest --sample
```

### Signal Detection
Scan for current AMD signals on BTCUSD:
```bash
python main.py --mode signals --symbol BTCUSD --lookback 12
```

### Live Trading
**Dry Run** (detects signals, does not place orders):
```bash
python main.py --mode live --symbol BTCUSD --dry-run
```

**Live Execution** (⚠️ places real orders):
```bash
python main.py --mode live --symbol BTCUSD --no-dry-run
```

### CLI Arguments

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--mode` | `backtest`, `signals`, or `live` | `backtest` |
| `--symbol` | Delta Exchange product symbol | `BTCUSD` |
| `--lookback` | Hours of candle history to fetch | `24` |
| `--resolution` | Candle interval (`1m`, `5m`, `15m`, `1h`, `1d`) | `5m` |
| `--csv` | Path to CSV file with OHLCV data | — |
| `--sample` | Use generated sample data for testing | `false` |
| `--chart` | Output path for visualization chart | `trading_chart.html` |
| `--no-chart` | Skip chart generation | `false` |
| `--balance` | Initial balance for backtesting | `10000.0` |
| `--dry-run` | Signal detection only (no orders) | `true` |
| `--no-dry-run` | Enable real order placement | — |
| `--loop-interval` | Seconds between live trading cycles | `300` |

### Strategy Tuning

| Argument | Description |
| :--- | :--- |
| `--rsi-period` | RSI lookback period |
| `--min-range` | Min candles for consolidation range |
| `--max-range` | Max candles for consolidation range |
| `--range-pct` | Range threshold % |
| `--breakout-pct` | Min % beyond range to confirm breakout |
| `--rr-ratio` | Risk:Reward ratio |
| `--risk-pct` | % of account risked per trade |

---

## ☁️ Deployment (Render)

This bot is configured to run 24/7 as a **Background Worker** on [Render](https://render.com).

### Features:
- Automatically restarts if the process crashes.
- Pulls configuration directly from the Render dashboard.

### Steps to Deploy:
1. Push this repository to GitHub.
2. Log into Render and click **New** > **Background Worker**.
3. Connect your GitHub repository.
4. Render will automatically detect the settings from `render.yaml`:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `bash start.sh`
5. **IMPORTANT:** In the Render dashboard for your worker, go to **Environment** and add:
   - `DELTA_API_KEY` (Your API Key)
   - `DELTA_API_SECRET` (Your API Secret)
     *(Do not commit these to GitHub. Add them directly in Render!)*
6. (Optional) Override defaults in the Environment tab:
   - `TRADE_SYMBOL` (default: BTCUSD)
   - `TRADE_INTERVAL` (default: 300)
7. Click **Deploy**. The bot will launch and stay alive indefinitely.

---

## 📂 Project Structure

```
Algo Trader/
├── main.py                         # CLI Entry Point
├── requirements.txt                # Dependencies
├── .env                            # Delta Exchange API Keys (private)
└── trading_bot/                    # Core Package
    ├── data/
    │   └── loader.py               # Data fetching (Delta API + CSV)
    ├── exchange/
    │   ├── delta_connector.py      # Delta Exchange API wrapper
    │   └── live_trader.py          # Automated live trading loop
    ├── strategy/
    │   ├── amd_strategy.py         # AMD signal generation pipeline
    │   ├── range_detector.py       # Consolidation range detection
    │   └── divergence_detector.py  # RSI divergence detection
    ├── indicators/
    │   └── rsi.py                  # RSI calculation + swing detection
    ├── backtest/
    │   ├── engine.py               # Backtesting simulation engine
    │   └── performance.py          # Performance metrics (Sharpe, PF, etc.)
    └── visualization/
        └── charts.py               # Plotly + Matplotlib charting
```

---

## 📊 Visualization

The bot generates a `trading_chart.html` by default, featuring:
-   **Candlestick Chart** with highlighted Accumulation Ranges
-   **Signal Markers** — Entry, SL, and TP levels
-   **RSI Subplot** with divergence confirmations
-   **Equity Curve** (backtest mode) showing account growth

---

## 🔗 Resources

- [Delta Exchange API Documentation](https://docs.delta.exchange/)
- [delta-rest-client (Python)](https://pypi.org/project/delta-rest-client/)
- [Delta Exchange India](https://www.delta.exchange)

---

## ⚠️ Disclaimer

Algorithmic trading involves significant financial risk. This software is for **educational purposes only**. Past performance does not guarantee future results. Use at your own risk.
