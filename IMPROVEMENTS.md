# 🔧 Strategy Improvement Suggestions

Targeted improvements to **increase win rate**, **improve signal accuracy**, and **reduce max drawdown** — based on a deep review of the current AMD strategy codebase.

---

## 🎯 Priority 1: Improve Win Rate

### 1.1 Add Volume Confirmation to Breakouts
**File:** `amd_strategy.py` (lines 150–184)

**Problem:** The bot confirms breakouts using only price + RSI divergence. Many false breakouts happen on low volume.

**Fix:** Require that the breakout candle's volume is **≥ 1.5× the average volume** of the consolidation range. This single filter can eliminate 30-40% of false signals.

```python
# In generate_signals(), after detecting a breakout:
range_avg_vol = np.mean(df["Volume"].values[rng.start_idx:rng.end_idx + 1])
breakout_vol = df["Volume"].values[i]

if breakout_vol < range_avg_vol * 1.5:
    continue  # Skip low-volume breakouts
```

---

### 1.2 Add Higher-Timeframe (HTF) Trend Filter
**Problem:** The bot trades both long and short regardless of the overall market trend. Counter-trend trades have a much lower win rate.

**Fix:** Before generating signals, fetch 1h or 4h candles and compute a simple trend bias (e.g., 50-period EMA direction). Only take LONG signals in uptrends and SHORT signals in downtrends.

```python
# Compute HTF bias using EMA on 1h data
htf_df = connector.fetch_candles(symbol, resolution="1h", lookback_hours=72)
ema_50 = htf_df["Close"].ewm(span=50).mean()

if ema_50.iloc[-1] > ema_50.iloc[-5]:
    bias = "bullish"  # Only take LONGs
else:
    bias = "bearish"  # Only take SHORTs
```

---

### 1.3 Wait for Re-entry Into Range (Displacement Confirmation)
**File:** `amd_strategy.py` (lines 150–212)

**Problem:** The bot enters on the **breakout candle itself**. In ICT theory, the ideal entry is when price **returns back inside the range** after the manipulation sweep — not during the sweep.

**Fix:** After detecting a breakout + divergence, don't enter immediately. Instead, wait for the **next candle that closes back inside the range**. This massively filters out continuation breakouts.

```python
# After detecting a SHORT signal at candle i:
# Wait for price to close back below range_high
for k in range(i + 1, min(i + 5, n)):
    if closes[k] < rng.range_high:
        entry = closes[k]  # Enter here, not at breakout candle
        break
```

---

## 📉 Priority 2: Reduce Max Drawdown

### 2.1 Add a Daily Loss Limit
**File:** `engine.py` / `live_trader.py`

**Problem:** The bot keeps trading even during losing streaks, compounding drawdown.

**Fix:** Implement a daily loss circuit breaker — stop trading for the day after losing **2% of account balance**.

```python
daily_loss = 0.0
MAX_DAILY_LOSS_PCT = 2.0

for sig in signals:
    if daily_loss >= (initial_balance * MAX_DAILY_LOSS_PCT / 100):
        break  # Stop trading for today
    # ... execute trade ...
    if trade.outcome == TradeOutcome.LOSS:
        daily_loss += abs(trade.pnl_dollar)
```

---

### 2.2 Scale Risk Based on Recent Performance
**File:** `amd_strategy.py` → `StrategyConfig`

**Problem:** Fixed 1% risk per trade regardless of recent performance.

**Fix:** After 2 consecutive losses, reduce risk to 0.5%. After 3 consecutive wins, increase to 1.5%. This is called **anti-martingale position sizing**.

---

### 2.3 Add a Trailing Stop Loss
**File:** `engine.py` (lines 150–181)

**Problem:** The SL is fixed at the manipulation wick. In volatile markets, price can hit SL before reaching TP, even when the trade direction is correct.

**Fix:** Once price moves **1R in your favor** (covers the risk distance), move SL to breakeven. Once **1.5R in favor**, trail the SL to lock in 0.5R profit.

```python
# Inside the walk-forward loop for a LONG trade:
current_pnl = lows[j] - sig.entry_price
risk = abs(sig.entry_price - sig.stop_loss)

if current_pnl >= risk * 1.5:
    trailing_sl = sig.entry_price + risk * 0.5  # Lock 0.5R
elif current_pnl >= risk:
    trailing_sl = sig.entry_price  # Breakeven

active_sl = max(sig.stop_loss, trailing_sl)
```

---

## 🔬 Priority 3: Improve Signal Accuracy

### 3.1 Require Minimum Divergence Strength
**File:** `divergence_detector.py`

**Problem:** Any RSI divergence triggers a signal, even if the RSI difference is tiny (e.g., 0.5 points).

**Fix:** Add a minimum RSI gap threshold — require at least **5 RSI points** of divergence.

```python
# In detect_bearish_divergence():
rsi_gap = abs(prior_rsi - current_rsi)
if rsi_gap < 5.0:
    return None  # Divergence too weak
```

---

### 3.2 Filter by Session Timing
**Problem:** Crypto markets have different volatility profiles. The best AMD setups on BTC occur during **London open (1:30 PM - 5:30 PM IST)** and **New York open (6:30 PM - 11:30 PM IST)**. Asian session signals tend to be lower quality.

**Fix:** Add a session filter that optionally skips signals outside high-liquidity windows.

```python
from datetime import time

LONDON_OPEN = time(8, 0)   # UTC
NY_OPEN = time(13, 0)      # UTC
NY_CLOSE = time(21, 0)     # UTC

signal_hour = df.index[i].time()
if not (LONDON_OPEN <= signal_hour <= NY_CLOSE):
    continue  # Skip low-liquidity signals
```

---

### 3.3 Add Multi-Candle Breakout Confirmation
**File:** `amd_strategy.py`

**Problem:** A single candle wick beyond the range is enough to trigger. This catches many wicks that don't represent true manipulation.

**Fix:** Require the breakout candle to **close** beyond the range level (not just wick), OR require 2 consecutive candles with highs/lows beyond the range.

---

### 3.4 Add OBV (On-Balance Volume) Divergence
**Problem:** RSI divergence alone can produce false signals. Adding OBV divergence as a **second confirmation** significantly filters noise.

**Fix:** Add OBV calculation to `indicators/` and check for OBV divergence alongside RSI divergence. Only trade when **both** divergences align.

---

## ⚙️ Priority 4: Strategy Config Tuning

### Recommended Parameter Adjustments

| Parameter | Current | Suggested | Reason |
| :--- | :---: | :---: | :--- |
| `rsi_period` | 14 | **9** | Faster RSI catches divergences sooner on 5m charts |
| `min_range_candles` | 10 | **12** | Slightly longer ranges = higher quality accumulation |
| `breakout_pct` | 0.10% | **0.15%** | Require stronger breakouts to filter noise |
| `risk_reward_ratio` | 2.0 | **2.5** | Higher R:R means fewer wins needed to be profitable |
| `sl_buffer_pct` | 0.05% | **0.10%** | Wider buffer avoids SL hunts on the manipulation wick |
| `divergence_lookback` | 10 | **15** | Wider window catches stronger divergence patterns |
| `max_scan_after_range` | 20 | **10** | Tighter scan window = fresher, more reliable breakouts |

---

## 📊 Implementation Priority

| # | Improvement | Impact | Effort |
| :---: | :--- | :---: | :---: |
| 1 | Volume confirmation on breakouts | 🟢 High | 🟢 Low |
| 2 | Wait for re-entry into range | 🟢 High | 🟡 Medium |
| 3 | Daily loss limit | 🟢 High | 🟢 Low |
| 4 | Trailing stop loss | 🟢 High | 🟡 Medium |
| 5 | Minimum divergence strength | 🟡 Medium | 🟢 Low |
| 6 | Parameter tuning | 🟡 Medium | 🟢 Low |
| 7 | Session timing filter | 🟡 Medium | 🟢 Low |
| 8 | HTF trend filter | 🟢 High | 🟡 Medium |
| 9 | Multi-candle breakout confirm | 🟡 Medium | 🟢 Low |
| 10 | OBV divergence | 🟡 Medium | 🟡 Medium |

> **Start with items 1, 3, 5, and 6** — they are low effort, high impact, and can be implemented in under an hour each. Then move to items 2, 4, and 8 for the biggest win rate and drawdown improvements.
