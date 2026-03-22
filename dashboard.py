"""
Trade Hunter Dashboard
Streamlit web UI for the ICT AMD Trading Bot.
Run: python -m streamlit run dashboard.py
"""

import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

BASE = Path(__file__).parent
_ENV = {**os.environ, "PYTHONIOENCODING": "utf-8"}

# ── Auto-start live bot if launched from start_dashboard.bat ─────────────────
# start_dashboard.bat sets AUTO_START_BOT=1 (and optionally BOT_SYMBOLS /
# BOT_TIMEFRAMES / BOT_MODE).  We fire this exactly once per Streamlit session.
def _maybe_auto_start():
    if st.session_state.get("_auto_start_done"):
        return
    st.session_state["_auto_start_done"] = True

    if os.environ.get("AUTO_START_BOT") != "1":
        return
    if st.session_state.get("live_proc") is not None:
        return  # Already running

    auto_symbols = os.environ.get("BOT_SYMBOLS", "ETHUSD,AVAXUSD,SOLUSD")
    auto_tfs     = os.environ.get("BOT_TIMEFRAMES", "5m,15m,5m")
    is_live      = os.environ.get("BOT_MODE", "dry-run") == "live"

    cmd = [sys.executable, "main.py", "--mode", "live",
           "--symbols", auto_symbols, "--timeframes", auto_tfs]
    cmd += ["--no-dry-run"] if is_live else ["--dry-run"]

    proc = subprocess.Popen(cmd, cwd=BASE, env=_ENV)
    st.session_state["live_proc"]   = proc
    st.session_state["live_symbol"] = auto_symbols
    st.session_state["live_tf"]     = auto_tfs


_maybe_auto_start()

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Trade Hunter",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .metric-card {
        background: #1e2130;
        border-radius: 8px;
        padding: 16px 20px;
        margin: 4px 0;
    }
    .win  { color: #26a69a; font-weight: 700; }
    .loss { color: #ef5350; font-weight: 700; }
    .neutral { color: #90a4ae; font-weight: 700; }
    div[data-testid="stStatusWidget"] { display: none; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📈 Trade Hunter")
    st.caption("ICT AMD Bot · Delta Exchange India")
    st.divider()

    page = st.radio(
        "Navigate",
        ["Backtest", "Portfolio", "Signals", "Live Trading", "Trade Journal"],
        label_visibility="collapsed",
    )

    st.divider()
    live_proc = st.session_state.get("live_proc")
    running = live_proc is not None and live_proc.poll() is None
    if running:
        st.success("Live trader running")
        if st.button("Stop Trader", use_container_width=True):
            live_proc.terminate()
            st.session_state["live_proc"] = None
            st.rerun()
    else:
        if live_proc is not None:
            # Process ended on its own — clean up
            st.session_state["live_proc"] = None
        st.info("No live trader running")


# ─────────────────────────────────────────────────────────────────────────────
# Helper: run a subprocess and stream the output to a st.code block
# ─────────────────────────────────────────────────────────────────────────────
def run_and_show(cmd: list[str], spinner_msg: str = "Running...") -> bool:
    with st.spinner(spinner_msg):
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=BASE,
            env=_ENV,
        )
    if result.returncode == 0:
        st.success("Done")
        if result.stdout:
            st.code(result.stdout, language="")
        return True
    else:
        st.error("Command failed")
        st.code(result.stderr or result.stdout, language="")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Page: Backtest
# ─────────────────────────────────────────────────────────────────────────────
if page == "Backtest":
    st.header("Backtest")
    st.caption("Simulate the strategy on historical candle data with full fee and slippage modelling.")

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        symbol = st.selectbox("Symbol", [
            "ETHUSD", "AVAXUSD", "SOLUSD", "BTCUSD",
            "BNBUSD", "ARBUSD", "LINKUSD", "DOTUSD",
            "ADAUSD", "SUIUSD", "LTCUSD", "BCHUSD", "APTUSD",
        ])
        resolution = st.selectbox("Resolution", ["15m", "5m", "1m", "1h", "4h"])
    with c2:
        lookback = st.slider("Lookback (hours)", 24, 2160, 720, step=24)
        balance = st.number_input("Starting Balance ($)", 1000, 500000, 10000, step=1000)
    with c3:
        st.write("")
        st.write("")
        st.write("")
        run_bt = st.button("Run Backtest", type="primary", use_container_width=True)

    st.divider()

    with st.expander("Advanced overrides (optional)"):
        oc1, oc2, oc3 = st.columns(3)
        with oc1:
            rsi_period = st.number_input("RSI Period", 0, 50, 0, help="0 = use default")
            min_range = st.number_input("Min Range Candles", 0, 50, 0, help="0 = use default")
        with oc2:
            rr_ratio = st.number_input("R:R Ratio", 0.0, 10.0, 0.0, step=0.1, help="0 = use default")
            risk_pct = st.number_input("Risk % per Trade", 0.0, 5.0, 0.0, step=0.1, help="0 = use default")
        with oc3:
            range_pct = st.number_input("Range Width %", 0.0, 5.0, 0.0, step=0.1, help="0 = use default")
            breakout_pct = st.number_input("Breakout %", 0.0, 2.0, 0.0, step=0.05, help="0 = use default")

    if run_bt:
        cmd = [
            sys.executable, "main.py",
            "--mode", "backtest",
            "--symbol", symbol,
            "--resolution", resolution,
            "--lookback", str(lookback),
            "--balance", str(balance),
            "--no-chart",
        ]
        if rsi_period > 0:
            cmd += ["--rsi-period", str(rsi_period)]
        if min_range > 0:
            cmd += ["--min-range", str(min_range)]
        if rr_ratio > 0:
            cmd += ["--rr-ratio", str(rr_ratio)]
        if risk_pct > 0:
            cmd += ["--risk-pct", str(risk_pct)]
        if range_pct > 0:
            cmd += ["--range-pct", str(range_pct)]
        if breakout_pct > 0:
            cmd += ["--breakout-pct", str(breakout_pct)]

        ok = run_and_show(cmd, f"Backtesting {symbol} {resolution} ({lookback}h)...")

        if ok:
            chart_path = BASE / "trading_chart.html"
            if chart_path.exists():
                st.info(f"Chart saved → open `trading_chart.html` in your browser for the interactive chart.")


# ─────────────────────────────────────────────────────────────────────────────
# Page: Portfolio Backtest
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Portfolio":
    st.header("Portfolio Backtest")
    st.caption("Multi-pair backtest with shared risk management. Recommended: ETHUSD 5m · AVAXUSD 15m · SOLUSD 5m.")

    pc1, pc2, pc3 = st.columns([1, 1, 1])
    with pc1:
        p_lookback = st.slider("Lookback (hours)", 168, 2160, 720, step=24)
    with pc2:
        p_balance = st.number_input("Starting Balance ($)", 1000, 500000, 10000, step=1000)
    with pc3:
        st.write("")
        st.write("")
        run_port = st.button("Run Portfolio Backtest", type="primary", use_container_width=True)

    st.divider()

    if run_port:
        cmd = [
            sys.executable, "portfolio_backtest.py",
            "--lookback", str(p_lookback),
            "--balance", str(p_balance),
        ]
        run_and_show(cmd, f"Running portfolio backtest ({p_lookback}h · ${p_balance:,.0f})...")


# ─────────────────────────────────────────────────────────────────────────────
# Page: Signals
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Signals":
    st.header("Signal Scanner")
    st.caption("Detect what the strategy sees right now on live candle data. No trades are placed.")

    sc1, sc2, sc3 = st.columns([1, 1, 1])
    with sc1:
        s_symbol = st.selectbox("Symbol", [
            "ETHUSD", "AVAXUSD", "SOLUSD", "BTCUSD",
            "BNBUSD", "ARBUSD", "LINKUSD", "DOTUSD",
            "ADAUSD", "SUIUSD", "LTCUSD", "BCHUSD", "APTUSD",
        ])
    with sc2:
        s_resolution = st.selectbox("Resolution", ["15m", "5m", "1m", "1h", "4h"])
    with sc3:
        s_lookback = st.slider("Lookback (hours)", 4, 168, 24, step=4)

    scan_btn = st.button("Scan for Signals", type="primary")

    if scan_btn:
        cmd = [
            sys.executable, "main.py",
            "--mode", "signals",
            "--symbol", s_symbol,
            "--resolution", s_resolution,
            "--lookback", str(s_lookback),
        ]
        run_and_show(cmd, f"Scanning {s_symbol} {s_resolution}...")


# ─────────────────────────────────────────────────────────────────────────────
# Page: Live Trading
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Live Trading":
    st.header("Live Trading")

    live_proc = st.session_state.get("live_proc")
    running = live_proc is not None and live_proc.poll() is None

    # ── Config panel (shown only when not running) ────────────────────────────
    if not running:
        st.subheader("Configure")

        lc1, lc2 = st.columns(2)
        with lc1:
            multi_pair = st.toggle("Multi-pair mode", value=True)
        with lc2:
            live_mode = st.selectbox("Trading mode", ["Dry Run (simulated)", "LIVE (real orders)"])
            is_live = live_mode.startswith("LIVE")
            if is_live:
                st.warning("LIVE mode places real orders on Delta Exchange with real money.")

        if multi_pair:
            mp1, mp2 = st.columns(2)
            with mp1:
                symbols_input = st.text_input("Symbols (comma-separated)", "ETHUSD,AVAXUSD,SOLUSD")
            with mp2:
                tfs_input = st.text_input("Timeframes (comma-separated)", "15m,5m,15m")
        else:
            sp1, sp2 = st.columns(2)
            with sp1:
                single_symbol = st.selectbox("Symbol", [
                    "ETHUSD", "AVAXUSD", "SOLUSD", "BTCUSD",
                    "BNBUSD", "ARBUSD", "LINKUSD", "DOTUSD",
                    "ADAUSD", "SUIUSD", "LTCUSD", "BCHUSD", "APTUSD",
                ])
            with sp2:
                single_tf = st.selectbox("Resolution", ["15m", "5m", "1m", "1h", "4h"])

        st.divider()
        start_btn = st.button("Start Trader", type="primary", use_container_width=True)

        if start_btn:
            cmd = [sys.executable, "main.py", "--mode", "live"]

            if multi_pair:
                cmd += ["--symbols", symbols_input.replace(" ", "")]
                cmd += ["--timeframes", tfs_input.replace(" ", "")]
            else:
                cmd += ["--symbol", single_symbol, "--resolution", single_tf]

            if is_live:
                cmd += ["--no-dry-run"]
            else:
                cmd += ["--dry-run"]

            proc = subprocess.Popen(cmd, cwd=BASE, env=_ENV)
            st.session_state["live_proc"] = proc
            st.session_state["live_symbol"] = symbols_input if multi_pair else single_symbol
            st.session_state["live_tf"] = tfs_input if multi_pair else single_tf
            st.rerun()

    # ── Running status panel ──────────────────────────────────────────────────
    else:
        sym_label = st.session_state.get("live_symbol", "?")
        tf_label = st.session_state.get("live_tf", "?")
        st.subheader(f"Running: {sym_label} @ {tf_label}")
        st.caption(f"Process PID: {live_proc.pid}")

        # Load state files
        state_files = list((BASE / "logs").glob("trade_state_*.json"))
        if state_files:
            cols = st.columns(len(state_files))
            for idx, sf in enumerate(state_files):
                try:
                    data = json.loads(sf.read_text())
                    s = data.get("state", {})
                    label = sf.stem.replace("trade_state_", "")
                    with cols[idx]:
                        st.markdown(f"**{label}**")
                        bal = s.get("balance", 0)
                        dpnl = s.get("daily_pnl", 0)
                        open_t = s.get("open_trades", 0)
                        cooldown = s.get("cooldown_until")
                        pnl_color = "win" if dpnl >= 0 else "loss"
                        st.markdown(
                            f"Balance: **${bal:,.2f}**  \n"
                            f"Daily P&L: <span class='{pnl_color}'>${dpnl:+.2f}</span>  \n"
                            f"Open trades: **{open_t}**  \n"
                            f"Cooldown: **{'Active' if cooldown else 'None'}**",
                            unsafe_allow_html=True,
                        )
                except Exception:
                    pass
        else:
            st.info("No state files found yet. State is written after the first trade cycle.")

        # Recent logs
        log_files = sorted((BASE / "logs").glob("trading_*.log"))
        log_path = log_files[-1] if log_files else None
        if log_path and log_path.exists():
            st.divider()
            st.caption("Recent log output (last 30 lines)")
            lines = log_path.read_text(errors="replace").splitlines()
            st.code("\n".join(lines[-30:]), language="")

        st.divider()
        st.caption("Page auto-refreshes every 10 seconds.")
        time.sleep(10)
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Page: Trade Journal
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Trade Journal":
    st.header("Trade Journal")
    st.caption("All trades recorded by the live/dry-run trader.")

    journal_files = glob.glob(str(BASE / "logs" / "trade_journal_*.csv"))
    if not journal_files:
        st.info(
            "No trade journal files found yet.\n\n"
            "They are created automatically when the live/dry-run trader closes a position. "
            "Run the trader in **Live Trading** mode to start recording trades."
        )
    else:
        dfs = []
        for f in journal_files:
            if os.path.getsize(f) > 0:
                try:
                    dfs.append(pd.read_csv(f))
                except Exception:
                    pass

        if not dfs:
            st.warning("Journal files exist but are empty — no trades closed yet.")
        else:
            df = pd.concat(dfs, ignore_index=True)

            # Parse dates
            if "closed_at" in df.columns:
                df["closed_at"] = pd.to_datetime(df["closed_at"], errors="coerce", utc=True)

            # ── Filters ───────────────────────────────────────────────────────
            fc1, fc2, fc3, fc4 = st.columns(4)
            with fc1:
                syms = ["All"] + sorted(df["symbol"].unique().tolist()) if "symbol" in df.columns else ["All"]
                f_sym = st.selectbox("Symbol", syms)
            with fc2:
                dirs = ["All"] + sorted(df["direction"].unique().tolist()) if "direction" in df.columns else ["All"]
                f_dir = st.selectbox("Direction", dirs)
            with fc3:
                outs = ["All"] + sorted(df["outcome"].unique().tolist()) if "outcome" in df.columns else ["All"]
                f_out = st.selectbox("Outcome", outs)
            with fc4:
                tfs_avail = ["All"] + sorted(df["resolution"].unique().tolist()) if "resolution" in df.columns else ["All"]
                f_tf = st.selectbox("Timeframe", tfs_avail)

            mask = pd.Series([True] * len(df))
            if f_sym != "All" and "symbol" in df.columns:
                mask &= df["symbol"] == f_sym
            if f_dir != "All" and "direction" in df.columns:
                mask &= df["direction"] == f_dir
            if f_out != "All" and "outcome" in df.columns:
                mask &= df["outcome"] == f_out
            if f_tf != "All" and "resolution" in df.columns:
                mask &= df["resolution"] == f_tf
            df_f = df[mask].copy()

            if df_f.empty:
                st.warning("No trades match the selected filters.")
            else:
                # ── Summary metrics ───────────────────────────────────────────
                total = len(df_f)
                wins = (df_f["outcome"].str.upper().str.contains("PROFIT|WIN") if "outcome" in df_f.columns else pd.Series([False] * total)).sum()
                total_pnl = df_f["pnl_dollar"].sum() if "pnl_dollar" in df_f.columns else 0.0
                avg_rr = df_f["rr_ratio"].mean() if "rr_ratio" in df_f.columns else 0.0
                avg_conf = df_f["confidence"].mean() if "confidence" in df_f.columns else 0.0
                win_rate = wins / total * 100 if total else 0.0

                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Total Trades", total)
                m2.metric("Win Rate", f"{win_rate:.1f}%")
                m3.metric("Total P&L", f"${total_pnl:+,.2f}")
                m4.metric("Avg R:R", f"{avg_rr:.2f}")
                m5.metric("Avg Confidence", f"{avg_conf:.2f}")

                st.divider()

                # ── Equity curve ──────────────────────────────────────────────
                if "pnl_dollar" in df_f.columns and "closed_at" in df_f.columns:
                    df_sorted = df_f.sort_values("closed_at")
                    df_sorted["cumulative_pnl"] = df_sorted["pnl_dollar"].cumsum()
                    st.subheader("Equity Curve")
                    st.line_chart(df_sorted.set_index("closed_at")["cumulative_pnl"])

                # ── Trade table ───────────────────────────────────────────────
                st.subheader(f"Trades ({total})")

                display_cols = [c for c in [
                    "closed_at", "symbol", "resolution", "direction",
                    "entry_price", "stop_loss", "take_profit",
                    "rr_ratio", "confidence", "session", "entry_type",
                    "outcome", "pnl_dollar", "balance_after",
                ] if c in df_f.columns]

                df_display = df_f[display_cols].sort_values(
                    "closed_at", ascending=False
                ).reset_index(drop=True)

                # Colour outcome column
                def _colour_outcome(val):
                    v = str(val).upper()
                    if "PROFIT" in v or "WIN" in v:
                        return "color: #26a69a; font-weight: bold"
                    if "LOSS" in v or "STOP" in v:
                        return "color: #ef5350; font-weight: bold"
                    return "color: #90a4ae"

                styled = df_display.style
                if "outcome" in df_display.columns:
                    styled = styled.applymap(_colour_outcome, subset=["outcome"])
                if "pnl_dollar" in df_display.columns:
                    styled = styled.format({"pnl_dollar": "${:+.2f}", "balance_after": "${:,.2f}"})

                st.dataframe(styled, use_container_width=True, height=500)

                # ── Download ──────────────────────────────────────────────────
                csv_bytes = df_f.to_csv(index=False).encode()
                st.download_button(
                    "Download filtered journal as CSV",
                    data=csv_bytes,
                    file_name="trade_journal_export.csv",
                    mime="text/csv",
                )
