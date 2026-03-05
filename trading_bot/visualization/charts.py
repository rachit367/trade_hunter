"""
Charts — Interactive visualization of the AMD trading strategy using Plotly.

Generates a comprehensive chart with:
  - Candlestick price chart
  - Shaded consolidation ranges
  - Trade entry / exit markers with SL and TP levels
  - RSI subplot with divergence annotations
  - Equity curve subplot
"""

from typing import List, Optional

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from trading_bot.indicators.rsi import calculate_rsi
from trading_bot.strategy.range_detector import detect_ranges, ConsolidationRange
from trading_bot.strategy.amd_strategy import TradeSignal, Direction, StrategyConfig
from trading_bot.backtest.engine import BacktestResult, CompletedTrade, TradeOutcome


def create_trading_chart(
    df: pd.DataFrame,
    signals: List[TradeSignal],
    config: StrategyConfig = None,
    result: Optional[BacktestResult] = None,
    save_path: str = "trading_chart.html",
    title: str = "ICT AMD Strategy — 5-Min Timeframe",
) -> str:
    """
    Generate an interactive Plotly chart.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data.
    signals : List[TradeSignal]
        Trade signals to visualize.
    config : StrategyConfig
        Strategy parameters.
    result : BacktestResult, optional
        Backtest results (for equity curve).
    save_path : str
        Output HTML file path.
    title : str
        Chart title.

    Returns
    -------
    str
        Path where the chart was saved.
    """
    if config is None:
        config = StrategyConfig()

    rsi = calculate_rsi(df["Close"], config.rsi_period)
    ranges = detect_ranges(
        df,
        min_candles=config.min_range_candles,
        max_candles=config.max_range_candles,
        range_threshold_pct=config.range_threshold_pct,
    )

    # Determine subplot layout
    has_equity = result is not None and len(result.equity_curve) > 1
    n_rows = 3 if has_equity else 2
    row_heights = [0.5, 0.25, 0.25] if has_equity else [0.65, 0.35]
    subplot_titles = ["Price", "RSI"]
    if has_equity:
        subplot_titles.append("Equity Curve")

    fig = make_subplots(
        rows=n_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights,
        subplot_titles=subplot_titles,
    )

    # === Row 1: Candlestick Chart ===
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Price",
            increasing_line_color="#00e676",
            decreasing_line_color="#ff1744",
            increasing_fillcolor="#00e676",
            decreasing_fillcolor="#ff1744",
        ),
        row=1, col=1,
    )

    # --- Consolidation ranges (shaded rectangles) ---
    for rng in ranges:
        t_start = df.index[rng.start_idx]
        t_end = df.index[rng.end_idx]
        fig.add_shape(
            type="rect",
            x0=t_start, x1=t_end,
            y0=rng.range_low, y1=rng.range_high,
            fillcolor="rgba(30, 136, 229, 0.12)",
            line=dict(color="rgba(66, 165, 245, 0.6)", width=1, dash="dash"),
            row=1, col=1,
        )
        # Label
        fig.add_annotation(
            x=t_start,
            y=rng.range_high,
            text="RANGE",
            showarrow=False,
            font=dict(size=9, color="rgba(66, 165, 245, 0.8)"),
            xanchor="left",
            yanchor="bottom",
            row=1, col=1,
        )

    # --- Trade entry markers ---
    for sig in signals:
        t = df.index[sig.entry_idx]

        if sig.direction == Direction.LONG:
            color = "#00e676"
            symbol = "triangle-up"
            label = "LONG"
        else:
            color = "#ff1744"
            symbol = "triangle-down"
            label = "SHORT"

        # Entry point
        fig.add_trace(
            go.Scatter(
                x=[t], y=[sig.entry_price],
                mode="markers+text",
                marker=dict(size=14, color=color, symbol=symbol, line=dict(width=1, color="white")),
                text=[label],
                textposition="top center" if sig.direction == Direction.LONG else "bottom center",
                textfont=dict(size=10, color=color),
                name=f"{label} @ {sig.entry_price:.2f}",
                showlegend=True,
            ),
            row=1, col=1,
        )

        # SL line
        fig.add_shape(
            type="line",
            x0=t, x1=t,
            y0=sig.entry_price, y1=sig.stop_loss,
            line=dict(color="#ff6f00", width=1.5, dash="dot"),
            row=1, col=1,
        )
        fig.add_annotation(
            x=t, y=sig.stop_loss,
            text=f"SL {sig.stop_loss:.2f}",
            showarrow=False,
            font=dict(size=8, color="#ff6f00"),
            xanchor="left",
            row=1, col=1,
        )

        # TP line
        fig.add_shape(
            type="line",
            x0=t, x1=t,
            y0=sig.entry_price, y1=sig.take_profit,
            line=dict(color="#00bfa5", width=1.5, dash="dot"),
            row=1, col=1,
        )
        fig.add_annotation(
            x=t, y=sig.take_profit,
            text=f"TP {sig.take_profit:.2f}",
            showarrow=False,
            font=dict(size=8, color="#00bfa5"),
            xanchor="left",
            row=1, col=1,
        )

    # --- Trade exit markers (if backtest results available) ---
    if result is not None:
        for trade in result.trades:
            if trade.outcome == TradeOutcome.OPEN:
                continue
            exit_t = df.index[trade.exit_idx] if trade.exit_idx < len(df) else trade.exit_time
            color = "#00e676" if trade.outcome == TradeOutcome.WIN else "#ff1744"
            fig.add_trace(
                go.Scatter(
                    x=[exit_t], y=[trade.exit_price],
                    mode="markers",
                    marker=dict(size=10, color=color, symbol="x", line=dict(width=2, color="white")),
                    name=f"Exit {trade.outcome.value} @ {trade.exit_price:.2f}",
                    showlegend=False,
                ),
                row=1, col=1,
            )

    # === Row 2: RSI ===
    fig.add_trace(
        go.Scatter(
            x=df.index, y=rsi,
            mode="lines",
            line=dict(color="#7c4dff", width=1.5),
            name="RSI",
        ),
        row=2, col=1,
    )

    # Overbought / Oversold zones
    fig.add_hline(y=70, line=dict(color="#ff1744", width=0.8, dash="dash"), row=2, col=1)
    fig.add_hline(y=30, line=dict(color="#00e676", width=0.8, dash="dash"), row=2, col=1)
    fig.add_hrect(y0=30, y1=70, fillcolor="rgba(124, 77, 255, 0.05)", line_width=0, row=2, col=1)

    # Divergence annotations on RSI
    for sig in signals:
        if sig.divergence is not None:
            div = sig.divergence
            color = "#ff1744" if div.divergence_type == "bearish" else "#00e676"
            # Draw line between prior and current RSI points
            t_prior = df.index[div.prior_idx]
            t_current = df.index[div.current_idx]
            fig.add_trace(
                go.Scatter(
                    x=[t_prior, t_current],
                    y=[div.prior_rsi, div.current_rsi],
                    mode="lines+markers",
                    line=dict(color=color, width=2, dash="dash"),
                    marker=dict(size=8, color=color),
                    name=f"{div.divergence_type.title()} Div",
                    showlegend=True,
                ),
                row=2, col=1,
            )

    # === Row 3: Equity Curve (if available) ===
    if has_equity:
        eq = result.equity_curve
        # Create x-axis: use trade indices mapped to dates, or just sequential
        eq_x = list(range(len(eq)))
        fig.add_trace(
            go.Scatter(
                x=eq_x, y=eq,
                mode="lines",
                line=dict(color="#00bcd4", width=2),
                fill="tozeroy",
                fillcolor="rgba(0, 188, 212, 0.1)",
                name="Equity",
            ),
            row=3, col=1,
        )
        fig.update_xaxes(title_text="Trade #", row=3, col=1)

    # === Layout ===
    fig.update_layout(
        template="plotly_dark",
        title=dict(text=title, font=dict(size=18)),
        height=900 if has_equity else 700,
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=10),
        ),
        margin=dict(l=60, r=30, t=80, b=40),
    )

    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)

    fig.write_html(save_path)
    print(f"  [OK] Chart saved to: {save_path}")
    return save_path


def create_matplotlib_chart(
    df: pd.DataFrame,
    signals: List[TradeSignal],
    config: StrategyConfig = None,
    save_path: str = "trading_chart.png",
) -> str:
    """
    Generate a static PNG chart using matplotlib (fallback).

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data.
    signals : List[TradeSignal]
        Trade signals.
    config : StrategyConfig
        Strategy parameters.
    save_path : str
        Output PNG path.

    Returns
    -------
    str
        Path where chart was saved.
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D

    if config is None:
        config = StrategyConfig()

    rsi_series = calculate_rsi(df["Close"], config.rsi_period)
    ranges = detect_ranges(
        df,
        min_candles=config.min_range_candles,
        max_candles=config.max_range_candles,
        range_threshold_pct=config.range_threshold_pct,
    )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12), height_ratios=[3, 1],
                                     gridspec_kw={"hspace": 0.05})
    fig.patch.set_facecolor("#0d1117")
    ax1.set_facecolor("#0d1117")
    ax2.set_facecolor("#0d1117")

    # Candlesticks
    dates = mdates.date2num(df.index.to_pydatetime())
    w = 0.0025
    for i in range(len(df)):
        o, h, l, c = df["Open"].iloc[i], df["High"].iloc[i], df["Low"].iloc[i], df["Close"].iloc[i]
        color = "#00e676" if c >= o else "#ff1744"
        ax1.plot([dates[i], dates[i]], [l, h], color=color, linewidth=0.7)
        body = min(o, c)
        rect = mpatches.FancyBboxPatch(
            (dates[i] - w / 2, body), w, abs(c - o) or 0.001,
            boxstyle="round,pad=0.0005", facecolor=color, edgecolor=color, linewidth=0.5,
        )
        ax1.add_patch(rect)

    # Ranges
    for rng in ranges:
        rect = mpatches.Rectangle(
            (dates[rng.start_idx], rng.range_low),
            dates[rng.end_idx] - dates[rng.start_idx],
            rng.range_high - rng.range_low,
            facecolor="#1e88e5", alpha=0.15, edgecolor="#42a5f5", linewidth=1.2, linestyle="--",
        )
        ax1.add_patch(rect)

    # Signals
    for sig in signals:
        x = dates[sig.entry_idx]
        mc = "#00e676" if sig.direction == Direction.LONG else "#ff1744"
        mk = "^" if sig.direction == Direction.LONG else "v"
        ax1.scatter(x, sig.entry_price, marker=mk, color=mc, s=150, zorder=5,
                    edgecolors="white", linewidths=0.8)
        ax1.hlines(sig.stop_loss, x - w * 8, x + w * 8, colors="#ff6f00", linewidth=1.5, linestyles="--")
        ax1.hlines(sig.take_profit, x - w * 8, x + w * 8, colors="#00bfa5", linewidth=1.5, linestyles="--")

    ax1.set_ylabel("Price", color="#e0e0e0", fontsize=12)
    ax1.tick_params(colors="#9e9e9e")
    ax1.grid(True, alpha=0.1, color="#424242")
    ax1.set_title("ICT AMD Strategy — 5-Min Timeframe", color="#e0e0e0", fontsize=14, fontweight="bold")

    # RSI
    ax2.plot(dates, rsi_series.values, color="#7c4dff", linewidth=1.2)
    ax2.axhline(70, color="#ff1744", linewidth=0.8, linestyle="--", alpha=0.6)
    ax2.axhline(30, color="#00e676", linewidth=0.8, linestyle="--", alpha=0.6)
    ax2.fill_between(dates, 30, 70, alpha=0.04, color="#7c4dff")
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("RSI", color="#e0e0e0", fontsize=12)
    ax2.tick_params(colors="#9e9e9e")
    ax2.grid(True, alpha=0.1, color="#424242")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right", fontsize=8)

    legend_elements = [
        Line2D([0], [0], marker="^", color="#0d1117", markerfacecolor="#00e676", markersize=10, label="LONG"),
        Line2D([0], [0], marker="v", color="#0d1117", markerfacecolor="#ff1744", markersize=10, label="SHORT"),
        Line2D([0], [0], color="#ff6f00", linewidth=1.5, linestyle="--", label="Stop Loss"),
        Line2D([0], [0], color="#00bfa5", linewidth=1.5, linestyle="--", label="Take Profit"),
        mpatches.Patch(facecolor="#1e88e5", alpha=0.3, edgecolor="#42a5f5", linestyle="--", label="Range"),
    ]
    ax1.legend(handles=legend_elements, loc="upper left", fontsize=9,
               facecolor="#1a1a2e", edgecolor="#424242", labelcolor="#e0e0e0")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [OK] Chart saved to: {save_path}")
    return save_path
