"""
Full multi-pair, multi-timeframe, multi-lookback backtest analysis.
Runs all combinations and prints a ranked summary table.
"""

import subprocess
import sys
import re
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional

PAIRS = [
    "BTCUSD", "ETHUSD", "SOLUSD", "BNBUSD", "ADAUSD",
    "AVAXUSD", "LINKUSD", "ARBUSD", "DOTUSD", "SUIUSD",
    "LTCUSD", "BCHUSD", "APTUSD",
]
TIMEFRAMES = ["5m", "15m", "1h"]
LOOKBACKS  = [720, 1440, 2160]   # 30d / 60d / 90d

ENV = {**os.environ, "PYTHONIOENCODING": "utf-8"}


@dataclass
class Result:
    pair: str
    tf: str
    lookback: int
    trades: int
    wr: float
    pf: float
    ret: float
    maxdd: float
    fees: float
    error: str = ""


def parse(pair: str, tf: str, lookback: int, stdout: str) -> Result:
    def grab(pattern, text, default=0.0):
        m = re.search(pattern, text)
        return float(m.group(1)) if m else default

    trades  = int(grab(r"Total Trades\s+(\d+)", stdout))
    wr_raw  = grab(r"Win Rate\s+([\d.]+)%", stdout)
    pf_raw  = stdout
    pf_m    = re.search(r"Profit Factor\s+([\d.]+|inf)", stdout)
    pf      = float(pf_m.group(1)) if pf_m and pf_m.group(1) != "inf" else 99.0
    ret     = grab(r"Total Return\s+([+-]?[\d.]+)%", stdout)
    maxdd   = grab(r"Max Drawdown %\s+([+-]?[\d.]+)%", stdout)
    fees    = grab(r"Total Fees\s+\$\s*([\d.]+)", stdout)
    return Result(pair, tf, lookback, trades, wr_raw, pf, ret, maxdd, fees)


def run_one(pair: str, tf: str, lookback: int) -> Result:
    cmd = [
        sys.executable, "main.py",
        "--mode", "backtest",
        "--symbol", pair,
        "--resolution", tf,
        "--lookback", str(lookback),
        "--no-chart",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=180, env=ENV)
        out = proc.stdout + proc.stderr
        if "Total Trades" not in out:
            return Result(pair, tf, lookback, 0, 0, 0, 0, 0, 0,
                          error="no trades / fetch error")
        return parse(pair, tf, lookback, out)
    except subprocess.TimeoutExpired:
        return Result(pair, tf, lookback, 0, 0, 0, 0, 0, 0, error="timeout")
    except Exception as e:
        return Result(pair, tf, lookback, 0, 0, 0, 0, 0, 0, error=str(e))


def main():
    jobs = [(p, tf, lb) for p in PAIRS for tf in TIMEFRAMES for lb in LOOKBACKS]
    total = len(jobs)
    print(f"Running {total} backtests  ({len(PAIRS)} pairs × {len(TIMEFRAMES)} TFs × {len(LOOKBACKS)} lookbacks)")
    print("Max workers: 8  |  this will take a few minutes ...\n")

    results = []
    done = 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(run_one, p, tf, lb): (p, tf, lb) for p, tf, lb in jobs}
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            done += 1
            tag = f"{r.pair} {r.tf} {r.lookback}h"
            if r.error:
                print(f"  [{done:3d}/{total}]  {tag:<22}  ERROR: {r.error}")
            else:
                sign = "+" if r.ret >= 0 else ""
                print(f"  [{done:3d}/{total}]  {tag:<22}  "
                      f"T={r.trades:2d}  WR={r.wr:4.0f}%  PF={r.pf:4.2f}  "
                      f"Ret={sign}{r.ret:.2f}%  DD={r.maxdd:.2f}%")

    # ------------------------------------------------------------------ #
    # Summary tables
    # ------------------------------------------------------------------ #
    ok = [r for r in results if not r.error and r.trades >= 3]

    print("\n")
    print("=" * 100)
    print("  FULL RESULTS  —  sorted by Return (min 3 trades)")
    print("=" * 100)
    header = f"{'Pair':<10} {'TF':<5} {'Days':>4}  {'T':>4}  {'WR':>6}  {'PF':>5}  {'Return':>8}  {'MaxDD':>7}  {'Fees':>7}"
    print(header)
    print("-" * 100)
    for r in sorted(ok, key=lambda x: x.ret, reverse=True):
        days = r.lookback // 24
        sign = "+" if r.ret >= 0 else ""
        print(f"{r.pair:<10} {r.tf:<5} {days:>4}  {r.trades:>4}  "
              f"{r.wr:>5.1f}%  {r.pf:>5.2f}  {sign}{r.ret:>7.2f}%  "
              f"{r.maxdd:>6.2f}%  ${r.fees:>6.0f}")

    # Per-pair best TF
    print("\n")
    print("=" * 80)
    print("  BEST TIMEFRAME per pair  (highest return, min 5 trades or best available)")
    print("=" * 80)
    from itertools import groupby
    ok_sorted_pair = sorted(ok, key=lambda x: x.pair)
    for pair, group in groupby(ok_sorted_pair, key=lambda x: x.pair):
        items = list(group)
        # prefer ≥5 trades, otherwise allow ≥3
        candidates = [r for r in items if r.trades >= 5] or items
        best = max(candidates, key=lambda x: x.ret)
        days = best.lookback // 24
        sign = "+" if best.ret >= 0 else ""
        verdict = "TRADE" if best.ret > 0 and best.pf > 1.2 else "SKIP"
        print(f"  {pair:<10} best: {best.tf} {days}d | "
              f"T={best.trades} WR={best.wr:.0f}% PF={best.pf:.2f} Ret={sign}{best.ret:.2f}%  [{verdict}]")

    # Best overall
    print("\n")
    print("=" * 60)
    print("  TOP 10 COMBINATIONS  (min 5 trades, sorted by PF)")
    print("=" * 60)
    top = sorted([r for r in ok if r.trades >= 5], key=lambda x: x.pf, reverse=True)[:10]
    for i, r in enumerate(top, 1):
        days = r.lookback // 24
        sign = "+" if r.ret >= 0 else ""
        print(f"  {i:2d}. {r.pair} {r.tf} {days}d  | "
              f"T={r.trades} WR={r.wr:.0f}% PF={r.pf:.2f} Ret={sign}{r.ret:.2f}% DD={r.maxdd:.2f}%")

    print("\nDone.")


if __name__ == "__main__":
    main()
