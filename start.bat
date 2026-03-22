@echo off
title Trade Hunter
color 0A

echo.
echo  ================================================
echo   Trade Hunter - ICT AMD Trading Bot
echo  ================================================
echo.

REM ── Settings (edit these) ──────────────────────────────────────────────────
set BOT_SYMBOLS=ETHUSD,AVAXUSD,SOLUSD
set BOT_TIMEFRAMES=5m,15m,5m
set BOT_MODE=dry-run
set AUTO_START_BOT=1
set PYTHONIOENCODING=utf-8

REM ── Create venv if missing ─────────────────────────────────────────────────
if not exist "venv\Scripts\python.exe" (
    echo  [1/2] Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo.
        echo  ERROR: Python not found. Install Python 3.10+ and retry.
        pause & exit /b 1
    )
    echo  Done.
    echo.
    goto :install
)

REM ── Check if dependencies are installed ────────────────────────────────────
venv\Scripts\python.exe -c "import streamlit, ta, delta_rest_client" 2>nul
if errorlevel 1 goto :install
goto :launch

:install
echo  [2/2] Installing dependencies into venv...
venv\Scripts\pip.exe install --upgrade pip --quiet
venv\Scripts\pip.exe install -r requirements.txt
echo  Done.
echo.

:launch
echo  Pairs     : %BOT_SYMBOLS%
echo  Timeframes: %BOT_TIMEFRAMES%
echo  Mode      : %BOT_MODE%
echo.
echo  Dashboard : http://localhost:8501
echo  Press Ctrl+C to stop.
echo.

venv\Scripts\python.exe -m streamlit run dashboard.py ^
    --server.port 8501 ^
    --server.headless false ^
    --browser.gatherUsageStats false

pause
