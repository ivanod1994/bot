import time
import logging
import asyncio
from datetime import datetime, timedelta, timezone as dt_tz
from typing import Dict, Tuple, List
import os

import numpy as np
import pandas as pd
import pytz
import yfinance as yf

from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, EMAIndicator, ADXIndicator
from ta.volatility import BollingerBands

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, JobQueue

"""
Pocket Option signals bot
-------------------------

This script implements a Telegram bot that monitors a set of currency pairs, computes
technical indicators on 3-minute aggregated data from Yahoo Finance, and sends trading
signals to a Telegram chat. It supports commands for retrieving recent accuracy
statistics (/status) and forcing an immediate analysis with top N signals (/force).

Key features:
  - Aggregates 1-minute OHLC data into 3-minute candles.
  - Computes indicators such as RSI, MACD, Stochastic, Bollinger Bands, EMAs, and ADX.
  - Votes across indicators to determine direction (UP/DOWN/WAIT) and signal strength.
  - Sends only the single best signal per cycle automatically, to avoid spam.
  - Logs every signal with timestamp, symbol, direction, strength label, and price.
  - Calculates approximate accuracy by checking price movement three minutes after signals.
  - Provides commands:
      /start   – explanation of bot behaviour and usage.
      /status  – display hit rate of recent signals.
      /force   – send up to top 3 signals immediately based on current analysis.

Note: This bot requires the `python-telegram-bot` library and `ta` technical analysis
library. Ensure dependencies are installed before running.
"""

# ================= CONFIGURATION =================
TELEGRAM_TOKEN = "8246979603:AAGSP7b-YRol151GlZpfxyyS34rW5ncZJo4"
CHAT_ID = "6677680988"

# Currency pairs to monitor (Yahoo Finance tickers)
SYMBOLS = ["EURUSD=X", "EURJPY=X", "USDJPY=X", "GBPUSD=X"]
# How often to run analysis in minutes
INTERVAL_MINUTES = 1
# Timezone for resampling and timestamps
TIMEZONE = "Africa/Algiers"
# Only send strong signals automatically
ONLY_STRONG = True

# File to store logged signals
LOG_CSV = "signals_log.csv"

# Timezone object for convenience
tz = pytz.timezone(TIMEZONE)

# ================= LOGGER SETUP =================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ================= HELPER FUNCTIONS =================
def now_tz() -> datetime:
    """Return current time aware of configured timezone."""
    return datetime.now(tz)


def to_pair(symbol: str) -> str:
    """Pretty-print a Yahoo ticker as currency pair (e.g. 'EURUSD=X' → 'EUR/USD')."""
    sym = symbol.replace("=X", "").upper()
    return f"{sym[:3]}/{sym[3:6]}"


def resample_to_3m(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate 1-minute OHLCV data into 3-minute candles.

    Pandas 1-minute frequency code 'T' is deprecated, so use 'min'. We ensure the
    DateTimeIndex is timezone-aware and converted to the configured timezone.
    """
    if df.empty:
        return df.copy()
    # Localize/convert index to timezone aware
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC").tz_convert(tz)
    else:
        df.index = df.index.tz_convert(tz)
    o = df["Open"].resample("3min").first()
    h = df["High"].resample("3min").max()
    l = df["Low"].resample("3min").min()
    c = df["Close"].resample("3min").last()
    v = df["Volume"].resample("3min").sum()
    out = pd.concat([o, h, l, c, v], axis=1)
    out.columns = ["Open", "High", "Low", "Close", "Volume"]
    return out.dropna().copy()


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a set of technical indicators for the DataFrame.

    Adds columns: rsi, macd, macd_signal, macd_hist, stoch_k, stoch_d,
    bb_high, bb_low, ema20, ema50, adx.
    """
    if df.empty:
        return df
    close, high, low = df["Close"], df["High"], df["Low"]
    df["rsi"] = RSIIndicator(close=close, window=14).rsi()
    macd = MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()
    stoch = StochasticOscillator(high=high, low=low, close=close, window=14, smooth_window=3)
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()
    bb = BollingerBands(close=close, window=20, window_dev=2)
    df["bb_high"] = bb.bollinger_hband()
    df["bb_low"] = bb.bollinger_lband()
    df["ema20"] = EMAIndicator(close=close, window=20).ema_indicator()
    df["ema50"] = EMAIndicator(close=close, window=50).ema_indicator()
    df["adx"] = ADXIndicator(high=high, low=low, close=close, window=14).adx()
    return df


def indicator_votes(row: pd.Series) -> Tuple[str, int, Dict[str, str]]:
    """
    Aggregate votes from multiple indicators for a single candle.

    Returns (direction, votes, reasons), where direction is 'UP', 'DOWN' or 'WAIT',
    votes is the absolute sum of votes, and reasons lists each indicator's result.
    Indicators vote as follows:
      - RSI: oversold (<30) → UP; overbought (>70) → DOWN; else WAIT.
      - MACD: MACD > signal & hist > 0 → UP; MACD < signal & hist < 0 → DOWN.
      - Stoch: %K < 20 and rising (K > D) → UP; %K > 80 and falling (K < D) → DOWN.
      - Bollinger Bands: Close at/below lower band → UP; Close at/above upper band → DOWN.
    A net vote of +2 or higher is required for UP, -2 or lower for DOWN.
    """
    votes = 0
    reasons: Dict[str, str] = {}
    # RSI
    if row["rsi"] < 30:
        votes += 1
        reasons["RSI"] = "UP"
    elif row["rsi"] > 70:
        votes -= 1
        reasons["RSI"] = "DOWN"
    else:
        reasons["RSI"] = "WAIT"
    # MACD
    if row["macd"] > row["macd_signal"] and row["macd_hist"] > 0:
        votes += 1
        reasons["MACD"] = "UP"
    elif row["macd"] < row["macd_signal"] and row["macd_hist"] < 0:
        votes -= 1
        reasons["MACD"] = "DOWN"
    else:
        reasons["MACD"] = "WAIT"
    # Stochastic
    if row["stoch_k"] < 20 and row["stoch_k"] > row["stoch_d"]:
        votes += 1
        reasons["Stoch"] = "UP"
    elif row["stoch_k"] > 80 and row["stoch_k"] < row["stoch_d"]:
        votes -= 1
        reasons["Stoch"] = "DOWN"
    else:
        reasons["Stoch"] = "WAIT"
    # Bollinger Band touches
    if row["Close"] <= row["bb_low"]:
        votes += 1
        reasons["BB"] = "UP"
    elif row["Close"] >= row["bb_high"]:
        votes -= 1
        reasons["BB"] = "DOWN"
    else:
        reasons["BB"] = "WAIT"
    direction = "WAIT"
    if votes >= 2:
        direction = "UP"
    elif votes <= -2:
        direction = "DOWN"
    return direction, abs(votes), reasons


def strength_label(direction: str, strength: int, row: pd.Series) -> str:
    """
    Determine a strength label for a signal given direction and vote count.

    Uses ADX as a boost: if ADX ≥ 25, adds 1 to the score. Score ≥ 3 → STRONG,
    score == 2 → MEDIUM, otherwise WEAK. 'WAIT' signals remain 'WAIT'.
    """
    if direction == "WAIT":
        return "WAIT"
    boost = 1 if row["adx"] >= 25 else 0
    score = strength + boost
    if score >= 3:
        return "STRONG"
    if score == 2:
        return "MEDIUM"
    return "WEAK"


# =============== LOGGING AND ACCURACY ===============
def append_log(symbol: str, direction: str, label: str, price: float, t_utc: datetime) -> None:
    """Append a signal record to the CSV log file."""
    exists = os.path.exists(LOG_CSV)
    row = {
        "time_utc": t_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "symbol": symbol,
        "direction": direction,
        "label": label,
        "price": price,
    }
    df = pd.DataFrame([row])
    if exists:
        df.to_csv(LOG_CSV, mode="a", index=False, header=False, encoding="utf-8")
    else:
        df.to_csv(LOG_CSV, index=False, encoding="utf-8")


def calc_accuracy(window: int = 50) -> Tuple[int, int, float]:
    """
    Compute a rough hit-rate for recent signals.

    For each logged signal in the last `window` entries (and within ~90 minutes),
    download recent price data and check if the price moved in the predicted
    direction after ~3 minutes. Returns (hits, total_signals, accuracy_percent).
    """
    if not os.path.exists(LOG_CSV):
        return 0, 0, 0.0
    df = pd.read_csv(LOG_CSV)
    if df.empty:
        return 0, 0, 0.0
    df = df.tail(window).copy()
    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True)
    # Only consider signals from the last 90 minutes
    min_time = datetime.now(dt_tz.utc) - timedelta(minutes=90)
    df = df[df["time_utc"] >= min_time]
    if df.empty:
        return 0, 0, 0.0
    hits = 0
    total = 0
    # Group by symbol to minimize downloads
    for sym, chunk in df.groupby("symbol"):
        data = yf.download(sym, period="90m", interval="1m", progress=False, auto_adjust=False)
        if data is None or data.empty:
            continue
        # Ensure index is UTC aware
        if data.index.tz is None:
            data.index = data.index.tz_localize("UTC")
        else:
            data.index = data.index.tz_convert("UTC")
        for _, r in chunk.iterrows():
            t3 = r["time_utc"] + timedelta(minutes=3)
            fut = data[data.index >= t3]
            if fut.empty:
                continue
            fut_close = float(fut["Close"].iloc[0])
            total += 1
            if r["direction"] == "UP" and fut_close > r["price"]:
                hits += 1
            elif r["direction"] == "DOWN" and fut_close < r["price"]:
                hits += 1
    acc = (hits / total * 100.0) if total else 0.0
    return hits, total, acc


# ================= ANALYSIS =================
async def analyze(context: ContextTypes.DEFAULT_TYPE) -> List[dict]:
    """
    Perform analysis for all configured symbols and return a list of signal candidates.

    Each candidate is a dictionary containing symbol, pair, direction, label,
    price, RSI, ADX, score, and timestamp. The list is sorted by descending
    score. Only signals meeting the ONLY_STRONG criteria are included.
    """
    end = datetime.now(dt_tz.utc)
    start = end - timedelta(hours=3)
    candidates: List[dict] = []
    for symbol in SYMBOLS:
        try:
            raw = yf.download(symbol, start=start, end=end, interval="1m", progress=False, auto_adjust=False)
            if raw.empty:
                continue
            df3 = resample_to_3m(raw)
            df3 = compute_indicators(df3)
            if len(df3) < 2:
                continue
            row = df3.iloc[-1]
            direction, votes, reasons = indicator_votes(row)
            label = strength_label(direction, votes, row)
            price = float(row["Close"])
            if direction in ("UP", "DOWN") and (label == "STRONG" or not ONLY_STRONG):
                score = votes + (1 if row["adx"] >= 25 else 0)
                candidates.append({
                    "symbol": symbol,
                    "pair": to_pair(symbol),
                    "direction": direction,
                    "label": label,
                    "price": price,
                    "rsi": float(row["rsi"]),
                    "adx": float(row["adx"]),
                    "score": int(score),
                    "t_utc": end,
                })
        except Exception as e:
            logging.exception(f"Ошибка анализа {symbol}: {e}")
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates


async def analyze_and_notify(context: ContextTypes.DEFAULT_TYPE, manual: bool = False) -> None:
    """
    Analyze symbols and send signals to Telegram.

    In manual mode (/force), up to top 3 signals are returned. In automatic mode,
    only the single best signal is sent. If no signals are found in manual mode,
    a message indicating no signals is sent. All sent signals are logged.
    """
    candidates = await analyze(context)
    if manual:
        if not candidates:
            await context.bot.send_message(chat_id=CHAT_ID, text="❌ Сейчас нет подходящих сигналов.")
            return
        top = candidates[:3]
        header = "🛑 РУЧНОЙ ЗАПРОС — ТОП-3 СИГНАЛА"
        lines: List[str] = []
        for i, s in enumerate(top, 1):
            lines.append(
                f"{i}) {s['pair']}: *{s['direction']}* ({s['label']}) | "
                f"Цена `{s['price']:.5f}` | RSI {s['rsi']:.1f} | ADX {s['adx']:.1f}"
            )
            append_log(s["symbol"], s["direction"], s["label"], s["price"], s["t_utc"])
        msg = header + "\n" + "\n".join(lines)
        await context.bot.send_message(chat_id=CHAT_ID, text=msg, parse_mode="Markdown")
    else:
        if not candidates:
            return
        s = candidates[0]
        msg = (
            f"🚨 ЛУЧШИЙ СИГНАЛ {s['pair']}\n"
            f"Направление: *{s['direction']}* | Сила: *{s['label']}*\n"
            f"Цена: `{s['price']:.5f}` | RSI: {s['rsi']:.1f} | ADX: {s['adx']:.1f}"
        )
        await context.bot.send_message(chat_id=CHAT_ID, text=msg, parse_mode="Markdown")
        append_log(s["symbol"], s["direction"], s["label"], s["price"], s["t_utc"])


# ================= TELEGRAM HANDLERS =================
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a welcome message and instructions when /start is used."""
    await update.message.reply_text(
        "Бот запущен. Планово шлёт 1 лучший сигнал/цикл.\n"
        "/status — точность, /force — ТОП-3 сигналов на сейчас."
    )


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Calculate and report the accuracy of recent signals."""
    hits, total, acc = calc_accuracy(window=50)
    if total == 0:
        await update.message.reply_text(
            "Недостаточно свежих данных для оценки точности."
        )
    else:
        await update.message.reply_text(
            f"🎯 Точность: {acc:.1f}% ({hits}/{total} попаданий)"
        )


async def cmd_force(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Trigger a manual analysis returning up to the top 3 signals."""
    await update.message.reply_text("Запускаю внеплановый анализ…")
    await analyze_and_notify(context, manual=True)


# ================= MAIN =================
def main() -> None:
    """Start the Telegram bot and schedule periodic analysis."""
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    application.add_handler(CommandHandler("start", cmd_start))
    application.add_handler(CommandHandler("status", cmd_status))
    application.add_handler(CommandHandler("force", cmd_force))
    jq: JobQueue = application.job_queue
    jq.run_repeating(analyze_and_notify, interval=INTERVAL_MINUTES * 60, first=5)
    logging.info("Бот запущен.")
    application.run_polling(close_loop=False)


if __name__ == "__main__":
    main()