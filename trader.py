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

# ================= НАСТРОЙКИ =================
TELEGRAM_TOKEN = "8246979603:AAGSP7b-YRol151GlZpfxyyS34rW5ncZJo4"
CHAT_ID = "6677680988"

SYMBOLS = ["EURUSD=X", "EURJPY=X", "USDJPY=X", "GBPUSD=X"]
INTERVAL_MINUTES = 1
TIMEZONE = "Africa/Algiers"
ONLY_STRONG = True

LOG_CSV = "signals_log.csv"
tz = pytz.timezone(TIMEZONE)

# ================= ЛОГИ =================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ================= УТИЛИТЫ =================
def now_tz():
    return datetime.now(tz)

def to_pair(symbol: str) -> str:
    return symbol.replace("=X", "").upper()[:3] + "/" + symbol.replace("=X", "").upper()[3:6]

def resample_to_3m(df: pd.DataFrame) -> pd.DataFrame:
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
    return out.dropna()

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close, high, low = df["Close"], df["High"], df["Low"]

    df["rsi"] = RSIIndicator(close=close, window=14).rsi()
    macd = MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
    df["macd"], df["macd_signal"], df["macd_hist"] = macd.macd(), macd.macd_signal(), macd.macd_diff()

    stoch = StochasticOscillator(high=high, low=low, close=close, window=14, smooth_window=3)
    df["stoch_k"], df["stoch_d"] = stoch.stoch(), stoch.stoch_signal()

    bb = BollingerBands(close=close, window=20, window_dev=2)
    df["bb_high"], df["bb_low"] = bb.bollinger_hband(), bb.bollinger_lband()

    df["ema20"] = EMAIndicator(close=close, window=20).ema_indicator()
    df["ema50"] = EMAIndicator(close=close, window=50).ema_indicator()
    df["adx"] = ADXIndicator(high=high, low=low, close=close, window=14).adx()
    return df

def indicator_votes(row: pd.Series) -> Tuple[str, int, Dict[str, str]]:
    votes = 0
    reasons = {}

    if row["rsi"] < 30:
        votes += 1; reasons["RSI"] = "UP"
    elif row["rsi"] > 70:
        votes -= 1; reasons["RSI"] = "DOWN"
    else:
        reasons["RSI"] = "WAIT"

    if row["macd"] > row["macd_signal"] and row["macd_hist"] > 0:
        votes += 1; reasons["MACD"] = "UP"
    elif row["macd"] < row["macd_signal"] and row["macd_hist"] < 0:
        votes -= 1; reasons["MACD"] = "DOWN"
    else:
        reasons["MACD"] = "WAIT"

    if row["stoch_k"] < 20 and row["stoch_k"] > row["stoch_d"]:
        votes += 1; reasons["Stoch"] = "UP"
    elif row["stoch_k"] > 80 and row["stoch_k"] < row["stoch_d"]:
        votes -= 1; reasons["Stoch"] = "DOWN"
    else:
        reasons["Stoch"] = "WAIT"

    if row["Close"] <= row["bb_low"]:
        votes += 1; reasons["BB"] = "UP"
    elif row["Close"] >= row["bb_high"]:
        votes -= 1; reasons["BB"] = "DOWN"
    else:
        reasons["BB"] = "WAIT"

    direction = "WAIT"
    if votes >= 2:
        direction = "UP"
    elif votes <= -2:
        direction = "DOWN"

    return direction, abs(votes), reasons

def strength_label(direction: str, strength: int, row: pd.Series) -> str:
    boost = 1 if row["adx"] >= 25 else 0
    score = strength + boost
    if direction == "WAIT": return "WAIT"
    if score >= 3: return "STRONG"
    if score == 2: return "MEDIUM"
    return "WEAK"

# =============== ЛОГИРОВАНИЕ И ТОЧНОСТЬ ===============
def append_log(symbol: str, direction: str, label: str, price: float, t_utc: datetime):
    exists = os.path.exists(LOG_CSV)
    row = {
        "time_utc": t_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "symbol": symbol,
        "direction": direction,
        "label": label,
        "price": float(price),
    }
    df = pd.DataFrame([row])
    if exists:
        df.to_csv(LOG_CSV, mode="a", index=False, header=False, encoding="utf-8")
    else:
        df.to_csv(LOG_CSV, index=False, encoding="utf-8")

def calc_accuracy(window: int = 50) -> Tuple[int, int, float]:
    if not os.path.exists(LOG_CSV):
        return 0, 0, 0.0
    df = pd.read_csv(LOG_CSV)
    if df.empty:
        return 0, 0, 0.0

    df = df.tail(window).copy()
    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True)
    min_time = datetime.now(dt_tz.utc) - timedelta(minutes=90)
    df = df[df["time_utc"] >= min_time]
    if df.empty:
        return 0, 0, 0.0

    hits, total = 0, 0
    for sym, chunk in df.groupby("symbol"):
        data = yf.download(
            sym, period="90m", interval="1m", progress=False, auto_adjust=False
        )
        if data is None or data.empty:
            continue
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

# ================= АНАЛИЗ =================
async def analyze(context: ContextTypes.DEFAULT_TYPE) -> List[dict]:
    end = datetime.now(dt_tz.utc)
    start = end - timedelta(hours=3)
    candidates: List[dict] = []

    for symbol in SYMBOLS:
        try:
            raw = yf.download(
                symbol,
                start=start,
                end=end,
                interval="1m",
                progress=False,
                auto_adjust=False
            )
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
            logging.exception(f"Ошибка {symbol}: {e}")

    # сортировка по силе (score) по убыванию
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates

async def analyze_and_notify(context: ContextTypes.DEFAULT_TYPE, manual: bool = False):
    candidates = await analyze(context)

    if manual:
        if not candidates:
            return await context.bot.send_message(chat_id=CHAT_ID, text="❌ Сейчас нет подходящих сигналов.")
        # ТОП-3
        top = candidates[:3]
        header = "🛑 РУЧНОЙ ЗАПРОС — ТОП-3 СИГНАЛА"
        lines = []
        for i, s in enumerate(top, 1):
            lines.append(
                f"{i}) {s['pair']}: *{s['direction']}* ({s['label']}) | "
                f"Цена `{s['price']:.5f}` | RSI {s['rsi']:.1f} | ADX {s['adx']:.1f}"
            )
            # логируем каждый сигнал из топа
            append_log(s["symbol"], s["direction"], s["label"], s["price"], s["t_utc"])
        msg = header + "\n" + "\n".join(lines)
        return await context.bot.send_message(chat_id=CHAT_ID, text=msg, parse_mode="Markdown")

    # плановый режим — один лучший
    if candidates:
        s = candidates[0]
        msg = (
            f"🚨 ЛУЧШИЙ СИГНАЛ {s['pair']}\n"
            f"Направление: *{s['direction']}* | Сила: *{s['label']}*\n"
            f"Цена: `{s['price']:.5f}` | RSI: {s['rsi']:.1f} | ADX: {s['adx']:.1f}"
        )
        await context.bot.send_message(chat_id=CHAT_ID, text=msg, parse_mode="Markdown")
        append_log(s["symbol"], s["direction"], s["label"], s["price"], s["t_utc"])

# ================= КОМАНДЫ =================
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Бот запущен. Планово шлёт 1 лучший сигнал/цикл.\n"
        "/status — точность, /force — ТОП-3 сигналов на сейчас."
    )

async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    hits, total, acc = calc_accuracy(window=50)
    if total == 0:
        return await update.message.reply_text("Недостаточно свежих данных для оценки точности.")
    await update.message.reply_text(f"🎯 Точность: {acc:.1f}% ({hits}/{total} попаданий)")

async def cmd_force(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Запускаю внеплановый анализ…")
    await analyze_and_notify(context, manual=True)

# ================= MAIN =================
def main():
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
