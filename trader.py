import time
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Tuple, List

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

SYMBOLS = ["EURUSD=X", "EURJPY=X", "USDJPY=X", "GBPUSD=X"]  # валютные пары
INTERVAL_MINUTES = 1  # как часто проверять (в минутах)
TIMEZONE = "Africa/Algiers"
ONLY_STRONG = True  # отправлять только сильные сигналы

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
    """Агрегируем минутные свечи в 3-минутные (без FutureWarning: use 'min')."""
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

# ================= АНАЛИЗ (ОДНО ЛУЧШЕЕ ОКНО) =================
async def analyze_and_notify(context: ContextTypes.DEFAULT_TYPE):
    end = datetime.utcnow()
    start = end - timedelta(hours=3)
    best_signal = None

    for symbol in SYMBOLS:
        try:
            raw = yf.download(symbol, start=start, end=end, interval="1m", progress=False)
            if raw.empty:
                continue
            df3 = resample_to_3m(raw)
            df3 = compute_indicators(df3)
            if len(df3) < 2:
                continue
            row = df3.iloc[-1]

            direction, votes, reasons = indicator_votes(row)
            label = strength_label(direction, votes, row)
            price = row["Close"]

            if direction in ("UP", "DOWN"):
                if label == "STRONG" or not ONLY_STRONG:
                    score = votes + (1 if row["adx"] >= 25 else 0)
                    if not best_signal or score > best_signal["score"]:
                        best_signal = {
                            "symbol": symbol,
                            "pair": to_pair(symbol),
                            "direction": direction,
                            "label": label,
                            "price": price,
                            "rsi": row["rsi"],
                            "adx": row["adx"],
                            "score": score
                        }
        except Exception as e:
            logging.exception(f"Ошибка {symbol}: {e}")

    if best_signal:
        msg = (
            f"🚨 ЛУЧШИЙ СИГНАЛ {best_signal['pair']}\n"
            f"Направление: *{best_signal['direction']}* | Сила: *{best_signal['label']}*\n"
            f"Цена: `{best_signal['price']:.5f}` | RSI: {best_signal['rsi']:.1f} | ADX: {best_signal['adx']:.1f}"
        )
        await context.bot.send_message(chat_id=CHAT_ID, text=msg, parse_mode="Markdown")

# ================= КОМАНДЫ =================
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Бот запущен. Буду присылать только один лучший сигнал за цикл.")

# ================= MAIN =================
def main():
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    application.add_handler(CommandHandler("start", cmd_start))

    jq: JobQueue = application.job_queue
    jq.run_repeating(analyze_and_notify, interval=INTERVAL_MINUTES * 60, first=5)

    logging.info("Бот запущен.")
    application.run_polling(close_loop=False)

if __name__ == "__main__":
    main()
