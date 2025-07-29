import requests
import pandas as pd
import numpy as np
import time
import csv
import os
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, EMAIndicator, ADXIndicator, CCIIndicator
from ta.volatility import BollingerBands
from datetime import datetime, timedelta, timezone
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pytz
import json

# === НАСТРОЙКИ ===
TELEGRAM_TOKEN = "8246979603:AAGSP7b-YRol151GlZpfxyyS34rW5ncZJo4"
CHAT_ID = "6677680988"
SYMBOLS = ["EURJPY=X", "EURUSD=X", "CHFJPY=X", "USDCAD=X", "CADJPY=X"]
INTERVAL = 60  # сек
CSV_FILE = "signals.csv"
DELETE_AFTER_MINUTES = 5
PREPARE_SECONDS = 90  # ⏳ Подготовка перед входом
RESULT_LOG_FILE = "results_log.csv"
# =================

def get_timezone():
    try:
        response = requests.get("https://ipinfo.io/json", timeout=5)
        data = response.json()
        timezone_str = data.get("timezone", "UTC")
        return pytz.timezone(timezone_str)
    except Exception as e:
        print(f"Ошибка определения часового пояса: {e}")
        return pytz.timezone("UTC")

LOCAL_TZ = get_timezone()

# Повторы запросов
session = requests.Session()
retry = Retry(total=3, backoff_factor=1, status_forcelist=[502, 503, 504, 429])
session.mount('https://', HTTPAdapter(max_retries=retry))

# Получить данные

def get_data(symbol):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range=5d&interval=1m"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        print(f"⏳ Получаем данные по {symbol.replace('=X','')}")
        response = session.get(url, headers=headers, timeout=10)
        data = response.json()
        if response.status_code != 200 or data['chart']['result'] is None:
            print(f"[{symbol}] Ошибка данных")
            return None
        ts = data['chart']['result'][0]['timestamp']
        prices = data['chart']['result'][0]['indicators']['quote'][0]['close']
        df = pd.DataFrame({"timestamp": ts, "close": prices})
        df.dropna(inplace=True)
        return df
    except Exception as e:
        print(f"[{symbol}] Ошибка: {e}")
        return None

# Анализ сигналов

def analyze(df):
    close = df['close']
    rsi = RSIIndicator(close).rsi()
    macd = MACD(close)
    bb = BollingerBands(close)
    ema12 = EMAIndicator(close, window=12).ema_indicator()
    ema26 = EMAIndicator(close, window=26).ema_indicator()
    ema200 = EMAIndicator(close, window=200).ema_indicator()
    adx = ADXIndicator(high=close, low=close, close=close).adx()
    stochastic = StochasticOscillator(close=close, high=close, low=close).stoch()

    rsi_v = rsi.iloc[-1]
    macd_val = macd.macd().iloc[-1]
    signal_val = macd.macd_signal().iloc[-1]
    ema12_v = ema12.iloc[-1]
    ema26_v = ema26.iloc[-1]
    ema200_v = ema200.iloc[-1]
    adx_v = adx.iloc[-1]
    stoch_v = stochastic.iloc[-1]
    price = close.iloc[-1]
    upper = bb.bollinger_hband().iloc[-1]
    lower = bb.bollinger_lband().iloc[-1]

    if adx_v < 15:
        return "WAIT", round(rsi_v, 2)

    signal_strength = 0

    if rsi_v < 40:
        signal_strength += 1
    if macd_val > signal_val:
        signal_strength += 1
    if ema12_v > ema26_v:
        signal_strength += 1
    if price > ema200_v:
        signal_strength += 1
    if stoch_v < 30:
        signal_strength += 1

    if signal_strength >= 4:
        return "BUY (Strong) ✅", round(rsi_v, 2)
    elif signal_strength == 3:
        return "BUY (Medium)", round(rsi_v, 2)

    signal_strength = 0
    if rsi_v > 60:
        signal_strength += 1
    if macd_val < signal_val:
        signal_strength += 1
    if ema12_v < ema26_v:
        signal_strength += 1
    if price < ema200_v:
        signal_strength += 1
    if stoch_v > 70:
        signal_strength += 1

    if signal_strength >= 4:
        return "SELL (Strong) ✅", round(rsi_v, 2)
    elif signal_strength == 3:
        return "SELL (Medium)", round(rsi_v, 2)

    return "WAIT", round(rsi_v, 2)

# Telegram

def send_telegram_message(msg):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": CHAT_ID, "text": msg}, timeout=10)
    except:
        print("Ошибка Telegram")

# Запись сигнала

def log_signal(symbol, signal, rsi, entry, exit):
    with open(CSV_FILE, 'a', newline='') as f:
        csv.writer(f).writerow([symbol, signal, rsi, entry, exit])

def log_result(symbol, signal, rsi, entry_time):
    with open(RESULT_LOG_FILE, 'a', newline='') as f:
        csv.writer(f).writerow([symbol, signal, rsi, entry_time, datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")])

def clean_old_signals():
    if not os.path.exists(CSV_FILE):
        return
    now = datetime.now(LOCAL_TZ)
    rows = []
    with open(CSV_FILE, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)
        for row in reader:
            try:
                entry_time = datetime.strptime(row[3], "%H:%M:%S")
                if now - entry_time <= timedelta(minutes=DELETE_AFTER_MINUTES):
                    rows.append(row)
            except:
                continue
    with open(CSV_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Symbol", "Signal", "RSI", "Entry Time", "Exit Time"])
        writer.writerows(rows)

# Агрегация и сигнал

def already_sent(symbol, signal, entry_str):
    if not os.path.exists(CSV_FILE):
        return False
    with open(CSV_FILE, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if row[0] == symbol and row[1] == signal and row[3] == entry_str:
                return True
    return False

def send_signal(symbol, signal, rsi):
    now = datetime.now(LOCAL_TZ)
    entry = now + timedelta(seconds=PREPARE_SECONDS)
    exit_ = entry + timedelta(minutes=1)
    entry_str = entry.strftime("%H:%M:%S")
    exit_str = exit_.strftime("%H:%M:%S")

    if already_sent(symbol.replace('=X',''), signal, entry_str):
        return

    msg = (
        f"🚨 СИГНАЛ по {symbol.replace('=X','')}
"
        f"📈 Прогноз: {signal}
"
        f"📊 RSI: {rsi}
"
        f"⏱ Вход: {entry_str} (через {PREPARE_SECONDS} сек)
"
        f"⏳ Выход: {exit_str} (через 1 мин после входа)"
    )

    print(msg)
    send_telegram_message(msg)
    log_signal(symbol.replace('=X',''), signal, rsi, entry_str, exit_str)
    log_result(symbol.replace('=X',''), signal, rsi, entry_str)

    # Уведомление об окончании сделки
    def notify_exit():
        time.sleep(PREPARE_SECONDS + 60)
        done_msg = f"✅ Завершена сделка по {symbol.replace('=X','')} ({signal.split()[0]}) в {exit_str}"
        print(done_msg)
        send_telegram_message(done_msg)

    import threading
    threading.Thread(target=notify_exit).start()

# Главный цикл

def main():
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, 'w', newline='') as f:
            csv.writer(f).writerow(["Symbol", "Signal", "RSI", "Entry Time", "Exit Time"])
    if not os.path.exists(RESULT_LOG_FILE):
        with open(RESULT_LOG_FILE, 'w', newline='') as f:
            csv.writer(f).writerow(["Symbol", "Signal", "RSI", "Entry Time", "Logged At"])

    while True:
        print("🌀 Новый цикл анализа...")
        clean_old_signals()
        for symbol in SYMBOLS:
            df = get_data(symbol)
            if df is not None:
                signal, rsi = analyze(df)
                if signal != "WAIT":
                    send_signal(symbol, signal, rsi)
        print("⏳ Ожидание...")
        time.sleep(INTERVAL)

if __name__ == '__main__':
    main()
