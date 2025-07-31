import requests
import pandas as pd
import numpy as np
import time
import csv
import os
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, EMAIndicator, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from datetime import datetime, timedelta, timezone
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pytz
import threading
try:
    from bs4 import BeautifulSoup
    BEAUTIFULSOUP_AVAILABLE = True
except ImportError:
    BEAUTIFULSOUP_AVAILABLE = False
    print("Библиотека BeautifulSoup не установлена. Установите: pip install beautifulsoup4")
try:
    from alpha_vantage.foreignexchange import ForeignExchange
    ALPHA_VANTAGE_AVAILABLE = True
except ImportError:
    ALPHA_VANTAGE_AVAILABLE = False
    print("Библиотека alpha_vantage не установлена. Установите: pip install alpha-vantage")

# === НАСТРОЙКИ ===
TELEGRAM_TOKEN = "8246979603:AAGSP7b-YRol151GlZpfxyyS34rW5ncZJo4"
CHAT_ID = "6677680988"
SYMBOLS = ["EURJPY=X", "EURUSD=X", "CHFJPY=X", "USDCAD=X", "CADJPY=X", "GBPUSD=X", "AUDUSD=X"]
SYMBOLS_ALPHA = ["EUR/JPY", "EUR/USD", "CHF/JPY", "USD/CAD", "CAD/JPY", "GBP/USD", "AUD/USD"]
INTERVAL = 300  # 5 минут
CSV_FILE = "signals.csv"
DELETE_AFTER_MINUTES = 5
PREPARE_SECONDS = 90
RESULT_LOG_FILE = "results_log.csv"
MANUAL_TZ = "Africa/Algiers"
CONFIRMATION_CANDLES = 4
PAYOUT = 0.85
TIMEOUT = 20
MIN_SIGNAL_INTERVAL = 300
VOLUME_MULTIPLIER = float('inf')  # Отключаем фильтр объема
ALPHA_VANTAGE_API_KEY = "YOUR_ALPHA_VANTAGE_API_KEY"  # Замените на ваш ключ
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
# =================

data_cache = {}
last_signal_time = {symbol: None for symbol in SYMBOLS}

def get_timezone():
    if MANUAL_TZ:
        return pytz.timezone(MANUAL_TZ)
    try:
        response = requests.get("https://ipinfo.io/json", timeout=5)
        data = response.json()
        timezone_str = data.get("timezone", "UTC")
        return pytz.timezone(timezone_str)
    except Exception as e:
        print(f"Ошибка определения часового пояса: {e}")
        return pytz.timezone("UTC")

LOCAL_TZ = get_timezone()

session = requests.Session()
retry = Retry(total=5, backoff_factor=2, status_forcelist=[429, 502, 503, 504])
session.mount('https://', HTTPAdapter(max_retries=retry))

def is_active_session():
    now = datetime.now(LOCAL_TZ)
    hour = now.hour
    return 8 <= hour <= 22

def is_news_time():
    if not BEAUTIFULSOUP_AVAILABLE:
        print("Новостной фильтр недоступен: установите beautifulsoup4")
        return False
    try:
        url = "https://www.investing.com/economic-calendar/"
        response = session.get(url, headers=HEADERS, timeout=TIMEOUT)
        soup = BeautifulSoup(response.text, 'html.parser')
        events = soup.find_all('tr', {'data-event-id': True})
        now = datetime.now(LOCAL_TZ)
        for event in events:
            time_elem = event.find('td', class_='time')
            if not time_elem:
                continue
            time_str = time_elem.text.strip()
            try:
                event_time = datetime.strptime(time_str, "%H:%M").replace(
                    year=now.year, month=now.month, day=now.day, tzinfo=LOCAL_TZ
                )
                if abs((now - event_time).total_seconds()) < 1800:  # 30 минут
                    return True
            except ValueError:
                continue
        return False
    except Exception as e:
        print(f"Ошибка проверки новостей: {e}")
        return False

def check_internet():
    try:
        requests.get("https://www.google.com", timeout=5)
        return True
    except:
        return False

def get_data(symbol, interval="5m", period="7d"):
    cache_key = f"{symbol}_{interval}"
    if cache_key in data_cache and (datetime.now() - data_cache[cache_key]['time']).seconds < 300:
        print(f"[{symbol}] Используем кэшированные данные")
        return data_cache[cache_key]['data']
    
    # Попытка через Yahoo Finance API
    for attempt in range(3):
        try:
            print(f"⏳ Получаем данные по {symbol.replace('=X','')} (интервал {interval}, Yahoo Finance, попытка {attempt+1})")
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range={period}&interval={interval}"
            response = session.get(url, headers=HEADERS, timeout=TIMEOUT)
            print(f"[{symbol}] HTTP статус: {response.status_code}")
            if response.status_code == 429:
                print(f"[{symbol}] Ошибка 429: Слишком много запросов, ожидание...")
                time.sleep(5 ** attempt)
                continue
            data = response.json()
            if data['chart']['result'] is None:
                print(f"[{symbol}] Ошибка данных: {data.get('chart', {}).get('error', 'Нет деталей ошибки')}")
                continue
            result = data['chart']['result'][0]
            ts = result['timestamp']
            quote = result['indicators']['quote'][0]
            df = pd.DataFrame({
                "timestamp": [datetime.fromtimestamp(t, tz=LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S") for t in ts],
                "close": quote['close'],
                "high": quote['high'],
                "low": quote['low'],
                "open": quote['open'],
                "volume": quote['volume']
            })
            df.dropna(inplace=True)
            if len(df) < 200:
                print(f"[{symbol}] Недостаточно данных (Yahoo Finance): {len(df)} свечей")
                continue
            data_cache[cache_key] = {'data': df, 'time': datetime.now()}
            return df
        except Exception as e:
            print(f"[{symbol}] Ошибка Yahoo Finance: {str(e)}")
            time.sleep(5 ** attempt)
    
    # Резервный источник: Alpha Vantage
    if ALPHA_VANTAGE_AVAILABLE and ALPHA_VANTAGE_API_KEY != "YOUR_ALPHA_VANTAGE_API_KEY":
        for attempt in range(3):
            try:
                print(f"⏳ Получаем данные по {symbol.replace('=X','')} (интервал 5min, Alpha Vantage, попытка {attempt+1})")
                alpha_symbol = SYMBOLS_ALPHA[SYMBOLS.index(symbol)]
                fx = ForeignExchange(key=ALPHA_VANTAGE_API_KEY)
                data, _ = fx.get_currency_exchange_intraday(symbol=alpha_symbol, interval="5min", outputsize="full")
                
                if not data:
                    print(f"[{symbol}] Пустые данные (Alpha Vantage): {data}")
                    continue
                
                df = pd.DataFrame(data).transpose().reset_index()
                df.columns = ['timestamp', 'open', 'high', 'low', 'close']
                df['open'] = df['open'].astype(float)
                df['high'] = df['high'].astype(float)
                df['low'] = df['low'].astype(float)
                df['close'] = df['close'].astype(float)
                df['volume'] = 0  # Alpha Vantage не предоставляет объем для forex
                df['timestamp'] = pd.to_datetime(df['timestamp']).dt.astimezone(LOCAL_TZ).dt.strftime("%Y-%m-%d %H:%M:%S")
                df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
                df.dropna(inplace=True)
                
                if len(df) < 200:
                    print(f"[{symbol}] Недостаточно данных (Alpha Vantage): {len(df)} свечей")
                    continue
                data_cache[cache_key] = {'data': df, 'time': datetime.now()}
                return df
            except Exception as e:
                print(f"[{symbol}] Ошибка Alpha Vantage: {str(e)}")
                time.sleep(5 ** attempt)
        send_telegram_message(f"⚠️ Ошибка получения данных для {symbol.replace('=X','')} ({interval}, Alpha Vantage): все попытки провалились")
    
    # Если все источники не сработали
    if cache_key in data_cache:
        print(f"[{symbol}] Используем старые кэшированные данные")
        return data_cache[cache_key]['data']
    send_telegram_message(f"⚠️ Ошибка получения данных для {symbol.replace('=X','')} ({interval}): все источники недоступны")
    return None

def analyze(symbol, df_5m, df_15m=None):
    # Проверка наличия достаточного количества данных
    if len(df_5m) < 50:
        reason = "Недостаточно данных для анализа (менее 50 свечей)"
        log_result(symbol.replace('=X',''), "WAIT", 0, datetime.now(LOCAL_TZ).strftime("%H:%M:%S"), reason, 0, 0, 0, 0, 0, 0, 0, 0.0)
        return "WAIT", 0, 0, 0, 0, reason, 0, 0, 0, 0, 0
    
    close = df_5m['close']
    high = df_5m['high']
    low = df_5m['low']
    open = df_5m['open']
    
    # Индикаторы
    rsi = RSIIndicator(close, window=14).rsi()
    macd = MACD(close, window_slow=26, window_fast=12, window_sign=9)
    bb = BollingerBands(close, window=20, window_dev=2)
    ema12 = EMAIndicator(close, window=12).ema_indicator()
    ema26 = EMAIndicator(close, window=26).ema_indicator()
    ema200 = EMAIndicator(close, window=200).ema_indicator()
    adx = ADXIndicator(high=high, low=low, close=close, window=14).adx()
    stochastic = StochasticOscillator(close=close, high=high, low=low, window=14, smooth_window=3).stoch()
    atr = AverageTrueRange(high=high, low=low, close=close, window=14).average_true_range()

    # Текущие значения
    rsi_v = rsi.iloc[-1]
    macd_val = macd.macd().iloc[-1]
    signal_val = macd.macd_signal().iloc[-1]
    ema12_v = ema12.iloc[-1]
    ema26_v = ema26.iloc[-1]
    ema200_v = ema200.iloc[-1]
    adx_v = adx.iloc[-1]
    stoch_v = stochastic.iloc[-1]
    price = close.iloc[-1]
    open_price = open.iloc[-1]
    upper_bb = bb.bollinger_hband().iloc[-1]
    lower_bb = bb.bollinger_lband().iloc[-1]
    atr_v = atr.iloc[-1]
    bb_width = (upper_bb - lower_bb) / price

    # Адаптивные пороги на основе последних 50 свечей
    rsi_mean = rsi[-50:].mean()
    rsi_std = rsi[-50:].std()
    adx_mean = adx[-50:].mean()
    bb_width_series = (bb.bollinger_hband()[-50:] - bb.bollinger_lband()[-50:]) / close[-50:]
    bb_width_mean = bb_width_series.mean()

    RSI_BUY_THRESHOLD = max(30, rsi_mean - rsi_std)  # Динамический RSI для покупки
    RSI_SELL_THRESHOLD = min(70, rsi_mean + rsi_std)  # Динамический RSI для продажи
    MIN_ADX = max(10, adx_mean * 0.8)  # Динамический ADX
    BB_WIDTH_MIN = max(0.001, bb_width_mean * 0.5)  # Динамическая ширина Bollinger Bands

    # Логирование индикаторов
    reason = f"RSI: {rsi_v:.2f}, ADX: {adx_v:.2f}, Stochastic: {stoch_v:.2f}, MACD: {macd_val:.4f}, Signal: {signal_val:.4f}, ATR: {atr_v:.4f}, BB_Width: {bb_width:.4f}"
    reason += f"; Адаптивные пороги: RSI_BUY={RSI_BUY_THRESHOLD:.2f}, RSI_SELL={RSI_SELL_THRESHOLD:.2f}, MIN_ADX={MIN_ADX:.2f}, BB_WIDTH_MIN={BB_WIDTH_MIN:.4f}"
    print(f"[{symbol}] {reason}")

    # Фильтры
    if adx_v < MIN_ADX:
        reason += f"; ADX слишком низкий (< {MIN_ADX})"
        log_result(symbol.replace('=X',''), "WAIT", round(rsi_v, 2), datetime.now(LOCAL_TZ).strftime("%H:%M:%S"), reason, rsi_v, adx_v, stoch_v, macd_val, signal_val, atr_v, price, 0.0)
        return "WAIT", round(rsi_v, 2), 0, price, atr_v, reason, rsi_v, adx_v, stoch_v, macd_val, signal_val
    if bb_width < BB_WIDTH_MIN:
        reason += "; Узкие Bollinger Bands"
        log_result(symbol.replace('=X',''), "WAIT", round(rsi_v, 2), datetime.now(LOCAL_TZ).strftime("%H:%M:%S"), reason, rsi_v, adx_v, stoch_v, macd_val, signal_val, atr_v, price, 0.0)
        return "WAIT", round(rsi_v, 2), 0, price, atr_v, reason, rsi_v, adx_v, stoch_v, macd_val, signal_val
    if not is_active_session() and "JPY" in symbol:
        reason += "; Торговля вне активной сессии для JPY"
        log_result(symbol.replace('=X',''), "WAIT", round(rsi_v, 2), datetime.now(LOCAL_TZ).strftime("%H:%M:%S"), reason, rsi_v, adx_v, stoch_v, macd_val, signal_val, atr_v, price, 0.0)
        return "WAIT", round(rsi_v, 2), 0, price, atr_v, reason, rsi_v, adx_v, stoch_v, macd_val, signal_val
    if is_news_time():
        reason += "; Новости, торговля приостановлена"
        log_result(symbol.replace('=X',''), "WAIT", round(rsi_v, 2), datetime.now(LOCAL_TZ).strftime("%H:%M:%S"), reason, rsi_v, adx_v, stoch_v, macd_val, signal_val, atr_v, price, 0.0)
        return "WAIT", round(rsi_v, 2), 0, price, atr_v, reason, rsi_v, adx_v, stoch_v, macd_val, signal_val

    # Подтверждение сигнала
    def is_confirmed(signal_type, candles=CONFIRMATION_CANDLES):
        if signal_type == "BUY":
            return all(macd.macd().iloc[-i] > macd.macd_signal().iloc[-i] for i in range(1, candles + 1))
        elif signal_type == "SELL":
            return all(macd.macd().iloc[-i] < macd.macd_signal().iloc[-i] for i in range(1, candles + 1))
        return False

    signal_strength = 0
    reason = ""
    # Условия для покупки
    if rsi_v < RSI_BUY_THRESHOLD:
        signal_strength += 1
        reason += "RSI перепродан; "
    if macd_val > signal_val + 0.01 and is_confirmed("BUY"):
        signal_strength += 2
        reason += "MACD бычий; "
    if ema12_v > ema26_v and ema12.iloc[-2] < ema26.iloc[-2] and ema12_v > ema200_v:
        signal_strength += 2
        reason += "EMA пересечение бычье + выше EMA200; "
    if stoch_v < 20:
        signal_strength += 1
        reason += "Stochastic перепродан; "
    if price < lower_bb * 1.01:
        signal_strength += 1
        reason += "Цена ниже Bollinger; "
    if df_15m is not None:
        prev_ema12 = EMAIndicator(df_15m['close'], window=12).ema_indicator().iloc[-1]
        prev_ema26 = EMAIndicator(df_15m['close'], window=26).ema_indicator().iloc[-1]
        if prev_ema12 > prev_ema26:
            signal_strength += 1
            reason += "Пред. EMA12 > EMA26; "
    if close.iloc[-1] > open_price:
        signal_strength += 1
        reason += "Бычья свеча; "

    if signal_strength >= 2:
        return "BUY (Adaptive)", round(rsi_v, 2), signal_strength, price, atr_v, reason, rsi_v, adx_v, stoch_v, macd_val, signal_val

    signal_strength = 0
    reason = ""
    # Условия для продажи
    if rsi_v > RSI_SELL_THRESHOLD:
        signal_strength += 1
        reason += "RSI перекуплен; "
    if macd_val < signal_val - 0.01 and is_confirmed("SELL"):
        signal_strength += 2
        reason += "MACD медвежий; "
    if ema12_v < ema26_v and ema12.iloc[-2] > ema26.iloc[-2] and ema12_v < ema200_v:
        signal_strength += 2
        reason += "EMA пересечение медвежье + ниже EMA200; "
    if stoch_v > 80:
        signal_strength += 1
        reason += "Stochastic перекуплен; "
    if price > upper_bb * 0.99:
        signal_strength += 1
        reason += "Цена выше Bollinger; "
    if df_15m is not None:
        prev_ema12 = EMAIndicator(df_15m['close'], window=12).ema_indicator().iloc[-1]
        prev_ema26 = EMAIndicator(df_15m['close'], window=26).ema_indicator().iloc[-1]
        if prev_ema12 < prev_ema26:
            signal_strength += 1
            reason += "Пред. EMA12 < EMA26; "
    if close.iloc[-1] < open_price:
        signal_strength += 1
        reason += "Медвежья свеча; "

    if signal_strength >= 2:
        return "SELL (Adaptive)", round(rsi_v, 2), signal_strength, price, atr_v, reason, rsi_v, adx_v, stoch_v, macd_val, signal_val

    reason += "; Недостаточно условий для сигнала"
    log_result(symbol.replace('=X',''), "WAIT", round(rsi_v, 2), datetime.now(LOCAL_TZ).strftime("%H:%M:%S"), reason, rsi_v, adx_v, stoch_v, macd_val, signal_val, atr_v, price, 0.0)
    return "WAIT", round(rsi_v, 2), 0, price, atr_v, reason, rsi_v, adx_v, stoch_v, macd_val, signal_val

def send_telegram_message(msg):
    if not check_internet():
        print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] Нет интернета для отправки Telegram")
        return False
    for attempt in range(3):
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
            response = requests.post(url, data={"chat_id": CHAT_ID, "text": msg}, timeout=TIMEOUT)
            if response.status_code != 200:
                print(f"Ошибка Telegram (попытка {attempt+1}): {response.json().get('description', 'Нет деталей')}")
            else:
                print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] Telegram сообщение отправлено: {msg[:50]}...")
                return True
            time.sleep(2 ** attempt)
        except Exception as e:
            print(f"Ошибка Telegram (попытка {attempt+1}): {e}")
    return False

def schedule(symbol, signal, rsi, entry_dt):
    def alert():
        alert_msg = f"⏰ ВХОД в сделку по {symbol} ({signal}) прямо сейчас ({entry_dt.strftime('%H:%M:%S')})"
        send_telegram_message(alert_msg)
        print(alert_msg)
    delay = (entry_dt - datetime.now(LOCAL_TZ)).total_seconds()
    if delay > 0:
        threading.Timer(delay, alert).start()

def log_signal(symbol, signal, rsi, entry, exit, entry_price, exit_price):
    try:
        with open(CSV_FILE, 'a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow([symbol, signal, rsi, entry, exit, entry_price, exit_price])
    except Exception as e:
        print(f"Ошибка записи в CSV: {e}")

def log_result(symbol, signal, rsi, entry_time, reason, rsi_v, adx_v, stoch_v, macd_val, signal_val, atr_v, entry_price, exit_price, outcome="PENDING"):
    for attempt in range(3):
        try:
            with open(RESULT_LOG_FILE, 'a', newline='', encoding='utf-8') as f:
                csv.writer(f).writerow([symbol, signal, rsi, entry_time, datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"), reason, outcome, rsi_v, adx_v, stoch_v, macd_val, signal_val, atr_v, entry_price, exit_price])
            return
        except Exception as e:
            print(f"Ошибка записи в лог результатов (попытка {attempt+1}): {e}")
            time.sleep(1)
    send_telegram_message(f"⚠️ Ошибка записи в results_log.csv для {symbol}: {str(e)}")

def clean_old_signals():
    if not os.path.exists(CSV_FILE):
        return
    now = datetime.now(LOCAL_TZ)
    rows = []
    try:
        with open(CSV_FILE, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            headers = next(reader, None)
            if headers is None:
                return
            for row in reader:
                try:
                    entry_time = datetime.strptime(row[4], "%H:%M:%S").replace(tzinfo=LOCAL_TZ)
                    entry_time = entry_time.replace(year=now.year, month=now.month, day=now.day)
                    if (now - entry_time).total_seconds() / 60 <= DELETE_AFTER_MINUTES:
                        rows.append(row)
                    else:
                        send_telegram_message(f"Завершена сделка по {row[0]} ({row[1]}) в {row[4]}")
                except:
                    continue
        with open(CSV_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Symbol", "Signal", "RSI", "Entry Time", "Exit Time", "Entry Price", "Exit Price"])
            writer.writerows(rows)
    except Exception as e:
        print(f"Ошибка очистки сигналов: {e}")

def calculate_win_rate():
    expected_columns = ["Symbol", "Signal", "RSI", "Entry Time", "Logged At", "Reason", "Outcome", "RSI_Value", "ADX_Value", "Stochastic_Value", "MACD_Value", "Signal_Value", "ATR_Value", "Entry_Price", "Exit_Price"]
    for attempt in range(3):
        try:
            if os.path.exists(RESULT_LOG_FILE):
                df = pd.read_csv(RESULT_LOG_FILE, on_bad_lines='skip')
                if not all(col in df.columns for col in expected_columns):
                    print(f"Ошибка: Неверная структура results_log.csv. Ожидаемые столбцы: {expected_columns}")
                    send_telegram_message(f"⚠️ Ошибка: Неверная структура results_log.csv. Создается новый файл.")
                    if os.path.exists(RESULT_LOG_FILE):
                        os.remove(RESULT_LOG_FILE)
                    with open(RESULT_LOG_FILE, 'w', newline='', encoding='utf-8') as f:
                        csv.writer(f).writerow(expected_columns)
                    return 0
                total = len(df[df['Signal'] != "WAIT"])
                wins = len(df[(df['Signal'] != "WAIT") & (df['Outcome'] == 'WIN')])
                win_rate = (wins / total * 100) if total > 0 else 0
                print(f"Общий Win Rate: {win_rate:.2f}% ({wins}/{total})")
                for symbol in SYMBOLS:
                    sym = symbol.replace('=X', '')
                    sym_df = df[df['Symbol'] == sym]
                    sym_total = len(sym_df[sym_df['Signal'] != "WAIT"])
                    sym_wins = len(sym_df[(sym_df['Signal'] != "WAIT") & (df['Outcome'] == 'WIN')])
                    sym_win_rate = (sym_wins / sym_total * 100) if sym_total > 0 else 0
                    print(f"Win Rate для {sym}: {sym_win_rate:.2f}% ({sym_wins}/{sym_total})")
                return win_rate
            else:
                print(f"Файл {RESULT_LOG_FILE} не существует. Создается новый.")
                with open(RESULT_LOG_FILE, 'w', newline='', encoding='utf-8') as f:
                    csv.writer(f).writerow(expected_columns)
                return 0
        except Exception as e:
            print(f"Ошибка расчета Win Rate (попытка {attempt+1}): {e}")
            time.sleep(1)
    send_telegram_message(f"⚠️ Ошибка расчета Win Rate: {str(e)}")
    return 0

def can_generate_signal(symbol):
    global last_signal_time
    now = datetime.now()
    if last_signal_time[symbol] is None or (now - last_signal_time[symbol]).total_seconds() > MIN_SIGNAL_INTERVAL:
        last_signal_time[symbol] = now
        return True
    return False

def send_signal(symbol, signal, rsi, price, atr_v, df_5m, reason, rsi_v, adx_v, stoch_v, macd_val, signal_val):
    try:
        now = datetime.now(LOCAL_TZ)
        TRADE_DURATION_MINUTES = 1  # Фиксированная экспирация 1 минута
        entry = now + timedelta(seconds=PREPARE_SECONDS)
        exit_ = entry + timedelta(minutes=TRADE_DURATION_MINUTES)
        entry_str = entry.strftime("%H:%M:%S")
        exit_str = exit_.strftime("%H:%M:%S")
        stop_loss = price - atr_v if "BUY" in signal else price + atr_v
        take_profit = price + 2 * atr_v if "BUY" in signal else price - 2 * atr_v
        entry_price = price
        exit_price = 0.0  # Будет обновлено после экспирации

        msg = (
            f"🚨 СИГНАЛ по {symbol.replace('=X','')}\n"
            f"📈 Прогноз: {signal}\n"
            f"📊 RSI: {rsi}\n"
            f"📝 Причина: {reason}\n"
            f"⏱ Вход: {entry_str} (через {PREPARE_SECONDS} сек)\n"
            f"⏳ Выход: {exit_str} (через {TRADE_DURATION_MINUTES} мин после входа)\n"
            f"🛑 Stop Loss: {stop_loss:.4f}\n"
            f"🎯 Take Profit: {take_profit:.4f}\n"
            f"💵 Цена входа: {entry_price:.4f}"
        )

        print(msg)
        if send_telegram_message(msg):
            log_signal(symbol.replace('=X',''), signal, rsi, entry_str, exit_str, entry_price, exit_price)
            log_result(symbol.replace('=X',''), signal, rsi, entry_str, reason, rsi_v, adx_v, stoch_v, macd_val, signal_val, atr_v, entry_price, exit_price)
            schedule(symbol.replace('=X',''), signal, rsi, entry)
    except Exception as e:
        print(f"❌ Ошибка в send_signal для {symbol}: {e}")
        send_telegram_message(f"❌ Ошибка отправки сигнала для {symbol.replace('=X','')}: {str(e)}")

def main():
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(["Symbol", "Signal", "RSI", "Entry Time", "Exit Time", "Entry Price", "Exit Price"])
    if not os.path.exists(RESULT_LOG_FILE):
        with open(RESULT_LOG_FILE, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(["Symbol", "Signal", "RSI", "Entry Time", "Logged At", "Reason", "Outcome", "RSI_Value", "ADX_Value", "Stochastic_Value", "MACD_Value", "Signal_Value", "ATR_Value", "Entry_Price", "Exit_Price"])

    while True:
        print("🌀 Новый цикл анализа...")
        clean_old_signals()
        signals = []
        for symbol in SYMBOLS:
            df_5m = get_data(symbol, interval="5m", period="7d")
            time.sleep(2)
            df_15m = get_data(symbol, interval="15m", period="10d")
            time.sleep(2)
            if df_5m is not None and can_generate_signal(symbol):
                signal, rsi, strength, price, atr_v, reason, rsi_v, adx_v, stoch_v, macd_val, signal_val = analyze(symbol, df_5m, df_15m)
                print(f"[{symbol}] Сигнал: {signal}, Сила: {strength}, Причина: {reason}")
                if signal != "WAIT" and strength >= 2:
                    signals.append((symbol, signal, rsi, price, atr_v, df_5m, reason, rsi_v, adx_v, stoch_v, macd_val, signal_val))
                else:
                    print(f"[{symbol}] Сигнал не сгенерирован: {reason}")
                    send_telegram_message(f"[{symbol.replace('=X','')}] Сигнал не сгенерирован: {reason}")
            else:
                print(f"[{symbol}] Пропуск анализа: отсутствуют данные 5m")
                send_telegram_message(f"[{symbol.replace('=X','')}] Пропуск анализа: отсутствуют данные 5m")
        if not signals:
            send_telegram_message("⚠️ Сигналы не сгенерированы в текущем цикле")
        for sig in signals:
            send_signal(*sig)
        calculate_win_rate()
        print("⏳ Ожидание...")
        time.sleep(INTERVAL)

if __name__ == '__main__':
    main()        return pytz.timezone(timezone_str)
    except Exception as e:
        print(f"Ошибка определения часового пояса: {e}")
        return pytz.timezone("UTC")

LOCAL_TZ = get_timezone()

session = requests.Session()
retry = Retry(total=5, backoff_factor=2, status_forcelist=[429, 502, 503, 504])
session.mount('https://', HTTPAdapter(max_retries=retry))

def is_active_session():
    now = datetime.now(LOCAL_TZ)
    hour = now.hour
    return 8 <= hour <= 22

def is_news_time():
    if not BEAUTIFULSOUP_AVAILABLE:
        print("Новостной фильтр недоступен: установите beautifulsoup4")
        return False
    try:
        url = "https://www.investing.com/economic-calendar/"
        response = session.get(url, headers=HEADERS, timeout=TIMEOUT)
        soup = BeautifulSoup(response.text, 'html.parser')
        events = soup.find_all('tr', {'data-event-id': True})
        now = datetime.now(LOCAL_TZ)
        for event in events:
            time_elem = event.find('td', class_='time')
            if not time_elem:
                continue
            time_str = time_elem.text.strip()
            try:
                event_time = datetime.strptime(time_str, "%H:%M").replace(
                    year=now.year, month=now.month, day=now.day, tzinfo=LOCAL_TZ
                )
                if abs((now - event_time).total_seconds()) < 1800:  # 30 минут
                    return True
            except ValueError:
                continue
        return False
    except Exception as e:
        print(f"Ошибка проверки новостей: {e}")
        return False

def check_internet():
    try:
        requests.get("https://www.google.com", timeout=5)
        return True
    except:
        return False

def get_data(symbol, interval="5m", period="7d"):
    cache_key = f"{symbol}_{interval}"
    if cache_key in data_cache and (datetime.now() - data_cache[cache_key]['time']).seconds < 300:
        print(f"[{symbol}] Используем кэшированные данные")
        return data_cache[cache_key]['data']
    
    # Попытка через Yahoo Finance API
    for attempt in range(3):
        try:
            print(f"⏳ Получаем данные по {symbol.replace('=X','')} (интервал {interval}, Yahoo Finance, попытка {attempt+1})")
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range={period}&interval={interval}"
            response = session.get(url, headers=HEADERS, timeout=TIMEOUT)
            print(f"[{symbol}] HTTP статус: {response.status_code}")
            if response.status_code == 429:
                print(f"[{symbol}] Ошибка 429: Слишком много запросов, ожидание...")
                time.sleep(5 ** attempt)
                continue
            data = response.json()
            if data['chart']['result'] is None:
                print(f"[{symbol}] Ошибка данных: {data.get('chart', {}).get('error', 'Нет деталей ошибки')}")
                continue
            result = data['chart']['result'][0]
            ts = result['timestamp']
            quote = result['indicators']['quote'][0]
            df = pd.DataFrame({
                "timestamp": [datetime.fromtimestamp(t, tz=LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S") for t in ts],
                "close": quote['close'],
                "high": quote['high'],
                "low": quote['low'],
                "open": quote['open'],
                "volume": quote['volume']
            })
            df.dropna(inplace=True)
            if len(df) < 200:
                print(f"[{symbol}] Недостаточно данных (Yahoo Finance): {len(df)} свечей")
                continue
            data_cache[cache_key] = {'data': df, 'time': datetime.now()}
            return df
        except Exception as e:
            print(f"[{symbol}] Ошибка Yahoo Finance: {str(e)}")
            time.sleep(5 ** attempt)
    
    # Резервный источник: Alpha Vantage
    if ALPHA_VANTAGE_AVAILABLE and ALPHA_VANTAGE_API_KEY != "YOUR_ALPHA_VANTAGE_API_KEY":
        for attempt in range(3):
            try:
                print(f"⏳ Получаем данные по {symbol.replace('=X','')} (интервал 5min, Alpha Vantage, попытка {attempt+1})")
                alpha_symbol = SYMBOLS_ALPHA[SYMBOLS.index(symbol)]
                fx = ForeignExchange(key=ALPHA_VANTAGE_API_KEY)
                data, _ = fx.get_currency_exchange_intraday(symbol=alpha_symbol, interval="5min", outputsize="full")
                
                if not data:
                    print(f"[{symbol}] Пустые данные (Alpha Vantage): {data}")
                    continue
                
                df = pd.DataFrame(data).transpose().reset_index()
                df.columns = ['timestamp', 'open', 'high', 'low', 'close']
                df['open'] = df['open'].astype(float)
                df['high'] = df['high'].astype(float)
                df['low'] = df['low'].astype(float)
                df['close'] = df['close'].astype(float)
                df['volume'] = 0  # Alpha Vantage не предоставляет объем для forex
                df['timestamp'] = pd.to_datetime(df['timestamp']).dt.astimezone(LOCAL_TZ).dt.strftime("%Y-%m-%d %H:%M:%S")
                df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
                df.dropna(inplace=True)
                
                if len(df) < 200:
                    print(f"[{symbol}] Недостаточно данных (Alpha Vantage): {len(df)} свечей")
                    continue
                data_cache[cache_key] = {'data': df, 'time': datetime.now()}
                return df
            except Exception as e:
                print(f"[{symbol}] Ошибка Alpha Vantage: {str(e)}")
                time.sleep(5 ** attempt)
        send_telegram_message(f"⚠️ Ошибка получения данных для {symbol.replace('=X','')} ({interval}, Alpha Vantage): все попытки провалились")
    
    # Если все источники не сработали
    if cache_key in data_cache:
        print(f"[{symbol}] Используем старые кэшированные данные")
        return data_cache[cache_key]['data']
    send_telegram_message(f"⚠️ Ошибка получения данных для {symbol.replace('=X','')} ({interval}): все источники недоступны")
    return None

def analyze(symbol, df, prev_df=None):
    close = df['close']
    high = df['high']
    low = df['low']
    open = df['open']
    volume = df['volume']
    
    # Индикаторы
    rsi = RSIIndicator(close, window=14).rsi()
    macd = MACD(close, window_slow=26, window_fast=12, window_sign=9)
    bb = BollingerBands(close, window=20, window_dev=2)
    ema12 = EMAIndicator(close, window=12).ema_indicator()
    ema26 = EMAIndicator(close, window=26).ema_indicator()
    ema200 = EMAIndicator(close, window=200).ema_indicator()
    adx = ADXIndicator(high=high, low=low, close=close, window=14).adx()
    stochastic = StochasticOscillator(close=close, high=high, low=low, window=14, smooth_window=3).stoch()
    atr = AverageTrueRange(high=high, low=low, close=close, window=14).average_true_range()

    # Текущие значения
    rsi_v = rsi.iloc[-1]
    macd_val = macd.macd().iloc[-1]
    signal_val = macd.macd_signal().iloc[-1]
    ema12_v = ema12.iloc[-1]
    ema26_v = ema26.iloc[-1]
    ema200_v = ema200.iloc[-1]
    adx_v = adx.iloc[-1]
    stoch_v = stochastic.iloc[-1]
    price = close.iloc[-1]
    open_price = open.iloc[-1]
    upper_bb = bb.bollinger_hband().iloc[-1]
    lower_bb = bb.bollinger_lband().iloc[-1]
    atr_v = atr.iloc[-1]
    bb_width = (upper_bb - lower_bb) / price
    volume_avg = volume.iloc[-20:].mean()
    volume_current = volume.iloc[-1]

    # Логирование индикаторов
    reason = f"RSI: {rsi_v:.2f}, ADX: {adx_v:.2f}, Stochastic: {stoch_v:.2f}, MACD: {macd_val:.4f}, Signal: {signal_val:.4f}, ATR: {atr_v:.4f}, BB_Width: {bb_width:.4f}"
    print(f"[{symbol}] {reason}")

    # Фильтры
    if adx_v < MIN_ADX:
        reason += f"; ADX слишком низкий (< {MIN_ADX})"
        log_result(symbol.replace('=X',''), "WAIT", round(rsi_v, 2), datetime.now(LOCAL_TZ).strftime("%H:%M:%S"), reason, rsi_v, adx_v, stoch_v, macd_val, signal_val, atr_v, price, 0.0)
        return "WAIT", round(rsi_v, 2), 0, price, atr_v, reason, rsi_v, adx_v, stoch_v, macd_val, signal_val
    if bb_width < BB_WIDTH_MIN:
        reason += "; Узкие Bollinger Bands"
        log_result(symbol.replace('=X',''), "WAIT", round(rsi_v, 2), datetime.now(LOCAL_TZ).strftime("%H:%M:%S"), reason, rsi_v, adx_v, stoch_v, macd_val, signal_val, atr_v, price, 0.0)
        return "WAIT", round(rsi_v, 2), 0, price, atr_v, reason, rsi_v, adx_v, stoch_v, macd_val, signal_val
    if not is_active_session() and "JPY" in symbol:
        reason += "; Торговля вне активной сессии для JPY"
        log_result(symbol.replace('=X',''), "WAIT", round(rsi_v, 2), datetime.now(LOCAL_TZ).strftime("%H:%M:%S"), reason, rsi_v, adx_v, stoch_v, macd_val, signal_val, atr_v, price, 0.0)
        return "WAIT", round(rsi_v, 2), 0, price, atr_v, reason, rsi_v, adx_v, stoch_v, macd_val, signal_val
    if is_news_time():
        reason += "; Новости, торговля приостановлена"
        log_result(symbol.replace('=X',''), "WAIT", round(rsi_v, 2), datetime.now(LOCAL_TZ).strftime("%H:%M:%S"), reason, rsi_v, adx_v, stoch_v, macd_val, signal_val, atr_v, price, 0.0)
        return "WAIT", round(rsi_v, 2), 0, price, atr_v, reason, rsi_v, adx_v, stoch_v, macd_val, signal_val

    # Подтверждение сигнала
    def is_confirmed(signal_type, candles=CONFIRMATION_CANDLES):
        if signal_type == "BUY":
            return all(macd.macd().iloc[-i] > macd.macd_signal().iloc[-i] for i in range(1, candles + 1))
        elif signal_type == "SELL":
            return all(macd.macd().iloc[-i] < macd.macd_signal().iloc[-i] for i in range(1, candles + 1))
        return False

    signal_strength = 0
    reason = ""
    # Условия для покупки
    if rsi_v < RSI_BUY_THRESHOLD:
        signal_strength += 1
        reason += "RSI перепродан; "
    if macd_val > signal_val + 0.007 and is_confirmed("BUY"):
        signal_strength += 2
        reason += "MACD бычий; "
    if ema12_v > ema26_v and ema12.iloc[-2] < ema26.iloc[-2] and ema12_v > ema200_v:
        signal_strength += 2
        reason += "EMA пересечение бычье + выше EMA200; "
    if stoch_v < 20:
        signal_strength += 1
        reason += "Stochastic перепродан; "
    if price < lower_bb * 1.01:
        signal_strength += 1
        reason += "Цена ниже Bollinger; "
    if prev_df is not None:
        prev_ema12 = EMAIndicator(prev_df['close'], window=12).ema_indicator().iloc[-1]
        prev_ema26 = EMAIndicator(prev_df['close'], window=26).ema_indicator().iloc[-1]
        if prev_ema12 > prev_ema26:
            signal_strength += 1
            reason += "Пред. EMA12 > EMA26; "
    if close.iloc[-1] > open_price:
        signal_strength += 1
        reason += "Бычья свеча; "

    if signal_strength >= 2:
        return "BUY (Strong)", round(rsi_v, 2), signal_strength, price, atr_v, reason, rsi_v, adx_v, stoch_v, macd_val, signal_val

    signal_strength = 0
    reason = ""
    # Условия для продажи
    if rsi_v > RSI_SELL_THRESHOLD:
        signal_strength += 1
        reason += "RSI перекуплен; "
    if macd_val < signal_val - 0.007 and is_confirmed("SELL"):
        signal_strength += 2
        reason += "MACD медвежий; "
    if ema12_v < ema26_v and ema12.iloc[-2] > ema26.iloc[-2] and ema12_v < ema200_v:
        signal_strength += 2
        reason += "EMA пересечение медвежье + ниже EMA200; "
    if stoch_v > 80:
        signal_strength += 1
        reason += "Stochastic перекуплен; "
    if price > upper_bb * 0.99:
        signal_strength += 1
        reason += "Цена выше Bollinger; "
    if prev_df is not None:
        prev_ema12 = EMAIndicator(prev_df['close'], window=12).ema_indicator().iloc[-1]
        prev_ema26 = EMAIndicator(prev_df['close'], window=26).ema_indicator().iloc[-1]
        if prev_ema12 < prev_ema26:
            signal_strength += 1
            reason += "Пред. EMA12 < EMA26; "
    if close.iloc[-1] < open_price:
        signal_strength += 1
        reason += "Медвежья свеча; "

    if signal_strength >= 2:
        return "SELL (Strong)", round(rsi_v, 2), signal_strength, price, atr_v, reason, rsi_v, adx_v, stoch_v, macd_val, signal_val

    reason += "; Недостаточно условий для сигнала"
    log_result(symbol.replace('=X',''), "WAIT", round(rsi_v, 2), datetime.now(LOCAL_TZ).strftime("%H:%M:%S"), reason, rsi_v, adx_v, stoch_v, macd_val, signal_val, atr_v, price, 0.0)
    return "WAIT", round(rsi_v, 2), 0, price, atr_v, reason, rsi_v, adx_v, stoch_v, macd_val, signal_val

def calculate_expiration(atr_v, avg_atr):
    if atr_v > avg_atr * 1.2:
        return 3
    elif atr_v > avg_atr:
        return 5
    elif atr_v > avg_atr * 0.8:
        return 7
    return 10

def send_telegram_message(msg):
    if not check_internet():
        print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] Нет интернета для отправки Telegram")
        return False
    for attempt in range(3):
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
            response = requests.post(url, data={"chat_id": CHAT_ID, "text": msg}, timeout=TIMEOUT)
            if response.status_code != 200:
                print(f"Ошибка Telegram (попытка {attempt+1}): {response.json().get('description', 'Нет деталей')}")
            else:
                print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] Telegram сообщение отправлено: {msg[:50]}...")
                return True
            time.sleep(2 ** attempt)
        except Exception as e:
            print(f"Ошибка Telegram (попытка {attempt+1}): {e}")
    return False

def schedule_entry_alert(symbol, signal, rsi, entry_dt):
    def alert():
        alert_msg = f"⏰ ВХОД в сделку по {symbol} ({signal}) прямо сейчас ({entry_dt.strftime('%H:%M:%S')})"
        send_telegram_message(alert_msg)
        print(alert_msg)
    delay = (entry_dt - datetime.now(LOCAL_TZ)).total_seconds()
    if delay > 0:
        threading.Timer(delay, alert).start()

def log_signal(symbol, signal, rsi, entry, exit, entry_price, exit_price):
    try:
        with open(CSV_FILE, 'a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow([symbol, signal, rsi, entry, exit, entry_price, exit_price])
    except Exception as e:
        print(f"Ошибка записи в CSV: {e}")

def log_result(symbol, signal, rsi, entry_time, reason, rsi_v, adx_v, stoch_v, macd_val, signal_val, atr_v, entry_price, exit_price, outcome="PENDING"):
    for attempt in range(3):
        try:
            with open(RESULT_LOG_FILE, 'a', newline='', encoding='utf-8') as f:
                csv.writer(f).writerow([symbol, signal, rsi, entry_time, datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"), reason, outcome, rsi_v, adx_v, stoch_v, macd_val, signal_val, atr_v, entry_price, exit_price])
            return
        except Exception as e:
            print(f"Ошибка записи в лог результатов (попытка {attempt+1}): {e}")
            time.sleep(1)
    send_telegram_message(f"⚠️ Ошибка записи в results_log.csv для {symbol}: {str(e)}")

def clean_old_signals():
    if not os.path.exists(CSV_FILE):
        return
    now = datetime.now(LOCAL_TZ)
    rows = []
    try:
        with open(CSV_FILE, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            headers = next(reader, None)
            if headers is None:
                return
            for row in reader:
                try:
                    entry_time = datetime.strptime(row[4], "%H:%M:%S").replace(tzinfo=LOCAL_TZ)
                    entry_time = entry_time.replace(year=now.year, month=now.month, day=now.day)
                    if (now - entry_time).total_seconds() / 60 <= DELETE_AFTER_MINUTES:
                        rows.append(row)
                    else:
                        send_telegram_message(f"Завершена сделка по {row[0]} ({row[1]}) в {row[4]}")
                except:
                    continue
        with open(CSV_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Symbol", "Signal", "RSI", "Entry Time", "Exit Time", "Entry Price", "Exit Price"])
            writer.writerows(rows)
    except Exception as e:
        print(f"Ошибка очистки сигналов: {e}")

def calculate_win_rate():
    expected_columns = ["Symbol", "Signal", "RSI", "Entry Time", "Logged At", "Reason", "Outcome", "RSI_Value", "ADX_Value", "Stochastic_Value", "MACD_Value", "Signal_Value", "ATR_Value", "Entry_Price", "Exit_Price"]
    for attempt in range(3):
        try:
            if os.path.exists(RESULT_LOG_FILE):
                df = pd.read_csv(RESULT_LOG_FILE, on_bad_lines='skip')
                if not all(col in df.columns for col in expected_columns):
                    print(f"Ошибка: Неверная структура results_log.csv. Ожидаемые столбцы: {expected_columns}")
                    send_telegram_message(f"⚠️ Ошибка: Неверная структура results_log.csv. Создается новый файл.")
                    if os.path.exists(RESULT_LOG_FILE):
                        os.remove(RESULT_LOG_FILE)
                    with open(RESULT_LOG_FILE, 'w', newline='', encoding='utf-8') as f:
                        csv.writer(f).writerow(expected_columns)
                    return 0
                total = len(df[df['Signal'] != "WAIT"])
                wins = len(df[(df['Signal'] != "WAIT") & (df['Outcome'] == 'WIN')])
                win_rate = (wins / total * 100) if total > 0 else 0
                print(f"Общий Win Rate: {win_rate:.2f}% ({wins}/{total})")
                for symbol in SYMBOLS:
                    sym = symbol.replace('=X', '')
                    sym_df = df[df['Symbol'] == sym]
                    sym_total = len(sym_df[sym_df['Signal'] != "WAIT"])
                    sym_wins = len(sym_df[(sym_df['Signal'] != "WAIT") & (sym_df['Outcome'] == 'WIN')])
                    sym_win_rate = (sym_wins / sym_total * 100) if sym_total > 0 else 0
                    print(f"Win Rate для {sym}: {sym_win_rate:.2f}% ({sym_wins}/{sym_total})")
                return win_rate
            else:
                print(f"Файл {RESULT_LOG_FILE} не существует. Создается новый.")
                with open(RESULT_LOG_FILE, 'w', newline='', encoding='utf-8') as f:
                    csv.writer(f).writerow(expected_columns)
                return 0
        except Exception as e:
            print(f"Ошибка расчета Win Rate (попытка {attempt+1}): {e}")
            time.sleep(1)
    send_telegram_message(f"⚠️ Ошибка расчета Win Rate: {str(e)}")
    return 0

def can_generate_signal(symbol):
    global last_signal_time
    now = datetime.now()
    if last_signal_time[symbol] is None or (now - last_signal_time[symbol]).total_seconds() > MIN_SIGNAL_INTERVAL:
        last_signal_time[symbol] = now
        return True
    return False

def send_signal(symbol, signal, rsi, price, atr_v, df_5m, reason, rsi_v, adx_v, stoch_v, macd_val, signal_val):
    try:
        now = datetime.now(LOCAL_TZ)
        avg_atr = df_5m['close'].std()
        TRADE_DURATION_MINUTES = calculate_expiration(atr_v, avg_atr)
        entry = now + timedelta(seconds=PREPARE_SECONDS)
        exit_ = entry + timedelta(minutes=TRADE_DURATION_MINUTES)
        entry_str = entry.strftime("%H:%M:%S")
        exit_str = exit_.strftime("%H:%M:%S")
        stop_loss = price - atr_v if "BUY" in signal else price + atr_v
        take_profit = price + 2 * atr_v if "BUY" in signal else price - 2 * atr_v
        entry_price = price
        exit_price = 0.0  # Будет обновлено после экспирации

        msg = (
            f"🚨 СИГНАЛ по {symbol.replace('=X','')}\n"
            f"📈 Прогноз: {signal}\n"
            f"📊 RSI: {rsi}\n"
            f"📝 Причина: {reason}\n"
            f"⏱ Вход: {entry_str} (через {PREPARE_SECONDS} сек)\n"
            f"⏳ Выход: {exit_str} (через {TRADE_DURATION_MINUTES} мин после входа)\n"
            f"🛑 Stop Loss: {stop_loss:.4f}\n"
            f"🎯 Take Profit: {take_profit:.4f}\n"
            f"💵 Цена входа: {entry_price:.4f}"
        )

        print(msg)
        if send_telegram_message(msg):
            log_signal(symbol.replace('=X',''), signal, rsi, entry_str, exit_str, entry_price, exit_price)
            log_result(symbol.replace('=X',''), signal, rsi, entry_str, reason, rsi_v, adx_v, stoch_v, macd_val, signal_val, atr_v, entry_price, exit_price)
            schedule_entry_alert(symbol.replace('=X',''), signal, rsi, entry)
    except Exception as e:
        print(f"❌ Ошибка в send_signal для {symbol}: {e}")
        send_telegram_message(f"❌ Ошибка отправки сигнала для {symbol.replace('=X','')}: {str(e)}")

def main():
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(["Symbol", "Signal", "RSI", "Entry Time", "Exit Time", "Entry Price", "Exit Price"])
    if not os.path.exists(RESULT_LOG_FILE):
        with open(RESULT_LOG_FILE, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(["Symbol", "Signal", "RSI", "Entry Time", "Logged At", "Reason", "Outcome", "RSI_Value", "ADX_Value", "Stochastic_Value", "MACD_Value", "Signal_Value", "ATR_Value", "Entry_Price", "Exit_Price"])

    while True:
        print("🌀 Новый цикл анализа...")
        clean_old_signals()
        signals = []
        for symbol in SYMBOLS:
            df_5m = get_data(symbol, interval="5m", period="7d")
            time.sleep(2)
            df_15m = get_data(symbol, interval="15m", period="10d")
            time.sleep(2)
            if df_5m is not None and can_generate_signal(symbol):
                signal, rsi, strength, price, atr_v, reason, rsi_v, adx_v, stoch_v, macd_val, signal_val = analyze(symbol, df_5m, df_15m)
                print(f"[{symbol}] Сигнал: {signal}, Сила: {strength}, Причина: {reason}")
                if signal != "WAIT" and strength >= 2:
                    signals.append((symbol, signal, rsi, price, atr_v, df_5m, reason, rsi_v, adx_v, stoch_v, macd_val, signal_val))
                else:
                    print(f"[{symbol}] Сигнал не сгенерирован: {reason}")
                    send_telegram_message(f"[{symbol.replace('=X','')}] Сигнал не сгенерирован: {reason}")
            else:
                print(f"[{symbol}] Пропуск анализа: отсутствуют данные 5m")
                send_telegram_message(f"[{symbol.replace('=X','')}] Пропуск анализа: отсутствуют данные 5m")
        if not signals:
            send_telegram_message("⚠️ Сигналы не сгенерированы в текущем цикле")
        for sig in signals:
            send_signal(*sig)
        calculate_win_rate()
        print("⏳ Ожидание...")
        time.sleep(INTERVAL)

if __name__ == '__main__':
    main()
