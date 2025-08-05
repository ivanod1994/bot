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
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
from telegram.error import TimedOut
try:
    from bs4 import BeautifulSoup
    BEAUTIFULSOUP_AVAILABLE = True
except ImportError:
    BEAUTIFULSOUP_AVAILABLE = False
    print("Библиотека BeautifulSoup не установлена. Установите: pip install beautifulsoup4")

# === НАСТРОЙКИ ===
TELEGRAM_TOKEN = "8246979603:AAGSP7b-YRol151GlZpfxyyS34rW5ncZJo4"
CHAT_ID = "6677680988"
SYMBOLS = ["EURJPY=X", "EURUSD=X", "CHFJPY=X", "USDCAD=X", "CADJPY=X", "GBPUSD=X", "AUDUSD=X"]
DEFAULT_TIMEFRAME = "5m"
CSV_FILE = "signals.csv"
DELETE_AFTER_MINUTES = 5
PREPARE_SECONDS = 90
RESULT_LOG_FILE = "/app/data/results_log.csv"
MANUAL_TZ = "Africa/Algiers"
CONFIRMATION_CANDLES = 2
PAYOUT = 0.85
TIMEOUT = 90
MIN_SIGNAL_INTERVAL = 60
VOLUME_MULTIPLIER = float('inf')
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
CORRELATION_PAIRS = {
    "EURUSD=X": ["GBPUSD=X", "AUDUSD=X"],
    "GBPUSD=X": ["EURUSD=X", "AUDUSD=X"],
    "AUDUSD=X": ["EURUSD=X", "GBPUSD=X"],
    "CADJPY=X": ["EURJPY=X", "CHFJPY=X"],
    "EURJPY=X": ["CADJPY=X", "CHFJPY=X"],
    "CHFJPY=X": ["CADJPY=X", "EURJPY=X"],
    "USDCAD=X": []
}
ACCOUNT_BALANCE = 10000
RISK_PER_TRADE = 0.005
MIN_SUCCESS_PROBABILITY = 0.65
MIN_SIGNAL_STRENGTH = 4.0
MAX_ACTIVE_TRADES = 3
MIN_REWARD_RISK_RATIO = 1.2
CSV_COLUMNS = ["Entry_Time", "Symbol", "Signal", "Entry_Price", "Stop_Loss", "Take_Profit", "Lot_Size", "Reason", "Success_Probability", "Outcome", "Exit_Price", "Profit"]
# =================

data_cache = {}
last_signal_time = {symbol: None for symbol in SYMBOLS}
user_selections = {}

def get_timezone():
    try:
        if MANUAL_TZ:
            return pytz.timezone(MANUAL_TZ)
        response = requests.get("https://ipinfo.io/json", timeout=5)
        response.raise_for_status()
        data = response.json()
        timezone_str = data.get("timezone", "UTC")
        return pytz.timezone(timezone_str)
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Ошибка определения часового пояса: {e}")
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
        print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] BeautifulSoup недоступен, пропускаем проверку новостей")
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
                if abs((now - event_time).total_seconds()) < 1800:
                    print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] Обнаружены новости, торговля приостановлена")
                    return True
            except ValueError:
                continue
        return False
    except Exception as e:
        print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] Ошибка проверки новостей: {e}")
        return False

def check_internet():
    try:
        requests.get("https://www.google.com", timeout=5)
        return True
    except Exception as e:
        print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] Нет интернета: {e}")
        return False

def send_telegram_message(msg, symbol="Unknown"):
    if not check_internet():
        print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] Нет интернета для отправки Telegram")
        return False
    for attempt in range(3):
        try:
            print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] Отправка сигнала для {symbol}: {msg[:50]}...")
            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
            response = requests.post(url, data={"chat_id": CHAT_ID, "text": msg}, timeout=120)
            if response.status_code != 200:
                print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] Ошибка Telegram (попытка {attempt+1}): {response.json().get('description', 'Нет деталей')}")
            else:
                print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] Telegram сообщение отправлено: {msg[:50]}...")
                return True
            time.sleep(2 ** attempt)
        except Exception as e:
            print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] Ошибка Telegram (попытка {attempt+1}): {e}")
    print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] Не удалось отправить сигнал для {symbol}")
    return False

def detect_fractals(df, window=3):
    if len(df) < 5:
        return pd.Series(False, index=df.index), pd.Series(False, index=df.index)
    
    bullish_fractals = pd.Series(False, index=df.index)
    bearish_fractals = pd.Series(False, index=df.index)
    volume = df['volume']
    avg_volume = volume.rolling(window=10).mean()

    for i in range(window, len(df) - window):
        if (df['low'].iloc[i] < df['low'].iloc[i-1] and
            df['low'].iloc[i] < df['low'].iloc[i-2] and
            df['low'].iloc[i] < df['low'].iloc[i+1] and
            df['low'].iloc[i] < df['low'].iloc[i+2] and
            volume.iloc[i] > avg_volume.iloc[i] * 1.0):
            bullish_fractals.iloc[i] = True
        
        if (df['high'].iloc[i] > df['high'].iloc[i-1] and
            df['high'].iloc[i] > df['high'].iloc[i-2] and
            df['high'].iloc[i] > df['high'].iloc[i+1] and
            df['high'].iloc[i] > df['high'].iloc[i+2] and
            volume.iloc[i] > avg_volume.iloc[i] * 1.0):
            bearish_fractals.iloc[i] = True
    
    return bullish_fractals, bearish_fractals

def is_bullish_pattern(df):
    return (df['close'].iloc[-1] > df['open'].iloc[-1] and
            df['close'].iloc[-1] > df['open'].iloc[-2] and
            (df['close'].iloc[-1] - df['open'].iloc[-1]) > (df['open'].iloc[-1] - df['low'].iloc[-1]) * 2)

def is_bearish_pattern(df):
    return (df['close'].iloc[-1] < df['open'].iloc[-1] and
            df['close'].iloc[-1] < df['open'].iloc[-2] and
            (df['open'].iloc[-1] - df['close'].iloc[-1]) > (df['high'].iloc[-1] - df['open'].iloc[-1]) * 2)

def get_correlation_confirmation(symbol, signal, df_5m, expiration=1):
    if symbol not in CORRELATION_PAIRS or not CORRELATION_PAIRS[symbol]:
        return False, "Нет коррелированных пар"
    
    confirmation = False
    reason = ""
    for corr_symbol in CORRELATION_PAIRS[symbol]:
        df_corr = get_data(corr_symbol, interval=DEFAULT_TIMEFRAME, period="3d")
        if df_corr is None:
            reason += f"Нет данных для {corr_symbol}; "
            continue
        
        close = df_corr['close']
        ema5 = EMAIndicator(close, window=5).ema_indicator().iloc[-1]
        ema12 = EMAIndicator(close, window=12).ema_indicator().iloc[-1]
        
        if signal == "BUY" and ema5 > ema12:
            confirmation = True
            reason += f"Бычий тренд подтверждён на {corr_symbol}; "
        elif signal == "SELL" and ema5 < ema12:
            confirmation = True
            reason += f"Медвежий тренд подтверждён на {corr_symbol}; "
    
    return confirmation, reason

def get_data(symbol, interval=DEFAULT_TIMEFRAME, period="1d"):
    cache_key = f"{symbol}_{interval}"
    if cache_key in data_cache and (datetime.now() - data_cache[cache_key]['time']).seconds < 300:
        print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] Использую кэшированные данные для {symbol} ({interval})")
        return data_cache[cache_key]['data']
    
    print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] Загружаю данные для {symbol} ({interval})")
    for attempt in range(3):
        try:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range={'3d' if interval in ['1m', '2m', '5m'] else period}&interval={interval}"
            response = session.get(url, headers=HEADERS, timeout=TIMEOUT)
            if response.status_code == 429:
                print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] Ошибка 429, жду перед повтором (попытка {attempt+1})")
                time.sleep(2 ** attempt)
                continue
            data = response.json()
            if data['chart']['result'] is None:
                print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] Нет данных от Yahoo Finance для {symbol}")
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
            print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] Загружено свечей для {symbol}: {len(df)}")
            if len(df) < 30:
                print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] Недостаточно данных для {symbol} ({len(df)} свечей)")
                continue
            data_cache[cache_key] = {'data': df, 'time': datetime.now()}
            print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] Данные для {symbol} ({interval}) успешно загружены")
            return df
        except Exception as e:
            print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] [{symbol}] Ошибка Yahoo Finance (попытка {attempt+1}): {str(e)}")
            time.sleep(2 ** attempt)
    print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] Не удалось загрузить данные для {symbol} ({interval})")
    return None

def clean_csv_file(file_path):
    if not os.path.exists(file_path):
        return
    
    temp_file = file_path + '.tmp'
    with open(file_path, 'r', newline='') as infile, open(temp_file, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        header_written = False
        
        for row in reader:
            if not header_written:
                writer.writerow(CSV_COLUMNS)
                header_written = True
            if len(row) == len(CSV_COLUMNS):
                writer.writerow(row)
            else:
                print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] Пропущена строка с некорректным количеством полей: {len(row)} вместо {len(CSV_COLUMNS)}")
    
    os.replace(temp_file, file_path)
    print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] Файл {file_path} очищен")

def update_outcome(symbol, entry_time, entry_price, signal, expiration, stop_loss, take_profit):
    try:
        df = get_data(symbol, interval="1m", period="1h")
        if df is None:
            return "PENDING", entry_price
        entry_dt = datetime.strptime(entry_time, "%H:%M:%S").replace(
            year=datetime.now(LOCAL_TZ).year,
            month=datetime.now(LOCAL_TZ).month,
            day=datetime.now(LOCAL_TZ).day,
            tzinfo=LOCAL_TZ
        )
        end_time = entry_dt + timedelta(minutes=expiration)
        future_data = df[df['timestamp'] <= end_time.strftime("%Y-%m-%d %H:%M:%S")]
        if signal == "BUY":
            if any(future_data['high'] >= take_profit):
                return "WIN", take_profit
            elif any(future_data['low'] <= stop_loss):
                return "LOSS", stop_loss
            elif any(future_data['high'] >= entry_price * (1.00005 + 0.00003 * expiration)):
                return "WIN", future_data['high'].max()
            elif any(future_data['low'] <= entry_price * (0.99995 - 0.00003 * expiration)):
                return "LOSS", future_data['low'].min()
        elif signal == "SELL":
            if any(future_data['low'] <= take_profit):
                return "WIN", take_profit
            elif any(future_data['high'] >= stop_loss):
                return "LOSS", stop_loss
            elif any(future_data['low'] <= entry_price * (0.99995 - 0.00003 * expiration)):
                return "WIN", future_data['low'].min()
            elif any(future_data['high'] >= entry_price * (1.00005 + 0.00003 * expiration)):
                return "LOSS", future_data['high'].max()
        return "PENDING", entry_price
    except Exception as e:
        print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] Ошибка обновления исхода для {symbol}: {e}")
        send_telegram_message(f"Ошибка обновления исхода для {symbol}: {e}")
        return "PENDING", entry_price

def analyze(symbol, df_5m, df_15m=None, df_1h=None, expiration=1):
    print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] Анализирую {symbol} для прогноза с экспирацией {expiration} мин...")
    if len(df_5m) < 30:
        reason = "Недостаточно данных для анализа (менее 30 свечей)"
        print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] {symbol}: {reason}")
        return "WAIT", 0, 0, 0, 0, reason, 0, 0, 0, 0, 0, 0, 0, 0, 0
    
    try:
        close = df_5m['close']
        high = df_5m['high']
        low = df_5m['low']
        open = df_5m['open']
        volume = df_5m['volume']
        
        rsi_window = 5 if expiration == 1 else 7 if expiration == 2 else 10
        rsi = RSIIndicator(close, window=rsi_window).rsi()
        macd = MACD(close, window_slow=26, window_fast=12, window_sign=9)
        bb = BollingerBands(close, window=20, window_dev=2)
        ema5 = EMAIndicator(close, window=5).ema_indicator()
        ema12 = EMAIndicator(close, window=12).ema_indicator()
        adx = ADXIndicator(high=high, low=low, close=close, window=14).adx()
        stochastic = StochasticOscillator(close=close, high=high, low=low, window=rsi_window, smooth_window=3).stoch()
        atr = AverageTrueRange(high=high, low=low, close=close, window=14).average_true_range()

        bullish_fractals, bearish_fractals = detect_fractals(df_5m)
        print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] Бычьи фракталы в последних 5 свечах: {bullish_fractals.iloc[-5:].any()}, Медвежьи фракталы: {bearish_fractals.iloc[-5:].any()}")

        rsi_v = rsi.iloc[-1]
        macd_val = macd.macd().iloc[-1]
        signal_val = macd.macd_signal().iloc[-1]
        ema5_v = ema5.iloc[-1]
        ema12_v = ema12.iloc[-1]
        adx_v = adx.iloc[-1]
        stoch_v = stochastic.iloc[-1]
        price = close.iloc[-1]
        open_price = open.iloc[-1]
        upper_bb = bb.bollinger_hband().iloc[-1]
        lower_bb = bb.bollinger_lband().iloc[-1]
        atr_v = atr.iloc[-1]
        bb_width = (upper_bb - lower_bb) / price

        bb_width_series = (bb.bollinger_hband()[-10:] - bb.bollinger_lband()[-10:]) / close[-10:]
        bb_width_mean = bb_width_series.mean()
        
        atr_mean = atr[-10:].mean()
        atr_historical = atr[-20:].mean()
        expected_move = atr_mean * (expiration / 5.0)
        price_high = price * (1.00005 + 0.00003 * expiration)
        price_low = price * (0.99995 - 0.00003 * expiration)

        # Адаптивные пороги
        market_volatility = atr_v / atr_historical
        trend_strength = adx_v / adx[-10:].mean()
        rsi_mean = rsi[-10:].mean()
        rsi_std = rsi[-10:].std()
        adx_mean = adx[-10:].mean()
        
        RSI_BUY_THRESHOLD = max(15, rsi_mean - rsi_std * (1 - 0.5 * market_volatility))
        RSI_SELL_THRESHOLD = min(85, rsi_mean + rsi_std * (1 - 0.5 * market_volatility))
        STOCH_BUY_THRESHOLD = max(10, 15 - 5 * (1 - market_volatility))
        STOCH_SELL_THRESHOLD = min(90, 85 + 5 * (1 - market_volatility))
        MIN_ADX = max(12, adx_mean * 0.6) if market_volatility > 1.5 else max(10, adx_mean * 0.5)
        BB_WIDTH_MIN = max(0.0001, bb_width_mean * 0.2) if market_volatility > 1.5 else max(0.00008, bb_width_mean * 0.15)
        MIN_ATR = atr_mean * 0.3 * (1 - 0.2 * market_volatility)

        print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] {symbol}: market_volatility={market_volatility:.2f}, trend_strength={trend_strength:.2f}, MIN_ADX={MIN_ADX:.2f}, BB_WIDTH_MIN={BB_WIDTH_MIN:.4f}")

        # Адаптивные веса условий
        weights = {
            'rsi': 1.0 + 0.3 * (1 - trend_strength),
            'macd': 3.0 if adx_v > 25 else 2.5,
            'ema': 2.0,
            'stoch': 1.0 + 0.3 * (1 - trend_strength),
            'bb': 1.0,
            'trend': 2.0 if adx_v > 25 else 1.5,
            'candle': 1.0,
            'price_trend': 1.0,
            'fractal': 1.2 + 0.3 * market_volatility,
            'volume': 1.5,
            'atr': 1.5,
            'correlation': 2.0,
            'pattern': 1.5
        }
        print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] Адаптивные веса: {weights}")

        # Определение тренда M15 и H1
        trend_m15 = "NEUTRAL"
        trend_h1 = "NEUTRAL"
        m15_macd_confirmed = False
        h1_trend_confirmed = False
        if df_15m is not None:
            try:
                ema5_m15 = EMAIndicator(df_15m['close'], window=5).ema_indicator().iloc[-1]
                ema12_m15 = EMAIndicator(df_15m['close'], window=12).ema_indicator().iloc[-1]
                macd_m15 = MACD(df_15m['close'], window_slow=26, window_fast=12, window_sign=9)
                macd_m15_val = macd_m15.macd().iloc[-1]
                signal_m15_val = macd_m15.macd_signal().iloc[-1]
                trend_m15 = "BULLISH" if ema5_m15 > ema12_m15 else "BEARISH" if ema5_m15 < ema12_m15 else "NEUTRAL"
                m15_macd_confirmed = (macd_m15_val > signal_m15_val) if trend_m15 == "BULLISH" else (macd_m15_val < signal_m15_val) if trend_m15 == "BEARISH" else False
            except Exception as e:
                print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] Ошибка расчёта тренда M15 для {symbol}: {e}")
                trend_m15 = "NEUTRAL"
                m15_macd_confirmed = False
        
        if df_1h is not None:
            try:
                ema5_h1 = EMAIndicator(df_1h['close'], window=5).ema_indicator().iloc[-1]
                ema12_h1 = EMAIndicator(df_1h['close'], window=12).ema_indicator().iloc[-1]
                trend_h1 = "BULLISH" if ema5_h1 > ema12_h1 else "BEARISH" if ema5_h1 < ema12_h1 else "NEUTRAL"
                h1_trend_confirmed = trend_h1 == trend_m15 and trend_m15 != "NEUTRAL"
            except Exception as e:
                print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] Ошибка расчёта тренда H1 для {symbol}: {e}")
                trend_h1 = "NEUTRAL"
                h1_trend_confirmed = False

        # Проверка резкого роста ATR и объёма
        atr_spike = atr_v > atr_historical * 1.4
        volume_trend = volume.iloc[-1] > volume[-10:].mean() * 1.3

        # Динамическая вероятность успеха
        success_probability = 0.50
        if adx_v > 25:
            success_probability += 0.15
        if bb_width > bb_width_mean * 1.2:
            success_probability += 0.10
        if bullish_fractals.iloc[-5:].any() or bearish_fractals.iloc[-5:].any():
            success_probability += 0.05
        if market_volatility > 1.2:
            success_probability += 0.05
        if trend_m15 in ["BULLISH", "BEARISH"]:
            success_probability += 0.10
        if h1_trend_confirmed:
            success_probability += 0.05
        if atr_spike:
            success_probability += 0.05
        if volume_trend:
            success_probability += 0.05
        if is_bullish_pattern(df_5m) or is_bearish_pattern(df_5m):
            success_probability += 0.05
        success_probability = min(success_probability, 0.90)

        # Формирование базовой причины
        reason = (f"RSI: {rsi_v:.2f}, ADX: {adx_v:.2f}, Stochastic: {stoch_v:.2f}, MACD: {macd_val:.4f}, "
                  f"Signal: {signal_val:.4f}, ATR: {atr_v:.4f}, BB_Width: {bb_width:.4f}, Trend M15: {trend_m15}, "
                  f"Trend H1: {trend_h1}, Expected Move: ±{expected_move:.4f}, Success Probability: {success_probability:.2%}")

        # Экранирование запятых в reason
        reason = reason.replace(',', ';')

        # Инициализация переменных
        signal = "WAIT"
        stop_loss = 0
        take_profit = 0
        lot_size = 0
        signal_strength = 0

        # Проверка активных сделок
        try:
            df_results = pd.read_csv(RESULT_LOG_FILE, usecols=CSV_COLUMNS) if os.path.exists(RESULT_LOG_FILE) else pd.DataFrame(columns=CSV_COLUMNS)
            active_trades = len(df_results[df_results['Outcome'] == 'PENDING']) if not df_results.empty else 0
        except Exception as e:
            print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] Ошибка чтения {RESULT_LOG_FILE}: {e}")
            send_telegram_message(f"Ошибка чтения {RESULT_LOG_FILE}: {e}")
            clean_csv_file(RESULT_LOG_FILE)
            df_results = pd.DataFrame(columns=CSV_COLUMNS)
            active_trades = 0

        if active_trades >= MAX_ACTIVE_TRADES:
            reason += f"; Превышен лимит активных сделок ({MAX_ACTIVE_TRADES})"
            print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] {symbol}: {reason}")
            return "WAIT", round(rsi_v, 2), signal_strength, price, atr_v, reason, rsi_v, adx_v, stoch_v, macd_val, signal_val, success_probability, stop_loss, take_profit, lot_size

        # Фильтры
        if adx_v < MIN_ADX:
            reason += f"; ADX слишком низкий (< {MIN_ADX:.2f})"
            print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] {symbol}: {reason}")
            return "WAIT", round(rsi_v, 2), signal_strength, price, atr_v, reason, rsi_v, adx_v, stoch_v, macd_val, signal_val, success_probability, stop_loss, take_profit, lot_size
        if bb_width < BB_WIDTH_MIN:
            reason += f"; Узкие Bollinger Bands (< {BB_WIDTH_MIN:.4f})"
            print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] {symbol}: {reason}")
            return "WAIT", round(rsi_v, 2), signal_strength, price, atr_v, reason, rsi_v, adx_v, stoch_v, macd_val, signal_val, success_probability, stop_loss, take_profit, lot_size
        if atr_v < MIN_ATR:
            reason += f"; ATR слишком низкий (< {MIN_ATR:.4f})"
            print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] {symbol}: {reason}")
            return "WAIT", round(rsi_v, 2), signal_strength, price, atr_v, reason, rsi_v, adx_v, stoch_v, macd_val, signal_val, success_probability, stop_loss, take_profit, lot_size
        if is_news_time():
            reason += "; Новости, торговля приостановлена"
            print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] {symbol}: {reason}")
            return "WAIT", round(rsi_v, 2), signal_strength, price, atr_v, reason, rsi_v, adx_v, stoch_v, macd_val, signal_val, success_probability, stop_loss, take_profit, lot_size
        if adx_v < 12 and bb_width < bb_width_mean * 0.8:
            reason += "; Рынок во флэте (низкий ADX и узкие Bollinger Bands)"
            print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] {symbol}: {reason}")
            return "WAIT", round(rsi_v, 2), signal_strength, price, atr_v, reason, rsi_v, adx_v, stoch_v, macd_val, signal_val, success_probability, stop_loss, take_profit, lot_size

        def is_confirmed(signal_type, candles=CONFIRMATION_CANDLES):
            if signal_type == "BUY":
                return all(macd.macd().iloc[-i] > macd.macd_signal().iloc[-i] for i in range(1, candles + 1))
            elif signal_type == "SELL":
                return all(macd.macd().iloc[-i] < macd.macd_signal().iloc[-i] for i in range(1, candles + 1))
            return False

        # Проверка корреляции для BUY
        corr_confirmed, corr_reason = get_correlation_confirmation(symbol, "BUY", df_5m, expiration)
        signal_strength = 0
        reason_add = ""
        if rsi_v < RSI_BUY_THRESHOLD:
            signal_strength += weights['rsi']
            reason_add += "RSI перепродан; "
        if macd_val > signal_val + 0.005 and is_confirmed("BUY") and m15_macd_confirmed:
            signal_strength += weights['macd']
            reason_add += "MACD бычий (подтверждён M15); "
        if ema5_v > ema12_v and ema5.iloc[-2] <= ema12.iloc[-2]:
            signal_strength += weights['ema']
            reason_add += "EMA5 пересекает EMA12 вверх; "
        if stoch_v < STOCH_BUY_THRESHOLD:
            signal_strength += weights['stoch']
            reason_add += "Stochastic перепродан; "
        if price < lower_bb * 1.005:
            signal_strength += weights['bb']
            reason_add += "Цена ниже Bollinger; "
        if trend_m15 == "BULLISH":
            signal_strength += weights['trend']
            reason_add += "Бычий тренд на M15; "
        if trend_h1 == "BULLISH" and h1_trend_confirmed:
            signal_strength += weights['trend'] * 0.5
            reason_add += "Бычий тренд на H1; "
        if close.iloc[-1] > open_price and macd_val > signal_val:
            signal_strength += weights['candle']
            reason_add += "Бычья свеча с MACD подтверждением; "
        if len(close) >= 3 and close.iloc[-1] > close.iloc[-2] > close.iloc[-3]:
            signal_strength += weights['price_trend']
            reason_add += "Рост цены последние 3 свечи; "
        if bullish_fractals.iloc[-5:-1].any() and close.iloc[-1] > close.iloc[-2]:
            signal_strength += weights['fractal']
            reason_add += "Бычий фрактал с подтверждением роста; "
        if volume_trend:
            signal_strength += weights['volume']
            reason_add += "Рост объёма; "
        if atr_spike:
            signal_strength += weights['atr']
            reason_add += "Спайк ATR; "
        if is_bullish_pattern(df_5m):
            signal_strength += weights['pattern']
            reason_add += "Бычий паттерн (поглощение/пин-бар); "
        if corr_confirmed:
            signal_strength += weights['correlation']
            reason_add += corr_reason
        elif trend_m15 == "BULLISH":
            signal_strength += weights['correlation'] * 0.5
            reason_add += "Корреляция недоступна, подтверждено трендом M15 (BULLISH); "

        # Экранирование запятых в reason_add
        reason_add = reason_add.replace(',', ';')

        if signal_strength >= MIN_SIGNAL_STRENGTH and success_probability >= MIN_SUCCESS_PROBABILITY:
            if price_high > price * (1.00005 + 0.00003 * expiration):
                signal_strength += 0.5
                reason_add += f"Прогноз роста на {expiration} мин; "
            else:
                reason_add += f"Прогноз не подтверждает рост на {expiration} мин; "
                signal_strength -= 0.5

            if signal_strength >= MIN_SIGNAL_STRENGTH:
                signal = "BUY"
                stop_loss = price * (0.999 - atr_v * 1.2)
                take_profit = price * (1.001 + atr_v * 1.5)
                reward_risk_ratio = abs(take_profit - price) / abs(price - stop_loss)
                lot_size = min(1.0, RISK_PER_TRADE * ACCOUNT_BALANCE / abs(price - stop_loss))
                if reward_risk_ratio < MIN_REWARD_RISK_RATIO:
                    reason_add += f"; Низкое соотношение риск/прибыль ({reward_risk_ratio:.2f} < {MIN_REWARD_RISK_RATIO})"
                    print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] {symbol}: {reason_add}")
                    return "WAIT", round(rsi_v, 2), 0, price, atr_v, reason + "; " + reason_add, rsi_v, adx_v, stoch_v, macd_val, signal_val, success_probability, 0, 0, 0
                reason = reason + "; " + reason_add
                print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] {symbol}: BUY сигнал, сила={signal_strength:.2f}, причина={reason}")
                return (signal, round(rsi_v, 2), signal_strength, price, atr_v, reason, rsi_v, adx_v, stoch_v,
                        macd_val, signal_val, success_probability, stop_loss, take_profit, lot_size)

        # Проверка корреляции для SELL
        corr_confirmed, corr_reason = get_correlation_confirmation(symbol, "SELL", df_5m, expiration)
        signal_strength = 0
        reason_add = ""
        if rsi_v > RSI_SELL_THRESHOLD:
            signal_strength += weights['rsi']
            reason_add += "RSI перекуплен; "
        if macd_val < signal_val - 0.005 and is_confirmed("SELL") and m15_macd_confirmed:
            signal_strength += weights['macd']
            reason_add += "MACD медвежий (подтверждён M15); "
        if ema5_v < ema12_v and ema5.iloc[-2] >= ema12.iloc[-2]:
            signal_strength += weights['ema']
            reason_add += "EMA5 пересекает EMA12 вниз; "
        if stoch_v > STOCH_SELL_THRESHOLD:
            signal_strength += weights['stoch']
            reason_add += "Stochastic перекуплен; "
        if price > upper_bb * 0.995:
            signal_strength += weights['bb']
            reason_add += "Цена выше Bollinger; "
        if trend_m15 == "BEARISH":
            signal_strength += weights['trend']
            reason_add += "Медвежий тренд на M15; "
        if trend_h1 == "BEARISH" and h1_trend_confirmed:
            signal_strength += weights['trend'] * 0.5
            reason_add += "Медвежий тренд на H1; "
        if close.iloc[-1] < open_price and macd_val < signal_val:
            signal_strength += weights['candle']
            reason_add += "Медвежья свеча с MACD подтверждением; "
        if len(close) >= 3 and close.iloc[-1] < close.iloc[-2] < close.iloc[-3]:
            signal_strength += weights['price_trend']
            reason_add += "Падение цены последние 3 свечи; "
        if bearish_fractals.iloc[-5:-1].any() and close.iloc[-1] < close.iloc[-2]:
            signal_strength += weights['fractal']
            reason_add += "Медвежий фрактал с подтверждением падения; "
        if volume_trend:
            signal_strength += weights['volume']
            reason_add += "Рост объёма; "
        if atr_spike:
            signal_strength += weights['atr']
            reason_add += "Спайк ATR; "
        if is_bearish_pattern(df_5m):
            signal_strength += weights['pattern']
            reason_add += "Медвежий паттерн (поглощение/пин-бар); "
        if corr_confirmed:
            signal_strength += weights['correlation']
            reason_add += corr_reason
        elif trend_m15 == "BEARISH":
            signal_strength += weights['correlation'] * 0.5
            reason_add += "Корреляция недоступна, подтверждено трендом M15 (BEARISH); "

        # Экранирование запятых в reason_add
        reason_add = reason_add.replace(',', ';')

        if signal_strength >= MIN_SIGNAL_STRENGTH and success_probability >= MIN_SUCCESS_PROBABILITY:
            if price_low < price * (0.99995 - 0.00003 * expiration):
                signal_strength += 0.5
                reason_add += f"Прогноз падения на {expiration} мин; "
            else:
                reason_add += f"Прогноз не подтверждает падение на {expiration} мин; "
                signal_strength -= 0.5

            if signal_strength >= MIN_SIGNAL_STRENGTH:
                signal = "SELL"
                stop_loss = price * (1.001 + atr_v * 1.2)
                take_profit = price * (0.999 - atr_v * 1.5)
                reward_risk_ratio = abs(take_profit - price) / abs(price - stop_loss)
                lot_size = min(1.0, RISK_PER_TRADE * ACCOUNT_BALANCE / abs(price - stop_loss))
                if reward_risk_ratio < MIN_REWARD_RISK_RATIO:
                    reason_add += f"; Низкое соотношение риск/прибыль ({reward_risk_ratio:.2f} < {MIN_REWARD_RISK_RATIO})"
                    print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] {symbol}: {reason_add}")
                    return "WAIT", round(rsi_v, 2), 0, price, atr_v, reason + "; " + reason_add, rsi_v, adx_v, stoch_v, macd_val, signal_val, success_probability, 0, 0, 0
                reason = reason + "; " + reason_add
                print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] {symbol}: SELL сигнал, сила={signal_strength:.2f}, причина={reason}")
                return (signal, round(rsi_v, 2), signal_strength, price, atr_v, reason, rsi_v, adx_v, stoch_v,
                        macd_val, signal_val, success_probability, stop_loss, take_profit, lot_size)

        reason = reason + "; " + reason_add + f"; Недостаточно условий для сигнала или низкая вероятность ({success_probability:.2%} < {MIN_SUCCESS_PROBABILITY:.2%})"
        print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] {symbol}: {reason}")
        return "WAIT", round(rsi_v, 2), signal_strength, price, atr_v, reason, rsi_v, adx_v, stoch_v, macd_val, signal_val, success_probability, stop_loss, take_profit, lot_size

    except Exception as e:
        print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] Ошибка анализа {symbol} ({DEFAULT_TIMEFRAME}): {e}")
        send_telegram_message(f"Ошибка анализа {symbol}: {e}")
        return "WAIT", 0, 0, 0, 0, f"Ошибка: {e}", 0, 0, 0, 0, 0, 0, 0, 0, 0

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [[InlineKeyboardButton(symbol, callback_data=symbol) for symbol in SYMBOLS[i:i+2]] for i in range(0, len(SYMBOLS), 2)]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text("Выберите актив для анализа:", reply_markup=reply_markup)

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    symbol = query.data
    user_id = query.from_user.id
    user_selections[user_id] = symbol
    df_5m = get_data(symbol, interval="5m", period="3d")
    df_15m = get_data(symbol, interval="15m", period="7d")
    df_1h = get_data(symbol, interval="1h", period="30d")
    if df_5m is None:
        await query.message.reply_text(f"Ошибка: Не удалось загрузить данные для {symbol}")
        return
    
    signal, rsi, signal_strength, price, atr, reason, rsi_v, adx_v, stoch_v, macd_v, signal_v, success_prob, stop_loss, take_profit, lot_size = analyze(symbol, df_5m, df_15m, df_1h, expiration=1)
    
    if signal in ["BUY", "SELL"]:
        entry_time = datetime.now(LOCAL_TZ).strftime("%H:%M:%S")
        msg = (f"📊 *Сигнал для {symbol}*\n"
               f"📈 Направление: {signal}\n"
               f"💰 Цена входа: {price:.5f}\n"
               f"🛑 Stop Loss: {stop_loss:.5f}\n"
               f"🎯 Take Profit: {take_profit:.5f}\n"
               f"📏 Размер лота: {lot_size:.2f}\n"
               f"⏰ Время: {entry_time}\n"
               f"📉 RSI: {rsi_v:.2f}, ADX: {adx_v:.2f}, Stochastic: {stoch_v:.2f}\n"
               f"📊 MACD: {macd_v:.4f}, Signal: {signal_v:.4f}\n"
               f"🔍 ATR: {atr:.4f}, Вероятность: {success_prob:.2%}\n"
               f"💡 Причина: {reason}")
        send_telegram_message(msg, symbol)
        
        # Сохранение сигнала в CSV
        with open(CSV_FILE, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([entry_time, symbol, signal, price, stop_loss, take_profit, lot_size, reason, success_prob, "PENDING", price, 0])
        
        # Сохранение в results_log.csv
        os.makedirs(os.path.dirname(RESULT_LOG_FILE), exist_ok=True)
        with open(RESULT_LOG_FILE, mode='a', newline='') as file:
            writer = csv.writer(file)
            if os.path.getsize(RESULT_LOG_FILE) == 0:
                writer.writerow(CSV_COLUMNS)
            writer.writerow([entry_time, symbol, signal, price, stop_loss, take_profit, lot_size, reason, success_prob, "PENDING", price, 0])
        
        last_signal_time[symbol] = datetime.now()
    else:
        await query.message.reply_text(f"Сигнал для {symbol}: {signal}\nПричина: {reason}")

async def check_results(context: ContextTypes.DEFAULT_TYPE):
    if not os.path.exists(CSV_FILE):
        return
    
    try:
        df = pd.read_csv(CSV_FILE, names=CSV_COLUMNS, usecols=CSV_COLUMNS)
    except Exception as e:
        print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] Ошибка чтения {CSV_FILE}: {e}")
        send_telegram_message(f"Ошибка чтения {CSV_FILE}: {e}")
        clean_csv_file(CSV_FILE)
        df = pd.DataFrame(columns=CSV_COLUMNS)
    
    current_time = datetime.now(LOCAL_TZ)
    
    for index, row in df.iterrows():
        if row['Outcome'] != "PENDING":
            continue
        entry_time = datetime.strptime(row['Entry_Time'], "%H:%M:%S").replace(
            year=current_time.year, month=current_time.month, day=current_time.day, tzinfo=LOCAL_TZ
        )
        if (current_time - entry_time).total_seconds() / 60 > DELETE_AFTER_MINUTES:
            outcome, exit_price = update_outcome(row['Symbol'], row['Entry_Time'], row['Entry_Price'], row['Signal'], 1, row['Stop_Loss'], row['Take_Profit'])
            profit = (exit_price - row['Entry_Price']) * row['Lot_Size'] * 100000 if row['Signal'] == "BUY" else (row['Entry_Price'] - exit_price) * row['Lot_Size'] * 100000
            df.at[index, 'Outcome'] = outcome
            df.at[index, 'Exit_Price'] = exit_price
            df.at[index, 'Profit'] = profit
            
            # Обновление results_log.csv
            try:
                df_results = pd.read_csv(RESULT_LOG_FILE, usecols=CSV_COLUMNS) if os.path.exists(RESULT_LOG_FILE) else pd.DataFrame(columns=CSV_COLUMNS)
                df_results.loc[df_results['Entry_Time'] == row['Entry_Time'], 'Outcome'] = outcome
                df_results.loc[df_results['Entry_Time'] == row['Entry_Time'], 'Exit_Price'] = exit_price
                df_results.loc[df_results['Entry_Time'] == row['Entry_Time'], 'Profit'] = profit
                df_results.to_csv(RESULT_LOG_FILE, index=False)
            except Exception as e:
                print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] Ошибка обновления {RESULT_LOG_FILE}: {e}")
                send_telegram_message(f"Ошибка обновления {RESULT_LOG_FILE}: {e}")
                clean_csv_file(RESULT_LOG_FILE)
            
            msg = f"🔍 Результат для {row['Symbol']} ({row['Entry_Time']}): {outcome}\nПрибыль: {profit:.2f}"
            send_telegram_message(msg, row['Symbol'])
    
    df = df[df['Outcome'] == "PENDING"]
    df.to_csv(CSV_FILE, index=False, header=False)

async def auto_analyze(context: ContextTypes.DEFAULT_TYPE):
    if not is_active_session():
        print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] Вне активной сессии (08:00-22:00)")
        return
    
    print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] Запуск автоматического анализа")
    for symbol in SYMBOLS:
        now = datetime.now()
        if last_signal_time[symbol] and (now - last_signal_time[symbol]).seconds < MIN_SIGNAL_INTERVAL:
            print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] Пропуск {symbol}: слишком частые сигналы")
            continue
        
        print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] Анализ {symbol} на таймфрейме {DEFAULT_TIMEFRAME} с экспирацией 1 мин")
        df_5m = get_data(symbol, interval="5m", period="3d")
        df_15m = get_data(symbol, interval="15m", period="7d")
        df_1h = get_data(symbol, interval="1h", period="30d")
        if df_5m is None:
            print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] Не удалось загрузить данные для {symbol}")
            continue
        
        signal, rsi, signal_strength, price, atr, reason, rsi_v, adx_v, stoch_v, macd_v, signal_v, success_prob, stop_loss, take_profit, lot_size = analyze(symbol, df_5m, df_15m, df_1h, expiration=1)
        
        if signal in ["BUY", "SELL"]:
            entry_time = datetime.now(LOCAL_TZ).strftime("%H:%M:%S")
            msg = (f"📊 *Сигнал для {symbol}*\n"
                   f"📈 Направление: {signal}\n"
                   f"💰 Цена входа: {price:.5f}\n"
                   f"🛑 Stop Loss: {stop_loss:.5f}\n"
                   f"🎯 Take Profit: {take_profit:.5f}\n"
                   f"📏 Размер лота: {lot_size:.2f}\n"
                   f"⏰ Время: {entry_time}\n"
                   f"📉 RSI: {rsi_v:.2f}, ADX: {adx_v:.2f}, Stochastic: {stoch_v:.2f}\n"
                   f"📊 MACD: {macd_v:.4f}, Signal: {signal_v:.4f}\n"
                   f"🔍 ATR: {atr:.4f}, Вероятность: {success_prob:.2%}\n"
                   f"💡 Причина: {reason}")
            send_telegram_message(msg, symbol)
            
            # Сохранение сигнала в CSV
            with open(CSV_FILE, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([entry_time, symbol, signal, price, stop_loss, take_profit, lot_size, reason, success_prob, "PENDING", price, 0])
            
            # Сохранение в results_log.csv
            os.makedirs(os.path.dirname(RESULT_LOG_FILE), exist_ok=True)
            with open(RESULT_LOG_FILE, mode='a', newline='') as file:
                writer = csv.writer(file)
                if os.path.getsize(RESULT_LOG_FILE) == 0:
                    writer.writerow(CSV_COLUMNS)
                writer.writerow([entry_time, symbol, signal, price, stop_loss, take_profit, lot_size, reason, success_prob, "PENDING", price, 0])
            
            last_signal_time[symbol] = now
        else:
            print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] {symbol}: {signal}, Причина: {reason}")

def main():
    print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] Бот запускается...")
    # Очистка CSV файлов при старте
    clean_csv_file(CSV_FILE)
    clean_csv_file(RESULT_LOG_FILE)
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CallbackQueryHandler(button_callback))
    application.job_queue.run_repeating(check_results, interval=60, first=10)
    application.job_queue.run_repeating(auto_analyze, interval=300, first=10)
    
    print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] Бот успешно запущен и начал анализ рынка!")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()
