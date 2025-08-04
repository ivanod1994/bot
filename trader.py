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
TIMEOUT = 30
MIN_SIGNAL_INTERVAL = 60
VOLUME_MULTIPLIER = float('inf')
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
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
            response = requests.post(url, data={"chat_id": CHAT_ID, "text": msg}, timeout=90)
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
            volume.iloc[i] > avg_volume.iloc[i] * 1.2):
            bullish_fractals.iloc[i] = True
        
        if (df['high'].iloc[i] > df['high'].iloc[i-1] and
            df['high'].iloc[i] > df['high'].iloc[i-2] and
            df['high'].iloc[i] > df['high'].iloc[i+1] and
            df['high'].iloc[i] > df['high'].iloc[i+2] and
            volume.iloc[i] > avg_volume.iloc[i] * 1.2):
            bearish_fractals.iloc[i] = True
    
    return bullish_fractals, bearish_fractals

def get_data(symbol, interval=DEFAULT_TIMEFRAME, period="1d"):
    cache_key = f"{symbol}_{interval}"
    if cache_key in data_cache and (datetime.now() - data_cache[cache_key]['time']).seconds < 300:
        print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] Использую кэшированные данные для {symbol} ({interval})")
        return data_cache[cache_key]['data']
    
    print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] Загружаю данные для {symbol} ({interval})")
    for attempt in range(3):
        try:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range={'3d' if interval == '5m' else period}&interval={interval}"
            response = session.get(url, headers=HEADERS, timeout=TIMEOUT)
            if response.status_code == 429:
                print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] Ошибка 429, жду перед повтором (попытка {attempt+1})")
                time.sleep(20 ** attempt)
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
            time.sleep(20 ** attempt)
    print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] Не удалось загрузить данные для {symbol} ({interval})")
    return None

def analyze(symbol, df_5m, df_15m=None, df_1h=None, expiration=1):
    print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] Анализирую {symbol} для прогноза с экспирацией {expiration} мин...")
    if len(df_5m) < 30:
        reason = "Недостаточно данных для анализа (менее 30 свечей)"
        print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] {symbol}: {reason}")
        return "WAIT", 0, 0, 0, 0, reason, 0, 0, 0, 0, 0, 0
    
    close = df_5m['close']
    high = df_5m['high']
    low = df_5m['low']
    open = df_5m['open']
    volume = df_5m['volume']
    
    rsi = RSIIndicator(close, window=14).rsi()
    macd = MACD(close, window_slow=26, window_fast=12, window_sign=9)
    bb = BollingerBands(close, window=20, window_dev=2)
    ema5 = EMAIndicator(close, window=5).ema_indicator()
    ema12 = EMAIndicator(close, window=12).ema_indicator()
    ema26 = EMAIndicator(close, window=26).ema_indicator()
    adx = ADXIndicator(high=high, low=low, close=close, window=14).adx()
    stochastic = StochasticOscillator(close=close, high=high, low=low, window=14, smooth_window=3).stoch()
    atr = AverageTrueRange(high=high, low=low, close=close, window=14).average_true_range()

    bullish_fractals, bearish_fractals = detect_fractals(df_5m)
    print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] Бычьи фракталы в последних 5 свечах: {bullish_fractals.iloc[-5:].any()}, Медвежьи фракталы: {bearish_fractals.iloc[-5:].any()}")

    rsi_v = rsi.iloc[-1]
    macd_val = macd.macd().iloc[-1]
    signal_val = macd.macd_signal().iloc[-1]
    ema5_v = ema5.iloc[-1]
    ema12_v = ema12.iloc[-1]
    ema26_v = ema26.iloc[-1]
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
    price_high = price + expected_move
    price_low = price - expected_move

    # Адаптивные пороги
    market_volatility = atr_v / atr_historical
    trend_strength = adx_v / adx[-10:].mean()
    rsi_mean = rsi[-10:].mean()
    rsi_std = rsi[-10:].std()
    adx_mean = adx[-10:].mean()
    
    RSI_BUY_THRESHOLD = max(30, rsi_mean - rsi_std * (1 - 0.3 * market_volatility))
    RSI_SELL_THRESHOLD = min(70, rsi_mean + rsi_std * (1 - 0.3 * market_volatility))
    MIN_ADX = max(12, adx_mean * 0.6 * (1 - 0.3 * trend_strength))  # Снижено с 15 до 12
    BB_WIDTH_MIN = max(0.0003, bb_width_mean * 0.4 * (1 + 0.3 * market_volatility))
    MIN_ATR = atr_mean * 0.5 * (1 - 0.2 * market_volatility)

    print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] {symbol}: market_volatility={market_volatility:.2f}, trend_strength={trend_strength:.2f}, MIN_ADX={MIN_ADX:.2f}, BB_WIDTH_MIN={BB_WIDTH_MIN:.4f}")

    # Адаптивные веса условий
    weights = {
        'rsi': 1.0 + 0.3 * (1 - trend_strength),
        'macd': 2.0 + 0.4 * market_volatility,
        'ema': 2.0,
        'stoch': 1.0 + 0.3 * (1 - trend_strength),
        'bb': 1.0,
        'trend': 1.5 + 0.3 * (1 - trend_strength),  # Увеличен вес тренда M15
        'candle': 1.0,
        'price_trend': 1.0,
        'fractal': 1.2 + 0.3 * market_volatility
    }
    print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] Адаптивные веса: {weights}")

    # Определение тренда M15
    trend = "NEUTRAL"
    m15_macd_confirmed = False
    if df_15m is not None:
        try:
            ema5_m15 = EMAIndicator(df_15m['close'], window=5).ema_indicator().iloc[-1]
            ema12_m15 = EMAIndicator(df_15m['close'], window=12).ema_indicator().iloc[-1]
            macd_m15 = MACD(df_15m['close'], window_slow=26, window_fast=12, window_sign=9)
            macd_m15_val = macd_m15.macd().iloc[-1]
            signal_m15_val = macd_m15.macd_signal().iloc[-1]
            trend = "BULLISH" if ema5_m15 > ema12_m15 else "BEARISH" if ema5_m15 < ema12_m15 else "NEUTRAL"
            m15_macd_confirmed = (macd_m15_val > signal_m15_val) if trend == "BULLISH" else (macd_m15_val < signal_m15_val) if trend == "BEARISH" else False
        except Exception as e:
            print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] Ошибка расчёта тренда M15 для {symbol}: {e}")
            trend = "NEUTRAL"
            m15_macd_confirmed = False

    # Динамическая вероятность успеха
    success_probability = 0.50
    if adx_v > 20:
        success_probability += 0.15
    if bb_width > bb_width_mean * 1.2:
        success_probability += 0.10
    if bullish_fractals.iloc[-5:].any() or bearish_fractals.iloc[-5:].any():
        success_probability += 0.05
    if market_volatility > 1.2:
        success_probability += 0.05
    if df_15m is not None and trend in ["BULLISH", "BEARISH"]:
        success_probability += 0.10
    success_probability = min(success_probability, 0.85)

    reason = (f"RSI: {rsi_v:.2f}, ADX: {adx_v:.2f}, Stochastic: {stoch_v:.2f}, MACD: {macd_val:.4f}, "
              f"Signal: {signal_val:.4f}, ATR: {atr_v:.4f}, BB_Width: {bb_width:.4f}, Trend M15: {trend}, "
              f"Expected Move: ±{expected_move:.4f}, Success Probability: {success_probability:.2%}")

    # Фильтр низкой волатильности
    if atr_v < atr_historical * 0.8:
        reason += "; Низкая текущая волатильность (ATR ниже исторического)"
        print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] {symbol}: {reason}")
        return "WAIT", round(rsi_v, 2), 0, price, atr_v, reason, rsi_v, adx_v, stoch_v, macd_val, signal_val, success_probability

    if adx_v < MIN_ADX:
        reason += f"; ADX слишком низкий (< {MIN_ADX:.2f})"
        print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] {symbol}: {reason}")
        return "WAIT", round(rsi_v, 2), 0, price, atr_v, reason, rsi_v, adx_v, stoch_v, macd_val, signal_val, success_probability
    if bb_width < BB_WIDTH_MIN:
        reason += f"; Узкие Bollinger Bands (< {BB_WIDTH_MIN:.4f})"
        print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] {symbol}: {reason}")
        return "WAIT", round(rsi_v, 2), 0, price, atr_v, reason, rsi_v, adx_v, stoch_v, macd_val, signal_val, success_probability
    if atr_v < MIN_ATR:
        reason += f"; ATR слишком низкий (< {MIN_ATR:.4f})"
        print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] {symbol}: {reason}")
        return "WAIT", round(rsi_v, 2), 0, price, atr_v, reason, rsi_v, adx_v, stoch_v, macd_val, signal_val, success_probability
    if is_news_time():
        reason += "; Новости, торговля приостановлена"
        print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] {symbol}: {reason}")
        return "WAIT", round(rsi_v, 2), 0, price, atr_v, reason, rsi_v, adx_v, stoch_v, macd_val, signal_val, success_probability

    def is_confirmed(signal_type, candles=CONFIRMATION_CANDLES):
        if signal_type == "BUY":
            return all(macd.macd().iloc[-i] > macd.macd_signal().iloc[-i] for i in range(1, candles + 1))
        elif signal_type == "SELL":
            return all(macd.macd().iloc[-i] < macd.macd_signal().iloc[-i] for i in range(1, candles + 1))
        return False

    signal_strength = 0
    reason = ""
    if rsi_v < RSI_BUY_THRESHOLD:
        signal_strength += weights['rsi']
        reason += "RSI перепродан; "
    if macd_val > signal_val + 0.005 and is_confirmed("BUY") and m15_macd_confirmed:
        signal_strength += weights['macd']
        reason += "MACD бычий (подтверждён M15); "
    if ema5_v > ema12_v and ema5.iloc[-2] <= ema12.iloc[-2]:
        signal_strength += weights['ema']
        reason += "EMA5 пересекает EMA12 вверх; "
    if stoch_v < 30:
        signal_strength += weights['stoch']
        reason += "Stochastic перепродан; "
    if price < lower_bb * 1.005:
        signal_strength += weights['bb']
        reason += "Цена ниже Bollinger; "
    if df_15m is not None and trend == "BULLISH":
        signal_strength += weights['trend']
        reason += "Бычий тренд на M15; "
    if close.iloc[-1] > open_price and macd_val > signal_val:
        signal_strength += weights['candle']
        reason += "Бычья свеча с MACD подтверждением; "
    if len(close) >= 3 and close.iloc[-1] > close.iloc[-2] > close.iloc[-3]:
        signal_strength += weights['price_trend']
        reason += "Рост цены последние 3 свечи; "
    if bullish_fractals.iloc[-5:].any():
        signal_strength += weights['fractal']
        reason += "Обнаружен бычий фрактал (подтверждён объёмом); "

    if signal_strength >= 3:
        if price_high > price * 1.0003:
            signal_strength += 1
            reason += f"Прогноз роста на {expiration} мин; "
        else:
            reason += f"Прогноз не подтверждает рост на {expiration} мин; "
            signal_strength -= 1

        if signal_strength >= 3:
            print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] {symbol}: BUY сигнал, сила={signal_strength:.2f}, причина={reason}")
            return "BUY", round(rsi_v, 2), signal_strength, price, atr_v, reason, rsi_v, adx_v, stoch_v, macd_val, signal_val, success_probability

    signal_strength = 0
    reason = ""
    if rsi_v > RSI_SELL_THRESHOLD:
        signal_strength += weights['rsi']
        reason += "RSI перекуплен; "
    if macd_val < signal_val - 0.005 and is_confirmed("SELL") and m15_macd_confirmed:
        signal_strength += weights['macd']
        reason += "MACD медвежий (подтверждён M15); "
    if ema5_v < ema12_v and ema5.iloc[-2] >= ema12.iloc[-2]:
        signal_strength += weights['ema']
        reason += "EMA5 пересекает EMA12 вниз; "
    if stoch_v > 70:
        signal_strength += weights['stoch']
        reason += "Stochastic перекуплен; "
    if price > upper_bb * 0.995:
        signal_strength += weights['bb']
        reason += "Цена выше Bollinger; "
    if df_15m is not None and trend == "BEARISH":
        signal_strength += weights['trend']
        reason += "Медвежий тренд на M15; "
    if close.iloc[-1] < open_price and macd_val < signal_val:
        signal_strength += weights['candle']
        reason += "Медвежья свеча с MACD подтверждением; "
    if len(close) >= 3 and close.iloc[-1] < close.iloc[-2] < close.iloc[-3]:
        signal_strength += weights['price_trend']
        reason += "Падение цены последние 3 свечи; "
    if bearish_fractals.iloc[-5:].any():
        signal_strength += weights['fractal']
        reason += "Обнаружен медвежий фрактал (подтверждён объёмом); "

    if signal_strength >= 3:
        if price_low < price * 0.9997:
            signal_strength += 1
            reason += f"Прогноз падения на {expiration} мин; "
        else:
            reason += f"Прогноз не подтверждает падение на {expiration} мин; "
            signal_strength -= 1

        if signal_strength >= 3:
            print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] {symbol}: SELL сигнал, сила={signal_strength:.2f}, причина={reason}")
            return "SELL", round(rsi_v, 2), signal_strength, price, atr_v, reason, rsi_v, adx_v, stoch_v, macd_val, signal_val, success_probability

    reason += "; Недостаточно условий для сигнала"
    print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] {symbol}: {reason}")
    return "WAIT", round(rsi_v, 2), 0, price, atr_v, reason, rsi_v, adx_v, stoch_v, macd_val, signal_val, success_probability

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] Получена команда /start от chat_id={update.message.chat_id}")
    for attempt in range(3):
        try:
            await update.message.reply_text(
                "Добро пожаловать в торгового бота! Прогнозы адаптируются к рынку. Используйте кнопки для выбора пары, экспирации или силы сигнала.",
                reply_markup=get_main_menu()
            )
            return
        except TimedOut as e:
            print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] Тайм-аут в /start (попытка {attempt+1}): {e}")
            if attempt < 2:
                time.sleep(5 * (2 ** attempt))
            continue
        except Exception as e:
            print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] Ошибка в /start (попытка {attempt+1}): {e}")
            if attempt < 2:
                time.sleep(5 * (2 ** attempt))
            continue
    print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] Не удалось отправить ответ на /start после 3 попыток")
    send_telegram_message("Ошибка: не удалось ответить на /start. Проверьте соединение или настройки бота.")

async def run_analysis(context: ContextTypes.DEFAULT_TYPE):
    if not check_internet():
        print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] Нет интернета, пропускаю анализ")
        return
    print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] Запуск автоматического анализа")
    expiration = 1
    min_signal_strength = context.bot_data.get('auto_signal_strength', 3)
    print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] Минимальная сила сигнала для автоанализа: {min_signal_strength}")
    for symbol in SYMBOLS:
        print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] Анализ {symbol} на таймфрейме {DEFAULT_TIMEFRAME} с экспирацией 1 мин")
        try:
            if last_signal_time[symbol] is not None:
                time_since_last = (datetime.now(LOCAL_TZ) - last_signal_time[symbol]).total_seconds()
                if time_since_last < MIN_SIGNAL_INTERVAL:
                    print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] Пропуск {symbol}: слишком рано после последнего сигнала ({time_since_last:.1f} сек)")
                    continue
            df = get_data(symbol, interval=DEFAULT_TIMEFRAME, period="3d")
            if df is None:
                print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] Не удалось получить данные для {symbol} ({DEFAULT_TIMEFRAME})")
                continue
            df_15m = get_data(symbol, interval="15m", period="3d")
            df_1h = get_data(symbol, interval="60m", period="7d")
            signal, rsi, strength, price, atr_v, reason, rsi_v, adx_v, stoch_v, macd_val, signal_val, success_probability = analyze(symbol, df, df_15m, df_1h, expiration)
            print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] {symbol}: Завершён анализ, результат: {signal}, сила={strength:.2f}, причина={reason}")
            if signal != "WAIT":
                print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] {symbol}: Потенциальный сигнал {signal}, сила={strength:.2f}, причина={reason}")
                if strength >= min_signal_strength:
                    msg = (
                        f"🚨 СИГНАЛ по {symbol.replace('=X','')}\n"
                        f"📈 Прогноз: {signal}\n"
                        f"📊 RSI: {rsi}\n"
                        f"💪 Сила сигнала: {strength:.2f}/9\n"
                        f"📝 Причина: {reason}\n"
                        f"💵 Цена: {price:.4f}\n"
                        f"⏱ Таймфрейм: {DEFAULT_TIMEFRAME}\n"
                        f"⏰ Прогноз на {expiration} мин\n"
                        f"🎯 Вероятность: {success_probability:.2%}"
                    )
                    log_result(symbol.replace('=X',''), signal, rsi, datetime.now(LOCAL_TZ).strftime("%H:%M:%S"), reason, rsi_v, adx_v, stoch_v, macd_val, signal_val, atr_v, price, 0.0, success_probability)
                    if send_telegram_message(msg, symbol):
                        last_signal_time[symbol] = datetime.now(LOCAL_TZ)
                    else:
                        print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] Не удалось отправить сигнал для {symbol}")
                else:
                    print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] Сигнал {signal} отклонён: сила={strength:.2f} < {min_signal_strength}")
        except Exception as e:
            print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] Ошибка анализа {symbol} ({DEFAULT_TIMEFRAME}): {e}")

def get_main_menu():
    keyboard = [
        [InlineKeyboardButton("Выбрать торговую пару", callback_data='select_pair')],
        [InlineKeyboardButton("Выбрать экспирацию", callback_data='select_expiration')],
        [InlineKeyboardButton("Выбрать силу сигнала для автоанализа", callback_data='select_signal_strength')],
        [InlineKeyboardButton("Получить сигнал", callback_data='get_signal')],
        [InlineKeyboardButton("Обновить данные", callback_data='refresh_data')]
    ]
    return InlineKeyboardMarkup(keyboard)

def get_pair_menu():
    keyboard = [[InlineKeyboardButton(symbol.replace('=X', ''), callback_data=f'pair_{symbol}')] for symbol in SYMBOLS]
    keyboard.append([InlineKeyboardButton("Назад", callback_data='back_to_main')])
    return InlineKeyboardMarkup(keyboard)

def get_expiration_menu():
    expirations = [1, 2, 5]
    keyboard = [[InlineKeyboardButton(f"{exp} мин", callback_data=f'expiration_{exp}')] for exp in expirations]
    keyboard.append([InlineKeyboardButton("Назад", callback_data='back_to_main')])
    return InlineKeyboardMarkup(keyboard)

def get_signal_strength_menu():
    strengths = [3, 4, 5]
    keyboard = [[InlineKeyboardButton(f"Сила {strength}/9", callback_data=f'signal_strength_{strength}')] for strength in strengths]
    keyboard.append([InlineKeyboardButton("Назад", callback_data='back_to_main')])
    return InlineKeyboardMarkup(keyboard)

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    chat_id = query.message.chat_id
    data = query.data
    print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] Получен callback от chat_id={chat_id}, data={data}")

    for attempt in range(3):
        try:
            if data == 'select_pair':
                await query.message.edit_text("Выберите торговую пару:", reply_markup=get_pair_menu())
                return
            elif data == 'select_expiration':
                await query.message.edit_text("Выберите экспирацию:", reply_markup=get_expiration_menu())
                return
            elif data == 'select_signal_strength':
                await query.message.edit_text("Выберите минимальную силу сигнала для автоанализа:", reply_markup=get_signal_strength_menu())
                return
            elif data.startswith('pair_'):
                symbol = data.split('_')[1]
                user_selections[chat_id] = user_selections.get(chat_id, {})
                user_selections[chat_id]['symbol'] = symbol
                await query.message.edit_text(f"Выбрана пара: {symbol.replace('=X', '')}\nВыберите действие:", reply_markup=get_main_menu())
                return
            elif data.startswith('expiration_'):
                expiration = int(data.split('_')[1])
                user_selections[chat_id] = user_selections.get(chat_id, {})
                user_selections[chat_id]['expiration'] = expiration
                context.bot_data['expiration'] = expiration
                await query.message.edit_text(f"Выбрана экспирация: {expiration} мин\nВыберите действие:", reply_markup=get_main_menu())
                return
            elif data.startswith('signal_strength_'):
                strength = int(data.split('_')[2])
                context.bot_data['auto_signal_strength'] = strength
                await query.message.edit_text(f"Выбрана минимальная сила сигнала для автоанализа: {strength}/9\nВыберите действие:", reply_markup=get_main_menu())
                return
            elif data == 'get_signal':
                if chat_id not in user_selections or 'symbol' not in user_selections[chat_id] or 'expiration' not in user_selections[chat_id]:
                    await query.message.edit_text("Пожалуйста, выберите торговую пару и экспирацию.", reply_markup=get_main_menu())
                    return
                symbol = user_selections[chat_id]['symbol']
                expiration = user_selections[chat_id]['expiration']
                print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] Запрос сигнала для {symbol} ({DEFAULT_TIMEFRAME}) с экспирацией {expiration} мин")
                df = get_data(symbol, interval=DEFAULT_TIMEFRAME, period="3d")
                if df is None:
                    await query.message.edit_text(f"Ошибка получения данных для {symbol.replace('=X', '')} ({DEFAULT_TIMEFRAME}). Попробуйте позже.")
                    return
                df_15m = get_data(symbol, interval="15m", period="3d")
                df_1h = get_data(symbol, interval="60m", period="7d")
                signal, rsi, strength, price, atr_v, reason, rsi_v, adx_v, stoch_v, macd_val, signal_val, success_probability = analyze(symbol, df, df_15m, df_1h, expiration)
                print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] {symbol}: Завершён анализ, результат: {signal}, сила={strength:.2f}, причина={reason}")
                if signal != "WAIT":
                    print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] {symbol}: Потенциальный сигнал {signal}, сила={strength:.2f}, причина={reason}")
                if signal != "WAIT" and strength >= 3:
                    msg = (
                        f"🚨 СИГНАЛ по {symbol.replace('=X','')}\n"
                        f"📈 Прогноз: {signal}\n"
                        f"📊 RSI: {rsi}\n"
                        f"💪 Сила сигнала: {strength:.2f}/9\n"
                        f"📝 Причина: {reason}\n"
                        f"💵 Цена: {price:.4f}\n"
                        f"⏱ Таймфрейм: {DEFAULT_TIMEFRAME}\n"
                        f"⏰ Прогноз на {expiration} мин\n"
                        f"🎯 Вероятность: {success_probability:.2%}"
                    )
                    log_result(symbol.replace('=X',''), signal, rsi, datetime.now(LOCAL_TZ).strftime("%H:%M:%S"), reason, rsi_v, adx_v, stoch_v, macd_val, signal_val, atr_v, price, 0.0, success_probability)
                    if send_telegram_message(msg, symbol):
                        last_signal_time[symbol] = datetime.now(LOCAL_TZ)
                else:
                    msg = f"⚠️ Сигнал для {symbol.replace('=X','')} ({DEFAULT_TIMEFRAME}): {signal}\nПричина: {reason}"
                await query.message.edit_text(msg, reply_markup=get_main_menu())
                return
            elif data == 'refresh_data':
                data_cache.clear()
                print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] Кэш данных очищен")
                await query.message.edit_text("Данные обновлены.", reply_markup=get_main_menu())
                return
            elif data == 'back_to_main':
                await query.message.edit_text("Выберите действие:", reply_markup=get_main_menu())
                return
        except TimedOut as e:
            print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] Тайм-аут в button_callback (попытка {attempt+1}): {e}")
            if attempt < 2:
                time.sleep(5 * (2 ** attempt))
            continue
        except Exception as e:
            print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] Ошибка в button_callback (попытка {attempt+1}): {e}")
            if attempt < 2:
                time.sleep(5 * (2 ** attempt))
            continue
    print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] Не удалось обработать callback {data} после 3 попыток")
    send_telegram_message(f"Ошибка: не удалось обработать действие {data}. Проверьте соединение.")

def log_result(symbol, signal, rsi, entry_time, reason, rsi_v, adx_v, stoch_v, macd_val, signal_val, atr_v, entry_price, exit_price, success_probability, outcome="PENDING"):
    expected_columns = ["Symbol", "Signal", "RSI", "Entry Time", "Logged At", "Reason", "Outcome", "RSI_Value", "ADX_Value", "Stochastic_Value", "MACD_Value", "Signal_Value", "ATR_Value", "Entry_Price", "Exit_Price", "Success_Probability"]
    try:
        os.makedirs(os.path.dirname(RESULT_LOG_FILE), exist_ok=True)
        with open(RESULT_LOG_FILE, 'a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow([symbol, signal, rsi, entry_time, datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"), reason, outcome, rsi_v, adx_v, stoch_v, macd_val, signal_val, atr_v, entry_price, exit_price, success_probability])
        print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] Результат записан в лог: {symbol}, {signal}")
    except Exception as e:
        print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] Ошибка записи в лог результатов: {e}")

def main():
    try:
        print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] Бот запускается...")
        if not os.path.exists(RESULT_LOG_FILE):
            os.makedirs(os.path.dirname(RESULT_LOG_FILE), exist_ok=True)
            with open(RESULT_LOG_FILE, 'w', newline='', encoding='utf-8') as f:
                csv.writer(f).writerow(["Symbol", "Signal", "RSI", "Entry Time", "Logged At", "Reason", "Outcome", "RSI_Value", "ADX_Value", "Stochastic_Value", "MACD_Value", "Signal_Value", "ATR_Value", "Entry_Price", "Exit_Price", "Success_Probability"])
        application = Application.builder().token(TELEGRAM_TOKEN).build()
        application.job_queue.scheduler.configure(timezone=LOCAL_TZ)
        application.add_handler(CommandHandler("start", start))
        application.add_handler(CallbackQueryHandler(button_callback))
        application.job_queue.run_repeating(run_analysis, interval=120, first=10)  # Увеличен интервал до 120 секунд
        send_telegram_message("Бот успешно запущен и начал анализ рынка!")
        print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] Бот запущен, ожидает команды...")
        application.run_polling()
    except Exception as e:
        print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] Ошибка при запуске бота: {e}")

if __name__ == '__main__':
    main()
