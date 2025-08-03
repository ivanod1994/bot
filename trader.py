import os
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, EMAIndicator, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from bs4 import BeautifulSoup
import pytz
import yfinance as yf

# Настройки
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "8246979603:AAGSP7b-YRol151GlZpfxyyS34rW5ncZJo4")
CHAT_ID = os.getenv("CHAT_ID", "6677680988")
MANUAL_TZ = os.getenv("MANUAL_TZ", "Africa/Algiers")
LOCAL_TZ = pytz.timezone(MANUAL_TZ)
DEFAULT_TIMEFRAME = "5m"
CONFIRMATION_CANDLES = 2
RESULT_LOG_FILE = "results_log.csv"
TIMEOUT = 30
data_cache = {}

def get_data(symbol, interval=DEFAULT_TIMEFRAME, period="1d"):
    cache_key = f"{symbol}_{interval}"
    if cache_key in data_cache and (datetime.now() - data_cache[cache_key]['time']).seconds < 300:
        print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] Использую кэшированные данные для {symbol} ({interval})")
        return data_cache[cache_key]['data']
    
    print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] Загружаю данные для {symbol} ({interval})")
    for attempt in range(3):
        try:
            df = yf.download(symbol, interval=interval, period=period, progress=False)
            if df.empty:
                raise ValueError(f"Данные для {symbol} ({interval}) пусты")
            df['datetime'] = df.index
            df['datetime'] = df['datetime'].dt.tz_localize(None)
            data_cache[cache_key] = {'data': df, 'time': datetime.now()}
            print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] Данные для {symbol} ({interval}) успешно загружены")
            return df
        except Exception as e:
            print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] [{symbol}] Ошибка Yahoo Finance (попытка {attempt+1}): {str(e)}")
            time.sleep(15 ** attempt)
    print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] [{symbol}] Не удалось загрузить данные после 3 попыток")
    return pd.DataFrame()

def is_active_session():
    now = datetime.now(LOCAL_TZ)
    hour = now.hour
    return not (hour < 8 or hour >= 22)

def is_news_time():
    try:
        response = requests.get("https://www.forexfactory.com/calendar", timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        events = soup.find_all('tr', class_='calendar_row')
        for event in events:
            time_elem = event.find('td', class_='calendar__time')
            impact_elem = event.find('td', class_='calendar__impact')
            if time_elem and impact_elem and 'High' in impact_elem.text:
                return True
        return False
    except Exception as e:
        print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] Ошибка проверки новостей: {str(e)}")
        return False

def log_result(symbol, signal, rsi, entry_time, reason, rsi_v, adx_v, stoch_v, macd_val, signal_val, atr_v, entry_price, exit_price, success_probability, outcome="PENDING"):
    try:
        entry_time_str = entry_time.strftime("%Y-%m-%d %H:%M:%S") if isinstance(entry_time, datetime) else entry_time
        logged_at = datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S")
        data = {
            'Symbol': symbol, 'Signal': signal, 'RSI': rsi, 'Entry Time': entry_time_str, 'Logged At': logged_at,
            'Reason': reason, 'Outcome': outcome, 'RSI_Value': rsi_v, 'ADX_Value': adx_v, 'Stoch_Value': stoch_v,
            'MACD_Value': macd_val, 'Signal_Value': signal_val, 'ATR_Value': atr_v, 'Entry_Price': entry_price,
            'Exit_Price': exit_price, 'Success_Probability': success_probability
        }
        df = pd.DataFrame([data])
        df.to_csv(RESULT_LOG_FILE, mode='a', header=not os.path.exists(RESULT_LOG_FILE), index=False)
        print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] Результат записан в лог: {symbol}, {signal}")
    except Exception as e:
        print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] Ошибка записи в лог: {str(e)}")

async def send_telegram_message(message):
    try:
        async with Application.builder().token(TELEGRAM_TOKEN).build() as app:
            await app.bot.send_message(chat_id=CHAT_ID, text=message, parse_mode="Markdown")
            print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] Telegram сообщение отправлено: {message[:50]}...")
    except Exception as e:
        print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] Ошибка Telegram: {str(e)}")

def detect_fractals(df, window=2):
    """
    Определяет бычьи и медвежьи фракталы на основе 5-свечной формации.
    Возвращает два Series: bullish_fractals и bearish_fractals (True, если фрактал найден).
    """
    if len(df) < 5:
        return pd.Series(False, index=df.index), pd.Series(False, index=df.index)
    
    bullish_fractals = pd.Series(False, index=df.index)
    bearish_fractals = pd.Series(False, index=df.index)
    
    for i in range(window, len(df) - window):
        # Бычий фрактал: минимум текущей свечи ниже минимумов двух предыдущих и двух последующих свечей
        if (df['Low'].iloc[i] < df['Low'].iloc[i-1] and
            df['Low'].iloc[i] < df['Low'].iloc[i-2] and
            df['Low'].iloc[i] < df['Low'].iloc[i+1] and
            df['Low'].iloc[i] < df['Low'].iloc[i+2]):
            bullish_fractals.iloc[i] = True
        
        # Медвежий фрактал: максимум текущей свечи выше максимумов двух предыдущих и двух последующих свечей
        if (df['High'].iloc[i] > df['High'].iloc[i-1] and
            df['High'].iloc[i] > df['High'].iloc[i-2] and
            df['High'].iloc[i] > df['High'].iloc[i+1] and
            df['High'].iloc[i] > df['High'].iloc[i+2]):
            bearish_fractals.iloc[i] = True
    
    return bullish_fractals, bearish_fractals

def analyze(symbol, df_5m, df_15m=None, df_1h=None, expiration=1):
    print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] Анализирую {symbol} для прогноза с экспирацией {expiration} мин...")
    if len(df_5m) < 50:
        reason = "Недостаточно данных для анализа (менее 50 свечей)"
        print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] {symbol}: {reason}")
        return "WAIT", 0, 0, 0, 0, reason, 0, 0, 0, 0, 0, 0
    
    close = df_5m['Close']
    high = df_5m['High']
    low = df_5m['Low']
    open = df_5m['Open']
    
    rsi = RSIIndicator(close, window=14).rsi()
    macd = MACD(close, window_slow=26, window_fast=12, window_sign=9)
    bb = BollingerBands(close, window=20, window_dev=2)
    ema5 = EMAIndicator(close, window=5).ema_indicator()
    ema12 = EMAIndicator(close, window=12).ema_indicator()
    ema26 = EMAIndicator(close, window=26).ema_indicator()
    adx = ADXIndicator(high=high, low=low, close=close, window=14).adx()
    stochastic = StochasticOscillator(close=close, high=high, low=low, window=14, smooth_window=3).stoch()
    atr = AverageTrueRange(high=high, low=low, close=close, window=14).average_true_range()

    # Вычисление фракталов
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

    # Прогноз движения цены на основе ATR
    atr_mean = atr[-10:].mean()
    expected_move = atr_mean * (expiration / 5.0)
    price_high = price + expected_move
    price_low = price - expected_move
    success_probability = 0.65

    rsi_mean = rsi[-10:].mean()
    rsi_std = rsi[-10:].std()
    adx_mean = adx[-10:].mean()
    bb_width_series = (bb.bollinger_hband()[-10:] - bb.bollinger_lband()[-10:]) / close[-10:]
    bb_width_mean = bb_width_series.mean()
    atr_mean = atr[-10:].mean()

    RSI_BUY_THRESHOLD = max(30, rsi_mean - rsi_std)
    RSI_SELL_THRESHOLD = min(70, rsi_mean + rsi_std)
    MIN_ADX = max(20, adx_mean * 0.8)
    BB_WIDTH_MIN = max(0.0005, bb_width_mean * 0.5)
    MIN_ATR = atr_mean * 0.5

    trend = "NEUTRAL"
    if df_15m is not None:
        ema5_m15 = EMAIndicator(df_15m['Close'], window=5).ema_indicator().iloc[-1]
        ema12_m15 = EMAIndicator(df_15m['Close'], window=12).ema_indicator().iloc[-1]
        trend = "BULLISH" if ema5_m15 > ema12_m15 else "BEARISH" if ema5_m15 < ema12_m15 else "NEUTRAL"

    reason = (f"RSI: {rsi_v:.2f}, ADX: {adx_v:.2f}, Stochastic: {stoch_v:.2f}, MACD: {macd_val:.4f}, "
              f"Signal: {signal_val:.4f}, ATR: {atr_v:.4f}, BB_Width: {bb_width:.4f}, Trend M15: {trend}")

    if adx_v < MIN_ADX:
        reason += f"; ADX слишком низкий (< {MIN_ADX})"
        print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] {symbol}: {reason}")
        return "WAIT", round(rsi_v, 2), 0, price, atr_v, reason, rsi_v, adx_v, stoch_v, macd_val, signal_val, success_probability
    if bb_width < BB_WIDTH_MIN:
        reason += "; Узкие Bollinger Bands"
        print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] {symbol}: {reason}")
        return "WAIT", round(rsi_v, 2), 0, price, atr_v, reason, rsi_v, adx_v, stoch_v, macd_val, signal_val, success_probability
    if atr_v < MIN_ATR:
        reason += f"; ATR слишком низкий (< {MIN_ATR})"
        print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] {symbol}: {reason}")
        return "WAIT", round(rsi_v, 2), 0, price, atr_v, reason, rsi_v, adx_v, stoch_v, macd_val, signal_val, success_probability
    if not is_active_session() and "JPY" in symbol:
        reason += "; Торговля вне активной сессии для JPY"
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
        signal_strength += 1
        reason += "RSI перепродан; "
    if macd_val > signal_val + 0.005 and is_confirmed("BUY"):
        signal_strength += 2
        reason += "MACD бычий; "
    if ema5_v > ema12_v and ema5.iloc[-2] <= ema12.iloc[-2]:
        signal_strength += 2
        reason += "EMA5 пересекает EMA12 вверх; "
    if stoch_v < 30:
        signal_strength += 1
        reason += "Stochastic перепродан; "
    if price < lower_bb * 1.005:
        signal_strength += 1
        reason += "Цена ниже Bollinger; "
    if df_15m is not None and trend == "BULLISH":
        signal_strength += 1
        reason += "Бычий тренд на M15; "
    if close.iloc[-1] > open_price:
        signal_strength += 1
        reason += "Бычья свеча; "
    if len(close) >= 3 and close.iloc[-1] > close.iloc[-2] > close.iloc[-3]:
        signal_strength += 1
        reason += "Рост цены последние 3 свечи; "
    if bullish_fractals.iloc[-5:].any():
        signal_strength += 1
        reason += "Обнаружен бычий фрактал; "

    if signal_strength >= 3:
        if price_high > price * 1.0005:
            signal_strength += 1
            reason += f"Прогноз роста на {expiration} мин; "
        else:
            reason += f"Прогноз не подтверждает рост на {expiration} мин; "
            signal_strength -= 1

        print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] {symbol}: BUY сигнал, сила={signal_strength}, причина={reason}")
        return "BUY", round(rsi_v, 2), signal_strength, price, atr_v, reason, rsi_v, adx_v, stoch_v, macd_val, signal_val, success_probability

    signal_strength = 0
    reason = ""
    if rsi_v > RSI_SELL_THRESHOLD:
        signal_strength += 1
        reason += "RSI перекуплен; "
    if macd_val < signal_val - 0.005 and is_confirmed("SELL"):
        signal_strength += 2
        reason += "MACD медвежий; "
    if ema5_v < ema12_v and ema5.iloc[-2] >= ema12.iloc[-2]:
        signal_strength += 2
        reason += "EMA5 пересекает EMA12 вниз; "
    if stoch_v > 70:
        signal_strength += 1
        reason += "Stochastic перекуплен; "
    if price > upper_bb * 0.995:
        signal_strength += 1
        reason += "Цена выше Bollinger; "
    if df_15m is not None and trend == "BEARISH":
        signal_strength += 1
        reason += "Медвежий тренд на M15; "
    if close.iloc[-1] < open_price:
        signal_strength += 1
        reason += "Медвежья свеча; "
    if len(close) >= 3 and close.iloc[-1] < close.iloc[-2] < close.iloc[-3]:
        signal_strength += 1
        reason += "Падение цены последние 3 свечи; "
    if bearish_fractals.iloc[-5:].any():
        signal_strength += 1
        reason += "Обнаружен медвежий фрактал; "

    if signal_strength >= 3:
        if price_low < price * 0.9995:
            signal_strength += 1
            reason += f"Прогноз падения на {expiration} мин; "
        else:
            reason += f"Прогноз не подтверждает падение на {expiration} мин; "
            signal_strength -= 1

        print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] {symbol}: SELL сигнал, сила={signal_strength}, причина={reason}")
        return "SELL", round(rsi_v, 2), signal_strength, price, atr_v, reason, rsi_v, adx_v, stoch_v, macd_val, signal_val, success_probability

    reason += "; Недостаточно условий для сигнала"
    print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] {symbol}: {reason}")
    return "WAIT", round(rsi_v, 2), 0, price, atr_v, reason, rsi_v, adx_v, stoch_v, macd_val, signal_val, success_probability

async def run_analysis(context: ContextTypes.DEFAULT_TYPE):
    print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] Запуск автоматического анализа")
    for symbol in ["EURJPY=X", "USDCAD=X", "CADJPY=X", "GBPUSD=X", "AUDUSD=X", "CHFJPY=X"]:
        print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] Анализ {symbol} на таймфрейме {DEFAULT_TIMEFRAME} с экспирацией 1 мин")
        df_5m = get_data(symbol, interval="5m", period="1d")
        df_15m = get_data(symbol, interval="15m", period="1d")
        df_1h = get_data(symbol, interval="60m", period="5d")
        
        expiration = 1  # Фиксированная экспирация 1 минута для автоматического анализа
        signal, rsi, strength, price, atr, reason, rsi_v, adx_v, stoch_v, macd_val, signal_val, success_probability = analyze(symbol, df_5m, df_15m, df_1h, expiration)
        
        if signal != "WAIT" and strength >= 4:
            entry_time = datetime.now(LOCAL_TZ)
            message = (f"🚨 *СИГНАЛ по {symbol}*\n"
                       f"📈 *Прогноз*: {signal}\n"
                       f"📊 *RSI*: {rsi:.2f}\n"
                       f"💪 *Сила сигнала*: {strength}/9\n"
                       f"📝 *Причина*: {reason}\n"
                       f"💵 *Цена*: {price:.4f}\n"
                       f"⏱ *Таймфрейм*: {DEFAULT_TIMEFRAME}\n"
                       f"⏰ *Прогноз на 1 мин*\n"
                       f"🎯 *Вероятность*: {success_probability:.2%}")
            await send_telegram_message(message)
            log_result(symbol, signal, rsi, entry_time, reason, rsi_v, adx_v, stoch_v, macd_val, signal_val, atr, price, None, success_probability)
            print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] {symbol}: Потенциальный сигнал {signal}, сила={strength}, причина={reason}")
        else:
            print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] {symbol}: Потенциальный сигнал {signal}, сила={strength}, причина={reason}")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("Выбрать торговую пару", callback_data='select_pair')],
        [InlineKeyboardButton("Выбрать экспирацию", callback_data='select_expiration')],
        [InlineKeyboardButton("Получить сигнал", callback_data='get_signal')],
        [InlineKeyboardButton("Обновить данные", callback_data='refresh_data')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(
        "Добро пожаловать в торгового бота! Прогнозы с выбранной экспирацией. Используйте кнопки ниже для выбора пары и экспирации.",
        reply_markup=reply_markup
    )
    print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] Получена команда /start от chat_id={update.effective_chat.id}")

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data
    print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] Получен callback от chat_id={query.from_user.id}, data={data}")
    
    if data == 'select_pair':
        keyboard = [
            [InlineKeyboardButton("EURJPY", callback_data='pair_EURJPY=X')],
            [InlineKeyboardButton("USDCAD", callback_data='pair_USDCAD=X')],
            [InlineKeyboardButton("CADJPY", callback_data='pair_CADJPY=X')],
            [InlineKeyboardButton("GBPUSD", callback_data='pair_GBPUSD=X')],
            [InlineKeyboardButton("AUDUSD", callback_data='pair_AUDUSD=X')],
            [InlineKeyboardButton("CHFJPY", callback_data='pair_CHFJPY=X')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.message.reply_text("Выберите торговую пару:", reply_markup=reply_markup)
    
    elif data.startswith('pair_'):
        symbol = data.split('pair_')[1]
        context.user_data['symbol'] = symbol
        await query.message.reply_text(f"Выбрана пара: {symbol.replace('=X', '')}")
    
    elif data == 'select_expiration':
        keyboard = [
            [InlineKeyboardButton("1 минута", callback_data='expiration_1')],
            [InlineKeyboardButton("2 минуты", callback_data='expiration_2')],
            [InlineKeyboardButton("5 минут", callback_data='expiration_5')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.message.reply_text("Выберите экспирацию:", reply_markup=reply_markup)
    
    elif data.startswith('expiration_'):
        expiration = int(data.split('expiration_')[1])
        context.bot_data['expiration'] = expiration
        await query.message.reply_text(f"Выбрана экспирация: {expiration} мин")
    
    elif data == 'get_signal':
        symbol = context.user_data.get('symbol', 'EURJPY=X')
        expiration = context.bot_data.get('expiration', 1)
        print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] Запрос сигнала для {symbol} ({DEFAULT_TIMEFRAME}) с экспирацией {expiration} мин")
        
        df_5m = get_data(symbol, interval="5m", period="1d")
        df_15m = get_data(symbol, interval="15m", period="1d")
        df_1h = get_data(symbol, interval="60m", period="5d")
        
        signal, rsi, strength, price, atr, reason, rsi_v, adx_v, stoch_v, macd_val, signal_val, success_probability = analyze(symbol, df_5m, df_15m, df_1h, expiration)
        
        if signal != "WAIT" and strength >= 3:
            entry_time = datetime.now(LOCAL_TZ)
            message = (f"🚨 *СИГНАЛ по {symbol}*\n"
                       f"📈 *Прогноз*: {signal}\n"
                       f"📊 *RSI*: {rsi:.2f}\n"
                       f"💪 *Сила сигнала*: {strength}/9\n"
                       f"📝 *Причина*: {reason}\n"
                       f"💵 *Цена*: {price:.4f}\n"
                       f"⏱ *Таймфрейм*: {DEFAULT_TIMEFRAME}\n"
                       f"⏰ *Прогноз на {expiration} мин*\n"
                       f"🎯 *Вероятность*: {success_probability:.2%}")
            await send_telegram_message(message)
            log_result(symbol, signal, rsi, entry_time, reason, rsi_v, adx_v, stoch_v, macd_val, signal_val, atr, price, None, success_probability)
        else:
            message = f"⚠️ Сигнал для {symbol} не сформирован (сила: {strength}/9). Причина: {reason}"
            await send_telegram_message(message)
        print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] {symbol}: Потенциальный сигнал {signal}, сила={strength}, причина={reason}")
    
    elif data == 'refresh_data':
        data_cache.clear()
        await query.message.reply_text("Кэш данных очищен")
        print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] Кэш данных очищен")

async def main():
    print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] Бот запускается...")
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CallbackQueryHandler(button_callback))
    
    print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] Запуск автоматического анализа")
    application.job_queue.run_repeating(run_analysis, interval=300, first=10)
    
    print(f"[{datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}] Бот успешно запущен и начал анализ рынка!")
    await application.run_polling(timeout=TIMEOUT)

if __name__ == '__main__':
    import asyncio
    import platform
    if platform.system() == "Emscripten":
        asyncio.ensure_future(main())
    else:
        asyncio.run(main())
