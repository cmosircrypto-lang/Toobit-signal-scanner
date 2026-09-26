import os
import requests
import pandas as pd
import numpy as np
# ============================================================
# SETTINGS
# ============================================================
BASE_URL = "https://api.toobit.com/quote/v1/klines"
SYMBOLS = [
    "BTC-SWAP-USDT",
    "ETH-SWAP-USDT",
    "SOL-SWAP-USDT",
    "XRP-SWAP-USDT",
    "BNB-SWAP-USDT",
    "DOGE-SWAP-USDT",
    "AVAX-SWAP-USDT",
    "LINK-SWAP-USDT",
]
LIMIT = 300
MIN_SCORE = 5
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
# ============================================================
# TELEGRAM
# ============================================================
def send_telegram(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram settings missing")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        response = requests.post(
            url,
            data={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": message
            },
            timeout=15
        )
        if response.status_code == 200:
            print("Telegram: message sent")
        else:
            print("Telegram ERROR:", response.status_code)
            print(response.text)
    except Exception as e:
        print("Telegram ERROR:", e)
# ============================================================
# GET KLINES
# ============================================================
def get_klines(symbol, interval):
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": LIMIT
    }
    response = requests.get(
        BASE_URL,
        params=params,
        timeout=15
    )
    response.raise_for_status()
    data = response.json()
    if isinstance(data, dict):
        if "data" in data:
            data = data["data"]
        elif "result" in data:
            data = data["result"]
    rows = []
    for x in data:
        rows.append([
            float(x[0]),
            float(x[1]),
            float(x[2]),
            float(x[3]),
            float(x[4]),
            float(x[5])
        ])
    df = pd.DataFrame(
        rows,
        columns=[
            "time",
            "open",
            "high",
            "low",
            "close",
            "volume"
        ]
    )
    return df
# ============================================================
# EMA
# ============================================================
def ema(series, length):
    return series.ewm(
        span=length,
        adjust=False
    ).mean()
# ============================================================
# RSI
# ============================================================
def calculate_rsi(series, length=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(
        alpha=1 / length,
        adjust=False
    ).mean()
    avg_loss = loss.ewm(
        alpha=1 / length,
        adjust=False
    ).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi
# ============================================================
# ATR
# ============================================================
def calculate_atr(df, length=14):
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)
    atr = tr.ewm(
        alpha=1 / length,
        adjust=False
    ).mean()
    return atr
# ============================================================
# SUPERTREND
# ============================================================
def calculate_supertrend(df, period=10, multiplier=3.0):
    atr = calculate_atr(df, period)
    hl2 = (df["high"] + df["low"]) / 2
    upperband = hl2 + multiplier * atr
    lowerband = hl2 - multiplier * atr
    final_upper = upperband.copy()
    final_lower = lowerband.copy()
    direction = pd.Series(
        1,
        index=df.index,
        dtype=float
    )
    for i in range(1, len(df)):
        if (
            upperband.iloc[i] < final_upper.iloc[i - 1]
            or df["close"].iloc[i - 1] > final_upper.iloc[i - 1]
        ):
            final_upper.iloc[i] = upperband.iloc[i]
        else:
            final_upper.iloc[i] = final_upper.iloc[i - 1]
        if (
            lowerband.iloc[i] > final_lower.iloc[i - 1]
            or df["close"].iloc[i - 1] < final_lower.iloc[i - 1]
        ):
            final_lower.iloc[i] = lowerband.iloc[i]
        else:
            final_lower.iloc[i] = final_lower.iloc[i - 1]
        if direction.iloc[i - 1] == 1:
            if df["close"].iloc[i] <= final_upper.iloc[i]:
                direction.iloc[i] = -1
            else:
                direction.iloc[i] = 1
        else:
            if df["close"].iloc[i] >= final_lower.iloc[i]:
                direction.iloc[i] = 1
            else:
                direction.iloc[i] = -1
    return direction
# ============================================================
# STOCHASTIC
# ============================================================
def calculate_stochastic(df, length=14, smooth=3):
    lowest_low = df["low"].rolling(length).min()
    highest_high = df["high"].rolling(length).max()
    denominator = (
        highest_high - lowest_low
    ).replace(0, np.nan)
    k = (
        100
        * (df["close"] - lowest_low)
        / denominator
    )
    d = k.rolling(smooth).mean()
    return k, d
# ============================================================
# ADX + DI
# ============================================================
def calculate_adx(df, length=14):
    high = df["high"]
    low = df["low"]
    close = df["close"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(
        np.where(
            (up_move > down_move) & (up_move > 0),
            up_move,
            0
        ),
        index=df.index
    )
    minus_dm = pd.Series(
        np.where(
            (down_move > up_move) & (down_move > 0),
            down_move,
            0
        ),
        index=df.index
    )
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ],
        axis=1
    ).max(axis=1)
    atr = tr.ewm(
        alpha=1 / length,
        adjust=False
    ).mean()
    plus_di = (
        100
        * plus_dm.ewm(
            alpha=1 / length,
            adjust=False
        ).mean()
        / atr
    )
    minus_di = (
        100
        * minus_dm.ewm(
            alpha=1 / length,
            adjust=False
        ).mean()
        / atr
    )
    dx = (
        100
        * (plus_di - minus_di).abs()
        / (plus_di + minus_di).replace(0, np.nan)
    )
    adx = dx.ewm(
        alpha=1 / length,
        adjust=False
    ).mean()
    return adx, plus_di, minus_di
# ============================================================
# PREPARE INDICATORS
# ============================================================
def prepare_dataframe(df):
    df = df.copy()
    df["ema20"] = ema(
        df["close"],
        20
    )
    df["ema200"] = ema(
        df["close"],
        200
    )
    df["rsi"] = calculate_rsi(
        df["close"],
        14
    )
    df["atr"] = calculate_atr(
        df,
        14
    )
    df["st"] = calculate_supertrend(
        df,
        10,
        3.0
    )
    df["k"], df["d"] = calculate_stochastic(
        df,
        14,
        3
    )
    (
        df["adx"],
        df["plus_di"],
        df["minus_di"]
    ) = calculate_adx(
        df,
        14
    )
    return df
# ============================================================
# TIMEFRAME STATE
# ============================================================
def timeframe_state(df):
    df = prepare_dataframe(df)
    # Last CLOSED candle
    row = df.iloc[-2]
    state = {
        "close": row["close"],
        "ema20": row["ema20"],
        "ema200": row["ema200"],
        "rsi": row["rsi"],
        "st": row["st"],
        "k": row["k"],
        "d": row["d"],
        "adx": row["adx"],
        "plus_di": row["plus_di"],
        "minus_di": row["minus_di"]
    }
    return state
# ============================================================
# CHECK SIGNAL
# ============================================================
def check_signal(df5, df15, df30):
    df5 = prepare_dataframe(df5)
    df15 = prepare_dataframe(df15)
    df30 = prepare_dataframe(df30)
    # Last CLOSED 5m candles
    cur = df5.iloc[-2]
    prev = df5.iloc[-3]
    # Last CLOSED 15m / 30m candles
    tf15 = df15.iloc[-2]
    tf30 = df30.iloc[-2]
    # ========================================================
    # 15m CONFIRMATION
    # ========================================================
    long_15 = (
        tf15["close"] > tf15["ema20"]
        and tf15["ema20"] > tf15["ema200"]
        and tf15["st"] < 0
    )
    short_15 = (
        tf15["close"] < tf15["ema20"]
        and tf15["ema20"] < tf15["ema200"]
        and tf15["st"] > 0
    )
    # ========================================================
    # 30m CONFIRMATION
    # ========================================================
    long_30 = (
        tf30["close"] > tf30["ema20"]
        and tf30["ema20"] > tf30["ema200"]
        and tf30["st"] < 0
    )
    short_30 = (
        tf30["close"] < tf30["ema20"]
        and tf30["ema20"] < tf30["ema200"]
        and tf30["st"] > 0
    )
    # ========================================================
    # LONG SCORE
    # ========================================================
    long_score = 0
    if long_30:
        long_score += 2
    if long_15:
        long_score += 1
    if cur["ema20"] > cur["ema200"]:
        long_score += 1
    if cur["close"] > cur["ema20"]:
        long_score += 1
    if cur["st"] < 0:
        long_score += 1
    if (
        cur["rsi"] > 50
        and cur["rsi"] > prev["rsi"]
    ):
        long_score += 1
    if (
        cur["k"] > cur["d"]
        and cur["k"] < 85
    ):
        long_score += 1
    if (
        cur["adx"] >= 18
        and cur["plus_di"] > cur["minus_di"]
    ):
        long_score += 1
    # ========================================================
    # SHORT SCORE
    # ========================================================
    short_score = 0
    if short_30:
        short_score += 2
    if short_15:
        short_score += 1
    if cur["ema20"] < cur["ema200"]:
        short_score += 1
    if cur["close"] < cur["ema20"]:
        short_score += 1
    if cur["st"] > 0:
        short_score += 1
    if (
        cur["rsi"] < 50
        and cur["rsi"] < prev["rsi"]
    ):
        short_score += 1
    if (
        cur["k"] < cur["d"]
        and cur["k"] > 15
    ):
        short_score += 1
    if (
        cur["adx"] >= 18
        and cur["minus_di"] > cur["plus_di"]
    ):
        short_score += 1
    # ========================================================
    # FRESH CORE
    # ========================================================
    current_long_core = (
        cur["ema20"] > cur["ema200"]
        and cur["close"] > cur["ema20"]
        and cur["st"] < 0
        and cur["rsi"] > 50
    )
    previous_long_core = (
        prev["ema20"] > prev["ema200"]
        and prev["close"] > prev["ema20"]
        and prev["st"] < 0
        and prev["rsi"] > 50
    )
    current_short_core = (
        cur["ema20"] < cur["ema200"]
        and cur["close"] < cur["ema20"]
        and cur["st"] > 0
        and cur["rsi"] < 50
    )
    previous_short_core = (
        prev["ema20"] < prev["ema200"]
        and prev["close"] < prev["ema20"]
        and prev["st"] > 0
        and prev["rsi"] < 50
    )
    fresh_long = (
        current_long_core
        and not previous_long_core
    )
    fresh_short = (
        current_short_core
        and not previous_short_core
    )
    # ========================================================
    # FINAL SIGNAL
    # ========================================================
    if (
        fresh_long
        and long_score >= MIN_SCORE
    ):
        return {
            "direction": "LONG",
            "score": long_score,
            "price": cur["close"],
            "rsi": cur["rsi"],
            "adx": cur["adx"]
        }
    if (
        fresh_short
        and short_score >= MIN_SCORE
    ):
        return {
            "direction": "SHORT",
            "score": short_score,
            "price": cur["close"],
            "rsi": cur["rsi"],
            "adx": cur["adx"]
        }
    return None
# ============================================================
# MAIN SCANNER
# ============================================================
def main():
    print("========================================")
    print("TOOBIT STRONG SIGNAL SCANNER")
    print("========================================")
    for symbol in SYMBOLS:
        try:
            df5 = get_klines(
                symbol,
                "5m"
            )
            df15 = get_klines(
                symbol,
                "15m"
            )
            df30 = get_klines(
                symbol,
                "30m"
            )
            signal = check_signal(
                df5,
                df15,
                df30
            )
            if signal:
                message = (
                    "🚨 TOOBIT STRONG SIGNAL\n\n"
                    f"Symbol: {symbol}\n"
                    f"Signal: STRONG {signal['direction']}\n"
                    f"Score: {signal['score']}/9\n"
                    f"Price: {signal['price']}\n"
                    f"RSI: {signal['rsi']:.2f}\n"
                    f"ADX: {signal['adx']:.2f}\n\n"
                    "Timeframe: 5m\n"
                    "Confirmation: 15m + 30m"
                )
                print(
                    f"{symbol} | "
                    f"STRONG {signal['direction']} | "
                    f"Score {signal['score']}/9"
                )
                send_telegram(message)
            else:
                print(
                    f"{symbol} | "
                    "No fresh strong signal"
                )
        except Exception as e:
            print(
                f"{symbol} | ERROR | {e}"
            )
# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    main()

بعد از جایگزینی

پیام Commit را بگذار:

Restore production scanner

بعد:

Actions → Toobit Signal Scanner → Run workflow → Run workflow

اگر خروجی مثلاً این باشد:

BTC-SWAP-USDT | No fresh strong signal
ETH-SWAP-USDT | No fresh strong signal
...

طبیعی است؛ یعنی فعلاً سیگنال تازه‌ای پیدا نشده.

اگر سیگنال پیدا شود، همان لحظه باید پیام STRONG LONG یا STRONG SHORT در Telegram بیاید.