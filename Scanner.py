import requests
import pandas as pd
import numpy as np
import os

BASE_URL = "https://api.toobit.com"

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

# =========================
# TELEGRAM
# =========================

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")


def send_telegram(message):

    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram settings are missing")
        return

    url = (
        f"https://api.telegram.org/bot"
        f"{TELEGRAM_BOT_TOKEN}/sendMessage"
    )

    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message
    }

    try:
        response = requests.post(
            url,
            data=payload,
            timeout=15
        )

        response.raise_for_status()

        print("Telegram notification sent")

    except Exception as e:

        print(
            "Telegram ERROR:",
            str(e)
        )


# =========================
# SETTINGS
# =========================

EMA_FAST = 20
EMA_SLOW = 200
RSI_LEN = 14

ST_ATR_PERIOD = 10
ST_FACTOR = 3.0

ADX_LEN = 14
ADX_MIN = 18

STOCH_LEN = 14
STOCH_SMOOTH = 3
STOCH_OB = 85
STOCH_OS = 15

MIN_SCORE = 5


# =========================
# GET KLINES
# =========================

def get_klines(symbol, interval, limit=300):

    url = f"{BASE_URL}/quote/v1/klines"

    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }

    response = requests.get(
        url,
        params=params,
        timeout=15
    )

    response.raise_for_status()

    data = response.json()

    if not data:
        return None

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


# =========================
# EMA
# =========================

def ema(series, length):

    return series.ewm(
        span=length,
        adjust=False
    ).mean()


# =========================
# RSI
# =========================

def rsi(series, length=14):

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

    rs = avg_gain / avg_loss.replace(
        0,
        np.nan
    )

    result = 100 - (
        100 / (1 + rs)
    )

    return result.fillna(50)


# =========================
# ATR
# =========================

def atr(df, length=14):

    high = df["high"]
    low = df["low"]
    close = df["close"]

    previous_close = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - previous_close).abs(),
        (low - previous_close).abs()
    ], axis=1).max(axis=1)

    return tr.ewm(
        alpha=1 / length,
        adjust=False
    ).mean()


# =========================
# SUPERTREND
# =========================

def supertrend(df, period=10, factor=3.0):

    atr_value = atr(df, period)

    hl2 = (
        df["high"] +
        df["low"]
    ) / 2

    upper_basic = (
        hl2 +
        factor * atr_value
    )

    lower_basic = (
        hl2 -
        factor * atr_value
    )

    upper = upper_basic.copy()
    lower = lower_basic.copy()

    direction = pd.Series(
        1,
        index=df.index,
        dtype="int64"
    )

    trend = pd.Series(
        np.nan,
        index=df.index,
        dtype="float64"
    )

    for i in range(1, len(df)):

        if (
            upper_basic.iloc[i] <
            upper.iloc[i - 1]
            or
            df["close"].iloc[i - 1] >
            upper.iloc[i - 1]
        ):

            upper.iloc[i] = upper_basic.iloc[i]

        else:

            upper.iloc[i] = upper.iloc[i - 1]

        if (
            lower_basic.iloc[i] >
            lower.iloc[i - 1]
            or
            df["close"].iloc[i - 1] <
            lower.iloc[i - 1]
        ):

            lower.iloc[i] = lower_basic.iloc[i]

        else:

            lower.iloc[i] = lower.iloc[i - 1]

        if direction.iloc[i - 1] == 1:

            if (
                df["close"].iloc[i]
                <= lower.iloc[i]
            ):

                direction.iloc[i] = -1

            else:

                direction.iloc[i] = 1

        else:

            if (
                df["close"].iloc[i]
                >= upper.iloc[i]
            ):

                direction.iloc[i] = 1

            else:

                direction.iloc[i] = -1

        if direction.iloc[i] == 1:

            trend.iloc[i] = lower.iloc[i]

        else:

            trend.iloc[i] = upper.iloc[i]

    return trend, direction


# =========================
# STOCHASTIC
# =========================

def stochastic(df, length=14, smooth=3):

    lowest = (
        df["low"]
        .rolling(length)
        .min()
    )

    highest = (
        df["high"]
        .rolling(length)
        .max()
    )

    denominator = (
        highest - lowest
    ).replace(
        0,
        np.nan
    )

    k = 100 * (
        (df["close"] - lowest)
        / denominator
    )

    d = (
        k
        .rolling(smooth)
        .mean()
    )

    return (
        k.fillna(50),
        d.fillna(50)
    )


# =========================
# ADX
# =========================

def adx(df, length=14):

    high = df["high"]
    low = df["low"]

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where(
            (
                (up_move > down_move)
                &
                (up_move > 0)
            ),
            up_move,
            0
        ),
        index=df.index
    )

    minus_dm = pd.Series(
        np.where(
            (
                (down_move > up_move)
                &
                (down_move > 0)
            ),
            down_move,
            0
        ),
        index=df.index
    )

    atr_value = atr(
        df,
        length
    )

    plus_di = (
        100 *
        plus_dm.ewm(
            alpha=1 / length,
            adjust=False
        ).mean()
        / atr_value
    )

    minus_di = (
        100 *
        minus_dm.ewm(
            alpha=1 / length,
            adjust=False
        ).mean()
        / atr_value
    )

    denominator = (
        plus_di +
        minus_di
    ).replace(
        0,
        np.nan
    )

    dx = (
        100 *
        (plus_di - minus_di).abs()
        / denominator
    )

    adx_value = dx.ewm(
        alpha=1 / length,
        adjust=False
    ).mean()

    return (
        adx_value.fillna(0),
        plus_di.fillna(0),
        minus_di.fillna(0)
    )


# =========================
# PREPARE INDICATORS
# =========================

def prepare(df):

    df["ema20"] = ema(
        df["close"],
        EMA_FAST
    )

    df["ema200"] = ema(
        df["close"],
        EMA_SLOW
    )

    df["rsi"] = rsi(
        df["close"],
        RSI_LEN
    )

    _, df["st_direction"] = supertrend(
        df,
        ST_ATR_PERIOD,
        ST_FACTOR
    )

    (
        df["stoch_k"],
        df["stoch_d"]
    ) = stochastic(
        df,
        STOCH_LEN,
        STOCH_SMOOTH
    )

    (
        df["adx"],
        df["plus_di"],
        df["minus_di"]
    ) = adx(
        df,
        ADX_LEN
    )

    return df


# =========================
# TIMEFRAME CONFIRMATION
# =========================

def timeframe_state(df):

    df = prepare(df)

    row = df.iloc[-2]

    bullish_ema = (
        row["ema20"] >
        row["ema200"]
    )

    bearish_ema = (
        row["ema20"] <
        row["ema200"]
    )

    bullish_st = (
        row["st_direction"] == -1
    )

    bearish_st = (
        row["st_direction"] == 1
    )

    return {
        "bullish":
            bullish_ema and bullish_st,

        "bearish":
            bearish_ema and bearish_st,

        "ema_bull":
            bullish_ema,

        "ema_bear":
            bearish_ema,

        "st_bull":
            bullish_st,

        "st_bear":
            bearish_st
    }


# =========================
# SIGNAL
# =========================

def check_signal(symbol):

    df5 = get_klines(
        symbol,
        "5m",
        300
    )

    df15 = get_klines(
        symbol,
        "15m",
        300
    )

    df30 = get_klines(
        symbol,
        "30m",
        300
    )

    if (
        df5 is None
        or df15 is None
        or df30 is None
    ):

        return None

    df5 = prepare(df5)
    df15 = prepare(df15)
    df30 = prepare(df30)

    row = df5.iloc[-2]
    prev = df5.iloc[-3]

    state15 = timeframe_state(df15)
    state30 = timeframe_state(df30)

    # =========================
    # LONG
    # =========================

    long_score = 0

    if state30["bullish"]:
        long_score += 2

    if state15["bullish"]:
        long_score += 1

    if row["ema20"] > row["ema200"]:
        long_score += 1

    if row["close"] > row["ema20"]:
        long_score += 1

    if row["st_direction"] == -1:
        long_score += 1

    if (
        row["rsi"] > 50
        and
        row["rsi"] > prev["rsi"]
    ):
        long_score += 1

    if (
        row["stoch_k"] >
        row["stoch_d"]
        and
        row["stoch_k"] < STOCH_OB
    ):
        long_score += 1

    if (
        row["adx"] >= ADX_MIN
        and
        row["plus_di"] >
        row["minus_di"]
    ):
        long_score += 1

    # =========================
    # SHORT
    # =========================

    short_score = 0

    if state30["bearish"]:
        short_score += 2

    if state15["bearish"]:
        short_score += 1

    if row["ema20"] < row["ema200"]:
        short_score += 1

    if row["close"] < row["ema20"]:
        short_score += 1

    if row["st_direction"] == 1:
        short_score += 1

    if (
        row["rsi"] < 50
        and
        row["rsi"] < prev["rsi"]
    ):
        short_score += 1

    if (
        row["stoch_k"] <
        row["stoch_d"]
        and
        row["stoch_k"] > STOCH_OS
    ):
        short_score += 1

    if (
        row["adx"] >= ADX_MIN
        and
        row["minus_di"] >
        row["plus_di"]
    ):
        short_score += 1

    # =========================
    # FRESH LONG
    # =========================

    previous_long_core = (
        state30["bullish"]
        and
        prev["ema20"] >
        prev["ema200"]
        and
        prev["close"] >
        prev["ema20"]
        and
        prev["st_direction"] == -1
        and
        prev["adx"] >= ADX_MIN
        and
        prev["plus_di"] >
        prev["minus_di"]
    )

    current_long_core = (
        state30["bullish"]
        and
        row["ema20"] >
        row["ema200"]
        and
        row["close"] >
        row["ema20"]
        and
        row["st_direction"] == -1
        and
        row["adx"] >= ADX_MIN
        and
        row["plus_di"] >
        row["minus_di"]
    )

    # =========================
    # FRESH SHORT
    # =========================

    previous_short_core = (
        state30["bearish"]
        and
        prev["ema20"] <
        prev["ema200"]
        and
        prev["close"] <
        prev["ema20"]
        and
        prev["st_direction"] == 1
        and
        prev["adx"] >= ADX_MIN
        and
        prev["minus_di"] >
        prev["plus_di"]
    )

    current_short_core = (
        state30["bearish"]
        and
        row["ema20"] <
        row["ema200"]
        and
        row["close"] <
        row["ema20"]
        and
        row["st_direction"] == 1
        and
        row["adx"] >= ADX_MIN
        and
        row["minus_di"] >
        row["plus_di"]
    )

    fresh_long = (
        current_long_core
        and
        not previous_long_core
    )

    fresh_short = (
        current_short_core
        and
        not previous_short_core
    )

    # =========================
    # RESULT
    # =========================

    if (
        fresh_long
        and
        long_score >= MIN_SCORE
    ):

        return {
            "symbol": symbol,
            "signal": "STRONG LONG",
            "score": long_score,
            "price": row["close"],
            "rsi": row["rsi"],
            "adx": row["adx"]
        }

    if (
        fresh_short
        and
        short_score >= MIN_SCORE
    ):

        return {
            "symbol": symbol,
            "signal": "STRONG SHORT",
            "score": short_score,
            "price": row["close"],
            "rsi": row["rsi"],
            "adx": row["adx"]
        }

    return None


# =========================
# MAIN
# =========================

print("========================================")
print("TOOBIT STRONG SIGNAL SCANNER")
print("========================================")

# Telegram test
send_telegram(
    "✅ Toobit Signal Scanner - Telegram Test"
)

for symbol in SYMBOLS:

    try:

        result = check_signal(symbol)

        if result:

            print(
                result["symbol"],
                "|",
                result["signal"],
                "| Score:",
                result["score"],
                "| Price:",
                result["price"],
                "| RSI:",
                round(
                    result["rsi"],
                    2
                ),
                "| ADX:",
                round(
                    result["adx"],
                    2
                )
            )

            message = (
                "🚨 TOOBIT SIGNAL 🚨\n\n"
                f"Symbol: {result['symbol']}\n"
                f"Signal: {result['signal']}\n"
                f"Score: {result['score']}/9\n"
                f"Price: {result['price']}\n"
                f"RSI: {result['rsi']:.2f}\n"
                f"ADX: {result['adx']:.2f}\n\n"
                "Timeframe: 5m\n"
                "Confirmation: 15m + 30m"
            )

            send_telegram(message)

        else:

            print(
                symbol,
                "| No fresh strong signal"
            )

    except Exception as e:

        print(
            symbol,
            "| ERROR |",
            str(e)
        )