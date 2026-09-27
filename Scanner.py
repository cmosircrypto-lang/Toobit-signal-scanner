import os
import requests
import pandas as pd
import numpy as np

# =========================================================
# CONFIG
# =========================================================

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

# VERY STRICT
MIN_SCORE = 8

# ADX must show a reasonably strong trend
MIN_ADX = 25

# ATR settings
ATR_LENGTH = 14

# Risk / Reward
SL_ATR_MULTIPLIER = 1.5
TP1_R = 1.0
TP2_R = 2.0

# Avoid duplicate signal on same candle
last_signal = {}


TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")


# =========================================================
# TELEGRAM
# =========================================================

def send_telegram(message):

    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram settings missing")
        return

    url = (
        f"https://api.telegram.org/"
        f"bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    )

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
            print(
                "Telegram ERROR:",
                response.status_code
            )
            print(response.text)

    except Exception as e:

        print(
            "Telegram ERROR:",
            e
        )


# =========================================================
# GET KLINES
# =========================================================

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

    return pd.DataFrame(
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


# =========================================================
# EMA
# =========================================================

def ema(series, length):

    return series.ewm(
        span=length,
        adjust=False
    ).mean()


# =========================================================
# RSI
# =========================================================

def calculate_rsi(
    series,
    length=14
):

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

    rs = (
        avg_gain /
        avg_loss.replace(0, np.nan)
    )

    return 100 - (
        100 / (1 + rs)
    )


# =========================================================
# ATR
# =========================================================

def calculate_atr(
    df,
    length=14
):

    prev_close = df["close"].shift(1)

    tr = pd.concat(
        [
            df["high"] - df["low"],

            (
                df["high"] -
                prev_close
            ).abs(),

            (
                df["low"] -
                prev_close
            ).abs()
        ],
        axis=1
    ).max(axis=1)

    return tr.ewm(
        alpha=1 / length,
        adjust=False
    ).mean()


# =========================================================
# SUPERTREND
# =========================================================

def calculate_supertrend(
    df,
    period=10,
    multiplier=3.0
):

    atr = calculate_atr(
        df,
        period
    )

    hl2 = (
        df["high"] +
        df["low"]
    ) / 2

    upperband = (
        hl2 +
        multiplier * atr
    )

    lowerband = (
        hl2 -
        multiplier * atr
    )

    final_upper = upperband.copy()
    final_lower = lowerband.copy()

    direction = pd.Series(
        1,
        index=df.index,
        dtype=float
    )

    for i in range(1, len(df)):

        if (
            upperband.iloc[i] <
            final_upper.iloc[i - 1]
            or
            df["close"].iloc[i - 1] >
            final_upper.iloc[i - 1]
        ):

            final_upper.iloc[i] = (
                upperband.iloc[i]
            )

        else:

            final_upper.iloc[i] = (
                final_upper.iloc[i - 1]
            )

        if (
            lowerband.iloc[i] >
            final_lower.iloc[i - 1]
            or
            df["close"].iloc[i - 1] <
            final_lower.iloc[i - 1]
        ):

            final_lower.iloc[i] = (
                lowerband.iloc[i]
            )

        else:

            final_lower.iloc[i] = (
                final_lower.iloc[i - 1]
            )

        if direction.iloc[i - 1] == 1:

            if (
                df["close"].iloc[i] <=
                final_upper.iloc[i]
            ):

                direction.iloc[i] = -1

            else:

                direction.iloc[i] = 1

        else:

            if (
                df["close"].iloc[i] >=
                final_lower.iloc[i]
            ):

                direction.iloc[i] = 1

            else:

                direction.iloc[i] = -1

    return direction


# =========================================================
# STOCHASTIC
# =========================================================

def calculate_stochastic(
    df,
    length=14,
    smooth=3
):

    lowest_low = (
        df["low"]
        .rolling(length)
        .min()
    )

    highest_high = (
        df["high"]
        .rolling(length)
        .max()
    )

    denominator = (
        highest_high -
        lowest_low
    ).replace(
        0,
        np.nan
    )

    k = (
        100 *
        (
            df["close"] -
            lowest_low
        ) /
        denominator
    )

    d = k.rolling(
        smooth
    ).mean()

    return k, d


# =========================================================
# ADX
# =========================================================

def calculate_adx(
    df,
    length=14
):

    high = df["high"]
    low = df["low"]
    close = df["close"]

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where(
            (
                up_move >
                down_move
            ) &
            (
                up_move > 0
            ),
            up_move,
            0
        ),
        index=df.index
    )

    minus_dm = pd.Series(
        np.where(
            (
                down_move >
                up_move
            ) &
            (
                down_move > 0
            ),
            down_move,
            0
        ),
        index=df.index
    )

    prev_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,

            (
                high -
                prev_close
            ).abs(),

            (
                low -
                prev_close
            ).abs()
        ],
        axis=1
    ).max(axis=1)

    atr = tr.ewm(
        alpha=1 / length,
        adjust=False
    ).mean()

    plus_di = (
        100 *
        plus_dm.ewm(
            alpha=1 / length,
            adjust=False
        ).mean() /
        atr
    )

    minus_di = (
        100 *
        minus_dm.ewm(
            alpha=1 / length,
            adjust=False
        ).mean() /
        atr
    )

    dx = (
        100 *
        (
            plus_di -
            minus_di
        ).abs() /
        (
            plus_di +
            minus_di
        ).replace(
            0,
            np.nan
        )
    )

    adx = dx.ewm(
        alpha=1 / length,
        adjust=False
    ).mean()

    return (
        adx,
        plus_di,
        minus_di
    )


# =========================================================
# PREPARE DATA
# =========================================================

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
        ATR_LENGTH
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

    # Volume confirmation
    df["volume_ma"] = (
        df["volume"]
        .rolling(20)
        .mean()
    )

    return df


# =========================================================
# SIGNAL
# =========================================================

def check_signal(
    df5,
    df15,
    df30
):

    df5 = prepare_dataframe(df5)
    df15 = prepare_dataframe(df15)
    df30 = prepare_dataframe(df30)

    # Last CLOSED candles
    cur = df5.iloc[-2]

    tf15 = df15.iloc[-2]
    tf30 = df30.iloc[-2]

    # -----------------------------------------------------
    # 30 MINUTE TREND
    # -----------------------------------------------------

    long_30 = (
        tf30["close"] >
        tf30["ema20"]
        and
        tf30["ema20"] >
        tf30["ema200"]
        and
        tf30["st"] < 0
        and
        tf30["adx"] >= MIN_ADX
        and
        tf30["plus_di"] >
        tf30["minus_di"]
    )

    short_30 = (
        tf30["close"] <
        tf30["ema20"]
        and
        tf30["ema20"] <
        tf30["ema200"]
        and
        tf30["st"] > 0
        and
        tf30["adx"] >= MIN_ADX
        and
        tf30["minus_di"] >
        tf30["plus_di"]
    )

    # -----------------------------------------------------
    # 15 MINUTE CONFIRMATION
    # -----------------------------------------------------

    long_15 = (
        tf15["close"] >
        tf15["ema20"]
        and
        tf15["ema20"] >
        tf15["ema200"]
        and
        tf15["st"] < 0
        and
        tf15["adx"] >= MIN_ADX
        and
        tf15["plus_di"] >
        tf15["minus_di"]
    )

    short_15 = (
        tf15["close"] <
        tf15["ema20"]
        and
        tf15["ema20"] <
        tf15["ema200"]
        and
        tf15["st"] > 0
        and
        tf15["adx"] >= MIN_ADX
        and
        tf15["minus_di"] >
        tf15["plus_di"]
    )

    # -----------------------------------------------------
    # LONG SCORE
    # -----------------------------------------------------

    long_score = 0

    if long_30:
        long_score += 2

    if long_15:
        long_score += 2

    if (
        cur["ema20"] >
        cur["ema200"]
    ):
        long_score += 1

    if (
        cur["close"] >
        cur["ema20"]
    ):
        long_score += 1

    if cur["st"] < 0:
        long_score += 1

    if (
        cur["rsi"] >= 55
        and
        cur["rsi"] <= 70
    ):
        long_score += 1

    if (
        cur["k"] >
        cur["d"]
        and
        cur["k"] < 85
    ):
        long_score += 1

    # ADX + DI is mandatory confirmation,
    # not an extra point in the 9-point score.
    long_adx_ok = (
        cur["adx"] >= MIN_ADX
        and
        cur["plus_di"] >
        cur["minus_di"]
    )

    # Volume confirmation
    long_volume_ok = (
        cur["volume"] >=
        cur["volume_ma"] * 0.8
    )

    # -----------------------------------------------------
    # SHORT SCORE
    # -----------------------------------------------------

    short_score = 0

    if short_30:
        short_score += 2

    if short_15:
        short_score += 2

    if (
        cur["ema20"] <
        cur["ema200"]
    ):
        short_score += 1

    if (
        cur["close"] <
        cur["ema20"]
    ):
        short_score += 1

    if cur["st"] > 0:
        short_score += 1

    if (
        cur["rsi"] <= 45
        and
        cur["rsi"] >= 30
    ):
        short_score += 1

    if (
        cur["k"] <
        cur["d"]
        and
        cur["k"] > 15
    ):
        short_score += 1

    short_adx_ok = (
        cur["adx"] >= MIN_ADX
        and
        cur["minus_di"] >
        cur["plus_di"]
    )

    short_volume_ok = (
        cur["volume"] >=
        cur["volume_ma"] * 0.8
    )

    # -----------------------------------------------------
    # LONG SIGNAL
    # -----------------------------------------------------

    if (
        long_score >= MIN_SCORE
        and
        long_30
        and
        long_15
        and
        long_adx_ok
        and
        long_volume_ok
    ):

        entry = float(cur["close"])
        atr = float(cur["atr"])

        sl = (
            entry -
            atr * SL_ATR_MULTIPLIER
        )

        risk = entry - sl

        tp1 = (
            entry +
            risk * TP1_R
        )

        tp2 = (
            entry +
            risk * TP2_R
        )

        return {
            "direction": "LONG",
            "score": long_score,
            "price": entry,
            "entry": entry,
            "tp1": tp1,
            "tp2": tp2,
            "sl": sl,
            "rsi": float(cur["rsi"]),
            "adx": float(cur["adx"]),
            "candle_time": cur["time"]
        }

    # -----------------------------------------------------
    # SHORT SIGNAL
    # -----------------------------------------------------

    if (
        short_score >= MIN_SCORE
        and
        short_30
        and
        short_15
        and
        short_adx_ok
        and
        short_volume_ok
    ):

        entry = float(cur["close"])
        atr = float(cur["atr"])

        sl = (
            entry +
            atr * SL_ATR_MULTIPLIER
        )

        risk = sl - entry

        tp1 = (
            entry -
            risk * TP1_R
        )

        tp2 = (
            entry -
            risk * TP2_R
        )

        return {
            "direction": "SHORT",
            "score": short_score,
            "price": entry,
            "entry": entry,
            "tp1": tp1,
            "tp2": tp2,
            "sl": sl,
            "rsi": float(cur["rsi"]),
            "adx": float(cur["adx"]),
            "candle_time": cur["time"]
        }

    return None


# =========================================================
# FORMAT PRICE
# =========================================================

def format_price(price):

    if price >= 1000:
        return f"{price:.2f}"

    if price >= 1:
        return f"{price:.4f}"

    if price >= 0.1:
        return f"{price:.5f}"

    return f"{price:.7f}"


# =========================================================
# MAIN
# =========================================================

def main():

    print("========================================")
    print("TOOBIT VERY STRICT SIGNAL SCANNER")
    print("========================================")

    signals_found = []

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

                candle_time = signal[
                    "candle_time"
                ]

                # Prevent duplicate signal
                # for the same symbol/candle
                if last_signal.get(symbol) == candle_time:

                    print(
                        f"{symbol} | "
                        "Duplicate signal ignored"
                    )

                    continue

                last_signal[symbol] = candle_time

                signals_found.append(
                    (
                        symbol,
                        signal
                    )
                )

                print(
                    f"{symbol} | "
                    f"STRONG "
                    f"{signal['direction']} | "
                    f"Score "
                    f"{signal['score']}/9"
                )

            else:

                print(
                    f"{symbol} | "
                    "No high-quality signal"
                )

        except Exception as e:

            print(
                f"{symbol} | "
                f"ERROR | {e}"
            )

    # =====================================================
    # ONLY SEND THE STRONGEST SIGNAL
    # =====================================================

    if not signals_found:

        print(
            "No high-quality signal found."
        )

        return

    # Sort by score
    signals_found.sort(
        key=lambda x: x[1]["score"],
        reverse=True
    )

    symbol, signal = signals_found[0]

    message = (
        "🚨 TOOBIT HIGH QUALITY SIGNAL\n\n"

        f"Position: "
        f"STRONG {signal['direction']}\n"

        f"Symbol: "
        f"{symbol}\n"

        f"Score: "
        f"{signal['score']}/9\n"

        f"Current Price: "
        f"{format_price(signal['price'])}\n\n"

        f"Entry: "
        f"{format_price(signal['entry'])}\n"

        f"TP1: "
        f"{format_price(signal['tp1'])}\n"

        f"TP2: "
        f"{format_price(signal['tp2'])}\n"

        f"SL: "
        f"{format_price(signal['sl'])}\n\n"

        f"RSI: "
        f"{signal['rsi']:.2f}\n"

        f"ADX: "
        f"{signal['adx']:.2f}\n\n"

        "Timeframe: 5m\n"
        "Confirmation: 15m + 30m\n"
        "Risk Filter: STRICT"
    )

    print("\n========================================")
    print("SELECTED SIGNAL")
    print("========================================")
    print(message)

    send_telegram(message)


# =========================================================

if __name__ == "__main__":
    main()