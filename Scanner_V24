import os
import json
import math
import requests
import pandas as pd
import numpy as np

# =========================================================
# TOOBIT V24 SCANNER
# Pine V24 logic ported to Python
# =========================================================

BASE_URL = "https://api.toobit.com/quote/v1/klines"
LIMIT = 300
STATE_FILE = "v24_state.json"

SYMBOLS = {
    "BTC": "BTC-SWAP-USDT",
    "SOL": "SOL-SWAP-USDT",
    "ETH": "ETH-SWAP-USDT",
    "XRP": "XRP-SWAP-USDT",
    "NEAR": "NEAR-SWAP-USDT",
    "SUI": "SUI-SWAP-USDT",
    "ADA": "ADA-SWAP-USDT",
    "LINK": "LINK-SWAP-USDT",
    "AVAX": "AVAX-SWAP-USDT",
    "BNB": "BNB-SWAP-USDT",
    "UNI": "UNI-SWAP-USDT",
    "ARB": "ARB-SWAP-USDT",
}

# V24 inputs
COOLDOWN_BARS = 5
RSI_LENGTH = 14
RSI_LONG_LEVEL = 55
RSI_SHORT_LEVEL = 45
ATR_PERIOD = 10
ST_FACTOR = 3.0
PIVOT_LENGTH = 5
SR_DISTANCE_PERCENT = 0.3
TRADE_ATR_PERIOD = 14
SL_ATR_MULTIPLIER = 1.5
TP1_R = 1.0
TP2_R = 2.0
MAX_DISTANCE_ATR = 1.8
MAX_CANDLE_ATR = 2.2
MIN_ADX = 25.0
MIN_SCORE = 5

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")


# =========================================================
# STATE
# =========================================================

def load_state():
    if not os.path.exists(STATE_FILE):
        return {"symbols": {}}

    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            state = json.load(f)
        if "symbols" not in state:
            state["symbols"] = {}
        return state
    except Exception:
        return {"symbols": {}}


def save_state(state):
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    os.replace(tmp, STATE_FILE)


def symbol_state(state, name):
    if name not in state["symbols"]:
        state["symbols"][name] = {
            "active": False,
            "direction": None,
            "entry": None,
            "sl": None,
            "tp1": None,
            "tp2": None,
            "signal_bar": None,
            "last_signal_bar": None,
        }
    return state["symbols"][name]


# =========================================================
# TELEGRAM
# =========================================================

def send_telegram(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram secrets are missing.")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

    try:
        r = requests.post(
            url,
            data={"chat_id": TELEGRAM_CHAT_ID, "text": message},
            timeout=20,
        )
        if r.ok:
            print("Telegram: message sent")
            return True
        print("Telegram ERROR:", r.status_code, r.text[:500])
    except Exception as e:
        print("Telegram ERROR:", e)
    return False


# =========================================================
# DATA
# =========================================================

def get_klines(symbol, interval):
    r = requests.get(
        BASE_URL,
        params={"symbol": symbol, "interval": interval, "limit": LIMIT},
        timeout=20,
    )
    r.raise_for_status()
    data = r.json()

    if isinstance(data, dict):
        data = data.get("data", data.get("result", data))

    rows = []
    for x in data:
        if len(x) < 6:
            continue
        rows.append([
            float(x[0]),
            float(x[1]),
            float(x[2]),
            float(x[3]),
            float(x[4]),
            float(x[5]),
        ])

    if not rows:
        raise ValueError(f"No kline data for {symbol} {interval}")

    return pd.DataFrame(
        rows,
        columns=["time", "open", "high", "low", "close", "volume"],
    ).sort_values("time").reset_index(drop=True)


# =========================================================
# INDICATORS
# =========================================================

def ema(s, length):
    return s.ewm(span=length, adjust=False).mean()


def rsi(s, length=14):
    delta = s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(df, length=14):
    pc = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - pc).abs(),
            (df["low"] - pc).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False).mean()


def supertrend(df, period=10, factor=3.0):
    a = atr(df, period)
    hl2 = (df["high"] + df["low"]) / 2

    upper_basic = hl2 + factor * a
    lower_basic = hl2 - factor * a

    upper = upper_basic.copy()
    lower = lower_basic.copy()

    direction = pd.Series(1.0, index=df.index)
    line = pd.Series(np.nan, index=df.index)

    for i in range(1, len(df)):
        if (
            upper_basic.iloc[i] < upper.iloc[i - 1]
            or df["close"].iloc[i - 1] > upper.iloc[i - 1]
        ):
            upper.iloc[i] = upper_basic.iloc[i]
        else:
            upper.iloc[i] = upper.iloc[i - 1]

        if (
            lower_basic.iloc[i] > lower.iloc[i - 1]
            or df["close"].iloc[i - 1] < lower.iloc[i - 1]
        ):
            lower.iloc[i] = lower_basic.iloc[i]
        else:
            lower.iloc[i] = lower.iloc[i - 1]

        if direction.iloc[i - 1] > 0:
            if df["close"].iloc[i] > upper.iloc[i]:
                direction.iloc[i] = -1.0
            else:
                direction.iloc[i] = 1.0
        else:
            if df["close"].iloc[i] < lower.iloc[i]:
                direction.iloc[i] = 1.0
            else:
                direction.iloc[i] = -1.0

        line.iloc[i] = lower.iloc[i] if direction.iloc[i] < 0 else upper.iloc[i]

    return direction, line


def stochastic(df, length=14, smooth=3):
    lo = df["low"].rolling(length).min()
    hi = df["high"].rolling(length).max()
    den = (hi - lo).replace(0, np.nan)
    k_raw = 100 * (df["close"] - lo) / den
    k = k_raw.rolling(3).mean()
    d = k.rolling(smooth).mean()
    return k, d


def dmi_adx(df, length=14):
    high = df["high"]
    low = df["low"]
    close = df["close"]

    up = high.diff()
    down = -low.diff()

    plus_dm = pd.Series(
        np.where((up > down) & (up > 0), up, 0.0), index=df.index
    )
    minus_dm = pd.Series(
        np.where((down > up) & (down > 0), down, 0.0), index=df.index
    )

    pc = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - pc).abs(),
            (low - pc).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr_w = tr.ewm(alpha=1 / length, adjust=False).mean()

    plus_di = 100 * plus_dm.ewm(
        alpha=1 / length, adjust=False
    ).mean() / atr_w

    minus_di = 100 * minus_dm.ewm(
        alpha=1 / length, adjust=False
    ).mean() / atr_w

    den = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / den
    adx = dx.ewm(alpha=1 / length, adjust=False).mean()

    return adx, plus_di, minus_di


def add_5m_indicators(df):
    out = df.copy()
    out["ema20"] = ema(out["close"], 20)
    out["ema200"] = ema(out["close"], 200)
    out["rsi"] = rsi(out["close"], RSI_LENGTH)
    out["atr14"] = atr(out, TRADE_ATR_PERIOD)
    out["atr10"] = atr(out, ATR_PERIOD)
    out["st_dir"], out["st_line"] = supertrend(
        out, ATR_PERIOD, ST_FACTOR
    )
    out["k"], out["d"] = stochastic(out, 14, 3)
    out["adx"], out["di_plus"], out["di_minus"] = dmi_adx(out, 14)
    return out


def add_tf_trend(df):
    out = df.copy()
    out["ema20"] = ema(out["close"], 20)
    out["ema200"] = ema(out["close"], 200)
    out["st_dir"], out["st_line"] = supertrend(
        out, ATR_PERIOD, ST_FACTOR
    )
    return out


# =========================================================
# SUPPORT / RESISTANCE
# =========================================================

def latest_pivot_high(df, length=5):
    if len(df) < (length * 2 + 1):
        return np.nan

    highs = df["high"].values
    pivots = []
    for i in range(length, len(df) - length):
        window = highs[i - length:i + length + 1]
        if highs[i] == np.max(window):
            pivots.append(highs[i])

    return pivots[-1] if pivots else np.nan


def latest_pivot_low(df, length=5):
    if len(df) < (length * 2 + 1):
        return np.nan

    lows = df["low"].values
    pivots = []
    for i in range(length, len(df) - length):
        window = lows[i - length:i + length + 1]
        if lows[i] == np.min(window):
            pivots.append(lows[i])

    return pivots[-1] if pivots else np.nan


# =========================================================
# V24 SIGNAL
# =========================================================

def evaluate_v24(df5_raw, df15_raw, df30_raw):
    df5 = add_5m_indicators(df5_raw)
    df15 = add_tf_trend(df15_raw)
    df30 = add_tf_trend(df30_raw)

    # Only CLOSED candles, matching Pine barstate.isconfirmed usage.
    if len(df5) < 220 or len(df15) < 220 or len(df30) < 220:
        return None

    cur = df5.iloc[-2]
    prev = df5.iloc[-3]
    t15 = df15.iloc[-2]
    t30 = df30.iloc[-2]

    # 30m trend
    bull30 = (
        t30["ema20"] > t30["ema200"]
        and t30["close"] > t30["ema20"]
        and t30["st_dir"] < 0
    )
    bear30 = (
        t30["ema20"] < t30["ema200"]
        and t30["close"] < t30["ema20"]
        and t30["st_dir"] > 0
    )

    # 15m trend
    bull15 = (
        t15["ema20"] > t15["ema200"]
        and t15["close"] > t15["ema20"]
        and t15["st_dir"] < 0
    )
    bear15 = (
        t15["ema20"] < t15["ema200"]
        and t15["close"] < t15["ema200"]
        and t15["st_dir"] > 0
    )

    # 5m conditions
    bull_local = cur["close"] > cur["ema20"]
    bear_local = cur["close"] < cur["ema20"]

    bull_ema = cur["ema20"] > cur["ema200"]
    bear_ema = cur["ema20"] < cur["ema200"]

    bull_st = cur["st_dir"] < 0
    bear_st = cur["st_dir"] > 0

    bull_rsi = (
        cur["rsi"] > RSI_LONG_LEVEL
        and cur["rsi"] > prev["rsi"]
    )
    bear_rsi = (
        cur["rsi"] < RSI_SHORT_LEVEL
        and cur["rsi"] < prev["rsi"]
    )

    bull_stoch = cur["k"] > cur["d"] and cur["k"] > 50
    bear_stoch = cur["k"] < cur["d"] and cur["k"] < 50

    bull_adx = (
        cur["adx"] >= MIN_ADX
        and cur["di_plus"] > cur["di_minus"]
    )
    bear_adx = (
        cur["adx"] >= MIN_ADX
        and cur["di_minus"] > cur["di_plus"]
    )

    # Score: 30m = 2; every other confirmation = 1. Max 9.
    long_score = (
        2 * int(bull30)
        + int(bull15)
        + int(bull_local)
        + int(bull_ema)
        + int(bull_st)
        + int(bull_rsi)
        + int(bull_stoch)
        + int(bull_adx)
    )

    short_score = (
        2 * int(bear30)
        + int(bear15)
        + int(bear_local)
        + int(bear_ema)
        + int(bear_st)
        + int(bear_rsi)
        + int(bear_stoch)
        + int(bear_adx)
    )

    # Core filters
    long_core = bull30 and bull_local and bull_adx
    short_core = bear30 and bear_local and bear_adx

    atr5 = float(cur["atr14"])
    if not np.isfinite(atr5) or atr5 <= 0:
        return None

    distance_from_ema = abs(cur["close"] - cur["ema20"]) / atr5
    distance_ok = distance_from_ema <= MAX_DISTANCE_ATR

    candle_size = abs(cur["close"] - cur["open"])
    candle_ok = candle_size <= atr5 * MAX_CANDLE_ATR

    resistance = latest_pivot_high(df5.iloc[:-2], PIVOT_LENGTH)
    support = latest_pivot_low(df5.iloc[:-2], PIVOT_LENGTH)

    near_resistance = (
        np.isfinite(resistance)
        and cur["close"] >= resistance * (1 - SR_DISTANCE_PERCENT / 100)
    )

    near_support = (
        np.isfinite(support)
        and cur["close"] <= support * (1 + SR_DISTANCE_PERCENT / 100)
    )

    long_signal = (
        long_score >= MIN_SCORE
        and long_core
        and not near_resistance
        and distance_ok
        and candle_ok
        and cur["close"] > cur["open"]
    )

    short_signal = (
        short_score >= MIN_SCORE
        and short_core
        and not near_support
        and distance_ok
        and candle_ok
        and cur["close"] < cur["open"]
    )

    if not long_signal and not short_signal:
        return None

    direction = "LONG" if long_signal else "SHORT"
    score = long_score if long_signal else short_score
    entry = float(cur["close"])
    risk = atr5 * SL_ATR_MULTIPLIER

    if direction == "LONG":
        sl = entry - risk
        tp1 = entry + risk * TP1_R
        tp2 = entry + risk * TP2_R
    else:
        sl = entry + risk
        tp1 = entry - risk * TP1_R
        tp2 = entry - risk * TP2_R

    return {
        "direction": direction,
        "score": int(score),
        "entry": entry,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "rsi": float(cur["rsi"]),
        "adx": float(cur["adx"]),
        "candle_time": int(cur["time"]),
    }


# =========================================================
# EXIT
# =========================================================

def check_exit(position, candle):
    if not position.get("active"):
        return None

    direction = position["direction"]
    sl = float(position["sl"])
    tp2 = float(position["tp2"])

    high = float(candle["high"])
    low = float(candle["low"])

    # If both SL and TP2 are touched in one candle, prioritize SL
    # because intrabar order is unknown from OHLC data.
    if direction == "LONG":
        if low <= sl:
            return "SL"
        if high >= tp2:
            return "TP2"
    else:
        if high >= sl:
            return "SL"
        if low <= tp2:
            return "TP2"

    return None


# =========================================================
# FORMAT
# =========================================================

def fmt(x):
    x = float(x)
    if x >= 1000:
        return f"{x:.2f}"
    if x >= 1:
        return f"{x:.4f}"
    if x >= 0.1:
        return f"{x:.5f}"
    return f"{x:.7f}"


def signal_message(name, s):
    arrow = "🟢 LONG" if s["direction"] == "LONG" else "🔴 SHORT"
    return (
        f"{arrow} — {name}\n\n"
        f"Score: {s['score']}/9\n"
        f"Entry: {fmt(s['entry'])}\n"
        f"TP1: {fmt(s['tp1'])}\n"
        f"TP2: {fmt(s['tp2'])}\n"
        f"SL: {fmt(s['sl'])}\n\n"
        f"RSI: {s['rsi']:.2f}\n"
        f"ADX: {s['adx']:.2f}\n"
        f"TF: 5m + 15m + 30m"
    )


def exit_message(name, position, reason):
    icon = "🟢" if reason == "TP2" else "🔴"
    return (
        f"{icon} EXIT — {name}\n\n"
        f"{position['direction']} hit {reason}\n"
        f"Entry: {fmt(position['entry'])}\n"
        f"Exit level: {fmt(position['tp2'] if reason == 'TP2' else position['sl'])}"
    )


# =========================================================
# MAIN
# =========================================================

def main():
    print("=" * 60)
    print("TOOBIT V24 SIGNAL SCANNER")
    print("5m + 15m + 30m | MIN SCORE 5/9 | ADX >= 25")
    print("=" * 60)

    state = load_state()
    new_messages = []

    for name, symbol in SYMBOLS.items():
        try:
            df5 = get_klines(symbol, "5m")
            df15 = get_klines(symbol, "15m")
            df30 = get_klines(symbol, "30m")

            if len(df5) < 220:
                print(f"{name}: not enough 5m data")
                continue

            sstate = symbol_state(state, name)

            # Current closed 5m candle
            cur = df5.iloc[-2]
            current_bar = int(cur["time"])

            # 1) Manage an already-active V24 trade.
            exit_reason = check_exit(sstate, cur)
            if exit_reason:
                msg = exit_message(name, sstate, exit_reason)
                print(msg)
                new_messages.append(msg)

                sstate.update({
                    "active": False,
                    "direction": None,
                    "entry": None,
                    "sl": None,
                    "tp1": None,
                    "tp2": None,
                    "signal_bar": None,
                })

            # 2) If no active trade, check for a new V24 signal.
            if not sstate["active"]:
                signal = evaluate_v24(df5, df15, df30)

                if signal:
                    signal_bar = int(signal["candle_time"])
                    last_bar = sstate.get("last_signal_bar")

                    # Pine cooldown: at least 5 completed 5m bars
                    cooldown_ok = True
                    if last_bar is not None:
                        cooldown_ok = (
                            signal_bar - int(last_bar)
                        ) >= COOLDOWN_BARS * 5 * 60 * 1000

                    # Never repeat the same signal candle.
                    duplicate_ok = signal_bar != last_bar

                    if cooldown_ok and duplicate_ok:
                        sstate.update({
                            "active": True,
                            "direction": signal["direction"],
                            "entry": signal["entry"],
                            "sl": signal["sl"],
                            "tp1": signal["tp1"],
                            "tp2": signal["tp2"],
                            "signal_bar": signal_bar,
                            "last_signal_bar": signal_bar,
                        })

                        msg = signal_message(name, signal)
                        print(msg)
                        new_messages.append(msg)
                    else:
                        print(f"{name}: signal blocked by cooldown/duplicate")
                else:
                    print(f"{name}: no V24 signal")
            else:
                print(f"{name}: active {sstate['direction']}")

        except Exception as e:
            print(f"{name}: ERROR: {e}")

    save_state(state)

    # Send every new signal/exit found in this scan.
    # This intentionally does NOT select only one strongest symbol.
    for msg in new_messages:
        send_telegram(msg)

    print("=" * 60)
    print(f"Events this run: {len(new_messages)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
