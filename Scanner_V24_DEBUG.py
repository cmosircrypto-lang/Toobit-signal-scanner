import os
import json
import math
import requests
import pandas as pd
import numpy as np

# =========================================================
# TOOBIT V24 SCANNER + FULL DEBUG
# V24 signal logic is kept unchanged.
# Debug is saved for every scan / every symbol, even no signal.
# =========================================================

BASE_URL = "https://api.toobit.com/quote/v1/klines"
LIMIT = 300
STATE_FILE = "v24_state.json"
DEBUG_FILE = "v24_debug.json"
DEBUG_MAX_PER_SYMBOL = 100

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
# DEBUG STORAGE
# =========================================================

def load_debug():
    if not os.path.exists(DEBUG_FILE):
        return {"symbols": {}}

    try:
        with open(DEBUG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "symbols" not in data:
            data["symbols"] = {}
        return data
    except Exception:
        return {"symbols": {}}


def save_debug(data):
    tmp = DEBUG_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, DEBUG_FILE)


def clean_value(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        v = float(v)
    if isinstance(v, float):
        if not np.isfinite(v):
            return None
        return v
    if pd.isna(v):
        return None
    return v


def add_debug_record(debug_data, name, record):
    if name not in debug_data["symbols"]:
        debug_data["symbols"][name] = []

    debug_data["symbols"][name].append(record)

    if len(debug_data["symbols"][name]) > DEBUG_MAX_PER_SYMBOL:
        debug_data["symbols"][name] = debug_data["symbols"][name][-DEBUG_MAX_PER_SYMBOL:]


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
# DEBUG EVALUATION
# This calculates the same V24 conditions and returns
# both the normal signal and a complete diagnostic record.
# =========================================================

def evaluate_v24_debug(df5_raw, df15_raw, df30_raw):
    df5 = add_5m_indicators(df5_raw)
    df15 = add_tf_trend(df15_raw)
    df30 = add_tf_trend(df30_raw)

    if len(df5) < 220 or len(df15) < 220 or len(df30) < 220:
        return None, {
            "status": "NOT_ENOUGH_DATA",
            "length_5m": len(df5),
            "length_15m": len(df15),
            "length_30m": len(df30),
        }

    # Only CLOSED candles, exactly as the original V24.
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

    # 5m
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

    long_core = bull30 and bull_local and bull_adx
    short_core = bear30 and bear_local and bear_adx

    atr5 = float(cur["atr14"])

    if not np.isfinite(atr5) or atr5 <= 0:
        return None, {
            "status": "INVALID_ATR",
            "candle_time": int(cur["time"]),
        }

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

    long_candle_direction = cur["close"] > cur["open"]
    short_candle_direction = cur["close"] < cur["open"]

    long_signal = (
        long_score >= MIN_SCORE
        and long_core
        and not near_resistance
        and distance_ok
        and candle_ok
        and long_candle_direction
    )

    short_signal = (
        short_score >= MIN_SCORE
        and short_core
        and not near_support
        and distance_ok
        and candle_ok
        and short_candle_direction
    )

    # Detailed failure reasons
    long_failures = []
    if long_score < MIN_SCORE:
        long_failures.append(f"score<{MIN_SCORE}")
    if not long_core:
        long_failures.append("core")
    if near_resistance:
        long_failures.append("near_resistance")
    if not distance_ok:
        long_failures.append("distance")
    if not candle_ok:
        long_failures.append("candle_size")
    if not long_candle_direction:
        long_failures.append("bearish_candle")

    short_failures = []
    if short_score < MIN_SCORE:
        short_failures.append(f"score<{MIN_SCORE}")
    if not short_core:
        short_failures.append("core")
    if near_support:
        short_failures.append("near_support")
    if not distance_ok:
        short_failures.append("distance")
    if not candle_ok:
        short_failures.append("candle_size")
    if not short_candle_direction:
        short_failures.append("bullish_candle")

    direction = None
    score = None
    entry = sl = tp1 = tp2 = None

    if long_signal:
        direction = "LONG"
        score = int(long_score)
        entry = float(cur["close"])
        risk = atr5 * SL_ATR_MULTIPLIER
        sl = entry - risk
        tp1 = entry + risk * TP1_R
        tp2 = entry + risk * TP2_R
    elif short_signal:
        direction = "SHORT"
        score = int(short_score)
        entry = float(cur["close"])
        risk = atr5 * SL_ATR_MULTIPLIER
        sl = entry + risk
        tp1 = entry - risk * TP1_R
        tp2 = entry - risk * TP2_R

    debug = {
        "status": "SIGNAL" if direction else "NO_SIGNAL",
        "candle_time": int(cur["time"]),
        "price": clean_value(cur["close"]),
        "open": clean_value(cur["open"]),
        "high": clean_value(cur["high"]),
        "low": clean_value(cur["low"]),

        "30m": {
            "bull": bool(bull30),
            "bear": bool(bear30),
            "close": clean_value(t30["close"]),
            "ema20": clean_value(t30["ema20"]),
            "ema200": clean_value(t30["ema200"]),
            "st_dir": clean_value(t30["st_dir"]),
        },

        "15m": {
            "bull": bool(bull15),
            "bear": bool(bear15),
            "close": clean_value(t15["close"]),
            "ema20": clean_value(t15["ema20"]),
            "ema200": clean_value(t15["ema200"]),
            "st_dir": clean_value(t15["st_dir"]),
        },

        "5m": {
            "close": clean_value(cur["close"]),
            "ema20": clean_value(cur["ema20"]),
            "ema200": clean_value(cur["ema200"]),
            "st_dir": clean_value(cur["st_dir"]),
            "st_line": clean_value(cur["st_line"]),
            "rsi": clean_value(cur["rsi"]),
            "rsi_prev": clean_value(prev["rsi"]),
            "k": clean_value(cur["k"]),
            "d": clean_value(cur["d"]),
            "adx": clean_value(cur["adx"]),
            "di_plus": clean_value(cur["di_plus"]),
            "di_minus": clean_value(cur["di_minus"]),
            "atr14": clean_value(cur["atr14"]),
            "distance_from_ema_atr": clean_value(distance_from_ema),
            "candle_size": clean_value(candle_size),
        },

        "LONG": {
            "score": int(long_score),
            "core": bool(long_core),
            "local": bool(bull_local),
            "ema": bool(bull_ema),
            "supertrend": bool(bull_st),
            "rsi": bool(bull_rsi),
            "stochastic": bool(bull_stoch),
            "adx_di": bool(bull_adx),
            "distance": bool(distance_ok),
            "candle_size": bool(candle_ok),
            "sr": not bool(near_resistance),
            "candle_direction": bool(long_candle_direction),
            "final": bool(long_signal),
            "failed": long_failures,
        },

        "SHORT": {
            "score": int(short_score),
            "core": bool(short_core),
            "local": bool(bear_local),
            "ema": bool(bear_ema),
            "supertrend": bool(bear_st),
            "rsi": bool(bear_rsi),
            "stochastic": bool(bear_stoch),
            "adx_di": bool(bear_adx),
            "distance": bool(distance_ok),
            "candle_size": bool(candle_ok),
            "sr": not bool(near_support),
            "candle_direction": bool(short_candle_direction),
            "final": bool(short_signal),
            "failed": short_failures,
        },

        "support_resistance": {
            "support": clean_value(support),
            "resistance": clean_value(resistance),
            "near_support": bool(near_support),
            "near_resistance": bool(near_resistance),
        },

        "signal": {
            "direction": direction,
            "score": score,
            "entry": entry,
            "sl": sl,
            "tp1": tp1,
            "tp2": tp2,
        },
    }

    # Return the same signal structure as the original V24.
    if not direction:
        return None, debug

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
    }, debug


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


def print_debug_summary(name, d):
    if not d:
        print(f"{name}: DEBUG unavailable")
        return

    if d.get("status") == "NOT_ENOUGH_DATA":
        print(
            f"{name}: DEBUG NOT_ENOUGH_DATA "
            f"5m={d.get('length_5m')} "
            f"15m={d.get('length_15m')} "
            f"30m={d.get('length_30m')}"
        )
        return

    if d.get("status") == "INVALID_ATR":
        print(f"{name}: DEBUG INVALID_ATR")
        return

    L = d["LONG"]
    S = d["SHORT"]
    x = d["5m"]

    print(
        f"{name}: DEBUG | "
        f"close={fmt(x['close'])} "
        f"RSI={x['rsi']:.2f} "
        f"ADX={x['adx']:.2f} "
        f"DI+={x['di_plus']:.2f} "
        f"DI-={x['di_minus']:.2f} "
        f"K={x['k']:.2f} "
        f"D={x['d']:.2f}"
    )

    print(
        f"  LONG  score={L['score']}/9 "
        f"core={'PASS' if L['core'] else 'FAIL'} "
        f"30m={'PASS' if d['30m']['bull'] else 'FAIL'} "
        f"15m={'PASS' if d['15m']['bull'] else 'FAIL'} "
        f"EMA={'PASS' if L['ema'] else 'FAIL'} "
        f"ST={'PASS' if L['supertrend'] else 'FAIL'} "
        f"RSI={'PASS' if L['rsi'] else 'FAIL'} "
        f"STOCH={'PASS' if L['stochastic'] else 'FAIL'} "
        f"ADX={'PASS' if L['adx_di'] else 'FAIL'} "
        f"DIST={'PASS' if L['distance'] else 'FAIL'} "
        f"CANDLE={'PASS' if L['candle_size'] and L['candle_direction'] else 'FAIL'} "
        f"SR={'PASS' if L['sr'] else 'FAIL'} "
        f"FINAL={'SIGNAL' if L['final'] else 'NO'}"
    )

    print(
        f"  SHORT score={S['score']}/9 "
        f"core={'PASS' if S['core'] else 'FAIL'} "
        f"30m={'PASS' if d['30m']['bear'] else 'FAIL'} "
        f"15m={'PASS' if d['15m']['bear'] else 'FAIL'} "
        f"EMA={'PASS' if S['ema'] else 'FAIL'} "
        f"ST={'PASS' if S['supertrend'] else 'FAIL'} "
        f"RSI={'PASS' if S['rsi'] else 'FAIL'} "
        f"STOCH={'PASS' if S['stochastic'] else 'FAIL'} "
        f"ADX={'PASS' if S['adx_di'] else 'FAIL'} "
        f"DIST={'PASS' if S['distance'] else 'FAIL'} "
        f"CANDLE={'PASS' if S['candle_size'] and S['candle_direction'] else 'FAIL'} "
        f"SR={'PASS' if S['sr'] else 'FAIL'} "
        f"FINAL={'SIGNAL' if S['final'] else 'NO'}"
    )

    if not L["final"]:
        print("  LONG FAILED:", ", ".join(L["failed"]) if L["failed"] else "none")

    if not S["final"]:
        print("  SHORT FAILED:", ", ".join(S["failed"]) if S["failed"] else "none")


# =========================================================
# MAIN
# =========================================================

def main():
    print("=" * 70)
    print("TOOBIT V24 SIGNAL SCANNER + FULL DEBUG")
    print("5m + 15m + 30m | MIN SCORE 5/9 | ADX >= 25")
    print("Debug: every scan / every symbol / last 100 records per symbol")
    print("=" * 70)

    state = load_state()
    debug_data = load_debug()
    new_messages = []

    for name, symbol in SYMBOLS.items():
        try:
            df5 = get_klines(symbol, "5m")
            df15 = get_klines(symbol, "15m")
            df30 = get_klines(symbol, "30m")

            if len(df5) < 220:
                print(f"{name}: not enough 5m data")
                add_debug_record(debug_data, name, {
                    "status": "NOT_ENOUGH_DATA",
                    "scan_time_utc": pd.Timestamp.utcnow().isoformat(),
                    "length_5m": len(df5),
                    "length_15m": len(df15),
                    "length_30m": len(df30),
                })
                continue

            sstate = symbol_state(state, name)

            # Current closed 5m candle
            cur = df5.iloc[-2]
            current_bar = int(cur["time"])

            # 1) Manage active trade.
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

            # 2) Evaluate V24 + full debug EVERY RUN.
            signal, debug_record = evaluate_v24_debug(
                df5, df15, df30
            )

            if debug_record:
                debug_record["scan_time_utc"] = pd.Timestamp.utcnow().isoformat()
                debug_record["symbol"] = name
                debug_record["toobit_symbol"] = symbol
                debug_record["active_before_new_signal"] = bool(sstate["active"])
                add_debug_record(debug_data, name, debug_record)

                print_debug_summary(name, debug_record)

            # 3) New signal handling remains the original V24 logic.
            if not sstate["active"]:
                if signal:
                    signal_bar = int(signal["candle_time"])
                    last_bar = sstate.get("last_signal_bar")

                    cooldown_ok = True
                    if last_bar is not None:
                        cooldown_ok = (
                            signal_bar - int(last_bar)
                        ) >= COOLDOWN_BARS * 5 * 60 * 1000

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

                        # Mark the debug record with final state handling.
                        if debug_record:
                            debug_record["state_result"] = "NEW_SIGNAL_ACCEPTED"
                            debug_record["cooldown_ok"] = True
                            debug_record["duplicate_ok"] = True
                    else:
                        print(f"{name}: signal blocked by cooldown/duplicate")
                        if debug_record:
                            debug_record["state_result"] = "SIGNAL_BLOCKED"
                            debug_record["cooldown_ok"] = bool(cooldown_ok)
                            debug_record["duplicate_ok"] = bool(duplicate_ok)
                else:
                    print(f"{name}: no V24 signal")
                    if debug_record:
                        debug_record["state_result"] = "NO_V24_SIGNAL"
            else:
                print(f"{name}: active {sstate['direction']}")
                if debug_record:
                    debug_record["state_result"] = "ACTIVE_TRADE"

        except Exception as e:
            print(f"{name}: ERROR: {e}")
            add_debug_record(debug_data, name, {
                "status": "ERROR",
                "scan_time_utc": pd.Timestamp.utcnow().isoformat(),
                "error": str(e),
            })

    save_state(state)
    save_debug(debug_data)

    # Send every new signal/exit found in this scan.
    for msg in new_messages:
        send_telegram(msg)

    print("=" * 70)
    print(f"Events this run: {len(new_messages)}")
    print(f"Debug saved to: {DEBUG_FILE}")
    print("=" * 70)


if __name__ == "__main__":
    main()
