import time
import requests
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator
from ta.volatility import AverageTrueRange

# =========================
# CONFIG
# =========================
SYMBOLS = ["AAPL","TSLA","MSFT","NVDA","AMZN","META","GC=F"]

TELEGRAM_TOKEN = "YOUR_BOT_TOKEN"
TELEGRAM_CHAT_ID = "YOUR_CHAT_ID"

COOLDOWN = 300  # seconds

last_alert_time = {}

# =========================
# TELEGRAM ALERT
# =========================
def send_telegram(msg):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {"chat_id": TELEGRAM_CHAT_ID, "text": msg}
    try:
        requests.post(url, data=data)
    except:
        pass

# =========================
# FETCH DATA
# =========================
def fetch(symbol):
    df = yf.download(symbol, period="5d", interval="5m", auto_adjust=True)

    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open","High","Low","Close","Volume"]]
    return df.dropna()

# =========================
# FEATURES
# =========================
def features(df):
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    df["ema20"] = close.ewm(span=20).mean()
    df["ema50"] = close.ewm(span=50).mean()
    df["rsi"] = RSIIndicator(close).rsi()
    df["atr"] = AverageTrueRange(high, low, close).average_true_range()

    df["trend_strength"] = abs(df["ema20"] - df["ema50"])
    df["volume_spike"] = volume / volume.rolling(20).mean()

    df["returns"] = close.pct_change()

    return df.dropna()

# =========================
# MODEL
# =========================
def train(df):
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df = df.dropna()

    feats = ["ema20","ema50","rsi","atr","trend_strength","volume_spike","returns"]

    if len(df) < 100:
        return None, feats

    X = df[feats]
    y = df["target"]

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)

    return model, feats

# =========================
# HIGHER TIMEFRAME
# =========================
def higher_trend(symbol):
    df = yf.download(symbol, period="1mo", interval="1h")

    if df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    close = df["Close"]

    ema20 = close.ewm(span=20).mean()
    ema50 = close.ewm(span=50).mean()

    return "UP" if ema20.iloc[-1] > ema50.iloc[-1] else "DOWN"

# =========================
# SIGNAL LOGIC
# =========================
def signal(row, prob, htf):
    if row["trend_strength"] < 0.05:
        return "HOLD"

    if row["volume_spike"] < 1.2:
        return "HOLD"

    if prob > 0.65 and row["ema20"] > row["ema50"] and htf == "UP":
        return "BUY"

    if prob < 0.35 and row["ema20"] < row["ema50"] and htf == "DOWN":
        return "SELL"

    return "HOLD"

# =========================
# MAIN LOOP
# =========================
def run():
    global last_alert_time

    while True:
        print("🔎 Scanning market...")

        for sym in SYMBOLS:
            try:
                df = fetch(sym)
                if df.empty:
                    continue

                df = features(df)
                if df.empty:
                    continue

                model, feats = train(df)
                if model is None:
                    continue

                latest = df.iloc[-1]
                prob = model.predict_proba([latest[feats]])[0][1]

                htf = higher_trend(sym)
                sig = signal(latest, prob, htf)

                now = time.time()

                # 🚨 ALERT LOGIC
                if sig != "HOLD":
                    if sym not in last_alert_time or now - last_alert_time[sym] > COOLDOWN:
                        msg = f"""
📊 {sym}
Signal: {sig}
Price: {latest['Close']:.2f}
Confidence: {prob:.2f}
Trend: {htf}
Volume Spike: {latest['volume_spike']:.2f}
"""
                        print(msg)
                        send_telegram(msg)
                        last_alert_time[sym] = now

            except Exception as e:
                print(f"Error {sym}: {e}")

        time.sleep(300)

# =========================
# START
# =========================
if __name__ == "__main__":
    run()