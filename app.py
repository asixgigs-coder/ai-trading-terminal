import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator
from ta.volatility import AverageTrueRange
import plotly.graph_objects as go

# SETTINGS
SYMBOLS = ["AAPL", "TSLA", "MSFT", "NVDA"]
INTERVAL = "5m"
PERIOD = "5d"

st.set_page_config(layout="wide")
st.title("🚀 AI Trading Terminal (Stable Version)")

# STATE
if "balance" not in st.session_state:
    st.session_state.balance = 10000
    st.session_state.positions = {}
    st.session_state.trades = []

# SIDEBAR
symbol = st.sidebar.selectbox("Select Stock", SYMBOLS)
auto_trade = st.sidebar.toggle("Auto Trade", False)

# FETCH DATA (SAFE)
@st.cache_data(ttl=60)
def fetch(symbol):
    df = yf.download(symbol, period=PERIOD, interval=INTERVAL, auto_adjust=True)

    if df is None or df.empty:
        return pd.DataFrame()

    # FIX MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    required = ["Open", "High", "Low", "Close", "Volume"]
    df = df[[col for col in required if col in df.columns]]

    df.dropna(inplace=True)
    return df

# FEATURES (SAFE)
def features(df):
    if df.empty or len(df) < 50:
        return pd.DataFrame()

    close = pd.to_numeric(df["Close"], errors="coerce")
    high = pd.to_numeric(df["High"], errors="coerce")
    low = pd.to_numeric(df["Low"], errors="coerce")

    df["ema20"] = EMAIndicator(close, window=20).ema_indicator()
    df["ema50"] = EMAIndicator(close, window=50).ema_indicator()
    df["rsi"] = RSIIndicator(close, window=14).rsi()
    df["atr"] = AverageTrueRange(high, low, close, window=14).average_true_range()

    df["returns"] = close.pct_change()

    df.dropna(inplace=True)
    return df

# TARGET
def target(df):
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df.dropna(inplace=True)
    return df

# MODEL (STABLE)
def train(df):
    feats = ["ema20","ema50","rsi","atr","returns"]

    if len(df) < 100:
        return None, feats

    X = df[feats]
    y = df["target"]

    split = int(len(df) * 0.8)

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X[:split], y[:split])

    return model, feats

# SIGNAL
def signal(row, prob):
    if prob > 0.6 and row["ema20"] > row["ema50"]:
        return "BUY"
    elif prob < 0.4 and row["ema20"] < row["ema50"]:
        return "SELL"
    return "HOLD"

# TRADE (paper)
def trade(sig, price, atr, symbol):
    balance = st.session_state.balance
    positions = st.session_state.positions

    risk = balance * 0.02

    if sig == "BUY" and symbol not in positions:
        qty = risk / (1.5 * atr)
        positions[symbol] = {"qty": qty, "entry": price}
        st.session_state.trades.append(f"BUY {symbol} @ {price:.2f}")

    elif sig == "SELL" and symbol in positions:
        entry = positions[symbol]["entry"]
        qty = positions[symbol]["qty"]
        profit = (price - entry) * qty

        st.session_state.balance += profit
        del positions[symbol]

        st.session_state.trades.append(f"SELL {symbol} @ {price:.2f} | PnL {profit:.2f}")

# ===== RUN =====

df = fetch(symbol)

if df.empty:
    st.error("No data loaded")
    st.stop()

df = features(df)

if df.empty:
    st.warning("Not enough data yet...")
    st.stop()

df = target(df)

model, feats = train(df)

if model is None:
    st.warning("Model not ready yet...")
    st.stop()

latest = df.iloc[-1]

prob = model.predict_proba([latest[feats]])[0][1]
sig = signal(latest, prob)

# UI
col1, col2, col3, col4 = st.columns(4)

col1.metric("Price", f"{latest['Close']:.2f}")
col2.metric("AI Confidence", f"{prob:.2f}")
col3.metric("Signal", sig)
col4.metric("Balance", f"{st.session_state.balance:.2f}")

# CHART
fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=df.index,
    open=df["Open"],
    high=df["High"],
    low=df["Low"],
    close=df["Close"]
))

fig.add_trace(go.Scatter(x=df.index, y=df["ema20"], name="EMA20"))
fig.add_trace(go.Scatter(x=df.index, y=df["ema50"], name="EMA50"))

st.plotly_chart(fig, use_container_width=True)

# AUTO TRADE
if auto_trade:
    trade(sig, latest["Close"], latest["atr"], symbol)

# POSITIONS
st.subheader("📂 Open Positions")
st.write(st.session_state.positions)

# LOG
st.subheader("📜 Trade Log")
st.write(st.session_state.trades)