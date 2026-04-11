import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator
from ta.volatility import AverageTrueRange
import plotly.graph_objects as go

USE_ALPACA = False

SYMBOLS = ["AAPL", "TSLA", "MSFT", "NVDA"]
INTERVAL = "5m"
PERIOD = "5d"

if "balance" not in st.session_state:
    st.session_state.balance = 10000
    st.session_state.positions = {}
    st.session_state.trades = []

st.set_page_config(layout="wide")
st.title("AI Trading Terminal")

symbol = st.sidebar.selectbox("Select Stock", SYMBOLS)
auto_trade = st.sidebar.toggle("Auto Trade", False)

@st.cache_data(ttl=60)
def fetch(symbol):
    df = yf.download(symbol, period=PERIOD, interval=INTERVAL, auto_adjust=True)

    # 🔥 FIX: flatten columns if MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open", "High", "Low", "Close", "Volume"]]

    df.dropna(inplace=True)
    return df

def features(df):
    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)

    df["ema20"] = EMAIndicator(close, window=20).ema_indicator()
    df["ema50"] = EMAIndicator(close, window=50).ema_indicator()
    df["rsi"] = RSIIndicator(close, window=14).rsi()
    df["atr"] = AverageTrueRange(high, low, close, window=14).average_true_range()

    df["returns"] = close.pct_change()

    df.dropna(inplace=True)
    return df

def target(df):
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df.dropna(inplace=True)
    return df

def train(df):
    feats = ["ema20","ema50","rsi","atr","returns"]
    X = df[feats]
    y = df["target"]
    split = int(len(df)*0.8)
    model = XGBClassifier(n_estimators=100,max_depth=4)
    model.fit(X[:split], y[:split])
    return model, feats

def signal(row, prob):
    if prob > 0.6 and row["ema20"] > row["ema50"]:
        return "BUY"
    elif prob < 0.4 and row["ema20"] < row["ema50"]:
        return "SELL"
    return "HOLD"

def trade(sig, price, atr, symbol):
    balance = st.session_state.balance
    positions = st.session_state.positions
    risk = balance * 0.02

    if sig == "BUY" and symbol not in positions:
        qty = risk / (1.5 * atr)
        positions[symbol] = {"qty": qty, "entry": price}
        st.session_state.trades.append(f"BUY {symbol} @ {price}")

    elif sig == "SELL" and symbol in positions:
        entry = positions[symbol]["entry"]
        qty = positions[symbol]["qty"]
        profit = (price - entry) * qty
        st.session_state.balance += profit
        del positions[symbol]
        st.session_state.trades.append(f"SELL {symbol} @ {price} | PnL {profit:.2f}")

df = fetch(symbol)
df = features(df)
df = target(df)

model, feats = train(df)

latest = df.iloc[-1]
prob = model.predict_proba([latest[feats]])[0][1]
sig = signal(latest, prob)

col1, col2, col3, col4 = st.columns(4)

col1.metric("Price", f"{latest['Close']:.2f}")
col2.metric("AI Confidence", f"{prob:.2f}")
col3.metric("Signal", sig)
col4.metric("Balance", f"{st.session_state.balance:.2f}")

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

if auto_trade:
    trade(sig, latest["Close"], latest["atr"], symbol)

st.subheader("Open Positions")
st.write(st.session_state.positions)

st.subheader("Trade Log")
st.write(st.session_state.trades)
