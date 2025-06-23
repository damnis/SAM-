import streamlit as st
import pandas as pd
import yfinance as yf

st.title("SAM Beleggingssignaal App")

# Ticker-selectie via dropdown
ticker = st.selectbox("Kies een aandeel of index", ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA"], index=0)

# Data ophalen via yfinance
data = yf.download(ticker, period="180d", interval="1d")

# Validatie
if data.empty or not all(col in data.columns for col in ["Open", "High", "Low", "Close"]):
    st.error(f"Geen geldige data gevonden voor ticker: {ticker}")
    st.stop()

# DataFrame bouwen
open_ = data["Open"]
high = data["High"]
low = data["Low"]
close = data["Close"]
df = pd.DataFrame({
    "Open": open_,
    "High": high,
    "Low": low,
    "Close": close
})

# SAMK (candles)
df["c1"] = df["Close"] > df["Open"]
df["c2"] = df["Close"].shift(1) > df["Open"].shift(1)
df["c3"] = df["Close"] > df["Close"].shift(1)
df["c4"] = df["Close"].shift(1) > df["Close"].shift(2)
df["c5"] = df["Close"] < df["Open"]
df["c6"] = df["Close"].shift(1) < df["Open"].shift(1)
