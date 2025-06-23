import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

st.title("SAM Signaal Generator")

# --- Ticker selectie ---
tickers = ["AAPL", "GOOG", "MSFT", "AMZN", "TSLA", "^GSPC", "BTC-USD", "ETH-USD"]
ticker = st.selectbox("Selecteer een ticker:", tickers)

# --- Data ophalen ---
data = yf.download(ticker, period="180d", interval="1d")
if data.empty:
    st.error(f"Geen data gevonden voor ticker: {ticker}")
    st.stop()

# --- Series extraheren ---
open_ = data["Open"].squeeze()
high = data["High"].squeeze()
low = data["Low"].squeeze()
close = data["Close"].squeeze()
dates = data.index

# --- DataFrame bouwen ---
df = pd.DataFrame({
    "Open": open_,
    "High": high,
    "Low": low,
    "Close": close
}, index=dates)

# --- SAMK berekenen ---
df["c1"] = df["Close"] > df["Open"]
df["c2"] = df["Close"].shift(1) > df["Open"].shift(1)
df["c3"] = df["Close"] > df["Close"].shift(1)
df["c4"] = df["Close"].shift(1) > df["Close"].shift(2)
df["c5"] = df["Close"] < df["Open"]
df["c6"] = df["Close"].shift(1) < df["Open"].shift(1)
df["c7"] = df["Close"] < df["Close"].shift(1)
df["c8"] = df["Close"].shift(1) < df["Close"].shift(2)

df["SAMK"] = 0.0
df.loc[(df["c1"] & df["c2"] & df["c3"] & df["c4"]).fillna(False), "SAMK"] = 1.25
df.loc[((df["c1"] & df["c3"] & df["c4"]) & ~df["c2"]).fillna(False), "SAMK"] = 1.0
df.loc[((df["c1"] & df["c3"]) & ~(df["c2"] | df["c4"])).fillna(False), "SAMK"] = 0.5
df.loc[((df["c1"] | df["c3"]) & ~(df["c1"] & df["c3"])).fillna(False), "SAMK"] = 0.25
df.loc[(df["c5"] & df["c6"] & df["c7"] & df["c8"]).fillna(False), "SAMK"] = -1.25
df.loc[((df["c5"] & df["c7"] & df["c8"]) & ~df["c6"]).fillna(False), "SAMK"] = -1.0
df.loc[((df["c5"] & df["c7"]) & ~(df["c6"] | df["c8"])).fillna(False), "SAMK"] = -0.5
df.loc[((df["c5"] | df["c7"]) & ~(df["c5"] & df["c7"])).fillna(False), "SAMK"] = -0.25

# --- SAMM (momentum) ---
df["SAMM"] = df["Close"].diff()

# --- SAMX (extremen) ---
df["High_shift"] = df["High"].shift(1)
df["Low_shift"] = df["Low"].shift(1)
df["range"] = df["High_shift"] - df["Low_shift"]
df["SAMX"] = (df["Close"] - df["Open"]) / df["range"]
df["SAMX"] = df["SAMX"].clip(-1, 1).fillna(0)

# --- SAMT (trend) ---
df["rolling_mean"] = df["Close"].rolling(window=5).mean()
df["SAMT"] = df["Close"] - df["rolling_mean"]

# --- SAMD (differentiatie/versnelling) ---
df["SAMD"] = df["SAMK"].diff()

# --- Totale SAM-score ---
df["SAM_score"] = df["SAMK"] + df["SAMM"] + df["SAMX"] + df["SAMT"] + df["SAMD"]

# --- Plotten ---
fig, ax = plt.subplots(figsize=(12, 4))
df["SAM_score"].plot(ax=ax, label="SAM score", color="blue")
ax.axhline(0, color="black", linestyle="--", linewidth=1)
ax.set_title(f"SAM Signaal voor {ticker}")
ax.set_ylabel("SAM-score")
ax.legend()
st.pyplot(fig)
