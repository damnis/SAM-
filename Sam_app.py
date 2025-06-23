import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

st.title("SAM Indicator App")

# --- Dropdown voor ticker selectie ---
ticker = st.selectbox("Kies een aandeel of index:", ["AAPL", "MSFT", "GOOGL", "^GSPC", "^NDX", "AMZN", "TSLA"])

# --- Data ophalen via yfinance ---
data = yf.download(ticker, period="180d", interval="1d")
if data.empty:
    st.error(f"Geen data gevonden voor ticker: {ticker}")
    st.stop()

close = data["Close"]
open_ = data["Open"]
high = data["High"]
low = data["Low"]
dates = data.index

# --- DataFrame maken ---
print("Close:", close.shape)
print("Open:", open_.shape)
print("High:", high.shape)
print("Low:", low.shape)
print("Dates:", dates.shape)
df = pd.DataFrame({
    "Close": close,
    "Open": open_,
    "High": high,
    "Low": low
}, index=dates)

# --- SAMK ---
df["c1"] = df["Close"] > df["Open"]
df["c2"] = df["Close"].shift(1) > df["Open"].shift(1)
df["c3"] = df["Close"] > df["Close"].shift(1)
df["c4"] = df["Close"].shift(1) > df["Close"].shift(2)
df["c5"] = df["Close"] < df["Open"]
df["c6"] = df["Close"].shift(1) < df["Open"].shift(1)
df["c7"] = df["Close"] < df["Close"].shift(1)
df["c8"] = df["Close"].shift(1) < df["Close"].shift(2)

SAMK = pd.Series(0.0, index=df.index)
SAMK.loc[(df["c1"] & df["c2"] & df["c3"] & df["c4"]).fillna(False)] = 1.25
SAMK.loc[((df["c1"] & df["c3"] & df["c4"]) & ~df["c2"]).fillna(False)] = 1.0
SAMK.loc[((df["c1"] & df["c3"]) & ~(df["c2"] | df["c4"])).fillna(False)] = 0.5
SAMK.loc[((df["c1"] | df["c3"]) & ~(df["c1"] & df["c3"])).fillna(False)] = 0.25
SAMK.loc[(df["c5"] & df["c6"] & df["c7"] & df["c8"]).fillna(False)] = -1.25
SAMK.loc[((df["c5"] & df["c7"] & df["c8"]) & ~df["c6"]).fillna(False)] = -1.0
SAMK.loc[((df["c5"] & df["c7"]) & ~(df["c6"] | df["c8"])).fillna(False)] = -0.5
SAMK.loc[((df["c5"] | df["c7"]) & ~(df["c5"] & df["c7"])).fillna(False)] = -0.25

df["SAMK"] = SAMK

# --- SAMX (volatiliteit) ---
range_ = df["High"] - df["Low"]
threshold = range_.rolling(20).mean()
df["SAMX"] = np.where(range_ > threshold, 1, 0)

# --- SAMA (afstand tot MA20) ---
ma20 = df["Close"].rolling(20).mean()
afstand = df["Close"] - ma20
norm = df["Close"].rolling(20).std()
df["SAMA"] = afstand / norm

# --- SAM totaal ---
df["SAM"] = df[["SAMK", "SAMX", "SAMA"]].sum(axis=1)

# --- SAMM, SAMT, SAMD ---
df["SAMM"] = df["SAM"].rolling(5).mean()
df["SAMT"] = df["SAMM"].diff()
df["SAMD"] = df["SAMT"].diff()

# --- Plotten ---
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(df.index, df["SAM"], color="black", label="SAM")
ax.plot(df["SAMM"], color="blue", linewidth=2.5, label="SAMM (Trend)")
ax.axhline(0, color="gray", linewidth=1, linestyle="--")
ax.legend()
ax.set_title(f"SAM-indicator voor {ticker}")
st.pyplot(fig)

# --- Toon data ---
st.dataframe(df[["Close", "SAM", "SAMM", "SAMT", "SAMD"]].tail(30))
