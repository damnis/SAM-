import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

st.title("📈 SAM Beleggingssignalen")

# --- Tickerselectie via dropdown ---
ticker = st.selectbox("Kies een aandeel (ticker):", ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"])

# --- Data ophalen ---
data = yf.download(ticker, period="180d", interval="1d")

# --- Check op data ---
if data.empty or len(data) < 10:
    st.error("Geen geldige data opgehaald. Kies een andere ticker.")
    st.stop()

# --- Kolommen ophalen ---
open_ = data["Open"]
high = data["High"]
low = data["Low"]
close = data["Close"]
dates = data.index

# --- DataFrame aanmaken ---
df = pd.DataFrame({
    "Open": open_,
    "High": high,
    "Low": low,
    "Close": close
}, index=dates)

# --- SAMK (candlestick signalen) ---
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

# --- SAMM (middenkoers vs. sluiting) ---
df["Mid"] = (df["High"] + df["Low"]) / 2
df["SAMM"] = ((df["Close"] - df["Mid"]) / df["Mid"]) * 2

# --- SAMT (trendvolger) ---
df["MA7"] = df["Close"].rolling(window=7).mean()
df["MA21"] = df["Close"].rolling(window=21).mean()
df["SAMT"] = 0.0
df.loc[df["MA7"] > df["MA21"], "SAMT"] = 1
df.loc[df["MA7"] < df["MA21"], "SAMT"] = -1

# --- SAMD (volatiliteit) ---
df["stddev"] = df["Close"].rolling(window=14).std()
df["SAMD"] = df["stddev"] / df["Close"]

# --- SAM totaal ---
df["SAM"] = df["SAMK"] + df["SAMM"] + df["SAMT"] - df["SAMD"]

# --- Plotten ---
fig, ax1 = plt.subplots(figsize=(12, 6))
ax1.plot(df.index, df["Close"], label="Koers", color="black")
ax1.set_ylabel("Koers")
ax1.set_title(f"{ticker} koers + SAM-indicator")

ax2 = ax1.twinx()
ax2.plot(df.index, df["SAM"], label="SAM", color="blue", linestyle="--")
ax2.set_ylabel("SAM-score")

fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9))
st.pyplot(fig)
