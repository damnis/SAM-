import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Functie om data op te halen ---
def fetch_data(ticker):
    df = yf.download(ticker, period="6mo", interval="1d")
    df = df[["Open", "High", "Low", "Close"]]
    df.dropna(inplace=True)
    return df

# --- SAM Indicatorberekeningen ---
def calculate_sam(df):
    df = df.copy()

    # Basiskolommen
    df["c1"] = df["Close"] > df["Open"]
    df["c2"] = df["Close"].shift(1) > df["Open"].shift(1)
    df["c3"] = df["Close"].shift(2) > df["Open"].shift(2)
    df["c4"] = df["Close"].shift(3) > df["Open"].shift(3)
    df["c5"] = df["Close"].shift(4) > df["Open"].shift(4)
    df["c6"] = df["Close"].shift(1) < df["Open"].shift(1)
    df["c7"] = df["Close"].shift(2) < df["Open"].shift(2)

    # SAMK
    df["SAMK"] = 0
    df.loc[(df["c1"] & df["c2"] & df["c3"] & df["c4"]).fillna(False), "SAMK"] = 1.25
    df.loc[(df["c1"] & df["c6"] & df["c7"]).fillna(False), "SAMK"] = -1

    # SAMG
    df["Change"] = df["Close"].pct_change()
    df["SAMG"] = 0
    df.loc[df["Change"] > 0.03, "SAMG"] = 1
    df.loc[df["Change"] < -0.03, "SAMG"] = -1

    # SAMT
    df["SMA5"] = df["Close"].rolling(window=5).mean()
    df["SMA20"] = df["Close"].rolling(window=20).mean()
    df["SAMT"] = 0
    df.loc[df["SMA5"] > df["SMA20"], "SAMT"] = 1
    df.loc[df["SMA5"] < df["SMA20"], "SAMT"] = -1

    # SAMD
    df["daily_range"] = df["High"] - df["Low"]
    avg_range = df["daily_range"].rolling(window=14).mean()
    df["SAMD"] = 0
    df.loc[df["daily_range"] > avg_range, "SAMD"] = 1
    df.loc[df["daily_range"] < avg_range, "SAMD"] = -1

    # SAMM
    df["SMA10"] = df["Close"].rolling(window=10).mean()
    df["SMA50"] = df["Close"].rolling(window=50).mean()
    df["SAMM"] = 0
    df.loc[df["SMA10"] > df["SMA50"], "SAMM"] = 1
    df.loc[df["SMA10"] < df["SMA50"], "SAMM"] = -1

    # SAMX (momentum op 3 dagen)
    df["Momentum"] = df["Close"] - df["Close"].shift(3)
    df["SAMX"] = 0
    df.loc[df["Momentum"] > 0, "SAMX"] = 1
    df.loc[df["Momentum"] < 0, "SAMX"] = -1

    # Totaal SAM-signaal
    df["SAM"] = df[["SAMK", "SAMG", "SAMT", "SAMD", "SAMM", "SAMX"]].sum(axis=1)

    # Trendberekening: rollend gemiddelde over 3 dagen
    df["Trend"] = df["SAM"].rolling(window=3).mean()

    return df

# --- Adviesfunctie ---
def geef_koop_verkoop_advies(df, gevoeligheid):
    trend = df["Trend"].dropna()
    advies_lijst = []
    vorig_advies = None

    for i in range(len(trend)):
        if i == 0:
            # Eerste advies op basis van trend
            vorig_advies = "KOOP" if trend.iloc[i] >= 0 else "VERKOOP"
        else:
            delta = trend.iloc[i] - trend.iloc[i - 1]
            if delta > gevoeligheid:
                vorig_advies = "KOOP"
            elif delta < -gevoeligheid:
                vorig_advies = "VERKOOP"
            # anders: behoud vorig advies
        advies_lijst.append(vorig_advies)

    df.loc[trend.index, "Advies"] = advies_lijst
    return df

# --- Streamlit UI ---
st.title("📊 SAM Trading Indicator")

ticker = st.selectbox("Selecteer een aandeel", ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"])
gevoeligheid = st.slider("Gevoeligheid trendverandering (0.1 = snel, 2.0 = traag)", 0.1, 2.0, 1.0, 0.1)

df = fetch_data(ticker)
df = calculate_sam(df)
df = geef_koop_verkoop_advies(df, gevoeligheid)

# Plot met histogram en trendlijn
st.subheader(f"SAM-indicator en Trend voor {ticker}")
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(df.index, df["SAM"], color="gray", label="SAM-signaal")
ax.plot(df.index, df["Trend"], color="orange", linewidth=2, label="Trend")
ax.set_title("SAM + Trend")
ax.legend()
st.pyplot(fig)

# Laatste signalen
st.subheader("Laatste signalen")
st.dataframe(df[["Close", "SAM", "Trend", "Advies"]].tail(10))
