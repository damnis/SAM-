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

    # Berekeningen (SAMK t/m SAMX)
    df["c1"] = df["Close"] > df["Open"]
    df["c2"] = df["Close"].shift(1) > df["Open"].shift(1)
    df["c3"] = df["Close"].shift(2) > df["Open"].shift(2)
    df["c4"] = df["Close"].shift(3) > df["Open"].shift(3)
    df["c5"] = df["Close"].shift(4) > df["Open"].shift(4)
    df["c6"] = df["Close"].shift(1) < df["Open"].shift(1)
    df["c7"] = df["Close"].shift(2) < df["Open"].shift(2)

    df["SAMK"] = 0
    df.loc[(df["c1"] & df["c2"] & df["c3"] & df["c4"]).fillna(False), "SAMK"] = 1.25
    df.loc[(df["c1"] & df["c6"] & df["c7"]).fillna(False), "SAMK"] = -1

    df["Change"] = df["Close"].pct_change()
    df["SAMG"] = 0
    df.loc[df["Change"] > 0.03, "SAMG"] = 1
    df.loc[df["Change"] < -0.03, "SAMG"] = -1

    df["SMA5"] = df["Close"].rolling(window=5).mean()
    df["SMA20"] = df["Close"].rolling(window=20).mean()
    df["SAMT"] = 0
    df.loc[df["SMA5"] > df["SMA20"], "SAMT"] = 1
    df.loc[df["SMA5"] < df["SMA20"], "SAMT"] = -1

    df["daily_range"] = df["High"] - df["Low"]
    avg_range = df["daily_range"].rolling(window=14).mean()
    df["SAMD"] = 0
    df.loc[df["daily_range"] > avg_range, "SAMD"] = 1
    df.loc[df["daily_range"] < avg_range, "SAMD"] = -1

    df["SMA10"] = df["Close"].rolling(window=10).mean()
    df["SMA50"] = df["Close"].rolling(window=50).mean()
    df["SAMM"] = 0
    df.loc[df["SMA10"] > df["SMA50"], "SAMM"] = 1
    df.loc[df["SMA10"] < df["SMA50"], "SAMM"] = -1

    df["Momentum"] = df["Close"] - df["Close"].shift(3)
    df["SAMX"] = 0
    df.loc[df["Momentum"] > 0, "SAMX"] = 1
    df.loc[df["Momentum"] < 0, "SAMX"] = -1

    df["SAM"] = df[["SAMK", "SAMG", "SAMT", "SAMD", "SAMM", "SAMX"]].sum(axis=1)
    df["Trend"] = df["SAM"].rolling(window=3).mean()

    return df

# --- Adviezen bepalen + rendementen ---
def generate_signals_with_returns(df, sensitivity):
    df = df.copy()
    df["Advies"] = ""
    df["SAM rendement"] = ""
    df["Marktrendement"] = ""
    df["Slotkoers"] = df["Close"]

    vorige_signaal = None
    start_koers = None
    start_index = None

    for idx in range(1, len(df)):
        huidige_trend = df["Trend"].iloc[idx]
        vorige_trend = df["Trend"].iloc[idx - 1]

        if huidige_trend - vorige_trend > sensitivity:
            nieuw_signaal = "Kopen"
        elif vorige_trend - huidige_trend > sensitivity:
            nieuw_signaal = "Verkopen"
        else:
            nieuw_signaal = vorige_signaal  # behoud vorige signaal

        df.at[df.index[idx], "Advies"] = nieuw_signaal

        # Als signaal verandert: bereken rendement vanaf vorige adviesmoment
        if nieuw_signaal != vorige_signaal and vorige_signaal is not None:
            eind_koers = df["Close"].iloc[idx]
            if start_koers and start_koers != 0:
            sam_rend = (eind_koers - start_koers) / start_koers
            markt_rend = (eind_koers - df["Close"].iloc[start_index]) / df["Close"].iloc[start_index]
    
    df.at[df.index[idx - 1], "SAM rendement"] = f"{sam_rend * 100:.2f}%"
    df.at[df.index[idx - 1], "Marktrendement"] = f"{markt_rend * 100:.2f}%"
        if nieuw_signaal != vorige_signaal:
            start_koers = df["Close"].iloc[idx]
            start_index = idx
            vorige_signaal = nieuw_signaal

    return df

# --- Streamlit UI ---
st.title("📊 SAM Trading Indicator")

ticker = st.selectbox("Selecteer een aandeel", ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"])
sensitivity = st.slider("Gevoeligheid trendwijziging", min_value=0.1, max_value=2.0, value=0.5, step=0.1)

df = fetch_data(ticker)
df = calculate_sam(df)
df = generate_signals_with_returns(df, sensitivity)

# --- Grafiek ---
st.subheader(f"SAM-indicator voor {ticker}")
fig, ax1 = plt.subplots(figsize=(10, 4))
ax1.plot(df.index, df["Trend"], label="SAM Trend", color="blue")
ax2 = ax1.twinx()
ax2.bar(df.index, df["SAM"], label="SAM Waarde", color="gray", alpha=0.3)
ax1.set_ylabel("Trendwaarde")
ax2.set_ylabel("SAM Score")
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")
st.pyplot(fig)

# --- Laatste signalen tabel ---
st.subheader("Laatste signalen")
cols = ["Slotkoers", "Advies", "SAM rendement", "Marktrendement"]
st.dataframe(df[cols].dropna(how='all').tail(15))
