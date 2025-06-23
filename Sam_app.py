import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Ophalen van marktdata ---
def fetch_data(ticker):
    df = yf.download(ticker, period="6mo", interval="1d")
    df = df[["Open", "High", "Low", "Close"]]
    df.dropna(inplace=True)
    return df

# --- Berekening van SAM-indicatoren ---
def calculate_sam(df):
    df = df.copy()

    # Signaalvoorwaarden
    df["c1"] = df["Close"] > df["Open"]
    df["c2"] = df["Close"].shift(1) > df["Open"].shift(1)
    df["c3"] = df["Close"].shift(2) > df["Open"].shift(2)
    df["c4"] = df["Close"].shift(3) > df["Open"].shift(3)
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

    # SAMX
    df["Momentum"] = df["Close"] - df["Close"].shift(3)
    df["SAMX"] = 0
    df.loc[df["Momentum"] > 0, "SAMX"] = 1
    df.loc[df["Momentum"] < 0, "SAMX"] = -1

    # Totaal SAM-signaal
    df["SAM"] = df[["SAMK", "SAMG", "SAMT", "SAMD", "SAMM", "SAMX"]].sum(axis=1)

    return df

# --- Advies + rendement berekening ---
def generate_signals_with_returns(df, sensitivity):
    df = df.copy()
    df["Signaal"] = ""
    df["Slotkoers"] = df["Close"]
    df["SAM rendement"] = ""
    df["Marktrendement"] = ""

    trend = df["SAM"].rolling(window=sensitivity).mean()
    df["Trend"] = trend

    prev_signal = None
    start_index = None
    start_koers = None

    for idx in range(sensitivity, len(df)):
        curr_trend = trend.iloc[idx]
        prev_trend = trend.iloc[idx - 1]

        if curr_trend > prev_trend:
            signal = "Kopen"
        elif curr_trend < prev_trend:
            signal = "Verkopen"
        else:
            signal = prev_signal

        df.at[df.index[idx], "Signaal"] = signal

        if signal != prev_signal and prev_signal is not None and start_index is not None:
            eind_koers = df["Close"].iloc[idx]
            if start_koers is not None and not pd.isna(start_koers) and float(start_koers) != 0.0:
                sam_rend = (eind_koers - start_koers) / start_koers
                markt_rend = (eind_koers - df["Close"].iloc[start_index]) / df["Close"].iloc[start_index]
                df.at[df.index[idx - 1], "SAM rendement"] = f"{sam_rend * 100:.2f}%"
                df.at[df.index[idx - 1], "Marktrendement"] = f"{markt_rend * 100:.2f}%"

        if signal != prev_signal:
            start_index = idx
            start_koers = df["Close"].iloc[idx]

        prev_signal = signal

    return df

# --- Streamlit interface ---
st.set_page_config(page_title="SAM Indicator", layout="centered")
st.title("📈 SAM Trading Indicator")

ticker = st.selectbox("Kies een aandeel", ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"])
sensitivity = st.slider("Gevoeligheid (trendlengte)", min_value=2, max_value=15, value=5)

df = fetch_data(ticker)
df = calculate_sam(df)
df = generate_signals_with_returns(df, sensitivity)

# --- Grafiek ---
st.subheader("SAM Indicator Grafiek")
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(df.index, df["SAM"], label="SAM", alpha=0.6)
ax.plot(df.index, df["Trend"], color="red", label="Trend")
ax.set_title(f"{ticker} - SAM Histogram + Trendlijn")
ax.legend()
st.pyplot(fig)

# --- Laatste signalen ---
st.subheader("📋 Laatste Signalen en Rendement")
columns_to_show = ["Close", "Slotkoers", "Signaal", "SAM", "Trend", "SAM rendement", "Marktrendement"]
st.dataframe(df[columns_to_show].tail(15))
