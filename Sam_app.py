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

    # Totaal SAM
    df["SAM"] = df[["SAMK", "SAMG", "SAMT", "SAMD", "SAMM", "SAMX"]].sum(axis=1)

    return df

# --- Signaalgeneratie met rendementen ---
def generate_signals_with_returns(df, sensitivity):
    df = df.copy()
    df["Trend"] = df["SAM"].rolling(window=3).mean()
    df["Advies"] = ""
    df["SAM rendement"] = ""
    df["Marktrendement"] = ""
    df["Slotkoers"] = df["Close"]

    prev_trend = 0
    current_signal = ""
    start_index = None
    start_koers = None

    for idx in range(len(df)):
        trend = df["Trend"].iloc[idx]
        if np.isnan(trend):
            continue

        if trend - prev_trend > sensitivity:
            nieuw_advies = "Kopen"
        elif trend - prev_trend < -sensitivity:
            nieuw_advies = "Verkopen"
        else:
            nieuw_advies = current_signal

        if nieuw_advies != current_signal:
            if current_signal in ["Kopen", "Verkopen"] and start_index is not None and idx > start_index:
                eind_koers = df["Close"].iloc[idx - 1]
                if start_koers and start_koers != 0:
                    try:
                        sam_rend = (eind_koers - start_koers) / start_koers
                        markt_rend = (eind_koers - df["Close"].iloc[start_index]) / df["Close"].iloc[start_index]
                        df.at[df.index[idx - 1], "SAM rendement"] = f"{sam_rend * 100:.2f}%"
                        df.at[df.index[idx - 1], "Marktrendement"] = f"{markt_rend * 100:.2f}%"
                    except:
                        df.at[df.index[idx - 1], "SAM rendement"] = "Fout"
                        df.at[df.index[idx - 1], "Marktrendement"] = "Fout"

            current_signal = nieuw_advies
            start_index = idx
            start_koers = df["Close"].iloc[idx]

        df.at[df.index[idx], "Advies"] = current_signal
        prev_trend = trend

    return df

# --- Streamlit Interface ---
st.title("📈 SAM Trading Indicator")

ticker = st.selectbox("Selecteer een aandeel", ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"])
sensitivity = st.slider("Gevoeligheid voor trendwijziging", min_value=0.1, max_value=2.0, value=0.5, step=0.1)

df = fetch_data(ticker)
df = calculate_sam(df)
df = generate_signals_with_returns(df, sensitivity)

# Plot
st.subheader(f"Grafiek SAM + trend voor {ticker}")
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(df.index, df["SAM"], label="SAM", alpha=0.6)
ax.plot(df.index, df["Trend"], label="Trend", color="red", linewidth=2)
ax.set_ylabel("Waarde")
ax.set_title("SAM-indicator en trendlijn")
ax.legend()
st.pyplot(fig)

# Signalen
st.subheader("Laatste signalen")
st.dataframe(df[["Slotkoers", "Advies", "SAM rendement", "Marktrendement"]].tail(15))
