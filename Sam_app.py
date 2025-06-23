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

# --- Signalen en rendementen genereren ---
def generate_signals_with_returns(df, sensitivity):
    df = df.copy()
    df["Advies"] = ""
    df["Slotkoers"] = df["Close"]
    df["SAM rendement"] = ""
    df["Marktrendement"] = ""

    prev_trend = 0
    start_index = 0
    signal = ""

    for idx in range(1, len(df)):
        huidige_trend = df["SAM"].iloc[idx]
        vorige_trend = df["SAM"].iloc[idx - 1]

        if abs(huidige_trend - vorige_trend) >= sensitivity:
            signal = "Kopen" if huidige_trend > vorige_trend else "Verkopen"
            df.at[df.index[idx], "Advies"] = signal

            try:
                eind_koers = float(df["Close"].iloc[idx])
                start_koers = float(df["Close"].iloc[start_index])

                if start_koers != 0.0:
                    sam_rend = (eind_koers - start_koers) / start_koers
                    markt_rend = (eind_koers - df["Close"].iloc[start_index]) / df["Close"].iloc[start_index]

                    df.at[df.index[idx], "SAM rendement"] = f"{sam_rend * 100:.2f}%"
                    df.at[df.index[idx], "Marktrendement"] = f"{markt_rend * 100:.2f}%"
            except Exception as e:
                df.at[df.index[idx], "SAM rendement"] = "n.v.t."
                df.at[df.index[idx], "Marktrendement"] = "n.v.t."

            start_index = idx

        else:
            df.at[df.index[idx], "Advies"] = signal

    return df
    
# --- Streamlit UI ---
st.set_page_config(page_title="SAM Beleggingsindicator", layout="wide")
st.title("📊 SAM Trading Indicator")

# Tickerkeuze en gevoeligheid
ticker = st.selectbox("Selecteer een aandeel", ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"])
sensitivity = st.slider("Gevoeligheid trendverandering", 1, 5, 2)

# Data ophalen en berekenen
df = fetch_data(ticker)
df = calculate_sam(df)
df = generate_signals_with_returns(df, sensitivity)

# Grafiek SAM + trendlijn
st.subheader(f"SAM-indicator & trendlijn voor {ticker}")
fig, ax = plt.subplots(figsize=(10, 4))
df["SAM"].plot(kind="bar", ax=ax, color="skyblue", label="SAM", width=1)
df["SAM"].rolling(window=5).mean().plot(ax=ax, color="red", label="Trend")
ax.legend()
st.pyplot(fig)

# Laatste adviezen en rendementen tonen
st.subheader("Laatste signalen & rendementen")
st.dataframe(df[["Slotkoers", "SAM", "Advies", "SAM rendement", "Marktrendement"]].tail(15))
