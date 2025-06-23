import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Functie om data op te halen ---
def fetch_data(ticker):
    # Haal historische data op van Yahoo Finance voor het opgegeven ticker (6 maanden)
    df = yf.download(ticker, period="6mo", interval="1d")
    df = df[["Open", "High", "Low", "Close"]]  # Selecteer de benodigde kolommen
    df.dropna(inplace=True)  # Verwijder eventuele rijen met ontbrekende waarden
    return df

# --- SAM Indicatorberekeningen ---
def calculate_sam(df):
    df = df.copy()  # Maak een kopie van de DataFrame om het origineel niet te wijzigen
    
    # Basiskolommen (c1 t/m c7 voor verschillende condities)
    df["c1"] = df["Close"] > df["Open"]  # Sluiting is hoger dan opening
    df["c2"] = df["Close"].shift(1) > df["Open"].shift(1)  # Vorige sluiting was hoger dan opening
    df["c3"] = df["Close"].shift(2) > df["Open"].shift(2)  # Twee dagen geleden was sluiting hoger
    df["c4"] = df["Close"].shift(3) > df["Open"].shift(3)  # Drie dagen geleden was sluiting hoger
    df["c5"] = df["Close"].shift(4) > df["Open"].shift(4)  # Vier dagen geleden was sluiting hoger
    df["c6"] = df["Close"].shift(1) < df["Open"].shift(1)  # Vorige sluiting was lager dan opening
    df["c7"] = df["Close"].shift(2) < df["Open"].shift(2)  # Twee dagen geleden was sluiting lager

    # SAMK (signalen voor koop en verkoop op basis van candles)
    df["SAMK"] = 0  # Initieel op 0
    # Zet SAMK op 1.25 als c1, c2, c3, en c4 waar zijn
    df.loc[(df["c1"] & df["c2"] & df["c3"] & df["c4"]).fillna(False), "SAMK"] = 1.25
    # Zet SAMK op -1 als c1, c6, en c7 waar zijn (verkopen)
    df.loc[(df["c1"] & df["c6"] & df["c7"]).fillna(False), "SAMK"] = -1

    # SAMG (verandering in koers)
    df["Change"] = df["Close"].pct_change()  # Procentuele verandering in de koers
    df["SAMG"] = 0  # Initieel op 0
    df.loc[df["Change"] > 0.03, "SAMG"] = 1  # Positieve verandering boven 3% (koop)
    df.loc[df["Change"] < -0.03, "SAMG"] = -1  # Negatieve verandering boven 3% (verkoop)

    # SAMT (basis van moving averages)
    df["SMA5"] = df["Close"].rolling(window=5).mean()  # 5-daags voortschrijdend gemiddelde
    df["SMA20"] = df["Close"].rolling(window=20).mean()  # 20-daags voortschrijdend gemiddelde
    df["SAMT"] = 0  # Initieel op 0
    df.loc[df["SMA5"] > df["SMA20"], "SAMT"] = 1  # Koop wanneer 5-daagse SMA boven 20-daagse is
    df.loc[df["SMA5"] < df["SMA20"], "SAMT"] = -1  # Verkoop wanneer 5-daagse SMA onder 20-daagse is

    # SAMD (dagelijks koersbereik)
    df["daily_range"] = df["High"] - df["Low"]  # Dagelijks bereik (hoogte - laagte)
    avg_range = df["daily_range"].rolling(window=14).mean()  # 14-daags gemiddeld bereik
    df["SAMD"] = 0  # Initieel op 0
    df.loc[df["daily_range"] > avg_range, "SAMD"] = 1  # Koop als het dagelijkse bereik groter is dan gemiddeld
    df.loc[df["daily_range"] < avg_range, "SAMD"] = -1  # Verkoop als het dagelijkse bereik kleiner is dan gemiddeld

    # SAMM (momentum op basis van moving averages)
    df["SMA10"] = df["Close"].rolling(window=10).mean()  # 10-daags voortschrijdend gemiddelde
    df["SMA50"] = df["Close"].rolling(window=50).mean()  # 50-daags voortschrijdend gemiddelde
    df["SAMM"] = 0  # Initieel op 0
    df.loc[df["SMA10"] > df["SMA50"], "SAMM"] = 1  # Koop wanneer 10-daagse SMA boven 50-daagse is
    df.loc[df["SMA10"] < df["SMA50"], "SAMM"] = -1  # Verkoop wanneer 10-daagse SMA onder 50-daagse is

    # SAMX (momentum over 3 dagen)
    df["Momentum"] = df["Close"] - df["Close"].shift(3)  # Momentum 3 dagen geleden
    df["SAMX"] = 0  # Initieel op 0
    df.loc[df["Momentum"] > 0, "SAMX"] = 1  # Koop wanneer momentum positief is
    df.loc[df["Momentum"] < 0, "SAMX"] = -1  # Verkoop wanneer momentum negatief is

    # Totaal SAM-signaal: som van alle indicatoren
    df["SAM"] = df[["SAMK", "SAMG", "SAMT", "SAMD", "SAMM", "SAMX"]].sum(axis=1)

    return df

# --- Streamlit UI ---
st.title("📊 SAM Trading Indicator")

# Dropdown menu voor ticker keuze
ticker = st.selectbox("Selecteer een aandeel", ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"])
df = fetch_data(ticker)  # Haal de data op voor het geselecteerde ticker
df = calculate_sam(df)  # Bereken de SAM-indicatoren

# Plotten van de SAM indicatoren met trend (SMA)
st.subheader(f"SAM-indicator voor {ticker}")

# Maak een figuur en as object voor de plot
fig, ax = plt.subplots(figsize=(10, 6))

# Plot de SAM-indicator als een bar chart
df["SAM"].plot(kind="bar", ax=ax, color="skyblue", width=1, label="SAM")

# Voeg de trendlijn toe (SMA)
df["SMA"] = df["Close"].rolling(window=10).mean()  # Bereken 10-daags voortschrijdend gemiddelde (trendlijn)
ax.plot(df.index, df["SMA"], label="Trendlijn (SMA)", color="orange")

# Voeg labels en titel toe aan de grafiek
ax.set_xlabel("Datum")
ax.set_ylabel("SAM Waarde")
ax.set_title(f"SAM-indicator & Trendlijn voor {ticker}")
ax.legend()

# Toon de grafiek in de Streamlit app
st.pyplot(fig)

# Laatste signalen en rendementen tonen in een tabel
st.subheader("Laatste signalen en rendementen")
df["Slotkoers"] = df["Close"].apply(lambda x: f"{x:.2f}")  # Voeg slotkoers kolom toe, geformatteerd op 2 decimalen
df["Marktrendement"] = df["Close"].pct_change().apply(lambda x: f"{x * 100:.2f}%")  # Rendement van de markt (percentage)
df["SAM rendement"] = df["SAM"].pct_change().apply(lambda x: f"{x * 100:.2f}%")  # Rendement van de SAM indicator

# Weergave van de laatste 10 signalen en rendementen
st.dataframe(df[["Close", "SAM", "Marktrendement", "SAM rendement", "Slotkoers"]].tail(10))
