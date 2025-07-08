# sam_indicator.py
import numpy as np
import pandas as pd
import streamlit as st
from ta.trend import ADXIndicator
import ta

# --- Weighted Moving Average functie ---
def weighted_moving_average(series, window):
    weights = np.arange(1, window + 1)
    return series.rolling(window).apply(lambda x: np.dot(x, weights)/weights.sum(), raw=True)

# --- SAM Indicatorberekeningen ---
@st.cache_data(ttl=900)
def calculate_sam(df):
    df = df.copy()

    # Basiskolommen
    # --- SAMG op basis van Weighted Moving Averages + Crossovers nodig ---
  #  def weighted_moving_average(series, window):  # reeds gedefinieerd 
  #      weights = np.arange(1, window + 1)
  #      return series.rolling(window).apply(lambda x: np.dot(x, weights)/weights.sum(), raw=True)
    
    # --- SAMK: candlestick score op basis van patronen Open/Close ---
    df["c1"] = df["Close"] > df["Open"]
    df["c2"] = df["Close"].shift(1) > df["Open"].shift(1)
    df["c3"] = df["Close"] > df["Close"].shift(1)
    df["c4"] = df["Close"].shift(1) > df["Close"].shift(2)
    df["c5"] = df["Close"] < df["Open"]
    df["c6"] = df["Close"].shift(1) < df["Open"].shift(1)
    df["c7"] = df["Close"] < df["Close"].shift(1)
    df["c8"] = df["Close"].shift(1) < df["Close"].shift(2)

    df["SAMK"] = 0.0  # standaard
    # Positieve patronen
    df.loc[
        (df["SAMK"] == 0.0) & (df["c1"] & df["c2"] & df["c3"] & df["c4"]).fillna(False),
        "SAMK"
    ] = 1.25

    df.loc[
        (df["SAMK"] == 0.0) & (df["c1"] & df["c3"] & df["c4"]).fillna(False),
        "SAMK"
    ] = 1.0

    df.loc[
        (df["SAMK"] == 0.0) & (df["c1"] & df["c3"]).fillna(False),
        "SAMK"
    ] = 0.5

    df.loc[
        (df["SAMK"] == 0.0) & (df["c1"] | df["c3"]).fillna(False),
        "SAMK"
    ] = 0.25

    # Negatieve patronen
    df.loc[
        (df["SAMK"] == 0.0) & (df["c5"] & df["c6"] & df["c7"] & df["c8"]).fillna(False),
        "SAMK"
    ] = -1.25

    df.loc[
        (df["SAMK"] == 0.0) & (df["c5"] & df["c7"] & df["c8"]).fillna(False),
        "SAMK"
    ] = -1.0

    df.loc[
        (df["SAMK"] == 0.0) & (df["c5"] & df["c7"]).fillna(False),
        "SAMK"
    ] = -0.5

    df.loc[
        (df["SAMK"] == 0.0) & (df["c5"] | df["c7"]).fillna(False),
        "SAMK"
    ] = -0.25
    

    # SAMK oud
#    df["SAMK"] = 0
#    df.loc[(df["c1"] & df["c2"] & df["c3"] & df["c4"]).fillna(False), "SAMK"] = 1.25
 #   df.loc[(df["c1"] & df["c6"] & df["c7"]).fillna(False), "SAMK"] = -1

    # --- SAMG (WMA-based trendanalyse met crossovers) ---
    df["WMA18"] = weighted_moving_average(df["Close"], 18)
    df["WMA35"] = weighted_moving_average(df["Close"], 35)
    df["WMA18_shifted"] = df["WMA18"].shift(1)
    df["WMA35_shifted"] = df["WMA35"].shift(1)

    df["SAMG"] = 0.0

    # Kleine trendbewegingen omhoog en omlaag
    df.loc[
        (df["WMA18"] > df["WMA18_shifted"] * 1.0015) & (df["WMA18"] > df["WMA18_shifted"]),
        "SAMG"
    ] = 0.5

    df.loc[
        (df["WMA18"] < df["WMA18_shifted"] * 1.0015) & (df["WMA18"] > df["WMA18_shifted"]),
        "SAMG"
    ] = -0.5

    df.loc[
        (df["WMA18"] > df["WMA18_shifted"] / 1.0015) & (df["WMA18"] <= df["WMA18_shifted"]),
        "SAMG"
    ] = 0.5

    df.loc[
        (df["WMA18"] < df["WMA18_shifted"] / 1.0015) & (df["WMA18"] <= df["WMA18_shifted"]),
        "SAMG"
    ] = -0.5

    # Grote crossovers tussen WMA18 en WMA35
    df.loc[
        (df["WMA18_shifted"] < df["WMA35_shifted"]) & (df["WMA18"] > df["WMA35"]),
        "SAMG"
    ] = 0.75 # oorspronkelijk 1.0, uit in nieuwere sam versie
    df.loc[
        (df["WMA18_shifted"] > df["WMA35_shifted"]) & (df["WMA18"] < df["WMA35"]),
        "SAMG"
    ] = -0.75 # zie vorige

#  samg oud
#    df["Change"] = df["Close"].pct_change()
#    df["SAMG"] = 0
#    df.loc[df["Change"] > 0.03, "SAMG"] = 1
#    df.loc[df["Change"] < -0.03, "SAMG"] = -1

    # SAMT oud
#    df["SMA5"] = df["Close"].rolling(window=5).mean()
#    df["SMA20"] = df["Close"].rolling(window=20).mean()
#    df["SAMT"] = 0
 #   df.loc[df["SMA5"] > df["SMA20"], "SAMT"] = 1
#    df.loc[df["SMA5"] < df["SMA20"], "SAMT"] = -1
# --- SAMT op basis van Weighted Moving Averages 6 en 80 ---
#    def weighted_moving_average(series, window): --> al eerder gedefineerd
 #       weights = np.arange(1, window + 1)
  #      return series.rolling(window).apply(lambda x: np.dot(x, weights)/weights.sum(), raw=True)

    df["WMA6"] = weighted_moving_average(df["Close"], 6)
    df["WMA6_shifted"] = df["WMA6"].shift(1)
    df["WMA80"] = weighted_moving_average(df["Close"], 80)

    df["SAMT"] = 0.0  # standaardwaarde

    df.loc[
        (df["WMA6"] > df["WMA6_shifted"]) & (df["WMA6"] > df["WMA80"]),
        "SAMT"
    ] = 0.5

    df.loc[
        (df["WMA6"] > df["WMA6_shifted"]) & (df["WMA6"] <= df["WMA80"]),
        "SAMT"
    ] = 0.25

    df.loc[
        (df["WMA6"] <= df["WMA6_shifted"]) & (df["WMA6"] <= df["WMA80"]),
        "SAMT"
    ] = -0.75

    df.loc[
        (df["WMA6"] <= df["WMA6_shifted"]) & (df["WMA6"] > df["WMA80"]),
        "SAMT"
    ] = -0.5

    # --- SAMD op basis van DI+ en DI- ---
    # ————————— Flatten MultiIndex kolommen ——————————
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    # 1) Haal de Series direct op (geen squeeze)
    high_series  = df["High"]
    low_series   = df["Low"]
    close_series = df["Close"]
    
    # 2) Optioneel: forceer numeric en drop NaNs vóór de indicator
    high_series  = pd.to_numeric(high_series, errors="coerce")
    low_series   = pd.to_numeric(low_series, errors="coerce")
    close_series = pd.to_numeric(close_series, errors="coerce")
    
    # 4) Roep ADXIndicator aan
    adx = ADXIndicator(
        high=high_series,
        low=low_series,
        close=close_series,
        window=14,
        fillna=True
    )
    
#    df["DI_PLUS"]  = adx.adx_pos()
#    df["DI_MINUS"] = adx.adx_neg()

    # SAMD - was goed
    # --- SAMD op basis van DI+ en DI- ---
#    high_series = df["High"].squeeze()
#    low_series = df["Low"].squeeze()
#    close_series = df["Close"].squeeze()

#    adx = ADXIndicator(high=high_series, low=low_series, close=close_series, window=14)

    df["DI_PLUS"] = adx.adx_pos()
    df["DI_MINUS"] = adx.adx_neg()
    df["SAMD"] = 0.0  # begin met 0.0

    # Epsilon-drempels instellen
    epsilonneg = 10.0  # vrijwel afwezig andere richting
    epsilonpos = 30.0  # sterke richting
    df["SAMD"] = 0.0
    # 1️⃣ Sterke positieve richting
    df.loc[(df["DI_PLUS"] > epsilonpos) & (df["DI_MINUS"] <= epsilonneg), "SAMD"] = 0.75 # was 1.0
    # 2️⃣ Sterke negatieve richting
    df.loc[(df["DI_MINUS"] > epsilonpos) & (df["DI_PLUS"] <= epsilonneg), "SAMD"] = -0.75 # was -1.0
    # 3️⃣ Lichte positieve richting
    df.loc[(df["DI_PLUS"] > df["DI_MINUS"]) & (df["DI_MINUS"] > epsilonneg), "SAMD"] = 0.5
    # 4️⃣ Lichte negatieve richting
    df.loc[(df["DI_MINUS"] > df["DI_PLUS"]) & (df["DI_PLUS"] > epsilonneg), "SAMD"] = -0.5


    # samd oud
#    df["daily_range"] = df["High"] - df["Low"]
 #   avg_range = df["daily_range"].rolling(window=14).mean()
#    df["SAMD"] = 0
#    df.loc[df["daily_range"] > avg_range, "SAMD"] = 1
#    df.loc[df["daily_range"] < avg_range, "SAMD"] = -1

    # SAMM
    # ✅ Correcte MACD-berekening met ta
    close_series = df["Close"].squeeze()  # Zorg dat het een Series is
    macd_ind = ta.trend.MACD(close=close_series, window_slow=26, window_fast=12, window_sign=9)

    df["MACD"] = macd_ind.macd()
    df["SIGNAL"] = macd_ind.macd_signal()

    # ✅ Detecteer MACD crossovers voor SAMM
    prev_macd = df["MACD"].shift(1)
    prev_signal = df["SIGNAL"].shift(1)

    conditions = [
        (prev_macd < prev_signal) & (df["MACD"] > df["SIGNAL"]),
        (df["MACD"] > df["SIGNAL"]),
        (prev_macd > prev_signal) & (df["MACD"] < df["SIGNAL"]),
        (df["MACD"] <= df["SIGNAL"])
    ]
    keuzes = [1.0, 0.5, -1.0, -0.5]

    df["SAMM"] = np.select(conditions, keuzes, default=0.0)

    # samm oud   
  #  df["SMA10"] = df["Close"].rolling(window=10).mean()
  #  df["SMA50"] = df["Close"].rolling(window=50).mean()
 #   df["SAMM"] = 0
  #  df.loc[df["SMA10"] > df["SMA50"], "SAMM"] = 1
 #   df.loc[df["SMA10"] < df["SMA50"], "SAMM"] = -1

    # SAMX
#    from ta.momentum import TRIXIndicator

    # --- SAMX op basis van TRIX ---
    # --- SAMX: handmatige TRIX-berekening en interpretatie ---
    def calculate_trix(series, period=15):
        ema1 = series.ewm(span=period, adjust=False).mean()
        ema2 = ema1.ewm(span=period, adjust=False).mean()
        ema3 = ema2.ewm(span=period, adjust=False).mean()
        trix = (ema3 - ema3.shift(1)) / ema3.shift(1) * 100
        return trix

    df["TRIX"] = calculate_trix(df["Close"], period=15)
    df["TRIX_PREV"] = df["TRIX"].shift(1)
    df["SAMX"] = 0.0

    # Sterke opwaartse trend
    df.loc[(df["TRIX"] > 0) & (df["TRIX"] > df["TRIX_PREV"]), "SAMX"] = 0.75
    # Zwakke opwaartse trend
    df.loc[(df["TRIX"] > 0) & (df["TRIX"] <= df["TRIX_PREV"]), "SAMX"] = 0.5
    # Sterke neerwaartse trend
    df.loc[(df["TRIX"] < 0) & (df["TRIX"] < df["TRIX_PREV"]), "SAMX"] = -0.75
    # Zwakke neerwaartse trend
    df.loc[(df["TRIX"] < 0) & (df["TRIX"] >= df["TRIX_PREV"]), "SAMX"] = -0.5

    # SAMX OUD
#    df["Momentum"] = df["Close"] - df["Close"].shift(3)
#    df["SAMX"] = 0
#    df.loc[df["Momentum"] > 0, "SAMX"] = 1
 #   df.loc[df["Momentum"] < 0, "SAMX"] = -1

    # Totale SAM
    df["SAM"] = df[["SAMK", "SAMG", "SAMT", "SAMD", "SAMM", "SAMX"]].sum(axis=1)

    return df
    
