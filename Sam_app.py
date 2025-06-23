# sam_app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

# --- App-configuratie ---
st.set_page_config(page_title="SAM Signaal App", layout="wide")
st.title("📈 Live SAM Beleggingssignalen")

# --- Dropdown voor tickerselectie ---
ticker = st.sidebar.selectbox(
    "Selecteer ticker:",
    ["ASML.AS", "AAPL", "^AEX", "BTC-USD"]
)

# --- Data ophalen via yfinance ---
data = yf.download(ticker, period="180d", interval="1d")
if data.empty:
    st.error(f"Geen data gevonden voor ticker: {ticker}")
    st.stop()

# Werk in één DataFrame
df = data.copy()

# --- SAM-componenten berekenen ---
# Candle condities
df["c1"] = df["Close"] > df["Open"]
df["c2"] = df["Close"].shift(1) > df["Open"].shift(1)
df["c3"] = df["Close"] > df["Close"].shift(1)
df["c4"] = df["Close"].shift(1) > df["Close"].shift(2)
df["c5"] = df["Close"] < df["Open"]
df["c6"] = df["Close"].shift(1) < df["Open"].shift(1)
df["c7"] = df["Close"] < df["Close"].shift(1)
df["c8"] = df["Close"].shift(1) < df["Close"].shift(2)

# SAMK berekening
df["SAMK"] = 0.0

df.loc[(df["c1"] & df["c2"] & df["c3"] & df["c4"]).fillna(False), "SAMK"] = 1.25
df.loc[((df["c1"] & df["c3"] & df["c4"]) & ~df["c2"]).fillna(False), "SAMK"] = 1.0
df.loc[((df["c1"] & df["c3"]) & ~(df["c2"] | df["c4"])).fillna(False), "SAMK"] = 0.5
df.loc[((df["c1"] | df["c3"]) & ~(df["c1"] & df["c3"])).fillna(False), "SAMK"] = 0.25

df.loc[(df["c5"] & df["c6"] & df["c7"] & df["c8"]).fillna(False), "SAMK"] = -1.25
df.loc[((df["c5"] & df["c7"] & df["c8"]) & ~df["c6"]).fillna(False), "SAMK"] = -1.0
df.loc[((df["c5"] & df["c7"]) & ~(df["c6"] | df["c8"])).fillna(False), "SAMK"] = -0.5
df.loc[((df["c5"] | df["c7"]) & ~(df["c5"] & df["c7"])).fillna(False), "SAMK"] = -0.25

# Indien gewenst: drop condities achteraf
df.drop(columns=["c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8"], inplace=True)

# SAMG (WMA18 toestand)
wma = lambda s, p: s.rolling(p).apply(lambda x: np.average(x, weights=range(1, p+1)), raw=True)
c9 = wma(close, 18)
c11 = wma(close.shift(1), 18)

SAMG = pd.Series(0.0, index=close.index)
SAMG[(c9 > c11 * 1.0015)] = 0.5
SAMG[(c9 < c11 * 1.0015) & (c9 > c11)] = -0.5
SAMG[(c9 > c11 / 1.0015) & (c9 <= c11)] = 0.5
SAMG[(c9 < c11 / 1.0015) & (c9 <= c11)] = -0.5

# SAMT (WMA6 vs WMA80)
c12 = wma(close, 6)
c13 = wma(close.shift(1), 6)
c19 = wma(close, 80)

SAMT = pd.Series(0.0, index=close.index)
SAMT[(c12 > c13) & (c12 > c19)] = 0.5
SAMT[(c12 > c13) & (c12 <= c19)] = 0.25
SAMT[(c12 <= c13) & (c12 <= c19)] = -0.75
SAMT[(c12 <= c13) & (c12 > c19)] = -0.5

# SAMM (MACD)
ema12 = close.ewm(span=12, adjust=False).mean()
ema26 = close.ewm(span=26, adjust=False).mean()
macd = ema12 - ema26
signal_line = macd.ewm(span=9, adjust=False).mean()

SAMM = pd.Series(0.0, index=close.index)
SAMM[(macd > signal_line) & (macd.shift(1) <= signal_line.shift(1))] = 1.0
SAMM[(macd > signal_line)] = 0.5
SAMM[(macd < signal_line) & (macd.shift(1) >= signal_line.shift(1))] = -1.0
SAMM[(macd < signal_line)] = -0.5

# SAMD (simpele DI)
SAMD = pd.Series(np.where(close.diff() > 0, 0.5, -0.5), index=close.index)

# SAMX (TRIX)
trix = close.pct_change().ewm(span=15).mean()
trix_prev = trix.shift(1)

SAMX = pd.Series(0.0, index=close.index)
SAMX[(trix > 0) & (trix > trix_prev)] = 0.75
SAMX[(trix > 0)] = 0.5
SAMX[(trix < 0) & (trix < trix_prev)] = -0.75
SAMX[(trix < 0)] = -0.5

# Totale SAM + Trendberekening
SAM = SAMK + SAMG + SAMT + SAMM + SAMD + SAMX
TrendSAM = SAM.rolling(window=12).mean()

# --- Sensitivity slider en signalen ---
sensitivity = st.sidebar.slider("Aantal bevestigingen voor signaal:", 1, 5, value=2)
signal = pd.Series("", index=SAM.index)
prev = ""
count = 0

for i in range(1, len(signal)):
    if SAM.iat[i] > 0 and TrendSAM.iat[i] > 0:
        count += 1
        if count >= sensitivity and prev != "KOOP":
            signal.iat[i] = "KOOP"
            prev = "KOOP"
            count = 0
    elif SAM.iat[i] < 0 and TrendSAM.iat[i] < 0:
        count += 1
        if count >= sensitivity and prev != "VERKOOP":
            signal.iat[i] = "VERKOOP"
            prev = "VERKOOP"
            count = 0
    else:
        count = 0

# DataFrame voor plot en weergave
sam_df = pd.DataFrame({
    "SAM": SAM,
    "TrendSAM": TrendSAM,
    "Signaal": signal
}, index=dates)

# --- Grafiekaanpassingen ---
fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(sam_df.index, sam_df["SAM"], color="black", label="SAM")
ax.plot(sam_df.index, sam_df["TrendSAM"], color="blue", linewidth=2.5, label="TrendSAM")
ax.axhline(0, color="gray", linestyle="--", linewidth=1)
ax.set_title(f"SAM & TrendSAM voor {ticker}")
ax.legend()
st.pyplot(fig)

# --- Laatste signalen ---
st.subheader("Laatste signalen")
st.dataframe(sam_df.tail(20))
