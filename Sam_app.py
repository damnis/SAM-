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

close = data["Close"]
open_ = data["Open"]
dates = data.index

# --- SAM-componenten berekenen ---
# SAMK (candles)
c1 = close > open_
c2 = close.shift(1) > open_.shift(1)
c3 = close > close.shift(1)
c4 = close.shift(1) > close.shift(2)
c5 = close < open_
c6 = close.shift(1) < open_.shift(1)
c7 = close < close.shift(1)
c8 = close.shift(1) < close.shift(2)

SAMK = pd.Series(0.0, index=close.index)

SAMK[(c1 & c2 & c3 & c4).fillna(False)] = 1.25
SAMK[((c1 & c3 & c4) & ~c2).fillna(False)] = 1.0
SAMK[((c1 & c3) & ~(c2 | c4)).fillna(False)] = 0.5
SAMK[((c1 | c3) & ~(c1 & c3)).fillna(False)] = 0.25

SAMK[(c5 & c6 & c7 & c8).fillna(False)] = -1.25
SAMK[((c5 & c7 & c8) & ~c6).fillna(False)] = -1.0
SAMK[((c5 & c7) & ~(c6 | c8)).fillna(False)] = -0.5
SAMK[((c5 | c7) & ~(c5 & c7)).fillna(False)] = -0.25

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
