#sam_app.py

import streamlit as st 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

#Simuleer koersdata

# Simulate price data (replace later with live data)
np.random.seed(0)
days = 120
price = np.cumsum(np.random.normal(0, 1, days)) + 100
close = pd.Series(price)
open_ = close.shift(1).fillna(method="bfill")

# --- SAMK calculation ---
c1 = close > open_
c2 = close.shift(1) > open_.shift(1)
c3 = close > close.shift(1)
c4 = close.shift(1) > close.shift(2)
c5 = close < open_
c6 = close.shift(1) < open_.shift(1)
c7 = close < close.shift(1)
c8 = close.shift(1) < close.shift(2)

SAMK = pd.Series(0.0, index=close.index)
SAMK[(c1 & c2 & c3 & c4)] = 1.25
SAMK[(c1 & c3 & c4) & ~c2] = 1.0
SAMK[(c1 & c3) & ~(c2 | c4)] = 0.5
SAMK[(c1 | c3) & ~(c1 & c3)] = 0.25
SAMK[(c5 & c6 & c7 & c8)] = -1.25
SAMK[(c5 & c7 & c8) & ~c6] = -1.0
SAMK[(c5 & c7) & ~(c6 | c8)] = -0.5
SAMK[(c5 | c7) & ~(c5 & c7)] = -0.25

# --- SAMG calculation ---
c9 = close.rolling(window=18).mean()
c11 = close.shift(1).rolling(window=18).mean()

SAMG = pd.Series(0.0, index=close.index)
SAMG[(c9 > c11 * 1.0015) & (c9 > c11)] = 0.5
SAMG[(c9 < c11 * 1.0015) & (c9 > c11)] = -0.5
SAMG[(c9 > c11 / 1.0015) & (c9 <= c11)] = 0.5
SAMG[(c9 < c11 / 1.0015) & (c9 <= c11)] = -0.5

# --- SAMT calculation ---
c12 = close.rolling(window=6).mean()
c13 = close.shift(1).rolling(window=6).mean()
c19 = close.rolling(window=80).mean()

SAMT = pd.Series(0.0, index=close.index)
SAMT[(c12 > c13) & (c12 > c19)] = 0.5
SAMT[(c12 > c13) & (c12 <= c19)] = 0.25
SAMT[(c12 <= c13) & (c12 <= c19)] = -0.75
SAMT[(c12 <= c13) & (c12 > c19)] = -0.5

# --- MACD-based SAMM calculation ---
ema12 = close.ewm(span=12, adjust=False).mean()
ema26 = close.ewm(span=26, adjust=False).mean()
macd_line = ema12 - ema26
signal_line = macd_line.ewm(span=9, adjust=False).mean()

SAMM = pd.Series(0.0, index=close.index)
SAMM[(macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))] = 1.0
SAMM[(macd_line > signal_line) & (macd_line.shift(1) > signal_line.shift(1))] = 0.5
SAMM[(macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))] = -1.0
SAMM[(macd_line < signal_line) & (macd_line.shift(1) < signal_line.shift(1))] = -0.5

# --- SAMD (placeholder for DI[14], replaced with random example) ---
# In live trading, use a library like TA-Lib or pandas-ta for real DI(14)
SAMD = pd.Series(0.0, index=close.index)
SAMD[close.diff() > 0] = 0.5
SAMD[close.diff() <= 0] = -0.5

# --- TRIX-based SAMX calculation ---
trix = close.pct_change().ewm(span=15).mean()
trix_prev = trix.shift(1)

SAMX = pd.Series(0.0, index=close.index)
SAMX[(trix > 0) & (trix > trix_prev)] = 0.75
SAMX[(trix > 0) & (trix <= trix_prev)] = 0.5
SAMX[(trix < 0) & (trix < trix_prev)] = -0.75
SAMX[(trix < 0) & (trix >= trix_prev)] = -0.5

# --- Final SAM and weighted average ---
SAM = SAMK + SAMG + SAMT + SAMM + SAMD + SAMX
WMASAM = SAM.rolling(window=12).mean()

# --- Sensitivity adjustment ---
st.title("SAM Indicator (Simple Alert Monitor)")
sensitivity = st.slider("Signal Sensitivity", min_value=1, max_value=5, value=1)

# --- Generate signals based on sensitivity ---
signal = pd.Series(index=SAM.index, dtype="object")
prev_signal = None
counter = 0

for i in range(1, len(SAM)):
    if SAM[i] > 0 and WMASAM[i] > 0:
        if prev_signal != "Buy":
            counter += 1
            if counter >= sensitivity:
                signal[i] = "Buy"
                prev_signal = "Buy"
                counter = 0
    elif SAM[i] < 0 and WMASAM[i] < 0:
        if prev_signal != "Sell":
            counter += 1
            if counter >= sensitivity:
                signal[i] = "Sell"
                prev_signal = "Sell"
                counter = 0
    else:
        counter = 0
        signal[i] = "Hold"

# --- Plotting ---
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(SAM.index, SAM, label="SAM", color="blue", alpha=0.5)
ax.plot(WMASAM.index, WMASAM, label="Trend SAM", color="green", linestyle="--")

buy_signals = signal[signal == "Buy"]
sell_signals = signal[signal == "Sell"]

ax.scatter(buy_signals.index, SAM[buy_signals.index], marker="^", color="green", label="Buy Signal")
ax.scatter(sell_signals.index, SAM[sell_signals.index], marker="v", color="red", label="Sell Signal")

ax.set_title("SAM Indicator and Signals")
ax.legend()


#SAM Kernberekening

c1 = close > open_ 
c2 = close.shift(1) > open_.shift(1) 
c3 = close > close.shift(1) 
c4 = close.shift(1) > close.shift(2) 
c5 = close < open_ 
c6 = close.shift(1) < open_.shift(1) 
c7 = close < close.shift(1) 
c8 = close.shift(1) < close.shift(2)

samk = np.select( [c1 & c2 & c3 & c4, c1 & c3 & c4, c1 & c3, c1 | c3, c5 & c6 & c7 & c8, c5 & c7 & c8, c5 & c7, c5 | c7], [1.25, 1.0, 0.5, 0.25, -1.25, -1.0, -0.5, -0.25], default=0.0 )

wma = lambda s, period: s.rolling(period).apply(lambda x: np.average(x, weights=range(1, period+1)), raw=True)

c9 = wma(close, 18) 
c11 = wma(close.shift(1), 18) 
samg = np.select(
 [
        (c9 > (c11 * 1.0015)),
        (c9 < (c11 * 1.0015)) & (c9 > c11),
        (c9 > (c11 / 1.0015)) & (c9 <= c11),
        (c9 < (c11 / 1.0015)) & (c9 <= c11)
    ],
    [0.5, -0.5, 0.5, -0.5],
    default=0.0
)
c12 = wma(close, 6) 
c13 = wma(close.shift(1), 6) 
c19 = wma(close, 80) 
samt = np.select( 
 [
        (c12 > c13) & (c12 > c19),
        (c12 > c13) & (c12 <= c19), 
        (c12 <= c13) & (c12 <= c19), 
        (c12 <= c13) & (c12 > c19)
    ], 
    [0.5, 0.25, -0.75, -0.5], 
    default=0.0
)

macd_fast = close.ewm(span=12, adjust=False).mean() 
macd_slow = close.ewm(span=26, adjust=False).mean() 
macd_line = macd_fast - macd_slow 
macd_signal = macd_line.ewm(span=9, adjust=False).mean() 
samm = np.select( [macd_line > macd_signal, macd_line < macd_signal], [0.5, -0.5], default=0.0 )

sams = samk + samg + samm + samt 
sam_df = pd.DataFrame({ 'SAM': sams, 'Trend SAM': wma(pd.Series(sams), 12), 'Close': close })

#Signaalgevoeligheid

st.title("SAM Beleggingssignalen") 
st.sidebar.header("Instellingen") 
signal_sensitivity = st.sidebar.slider("Aantal opeenvolgende signalen voor melding", 1, 5, 1)

#Signaallogica
sam_signal = [] 
last_signal = None 
counter = 0 

for i in range(len(sam_df)): 
    if i < signal_sensitivity: 
        sam_signal.append("") 
        continue

    recent = sam_df["SAM"].iloc[i - signal_sensitivity + 1 : i + 1] 
    avg = recent.mean() 

    if avg > 0.5: 
        signal = "KOOP" 
    elif avg < -0.5: 
        signal = "VERKOOP" 
    else: 
        signal = "" 

    if signal != last_signal and signal != "": 
        sam_signal.append(signal) 
        last_signal = signal 
    else: 
        sam_signal.append("")

sam_df["Signaal"] = sam_signal

#Plot
fig, ax = plt.subplots()

# SAM als zwart histogram (staafdiagram)
ax.bar(sam_df.index, sam_df["SAM"], 
       color="black", label="SAM")

# TrendSAM als dikkere blauwe lijn
# Trendlijn voor TrendSAM met juiste X-waarden
ax.plot(sam_df.index, sam_df["TrendSAM"], 
        color="blue", label="TrendSAM")

ax.set_title("SAM Indicator")
ax.legend()
st.pyplot(fig)

#fig, ax = plt.subplots(figsize=(10, 5)) 
#ax.plot(sam_df['SAM'], 
#        label='SAM', color='black) 
#ax.plot(sam_df['Trend SAM'], 
#        label='Trend SAM', color='blue) 
#ax.set_title('SAM en Trend') 
#ax.legend() 
#st.pyplot(fig)

#Laat signalen zien

st.subheader("Laatste signalen") 
st.dataframe(sam_df[["SAM", "Trend SAM", "Signaal"]].tail(20))

