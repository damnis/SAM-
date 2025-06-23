sam_app.py

import streamlit as st 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

Simuleer koersdata

np.random.seed(0) days = 120 price = np.cumsum(np.random.normal(0, 1, days)) + 100 close = pd.Series(price) open_ = close.shift(1).fillna(method="bfill")

SAM Kernberekening

c1 = close > open_ c2 = close.shift(1) > open_.shift(1) c3 = close > close.shift(1) c4 = close.shift(1) > close.shift(2) c5 = close < open_ c6 = close.shift(1) < open_.shift(1) c7 = close < close.shift(1) c8 = close.shift(1) < close.shift(2)

samk = np.select( [c1 & c2 & c3 & c4, c1 & c3 & c4, c1 & c3, c1 | c3, c5 & c6 & c7 & c8, c5 & c7 & c8, c5 & c7, c5 | c7], [1.25, 1.0, 0.5, 0.25, -1.25, -1.0, -0.5, -0.25], default=0.0 )

wma = lambda s, period: s.rolling(period).apply(lambda x: np.average(x, weights=range(1, period+1)), raw=True)

c9 = wma(close, 18) c11 = wma(close.shift(1), 18) samg = np.select( [c9 > c11 * 1.0015, c9 < c11 * 1.0015 & (c9 > c11), c9 > c11 / 1.0015 & (c9 <= c11), c9 < c11 / 1.0015 & (c9 <= c11)], [0.5, -0.5, 0.5, -0.5], default=0.0 )

c12 = wma(close, 6) c13 = wma(close.shift(1), 6) c19 = wma(close, 80) samt = np.select( [c12 > c13 & c12 > c19, c12 > c13 & c12 <= c19, c12 <= c13 & c12 <= c19, c12 <= c13 & c12 > c19], [0.5, 0.25, -0.75, -0.5], default=0.0 )

macd_fast = close.ewm(span=12, adjust=False).mean() macd_slow = close.ewm(span=26, adjust=False).mean() macd_line = macd_fast - macd_slow macd_signal = macd_line.ewm(span=9, adjust=False).mean() samm = np.select( [macd_line > macd_signal, macd_line < macd_signal], [0.5, -0.5], default=0.0 )

sams = samk + samg + samm + samt sam_df = pd.DataFrame({ 'SAM': sams, 'Trend SAM': wma(pd.Series(sams), 12), 'Close': close })

Signaalgevoeligheid

st.title("SAM Beleggingssignalen") st.sidebar.header("Instellingen") signal_sensitivity = st.sidebar.slider("Aantal opeenvolgende signalen voor melding", 1, 5, 1)

Signaallogica

sam_signal = [] last_signal = None counter = 0 for i in range(len(sam_df)): if i < signal_sensitivity: sam_signal.append("") continue recent = sam_df["SAM"].iloc[i-signal_sensitivity+1:i+1] avg = recent.mean() if avg > 0.5: signal = "KOOP" elif avg < -0.5: signal = "VERKOOP" else: signal = "" if signal != last_signal and signal != "": sam_signal.append(signal) last_signal = signal else: sam_signal.append("")

sam_df["Signaal"] = sam_signal

Plot

fig, ax = plt.subplots(figsize=(10, 5)) ax.plot(sam_df['SAM'], label='SAM', color='blue') ax.plot(sam_df['Trend SAM'], label='Trend SAM', color='orange') ax.set_title('SAM en Trend') ax.legend() st.pyplot(fig)

Laat signalen zien

st.subheader("Laatste signalen") st.dataframe(sam_df[["SAM", "Trend SAM", "Signaal"]].tail(20))

