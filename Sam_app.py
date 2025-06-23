import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

st.title("📈 SAM Indicator App")
st.markdown("Een eenvoudige trading indicator gebaseerd op candles, prijs, momentum, trend en dynamiek.")

# --- Ticker selectie ---
ticker = st.selectbox("Selecteer een ticker:", ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA"], index=0)

# --- Data ophalen ---
data = yf.download(ticker, period="180d", interval="1d")

if data.empty:
    st.error(f"❌ Geen data gevonden voor ticker: {ticker}")
    st.stop()

required_cols = ["Open", "High", "Low", "Close"]
missing_cols = [col for col in required_cols if col not in data.columns]
if missing_cols:
    st.error(f"❌ Ontbrekende kolommen in koersdata: {', '.join(missing_cols)}")
    st.stop()

# --- Variabelen toewijzen ---
open_ = data["Open"]
high = data["High"]
low = data["Low"]
close = data["Close"]
dates = data.index

# --- DataFrame opbouwen ---
df = pd.DataFrame({
    "Open": open_,
    "High": high,
    "Low": low,
    "Close": close
})

# --- SAMK ---
df["c1"] = df["Close"] > df["Open"]
df["c2"] = df["Close"].shift(1) > df["Open"].shift(1)
df["c3"] = df["Close"] > df["Close"].shift(1)
df["c4"] = df["Close"].shift(1) > df["Close"].shift(2)
df["c5"] = df["Close"] < df["Open"]
df["c6"] = df["Close"].shift(1) < df["Open"].shift(1)
df["c7"] = df["Close"] < df["Close"].shift(1)
df["c8"] = df["Close"].shift(1) < df["Close"].shift(2)

df["SAMK"] = 0.0
df.loc[(df["c1"] & df["c2"] & df["c3"] & df["c4"]).fillna(False), "SAMK"] = 1.25
df.loc[((df["c1"] & df["c3"] & df["c4"]) & ~df["c2"]).fillna(False), "SAMK"] = 1.0
df.loc[((df["c1"] & df["c3"]) & ~(df["c2"] | df["c4"])).fillna(False), "SAMK"] = 0.5
df.loc[((df["c1"] | df["c3"]) & ~(df["c1"] & df["c3"])).fillna(False), "SAMK"] = 0.25
df.loc[(df["c5"] & df["c6"] & df["c7"] & df["c8"]).fillna(False), "SAMK"] = -1.25
df.loc[((df["c5"] & df["c7"] & df["c8"]) & ~df["c6"]).fillna(False), "SAMK"] = -1.0
df.loc[((df["c5"] & df["c7"]) & ~(df["c6"] | df["c8"])).fillna(False), "SAMK"] = -0.5
df.loc[((df["c5"] | df["c7"]) & ~(df["c5"] & df["c7"])).fillna(False), "SAMK"] = -0.25

# --- SAMX ---
df["price_change"] = df["Close"] - df["Close"].shift(1)
df["SAMX"] = 0.0
df.loc[(df["price_change"] > 1.5).fillna(False), "SAMX"] = 1.25
df.loc[((df["price_change"] > 1.0) & (df["price_change"] <= 1.5)).fillna(False), "SAMX"] = 1.0
df.loc[((df["price_change"] > 0.5) & (df["price_change"] <= 1.0)).fillna(False), "SAMX"] = 0.5
df.loc[((df["price_change"] < -1.5)).fillna(False), "SAMX"] = -1.25
df.loc[((df["price_change"] < -1.0) & (df["price_change"] >= -1.5)).fillna(False), "SAMX"] = -1.0
df.loc[((df["price_change"] < -0.5) & (df["price_change"] >= -1.0)).fillna(False), "SAMX"] = -0.5

# --- SAMM ---
df["momentum"] = df["Close"] - df["Close"].shift(3)
df["SAMM"] = 0.0
df.loc[(df["momentum"] > 2.0).fillna(False), "SAMM"] = 1.25
df.loc[((df["momentum"] > 1.0) & (df["momentum"] <= 2.0)).fillna(False), "SAMM"] = 1.0
df.loc[((df["momentum"] > 0.5) & (df["momentum"] <= 1.0)).fillna(False), "SAMM"] = 0.5
df.loc[((df["momentum"] < -2.0)).fillna(False), "SAMM"] = -1.25
df.loc[((df["momentum"] < -1.0) & (df["momentum"] >= -2.0)).fillna(False), "SAMM"] = -1.0
df.loc[((df["momentum"] < -0.5) & (df["momentum"] >= -1.0)).fillna(False), "SAMM"] = -0.5

# --- SAMT ---
df["trend"] = df["Close"].rolling(window=5).mean() - df["Close"].rolling(window=20).mean()
df["SAMT"] = 0.0
df.loc[(df["trend"] > 1.0).fillna(False), "SAMT"] = 1
df.loc[(df["trend"] < -1.0).fillna(False), "SAMT"] = -1

# --- SAMD ---
df["volatility"] = df["Close"].rolling(window=5).std()
df["SAMD"] = 0.0
df.loc[(df["volatility"] > 2.0).fillna(False), "SAMD"] = 1
df.loc[(df["volatility"] < 1.0).fillna(False), "SAMD"] = -1

# --- SAM = som van alle componenten ---
df["SAM"] = df[["SAMK", "SAMX", "SAMM", "SAMT", "SAMD"]].sum(axis=1)
df["TrendSAM"] = df["SAM"].rolling(window=5).mean()

# --- Plotten ---
fig, ax = plt.subplots(figsize=(12, 6))

ax.bar(df.index, df["SAM"], color="black", label="SAM Histogram")
ax.plot(df.index, df["TrendSAM"], color="blue", linewidth=2.5, label="TrendSAM")

ax.set_title(f"SAM Indicator voor {ticker}")
ax.set_ylabel("Waarde")
ax.legend()
ax.grid(True)

st.pyplot(fig)
