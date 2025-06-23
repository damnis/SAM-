import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

st.title("SAM Indicator - Koop/Verkoop Advies")

# Sidebar instellingen
st.sidebar.header("Instellingen")
ticker = st.sidebar.text_input("Ticker (bv. AAPL):", value="AAPL")
start_date = st.sidebar.date_input("Startdatum:", value=pd.to_datetime("2013-01-01"))
end_date = st.sidebar.date_input("Einddatum:", value=pd.to_datetime("today"))
sensitivity = st.sidebar.slider("Gevoeligheid (hogere waarde = trager signaal)", min_value=1, max_value=20, value=5)

@st.cache_data
def download_data(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date)
    df = df[['Open', 'High', 'Low', 'Close']]
    return df

def calculate_sam(df):
    df = df.copy()

    # Componenten van SAM
    df["c1"] = df["Close"] > df["Open"]
    df["c2"] = df["Close"].shift(1) > df["Open"].shift(1)
    df["c3"] = df["Close"] > df["Close"].shift(1)
    df["c4"] = df["Open"] > df["Open"].shift(1)
    df["c5"] = df["High"] > df["High"].shift(1)
    df["c6"] = df["Low"] > df["Low"].shift(1)

    # SAMK
    df["SAMK"] = 0.0
    df.loc[(df["c1"] & df["c2"] & df["c3"] & df["c4"] & df["c5"] & df["c6"]).fillna(False), "SAMK"] = 1.25

    # SAMG - groei
    df["SAMG"] = ((df["Close"] - df["Close"].shift(1)) / df["Close"].shift(1)).rolling(window=3).mean()

    # SAMT - trend
    df["SAMT"] = df["Close"].rolling(window=5).mean().diff()

    # SAMD - verschil tussen high en low
    df["SAMD"] = (df["High"] - df["Low"]).rolling(window=3).mean()

    # SAMM - momentum
    df["SAMM"] = df["Close"] - df["Close"].rolling(window=10).mean()

    # SAMX - extreme bewegingen
    df["SAMX"] = df["Close"].pct_change().abs().rolling(window=5).mean()

    # SAM score (gestandaardiseerd)
    df["SAM"] = (df["SAMK"] + df["SAMG"] + df["SAMT"] + df["SAMD"] + df["SAMM"] + df["SAMX"]) / 6
    df["SAM"] = df["SAM"].fillna(0)

    return df

def generate_signals_with_returns(df, sensitivity):
    df = df.copy()
    df["Advies"] = ""
    df["Rendement"] = ""
    df["Marktrendement"] = ""
    df["Slotkoers"] = df["Close"]

    previous_signal = ""
    start_koers = None
    start_index = None

    for idx in range(1, len(df)):
        change = df.iloc[idx]["SAM"] - df.iloc[idx - 1]["SAM"]
        signal = previous_signal

        if change > 1 / sensitivity:
            signal = "Kopen"
        elif change < -1 / sensitivity:
            signal = "Verkopen"

        df.at[df.index[idx], "Advies"] = signal

        if signal != previous_signal and previous_signal != "":
            if start_index is not None:
                eind_koers = df.iloc[idx - 1]["Close"]
                if start_koers and eind_koers:
                    sam_rend = (eind_koers - start_koers) / start_koers * (1 if previous_signal == "Kopen" else -1)
                    markt_rend = (eind_koers - start_koers) / start_koers
                    df.at[df.index[idx - 1], "Rendement"] = f"{sam_rend * 100:.2f}%"
                    df.at[df.index[idx - 1], "Marktrendement"] = f"{markt_rend * 100:.2f}%"
            start_koers = df.iloc[idx]["Close"]
            start_index = idx

        elif previous_signal == "":
            start_koers = df.iloc[idx]["Close"]
            start_index = idx

        previous_signal = signal

    return df

# Ophalen en berekenen
df = download_data(ticker, start_date, end_date)
df = calculate_sam(df)
df = generate_signals_with_returns(df, sensitivity)

# Plot
st.subheader("SAM Histogram")
fig, ax = plt.subplots(figsize=(10, 4))
df["SAM"].plot(kind="bar", ax=ax, color="skyblue", width=1)
plt.xticks([], [])
plt.xlabel("Tijd")
plt.ylabel("SAM Waarde")
plt.tight_layout()
st.pyplot(fig)

# Tabel
st.subheader("Adviezen en Resultaten")
st.dataframe(df[["Close", "Advies", "Rendement", "Marktrendement"]].dropna(how="all"))
