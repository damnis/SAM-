import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

def fetch_data(ticker, period):
    df = yf.download(ticker, period=period)
    return df

def calculate_indicators(df):
    df["SAMK"] = df["Close"].diff().rolling(window=3).mean()
    df["SAMG"] = df["Close"].pct_change().rolling(window=5).mean()
    df["SAMT"] = (df["Close"] - df["Close"].shift(5)) / df["Close"].shift(5)
    df["SAMD"] = df["Close"] - df["Low"].rolling(window=10).min()
    df["SAMM"] = df["High"].rolling(window=10).max() - df["Close"]
    df["SAMX"] = df["Volume"].rolling(window=5).mean() / df["Volume"].rolling(window=20).mean()

    df["SAM"] = (
        df[["SAMK", "SAMG", "SAMT", "SAMD", "SAMM", "SAMX"]]
        .mean(axis=1)
        .fillna(0)
    )
    return df

def generate_signals_with_returns(df, sensitivity):
    df = df.copy()
    df["Advies"] = ""
    df["Slotkoers"] = df["Close"]
    df["SAM rendement"] = ""
    df["Marktrendement"] = ""

    previous_signal = None
    start_index = None
    start_sam = None
    start_koers = None

    for idx in range(1, len(df)):
        change = df["SAM"].iloc[idx] - df["SAM"].iloc[idx - 1]
        signal = previous_signal

        if abs(change) > sensitivity:
            signal = "Kopen" if change > 0 else "Verkopen"

        df.at[df.index[idx], "Advies"] = signal

        if signal != previous_signal:
            if (
                previous_signal is not None
                and start_index is not None
                and start_koers is not None
                and start_sam is not None
                and float(start_koers) != 0.0
            ):
                eind_koers = df["Close"].iloc[idx]
                eind_sam = df["SAM"].iloc[idx]
                sam_rend = (eind_sam - start_sam) / abs(start_sam) if abs(start_sam) > 0 else 0
                markt_rend = (eind_koers - start_koers) / start_koers if start_koers != 0 else 0

                df.at[df.index[idx], "SAM rendement"] = f"{sam_rend * 100:.2f}%"
                df.at[df.index[idx], "Marktrendement"] = f"{markt_rend * 100:.2f}%"

            start_index = idx
            start_sam = df["SAM"].iloc[idx]
            start_koers = df["Close"].iloc[idx]

        previous_signal = signal

    return df

def main():
    st.title("SAM Beleggingsadvies App")

    ticker = st.text_input("Ticker (bijv. AAPL)", "AAPL")
    period = st.selectbox("Periode", ["1mo", "3mo", "6mo", "1y", "2y"], index=1)
    sensitivity = st.slider("Gevoeligheid (hoe lager, hoe sneller een signaal)", 0.001, 1.0, 0.01)

    df = fetch_data(ticker, period)

    if df.empty:
        st.error("Geen data gevonden voor deze ticker.")
        return

    df = calculate_indicators(df)
    df = generate_signals_with_returns(df, sensitivity)

    st.subheader("Adviezen en rendement")
    st.dataframe(df[["Close", "Advies", "Slotkoers", "SAM rendement", "Marktrendement"]].dropna(how="all"))

    # SAM-grafiek alleen tonen als er bruikbare waarden zijn
    if not df.empty and "SAM" in df.columns and df["SAM"].dropna().shape[0] > 0:
        fig, ax = plt.subplots()
        df["SAM"].plot(kind="bar", ax=ax, color="skyblue", width=1)
        ax.set_title("SAM Indicator")
        ax.set_ylabel("Waarde")
        st.pyplot(fig)
    else:
        st.warning("Niet genoeg gegevens om de SAM-indicator weer te geven.")

if __name__ == "__main__":
    main()
