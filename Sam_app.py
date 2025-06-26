import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Functie om data op te halen ---
def fetch_data(ticker, interval):
    import yfinance as yf
    import pandas as pd

    # Intervalspecifieke periode instellen
    if interval == "15m":
        period = "7d"
    elif interval == "1h":
        period = "30d"
    elif interval == "4h":
        period = "60d"
    elif interval == "1d":
        period = "360d"
    else:
        period = "360wk"

    # Data ophalen
    df = yf.download(ticker, interval=interval, period=period)

    # Verwijder rijen zonder volume of zonder koersverandering
    df = df[
        (df["Volume"] > 0) &
        ((df["Open"] != df["Close"]) | (df["High"] != df["Low"]))
    ]

    # Zorg dat de index datetime is en verwijder ongeldige datums
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()]

    return df

    
# --- SAM Indicatorberekeningen ---
def calculate_sam(df):
    df = df.copy()

    # Basiskolommen
    df["c1"] = df["Close"] > df["Open"]
    df["c2"] = df["Close"].shift(1) > df["Open"].shift(1)
    df["c3"] = df["Close"].shift(2) > df["Open"].shift(2)
    df["c4"] = df["Close"].shift(3) > df["Open"].shift(3)
    df["c5"] = df["Close"].shift(4) > df["Open"].shift(4)
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

    # Totale SAM
    df["SAM"] = df[["SAMK", "SAMG", "SAMT", "SAMD", "SAMM", "SAMX"]].sum(axis=1)

    return df

# --- Advies en rendementen ---
def determine_advice(df, threshold):
    df = df.copy()
    df["Trend"] = df["SAM"].rolling(window=3).mean()
    df["TrendChange"] = df["Trend"] - df["Trend"].shift(1)

    df["Advies"] = np.nan
    df.loc[df["TrendChange"] > threshold, "Advies"] = "Kopen"
    df.loc[df["TrendChange"] < -threshold, "Advies"] = "Verkopen"
    df["Advies"] = df["Advies"].ffill()

    df["AdviesGroep"] = (df["Advies"] != df["Advies"].shift()).cumsum()
    rendementen = []
    sam_rendementen = []

    for _, groep in df.groupby("AdviesGroep"):
        start = groep["Close"].iloc[0]
        eind = groep["Close"].iloc[-1]
        advies = groep["Advies"].iloc[0]

        markt_rendement = (eind - start) / start
        sam_rendement = markt_rendement if advies == "Kopen" else -markt_rendement

        rendementen.extend([markt_rendement] * len(groep))
        sam_rendementen.extend([sam_rendement] * len(groep))

    df["Markt-%"] = rendementen
    df["SAM-%"] = sam_rendementen

    # Huidig advies bepalen
    if "Advies" in df.columns and df["Advies"].notna().any():
        huidig_advies = df["Advies"].dropna().iloc[-1]
    else:
        huidig_advies = "Niet beschikbaar"

    return df, huidig_advies
    
# --- Streamlit UI ---
st.title("ðŸ“Š SAM Trading Indicator")

# --- Volledige tickerlijsten ---
aex_tickers = {
    "ABN.AS": "ABN AMRO",
    "ADYEN.AS": "Adyen",
    "AGN.AS": "Aegon",
    "AD.AS": "Ahold Delhaize",
    "AKZA.AS": "Akzo Nobel",
    "MT.AS": "ArcelorMittal",
    "ASM.AS": "ASMI",
    "ASML.AS": "ASML",
    "ASRNL.AS": "ASR Nederland",
    "BESI.AS": "BESI",
    "DSFIR.AS": "DSM-Firmenich",
    "GLPG.AS": "Galapagos",
    "HEIA.AS": "Heineken",
    "IMCD.AS": "IMCD",
    "INGA.AS": "ING Groep",
    "TKWY.AS": "Just Eat Takeaway",
    "KPN.AS": "KPN",
    "NN.AS": "NN Group",
    "PHIA.AS": "Philips",
    "PRX.AS": "Prosus",
    "RAND.AS": "Randstad",
    "REN.AS": "Relx",
    "SHELL.AS": "Shell",
    "UNA.AS": "Unilever",
    "WKL.AS": "Wolters Kluwer"
}

dow_tickers = {
    'MMM': '3M', 'AXP': 'American Express', 'AMGN': 'Amgen', 'AAPL': 'Apple', 'BA': 'Boeing',
    'CAT': 'Caterpillar', 'CVX': 'Chevron', 'CSCO': 'Cisco', 'KO': 'Coca-Cola', 'DIS': 'Disney',
    'GS': 'Goldman Sachs', 'HD': 'Home Depot', 'HON': 'Honeywell', 'IBM': 'IBM', 'INTC': 'Intel',
    'JPM': 'JPMorgan Chase', 'JNJ': 'Johnson & Johnson', 'MCD': 'McDonaldâ€™s', 'MRK': 'Merck',
    'MSFT': 'Microsoft', 'NKE': 'Nike', 'PG': 'Procter & Gamble', 'CRM': 'Salesforce',
    'TRV': 'Travelers', 'UNH': 'UnitedHealth', 'VZ': 'Verizon', 'V': 'Visa', 'WMT': 'Walmart',
    'DOW': 'Dow', 'RTX': 'RTX Corp.', 'WBA': 'Walgreens Boots'
}
nasdaq_tickers = {
    'MSFT': 'Microsoft', 'NVDA': 'NVIDIA', 'AAPL': 'Apple', 'AMZN': 'Amazon', 'META': 'Meta',
    'NFLX': 'Netflix', 'GOOG': 'Google', 'GOOGL': 'Alphabet', 'TSLA': 'Tesla', 'CSCO': 'Cisco',
    'INTC': 'Intel', 'ADBE': 'Adobe', 'CMCSA': 'Comcast', 'PEP': 'PepsiCo', 'COST': 'Costco',
    'AVGO': 'Broadcom', 'QCOM': 'Qualcomm', 'TMUS': 'T-Mobile', 'TXN': 'Texas Instruments',
    'AMAT': 'Applied Materials'
}

ustech_tickers = {
    "SMCI": "Super Micro Computer",
    "PLTR": "Palantir",
    "ORCL": "Oracle",
    "SNOW": "Snowflake",
    "NVDA": "NVIDIA",
    "AMD": "AMD",
    "MDB": "MongoDB",
    "DDOG": "Datadog",
    "CRWD": "CrowdStrike",
    "ZS": "Zscaler",
    "TSLA": "Tesla",
    "AAPL": "Apple",
    "GOOGL": "Alphabet (GOOGL)",
    "MSFT": "Microsoft"
}

# --- Toevoeging tickers AMX & Crypto ---
amx_tickers = {
    "AMG.AS": "AMG", "ARCAD.AS": "Arcadis", "BAMNB.AS": "BAM Groep",
    "BPOST.AS": "BPost", "FAGR.AS": "Fagron", "FUR.AS": "Fugro", "KENDR.AS": "Kendrion",
    "SBMO.AS": "SBM Offshore", "TKWY.AS": "Just Eat", "VASTN.AS": "Vastned Retail"
}

crypto_tickers = {
    "BTC-USD": "Bitcoin", "ETH-USD": "Ethereum", "SOL-USD": "Solana",
    "BNB-USD": "BNB", "XRP-USD": "XRP", "DOGE-USD": "Dogecoin"
}
# --- Update tab labels en bijbehorende mapping ---
tabs_mapping = {
    "ðŸ‡ºðŸ‡¸ Dow Jones": dow_tickers,
    "ðŸ‡ºðŸ‡¸ Nasdaq": nasdaq_tickers,
    "ðŸ‡ºðŸ‡¸ US Tech": ustech_tickers,
    "ðŸ‡³ðŸ‡± AEX": aex_tickers,
    "ðŸ‡³ðŸ‡± AMX": amx_tickers,
    "ðŸŒ Crypto": crypto_tickers
}

tab_labels = list(tabs_mapping.keys())
selected_tab = st.radio("Kies beurs", tab_labels, horizontal=True)

valutasymbool = {
    "ðŸ‡³ðŸ‡± AEX": "â‚¬ ",
    "ðŸ‡³ðŸ‡± AMX": "â‚¬ ",
    "ðŸ‡ºðŸ‡¸ Dow Jones": "$ ",
    "ðŸ‡ºðŸ‡¸ Nasdaq": "$ ",
    "ðŸ‡ºðŸ‡¸ US Tech": "$ ",
    "ðŸŒ Crypto": "",  # Geen symbool
}.get(selected_tab, "")

# Dan in je tekst:
#f"{ticker_name} â€“ {valutasymbool}{last:.2f}"

# --- Data ophalen voor dropdown live view ---
#def get_live_ticker_data(tickers_dict):
# --- Data ophalen voor dropdown live view ---
def get_live_ticker_data(tickers_dict):
    tickers = list(tickers_dict.keys())
    data = yf.download(tickers, period="1d", interval="1d", progress=False, group_by='ticker')
    result = []

    for ticker in tickers:
        try:
            last = data[ticker]['Close'].iloc[-1]
            prev = data[ticker]['Open'].iloc[-1]
            change = (last - prev) / prev * 100
            kleur = "#00FF00" if change > 0 else "#FF0000" if change < 0 else "#808080"
            naam = tickers_dict[ticker]
            result.append((ticker, naam, last, change, kleur))
        except Exception:
            continue

    return result

# --- Weergave dropdown met live info ---
live_info = get_live_ticker_data(tabs_mapping[selected_tab])
# --- Dropdown dictionary voorbereiden ---
dropdown_dict = {}  # key = ticker, value = (display_tekst, naam)

for t, naam, last, change, kleur in live_info:
    emoji = "ðŸŸ¢" if change > 0 else "ðŸ”´" if change < 0 else "âšª"
    display = f"{t} - {naam} | {valutasymbool}{last:.2f} {emoji} {change:+.2f}%"
    dropdown_dict[t] = (display, naam)

# --- Bepalen van de juiste default key voor selectie
# Herstel vorige selectie als deze nog bestaat
default_ticker_key = st.session_state.get(f"ticker_select_{selected_tab}")
if default_ticker_key not in dropdown_dict:
    default_ticker_key = list(dropdown_dict.keys())[0]  # fallback naar eerste

# --- Dropdown zelf ---
selected_ticker = st.selectbox(
    f"Selecteer {selected_tab} ticker:",
    options=list(dropdown_dict.keys()),  # alleen tickers als stabiele optie-key
    format_func=lambda x: dropdown_dict[x][0],  # toon koers etc.
    key=f"ticker_select_{selected_tab}",
    index=list(dropdown_dict.keys()).index(default_ticker_key)
)

# --- Ophalen ticker info ---
ticker = selected_ticker
ticker_name = dropdown_dict[ticker][1]


# --- Opgehaalde waarden ---
#ticker = selected_ticker
#ticker_name = dropdown_dict[ticker][1]

# --- Live koers opnieuw ophalen voor de geselecteerde ticker ---
try:
    live_data = yf.download(ticker, period="1d", interval="1d", progress=False)
    last = live_data["Close"].iloc[-1]
except Exception:
    last = 0.0  # fallback


# --- Andere instellingen ---
# --- Intervalopties ---
interval_optie = st.selectbox(
    "Kies de interval",
    ["Dagelijks", "Wekelijks", "4-uur", "1-uur", "15-minuten"]
)

# Vertaal gebruikerskeuze naar Yahoo Finance intervalcode
interval_mapping = {
    "Dagelijks": "1d",
    "Wekelijks": "1wk",
    "4-uur": "4h",
    "1-uur": "1h",
    "15-minuten": "15m"
}

interval = interval_mapping[interval_optie]


thresh = st.slider("Gevoeligheid van trendverandering", 0.01, 2.0, 0.5, step=0.01)

# Berekening
df = fetch_data(ticker, interval)
df = calculate_sam(df)
# df = determine_advice(df, threshold=thresh)
df, huidig_advies = determine_advice(df, threshold=thresh)

# Grafieken
#st.subheader(f"SAM-indicator en trend voor {ticker}")
# Huidige advies ophalen
# huidig_advies = df["Advies"].dropna().iloc[-1]

# Kleur bepalen op basis van advies
advies_kleur = "green" if huidig_advies == "Kopen" else "red" if huidig_advies == "Verkopen" else "gray"

# Titel met kleur en grootte tonen
st.markdown(
#    f"""
#    <h3>SAM-indicator en trend voor <span style='color:#3366cc'>{ticker_name} - ${last:.2f}</span></h3>
#    <h2 style='color:{advies_kleur}'>Huidig advies: {huidig_advies}</h2>
#    """,
#    unsafe_allow_html=True
#)
#st.markdown(
    f"""
    <h3>SAM-indicator en trend voor <span style='color:#3366cc'>{ticker_name}</span></h3>
    <h2 style='color:{advies_kleur}'>Huidig advies: {huidig_advies}</h2>
    """,
    unsafe_allow_html=True
)

import matplotlib.pyplot as plt
import streamlit as st

# --- Grafiek met SAM en Trend ---
fig, ax1 = plt.subplots(figsize=(10, 4))
ax1.bar(df.index, df["SAM"], color="lightblue", label="SAM")
ax2 = ax1.twinx()
ax2.plot(df.index, df["Trend"], color="red", label="Trend")
ax1.set_ylabel("SAM")
ax2.set_ylabel("Trend")
fig.tight_layout()
st.pyplot(fig)

# --- Tabel met signalen en rendement ---
st.subheader("Laatste signalen en rendement")

# Kolommen selecteren en formatteren
kolommen = ["Close", "Advies", "SAM", "Trend", "Markt-%", "SAM-%"]
#tabel = df[kolommen].dropna().tail(30).round(3).copy()
tabel = df[kolommen].dropna().copy()
tabel = tabel.sort_index(ascending=False).head(20)

# Datumkolom aanmaken vanuit index
if not isinstance(tabel.index, pd.DatetimeIndex):
    tabel.index = pd.to_datetime(tabel.index, errors="coerce")
tabel = tabel[~tabel.index.isna()]
tabel["Datum"] = tabel.index.strftime("%d-%m-%Y")

# Zet kolomvolgorde
tabel = tabel[["Datum"] + kolommen]

# Afronding en formatting
if selected_tab == "ðŸŒ Crypto":
    tabel["Close"] = tabel["Close"].map("{:.3f}".format)
else:
    tabel["Close"] = tabel["Close"].map("{:.2f}".format)

tabel["SAM"] = tabel["SAM"].map("{:.2f}".format)
tabel["Trend"] = tabel["Trend"].map("{:.3f}".format)

tabel["Markt-%"] = (tabel["Markt-%"].astype(float) * 100).map("{:+.2f}%".format)
tabel["SAM-%"] = (tabel["SAM-%"].astype(float) * 100).map("{:+.2f}%".format)


# HTML-rendering
html = """
<style>
    table {
        border-collapse: collapse;
        width: 100%;
        font-family: Arial, sans-serif;
        font-size: 14px;
    }
    th {
        background-color: #004080;
        color: white;
        padding: 6px;
        text-align: center;
    }
    td {
        border: 1px solid #ddd;
        padding: 6px;
        text-align: right;
        background-color: #f9f9f9;
        color: #222222;
    }
    tr:nth-child(even) td {
        background-color: #eef2f7;
    }
    tr:hover td {
        background-color: #d0e4f5;
    }
</style>
<table>
    <thead>
        <tr>
            <th style='width: 110px;'>Datum</th>
            <th style='width: 80px;'>Close</th>
            <th style='width: 90px;'>Advies</th>
            <th style='width: 60px;'>SAM</th>
            <th style='width: 70px;'>Trend</th>
            <th style='width: 90px;'>Markt-%</th>
            <th style='width: 90px;'>SAM-%</th>
        </tr>
    </thead>
    <tbody>
"""

# Voeg rijen toe aan de tabel
for _, row in tabel.iterrows():
    html += "<tr>"
    for value in row:
        html += f"<td>{value}</td>"
    html += "</tr>"

html += "</tbody></table>"

#Weergave in Streamlit
st.markdown(html, unsafe_allow_html=True)

###--- Toevoeging: Backtestfunctie ---

from datetime import date
import pandas as pd
import streamlit as st

# ðŸ§¼ Zorg dat index datetime is
df = df.copy()
df.index = pd.to_datetime(df.index)

# ðŸ“… 1. Datumkeuze
st.subheader("ðŸ“… Vergelijk Marktrendement en SAM-rendement")

default_start = df.index.min().date()
default_end = df.index.max().date()

start_date = st.date_input("Startdatum analyse", default_start)
end_date = st.date_input("Einddatum analyse", default_end)

# ðŸ“ 2. Signaalkeuze
signaalkeuze = st.selectbox(
    "Welke signalen tellen mee voor SAM-rendement?",
    options=["Beide", "Koop", "Verkoop"],
    index=0
)

# ðŸ§½ 3. Filter op periode
df_period = df.loc[
    (df.index.date >= start_date) & (df.index.date <= end_date)
].copy()

# ðŸ”§ Flat multi-index kolommen (indien nodig)
if isinstance(df_period.columns, pd.MultiIndex):
    df_period.columns = ["_".join([str(i) for i in col if i]) for col in df_period.columns]

# ðŸ”Ž Zoek juiste Close-kolom
close_col = next((col for col in df_period.columns if col.lower().startswith("close")), None)

# ðŸ” Debug
st.write("âœ… DEBUG: df_period shape:", df_period.shape)
#st.write("âœ… DEBUG: Columns in df_period:", df_period.columns.tolist())
st.write("âœ… DEBUG: Gekozen kolom voor 'Close':", close_col)
st.write("âœ… DEBUG: Eerste rijen df_period:")
st.dataframe(df_period.head(20))

# â›‘ï¸ Verwerk 'Close'-data
df_valid = pd.Series([], dtype=float)

if close_col:
    df_period[close_col] = pd.to_numeric(df_period[close_col], errors="coerce")
    df_valid = df_period[close_col].dropna()
    df_period = df_period.dropna(subset=[close_col])
else:
    st.warning("â— Geen geldige 'Close'-kolom gevonden in de data.")
    df_period = pd.DataFrame(columns=df.columns)

st.write("âœ… DEBUG: Lengte df_valid:", len(df_valid))
#st.write("âœ… DEBUG: Eerste 5 waarden in df_valid:", df_valid.head())

# ðŸ“Š 4. Marktrendement
marktrendement = None
if not df_valid.empty and len(df_valid) >= 2:
    koers_start = df_valid.iloc[0]
    koers_eind = df_valid.iloc[-1]
    if koers_start != 0.0:
        marktrendement = ((koers_eind - koers_start) / koers_start) * 100

# âœ‚ï¸ 5. Filter op geldige adviezen
advies_col = "Advies"
df_signalen = df_period[df_period[advies_col].isin(["Kopen", "Verkopen"])].copy()
# ðŸ”§ Zorg dat df_signalen een 'Close'-kolom bevat
# Herhaal eventueel de kolomextractie (zoals eerder gedaan bij df_period)
close_col = [col for col in df_signalen.columns if str(col).startswith("Close")]
if close_col:
    df_signalen = df_signalen.rename(columns={close_col[0]: "Close"})
else:
    st.error("â— Geen kolom gevonden die begint met 'Close' in df_signalen.")

if signaalkeuze == "Koop":
    df_signalen = df_signalen[df_signalen[advies_col] == "Kopen"]
elif signaalkeuze == "Verkoop":
    df_signalen = df_signalen[df_signalen[advies_col] == "Verkopen"]

# TEST: handmatig test-signalen invoegen
#df_signalen = pd.DataFrame({
#    "Advies": ["Kopen", "Kopen", "Verkopen", "Verkopen", "Kopen", "Verkopen", "Kopen", "Verkopen", "Kopen", "Kopen", "Verkopen"],
#    "Close": [100, 105, 104, 102, 98, 100, 105, 104, 102, 98, 95]
#}, index=pd.date_range("2025-01-01", periods=11))
#close_col = "Close"

# ðŸ§  6. SAM-berekening
def bereken_sam_rendement(df_signalen, signaal_type="Beide"):
    rendementen = []
    trades = []
    entry_price = None
    entry_date = None
    entry_type = None

    type_map = {"Koop": "Kopen", "Verkoop": "Verkopen", "Beide": "Beide"}
    mapped_type = type_map.get(signaal_type, "Beide")

    for datum, row in df_signalen.iterrows():
        advies = row["Advies"]
        close = row["Close"]

        if entry_type is None:
            if mapped_type == "Beide" or advies == mapped_type:
                entry_type = advies
                entry_price = close
                entry_date = datum
        else:
            if advies != entry_type and (mapped_type == "Beide" or entry_type == mapped_type):
                # Sluit trade
                if entry_type == "Kopen":
                    rendement = (close - entry_price) / entry_price * 100
                else:
                    rendement = (entry_price - close) / entry_price * 100

                rendementen.append(rendement)
                trades.append({
                    "Type": entry_type,
                    "Open datum": entry_date.strftime("%d-%m-%Y"),
                    "Open prijs": round(entry_price, 2),
                    "Sluit datum": datum.strftime("%d-%m-%Y"),
                    "Sluit prijs": round(close, 2),
                    "Rendement (%)": round(rendement, 2)
                })

                # Start nieuwe trade als huidige advies binnen mapping valt
                if mapped_type == "Beide" or advies == mapped_type:
                    entry_type = advies
                    entry_price = close
                    entry_date = datum
                else:
                    entry_type = None
                    entry_price = None
                    entry_date = None

    # âš ï¸ Sluit openstaande trade aan het einde van de periode
    if entry_type is not None and entry_price is not None:
        laatste_datum = df_signalen.index[-1]
        laatste_koers = df_signalen["Close"].iloc[-1]

        if entry_type == "Kopen":
            rendement = (laatste_koers - entry_price) / entry_price * 100
        else:
            rendement = (entry_price - laatste_koers) / entry_price * 100

        rendementen.append(rendement)
        trades.append({
            "Type": entry_type,
            "Open datum": entry_date.strftime("%d-%m-%Y"),
            "Open prijs": round(entry_price, 2),
            "Sluit datum": laatste_datum.strftime("%d-%m-%Y"),
            "Sluit prijs": round(laatste_koers, 2),
            "Rendement (%)": round(rendement, 2)
        })

    sam_rendement = sum(rendementen) if rendementen else 0.0
    return sam_rendement, trades, rendementen
    
sam_rendement, trades, rendementen = bereken_sam_rendement(df_signalen, signaalkeuze)

# ðŸ“ˆ 7. Resultaten tonen
col1, col2 = st.columns(2)

if isinstance(marktrendement, (int, float)):
    col1.metric("Marktrendement (Buy & Hold)", f"{marktrendement:+.2f}%")
else:
    col1.metric("Marktrendement (Buy & Hold)", "n.v.t.")

if isinstance(sam_rendement, (int, float)):
    col2.metric("ðŸ“ˆ SAM-rendement", f"{sam_rendement:+.2f}%")
    st.caption(f"Aantal afgeronde trades: **{len(trades)}** binnen deze periode.")
else:
    col2.metric("ðŸ“ˆ SAM-rendement", "n.v.t.")

# ðŸ§ª Debug
st.write("ðŸ” DEBUG - Signaalkeuze:", signaalkeuze)
st.write("ðŸ” Aantal signalen:", len(df_signalen))
st.write("ðŸ” Unieke adviezen:", df_signalen[advies_col].unique())
st.write("ðŸ” Aantal trades:", len(trades))
st.write("ðŸ” Rendementenlijst:", rendementen)
st.dataframe(pd.DataFrame(trades))












# extra witregels 


