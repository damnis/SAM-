import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Functie om data op te halen ---
def fetch_data(ticker, interval):
    # Bepaal de periode op basis van het gekozen interval
    period = f"{360}{'d' if interval == '1d' else 'wk'}"  # "360d" of "360wk"
    
    # Download koersdata met de juiste interval en periode
    df = yf.download(ticker, period=period, interval=interval)
    df = df[["Open", "High", "Low", "Close"]]
    df.dropna(inplace=True)
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
st.title("📊 SAM Trading Indicator")

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
    'JPM': 'JPMorgan Chase', 'JNJ': 'Johnson & Johnson', 'MCD': 'McDonald’s', 'MRK': 'Merck',
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


# --- Tabs met selecties ---

tab_labels = ["🇺🇸 Dow Jones", "🇺🇸 Nasdaq", "🇺🇸 US Tech", "🇳🇱 AEX"]
selected_tab = st.radio("Kies beurs", tab_labels, horizontal=True)

if selected_tab == "🇺🇸 Dow Jones":
    ticker_label = st.selectbox("Dow Jones aandeel", [f"{v}, {k}" for k, v in dow_tickers.items()], key="dow")
    ticker, ticker_name = ticker_label.split(", ")

elif selected_tab == "🇺🇸 Nasdaq":
    ticker_label = st.selectbox("Nasdaq aandeel", [f"{v}, {k}" for k, v in nasdaq_tickers.items()], key="nasdaq")
    ticker, ticker_name = ticker_label.split(", ")

elif selected_tab == "🇺🇸 US Tech":
    ticker_label = st.selectbox("US Tech aandeel", [f"{v}, {k}" for k, v in ustech_tickers.items()], key="ustech")
    ticker, ticker_name = ticker_label.split(", ")

else:  # AEX
    ticker_label = st.selectbox("AEX aandeel", [f"{v}, {k}" for k, v in aex_tickers.items()], key="aex")
    ticker, ticker_name = ticker_label.split(", ")

# --- Andere instellingen ---
interval_optie = st.selectbox("Kies de interval", ["Dagelijks", "Wekelijks"])
interval = "1d" if interval_optie == "Dagelijks" else "1wk"
thresh = st.slider("Gevoeligheid van trendverandering", 0.01, 2.0, 0.5, step=0.01)

# vanaf hier
#import streamlit as st

# --- Aandelen per beurs met volledige naam ---
#aex_tickers = [
#    ('ABN', 'ABN AMRO'), ('ADYEN', 'Adyen'), ('AEGN', 'Aegon'), ('AD', 'Koninklijke Ahold Delhaize'),
#    ('AKZA', 'Akzo Nobel'), ('MT', 'ArcelorMittal'), ('ASM', 'ASM International'), ('ASML', 'ASML Holding'),
#    ('ASRNL', 'ASR Nederland'), ('BESI', 'BE Semiconductor'), ('DSFIR', 'DSM-Firmenich'),
#    ('GALAP', 'Galapagos'), ('HEIA', 'Heineken'), ('IMCD', 'IMCD Group'), ('INGA', 'ING Groep'),
#    ('JUST', 'Just Eat Takeaway'), ('KPN', 'KPN'), ('NN', 'NN Group'), ('PHIA', 'Philips'),
#    ('PRX', 'Prosus'), ('RAND', 'Randstad'), ('REN', 'Renewi'), ('SHELL', 'Shell'),
#    ('UNA', 'Unilever'), ('WKL', 'Wolters Kluwer')
#]

#dow_tickers = [
#    ('MMM', '3M'), ('AXP', 'American Express'), ('AMGN', 'Amgen'), ('AAPL', 'Apple'), ('BA', 'Boeing'),
#    ('CAT', 'Caterpillar'), ('CVX', 'Chevron'), ('CSCO', 'Cisco'), ('KO', 'Coca-Cola'), ('DIS', 'Disney'),
#    ('GS', 'Goldman Sachs'), ('HD', 'Home Depot'), ('HON', 'Honeywell'), ('IBM', 'IBM'),
 #   ('INTC', 'Intel'), ('JPM', 'JPMorgan Chase'), ('JNJ', 'Johnson & Johnson'), ('MCD', 'McDonald’s'),
#    ('MRK', 'Merck'), ('MSFT', 'Microsoft'), ('NKE', 'Nike'), ('PG', 'Procter & Gamble'),
#    ('CRM', 'Salesforce'), ('TRV', 'Travelers'), ('UNH', 'UnitedHealth'), ('VZ', 'Verizon'),
#    ('V', 'Visa'), ('WMT', 'Walmart'), ('DOW', 'Dow Inc.'), ('RTX', 'Raytheon Technologies'),
#    ('WBA', 'Walgreens Boots Alliance')
#]

#nasdaq_tickers = [
#    ('MSFT', 'Microsoft'), ('NVDA', 'NVIDIA'), ('AAPL', 'Apple'), ('AMZN', 'Amazon'),
#    ('META', 'Meta'), ('NFLX', 'Netflix'), ('GOOG', 'Google'), ('GOOGL', 'Alphabet'),
#    ('TSLA', 'Tesla'), ('CSCO', 'Cisco'), ('INTC', 'Intel'), ('ADBE', 'Adobe'),
#    ('CMCSA', 'Comcast'), ('PEP', 'PepsiCo'), ('COST', 'Costco'), ('AVGO', 'Broadcom'),
#    ('QCOM', 'Qualcomm'), ('TMUS', 'T-Mobile US'), ('TXN', 'Texas Instruments'),
#    ('AMAT', 'Applied Materials'), ('AMD', 'AMD'), ('CHTR', 'Charter Communications'),
#    ('SBUX', 'Starbucks'), ('MDLZ', 'Mondelez'), ('PYPL', 'PayPal'), ('INTU', 'Intuit'),
#    ('BKNG', 'Booking Holdings'), ('ISRG', 'Intuitive Surgical'), ('ADP', 'ADP'),
#    ('GILD', 'Gilead'), ('CSX', 'CSX'), ('MU', 'Micron'), ('LRCX', 'Lam Research'),
#    ('MELI', 'MercadoLibre'), ('MRVL', 'Marvell'), ('PANW', 'Palo Alto Networks'),
#    ('MCHP', 'Microchip Tech'), ('NXPI', 'NXP Semiconductors'), ('ORLY', 'O’Reilly'),
#    ('VRTX', 'Vertex'), ('ROST', 'Ross Stores'), ('MAR', 'Marriott'), ('DOCU', 'DocuSign'),
#    ('SNPS', 'Synopsys'), ('ZM', 'Zoom'), ('WDAY', 'Workday'), ('KHC', 'Kraft Heinz'),
#    ('REGN', 'Regeneron')
#]

# --- Tabs: Dow, Nasdaq, AEX ---
#tab_dow, tab_nasdaq, tab_aex = st.tabs(["🇺🇸 Dow Jones", "📈 Nasdaq", "🇳🇱 AEX"])

# --- Functie voor selecteren ---
#def toon_dropdown(tickers):
#    opties = [f"{sym} - {naam}" for sym, naam in tickers]
#    keuze = st.selectbox("Selecteer aandeel", opties, index=0)
#    return keuze.split(" - ")[0]  # Alleen ticker symbool teruggeven

# --- Initialiseer ticker via actieve tab ---
#ticker = None

#with tab_dow:
#    ticker = toon_dropdown(dow_tickers)
#with tab_nasdaq:
#    ticker = toon_dropdown(nasdaq_tickers)
#with tab_aex:
#    ticker = toon_dropdown(aex_tickers)

# Nu kun je 'ticker' gebruiken zoals voorheen in je code
# selectbox(label, opties, index=index)
#    return keuze.split(" - ")[0] if keuze != "(Geen selectie)" else None
    
#def make_dropdown(tickers, label):
 #   display_names = [f"{symbol} - {name}" for symbol, name in tickers]
  #  keuze = st.selectbox(label, ["(Geen selectie)"] + display_names)
 #   return keuze.split(" - ")[0] if keuze != "(Geen selectie)" else None

# dow krijgt standaard een waarde (bijv. ABN, index 0)
#ticker_dow = make_dropdown(dow_tickers, "Dow Jones Selectie", default_index=0, initial_active=True)

# Andere dropdowns starten leeg
#ticker_nasdaq = make_dropdown(nasdaq_tickers, "Nasdaq Selectie")
#ticker_aex = make_dropdown(aex_tickers, "AEX Selectie")
#ticker_dow = make_dropdown(dow_tickers, "Dow Jones Selectie")
#ticker_nasdaq = make_dropdown(nasdaq_tickers, "Nasdaq Selectie")
#ticker_aex = make_dropdown(aex_tickers, "AEX Selectie")

# --- Kies de geselecteerde ticker (één tegelijk) ---
#ticker =  ticker_dow or ticker_nasdaq or ticker_aex 

# --- Alleen doorgaan als er een is gekozen ---
#if ticker:
#    interval_optie = st.selectbox("Kies de interval", ["Dagelijks", "Wekelijks"])
#    interval = "1d" if interval_optie == "Dagelijks" else "1wk"
#    thresh = st.slider("Gevoeligheid van trendverandering", 0.01, 2.0, 0.5, step=0.01)

    # Je kunt nu `ticker` gebruiken in je download/verwerking

#st.title("📊 SAM Trading Indicator")
#all_tickers = [
    # AEX
#    'ABN', 'ADYEN', 'AEGN', 'AD', 'AKZA', 'MT', 'ASM', 'ASML',
#    'ASRNL', 'BESI', 'DSFIR', 'GALAP', 'HEIA', 'IMCD', 'INGA',
#    'JUST', 'KPN', 'NN', 'PHIA', 'PRX', 'RAND', 'REN', 'SHELL',
 #   'UNA', 'WKL',
    # Dow Jones
 #   'MMM', 'AXP', 'AMGN', 'AAPL', 'BA', 'CAT', 'CVX', 'CSCO', 'KO',
 #   'DIS', 'GS', 'HD', 'HON', 'IBM', 'INTC', 'JPM', 'JNJ', 'MCD',
 #   'MRK', 'MSFT', 'NKE', 'PG', 'CRM', 'TRV', 'UNH', 'VZ', 'V',
 #   'WMT', 'DOW', 'RTX', 'WBA',
    # Nasdaq‑100 (voorbeeldsubset – uitbreiden naar ~100)
 #   'MSFT', 'NVDA', 'AAPL', 'AMZN', 'META', 'NFLX', 'GOOG', 'GOOGL',
 #   'TSLA', 'CSCO', 'INTC', 'ADBE', 'CMCSA', 'PEP', 'COST', 'AVGO',
 #   'QCOM', 'TMUS', 'TXN', 'AMAT', 'AMD', 'CHTR', 'SBUX', 'MDLZ',
 #   'PYPL', 'INTU', 'BKNG', 'ISRG', 'ADP', 'GILD', 'CSX', 'MU',
 #   'LRCX', 'MELI', 'MRVL', 'PANW', 'MCHP', 'NXPI', 'ORLY', 'VRTX',
 #   'ROST', 'MAR', 'DOCU', 'SNPS', 'ZM', 'WDAY', 'KHC', 'REGN'
    # vul verder aan tot ~150 tickers
#]

#ticker = st.selectbox("Selecteer een aandeel (AEX, Dow, Nasdaq)", all_tickers)
#interval_optie = st.selectbox("Kies de interval", ["Dagelijks", "Wekelijks"])
#interval = "1d" if interval_optie == "Dagelijks" else "1wk"
#thresh = st.slider("Gevoeligheid van trendverandering", 0.01, 2.0, 0.5, step=0.01)

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
    f"""
    <h2>SAM-indicator en trend voor <span style='color:#3366cc'>{ticker_name}</span></h2>
    <h3 style='color:{advies_kleur}'>Huidig advies: {huidig_advies}</h3>
    """,
    unsafe_allow_html=True
)
#st.markdown(
#    f"""
#    <h2>SAM-indicator en trend voor <span style='color:#3366cc'>{ticker}</span></h2>
#    <h3 style='color:{advies_kleur}'>Huidig advies: {huidig_advies}</h3>
#    """,
#    unsafe_allow_html=True
#)

fig, ax1 = plt.subplots(figsize=(10, 4))
ax1.bar(df.index, df["SAM"], color="lightblue", label="SAM")
ax2 = ax1.twinx()
ax2.plot(df.index, df["Trend"], color="red", label="Trend")
ax1.set_ylabel("SAM")
ax2.set_ylabel("Trend")
fig.tight_layout()
st.pyplot(fig)

# Tabel met advies
st.subheader("Laatste signalen en rendement")
# --- Laatste rijen en formattering ---
kolommen = ["Close", "Advies", "SAM", "Trend", "Markt-%", "SAM-%"]
tabel = df[kolommen].dropna().tail(30).round(3).copy()

# Voeg datum toe vanuit index (in formaat dd-mm-jjjj)
tabel["Datum"] = tabel.index.strftime("%d-%m-%Y")

# Zet kolomvolgorde: eerst Datum
tabel = tabel[["Datum"] + kolommen]

# Rond andere kolommen af op 3 decimalen (behalve percentages)
for kolom in ["Close", "SAM", "Trend"]:
    tabel[kolom] = tabel[kolom].round(3)

# Format percentagekolommen netjes als 1.24% of -1.24%
tabel["Markt-%"] = (tabel["Markt-%"].astype(float) * 100).map("{:+.2f}%".format)
tabel["SAM-%"] = (tabel["SAM-%"].astype(float) * 100).map("{:+.2f}%".format)
#tabel["Markt-%"] = (tabel["Markt-%"] * 100).map("{:+.2f}%".format)
#tabel["SAM-%"] = (tabel["SAM-%"] * 100).map("{:+.2f}%".format)

# HTML-tabel bouwen met aangepaste styling
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
        text-align: right; color: #222222;
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

# Rijen toevoegen aan HTML
for _, row in tabel.iterrows():
    html += "<tr>"
    for value in row:
        html += f"<td>{value}</td>"
    html += "</tr>"

html += "</tbody></table>"

# HTML weergeven in Streamlit
import streamlit as st
#st.subheader("Laatste signalen en rendement (HTML)")
#st.markdown(
#    tabel.to_html(index=False, escape=False),
#    unsafe_allow_html=True
# )

st.markdown(html, unsafe_allow_html=True)







