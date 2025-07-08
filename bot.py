# bot.py
import yfinance as yf
import streamlit as st
import pandas as pd
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest, TrailingStopOrderRequest

def verbind_met_alpaca():
    try:
        api_key = st.secrets["ALPACA_API_KEY"]
        secret_key = st.secrets["ALPACA_SECRET_KEY"]
        client = TradingClient(api_key, secret_key, paper=True)
        account = client.get_account()
        return client, account
    except Exception as e:
        st.error(f"❌ Fout bij verbinden met Alpaca: {e}")
        return None, None

def haal_laatste_koers(ticker):
    try:
        live_data = yf.download(ticker, period="1d", interval="1d", progress=False)
        if isinstance(live_data, pd.DataFrame) and "Close" in live_data.columns:
            return float(live_data["Close"].dropna().iloc[-1].squeeze())
    except:
        return None
    return None

def plaats_order(client, ticker, bedrag, last_price, advies, order_type="Market", trailing_pct=None):
    aantal = int(bedrag / last_price)
    if aantal <= 0:
        st.warning("❌ Bedrag is te klein voor aankoop.")
        return
    side = OrderSide.BUY if advies == "Kopen" else OrderSide.SELL
    try:
        if order_type == "Market":
            order = MarketOrderRequest(
                symbol=ticker,
                qty=aantal,
                side=side,
                time_in_force=TimeInForce.GTC
            )
        else:
            order = TrailingStopOrderRequest(
                symbol=ticker,
                qty=aantal,
                side=side,
                trail_percent=trailing_pct,
                time_in_force=TimeInForce.GTC
            )
        response = client.submit_order(order)
        st.success(f"✅ Order geplaatst: {aantal}x {ticker} ({advies}) via {order_type}")
        st.write(response)
    except Exception as e:
        st.error(f"❌ Order kon niet worden geplaatst: {e}")

def sluit_positie(client, ticker, advies, force=False):
    try:
        positie = client.get_open_position(ticker)
        aantal = int(float(positie.qty))
        if not force and advies != "Verkopen":
            st.info("ℹ️ Huidig advies is geen 'Verkopen'. Geen actie ondernomen.")
            return
        order = MarketOrderRequest(
            symbol=ticker,
            qty=aantal,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY
        )
        response = client.submit_order(order)
        st.success(f"✅ Verkooporder geplaatst voor {aantal}x {ticker}")
        st.write(response)
    except Exception as e:
        st.info("📭 Geen open positie of fout bij ophalen: " + str(e))

def toon_trading_bot_interface(ticker, huidig_advies):
    st.subheader("📥 Plaats live paper trade op basis van advies")

    modus = st.radio("🎛️ Kies handelsmodus", ["Handmatig", "Automatisch", "Beide"], horizontal=True)

    client, account = verbind_met_alpaca()
    if client is None:
        return

    if account:
        st.success(f"✅ Verbonden met Alpaca-account ({account.status})")
        st.write(f"👤 Account-ID: {account.id}")
        st.write(f"💰 Beschikbaar cash: ${float(account.cash):,.2f}")
        st.write(f"📈 Portfolio waarde: ${float(account.portfolio_value):,.2f}")

    with st.expander("🧪 Virtuele testorder plaatsen via Alpaca Paper Account"):
        last = haal_laatste_koers(ticker)
        if last:
            st.write(f"📉 Laatste koers voor {ticker}: **${last:.2f}**")
        else:
            st.warning("⚠️ Geen geldige koers beschikbaar voor dit aandeel.")
            return

        bedrag = st.number_input("💰 Te investeren bedrag ($)", min_value=10.0, value=1000.0, step=10.0)
        st.write(f"📌 Actueel advies voor {ticker}: **{huidig_advies}**")

        order_type = st.radio("🛒 Kies ordertype", ["Market", "Trailing Stop"], horizontal=True)
        trailing_pct = None
        if order_type == "Trailing Stop":
            trailing_pct = st.slider("📉 Trailing stop (% vanaf hoogste koers)", 1.0, 20.0, 5.0)

        handmatig = modus in ["Handmatig", "Beide"]
        automatisch = modus in ["Automatisch", "Beide"]

        if handmatig and st.button("📤 Handmatig order plaatsen"):
            plaats_order(client, ticker, bedrag, last, huidig_advies, order_type, trailing_pct)

        if automatisch and huidig_advies in ["Kopen", "Verkopen"]:
            st.info("🤖 Automatisch advies actief...")
            plaats_order(client, ticker, bedrag, last, huidig_advies, order_type, trailing_pct)

    st.markdown("---")

    st.subheader("📤 Verkooppositie controleren en sluiten")
    with st.expander("📊 Positie check en verkoopactie"):
        try:
            positie = client.get_open_position(ticker)
            huidige_qty = int(float(positie.qty))
            avg_price = float(positie.avg_entry_price)
            st.write(f"📦 Je bezit momenteel **{huidige_qty}x {ticker}** @ ${avg_price:.2f} gemiddeld.")
        except Exception:
            st.info("📭 Geen open positie gevonden in deze ticker.")
            return

        st.write(f"📌 Huidig advies: **{huidig_advies}**")
        force_verkoop = st.checkbox("🔒 Forceer verkoop, ongeacht advies")

        if st.button("❗ Verkooppositie sluiten"):
            sluit_positie(client, ticker, huidig_advies, force=force_verkoop)










#  wir
# trading bot
# ALPACA_API_KEY = "PK8IAXXDXQEO9QLVNSCV"
# ALPACA_SECRET_KEY = "ooAURWeE0c2gp336eq5oHC1bqrRAVcCDlWpTbJDJ"
# trading bot code

