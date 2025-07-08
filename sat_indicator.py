# sat_indicator.py
import streamlit as st
import yfinance as yf
import pandas as pd
import ta
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta, date
import matplotlib.pyplot as plt
from ta.trend import ADXIndicator

# ✅ Helperfunctie voor veilige conversie naar float
def safe_float(val):
    try:
        return float(val) if pd.notna(val) else 0.0
    except:
        return 0.0

# ✅ Verbeterde SAT-berekening met debug en fallback
@st.cache_data(ttl=900)
def calculate_sat(df):
    # ✅ Controle op MultiIndex en 'Close'-fallback
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if "Close" not in df.columns:
        mogelijke_close = [col for col in df.columns if col.lower() == "close" or "close" in col.lower()]
        if mogelijke_close:
            df["Close"] = df[mogelijke_close[0]]
        else:
            st.error("❌ Kon geen geldige 'Close'-kolom vinden voor SAT-berekening.")
            return df

    # ✅ Berekeningen
    df["MA150"] = df["Close"].rolling(window=150).mean()
    df["MA30"] = df["Close"].rolling(window=30).mean()
    df["SAT_Stage"] = np.nan  # eerst lege kolom

    for i in range(1, len(df)):
        ma150 = safe_float(df["MA150"].iloc[i])
        ma150_prev = safe_float(df["MA150"].iloc[i - 1])
        ma30 = safe_float(df["MA30"].iloc[i])
        ma30_prev = safe_float(df["MA30"].iloc[i - 1])
        close = safe_float(df["Close"].iloc[i])
        stage_prev = safe_float(df["SAT_Stage"].iloc[i - 1]) if i > 1 else 0.0
        stage = stage_prev  # start met vorige stage-waarde

#        if i > len(df) - 10:
#            st.write(f"🔍 i={i} | Close={close:.2f}, MA150={ma150:.2f}, MA150_prev={ma150_prev:.2f}, MA30={ma30:.2f}, MA30_prev={ma30_prev:.2f}")

        if ((ma150 > ma150_prev and close > ma150 and ma30 > close) or
              (close > ma150 and ma30 < ma30_prev and ma30 > close)):
            stage = -1
 #           if i > len(df) - 10:
 #               st.write(f"🔥 i={i}: Oververhitting of correctie → stage = -1")
        
        elif (ma150 < ma150_prev and close < ma150 and close > ma30 and ma30 > ma30_prev):
            stage = 1
 #           if i > len(df) - 10:
 #               st.write(f"🌀 i={i}: Koers tussen MA150 en MA30, MA150 daalt → stage = 1")
        
        elif (ma150 > close and ma150 > ma150_prev):
            stage = -1
#            if i > len(df) - 10:
#                st.write(f"😐 i={i}: MA150 stijgt, ligt boven koers → stage = -1")

        elif (ma150 < close and ma150 < ma150_prev and ma30 > ma30_prev):
            stage = 1
  #          if i > len(df) - 10:
 #               st.write(f"🌱 i={i}: MA150 en MA30 stijgen onder koers → stage = 1")

        elif (ma150 > close and ma150 < ma150_prev):
            stage = -2
  #          if i > len(df) - 10:
#                st.write(f"📉 i={i}: MA150 > Close en MA150 daalt → stage = -2")
        
        elif (ma150 < close and ma150 > ma150_prev and ma30 > ma30_prev):
            stage = 2
#            if i > len(df) - 10:
#                st.write(f"📈 i={i}: MA150 stijgt richting koers, MA30 stijgt → stage = 2")

        else:
            stage = stage_prev
#            if i > len(df) - 10:
#                st.write(f"⚪️ i={i}: Geen duidelijke verandering → stage = stage_prev ({stage_prev})")

        df.at[df.index[i], "SAT_Stage"] = stage

    df["SAT_Stage"] = df["SAT_Stage"].astype(float)
    df["SAT_Trend"] = df["SAT_Stage"].rolling(window=25).mean()
    return df
    
    # 📊 Debug: Laatste waarden MA150 en MA30
 #   st.write("Laatste 5 waarden van MA150:", df["MA150"].tail())
 # ×  st.write("Laatste 5 waarden van MA30:", df["MA30"].tail())
# ÷   st.write("📈 Laatste Close-waarden:", df["Close"].tail(10))
  #  return df
    
#st.write("MA150 laatste waarden:", df["MA150"].tail())
#st.write("MA30 laatste waarden:", df["MA30"].tail())
#st.write("SAT_Stage laatste waarden:", df["SAT_Stage"].tail())    

