# -*- coding: utf-8 -*-
"""
GEX Focused Pro v18.2 (Final Explicit Edition)
Features:
- Hybrid Dealer Model (Default)
- Vectorized Math (Fast)
- High Contrast Visuals
- Explicit AI Market Analysis (Rialzo/Ribasso/Trappola)
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.stats import norm
from datetime import datetime, timezone
from io import BytesIO

# Configurazione pagina
st.set_page_config(page_title="GEX Pro v18.2", layout="wide", page_icon="⚡")

# -----------------------------------------------------------------------------
# 1. MOTORE MATEMATICO & DATI
# -----------------------------------------------------------------------------

def get_spot_price(ticker):
    """Recupera il prezzo spot corrente."""
    try:
        tk = yf.Ticker(ticker)
        price = tk.fast_info.get("last_price")
        if not price:
            hist = tk.history(period="1d")
            if not hist.empty:
                price = hist["Close"].iloc[-1]
        return float(price) if price else None
    except Exception:
        return None

def vectorized_bs_gamma(S, K, T, r, sigma):
    """Calcolo Black-Scholes Gamma vettorializzato (NumPy)."""
    T = np.maximum(T, 1e-5)
    sigma = np.maximum(sigma, 1e-5)
    S = float(S)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        pdf = norm.pdf(d1)
        gamma = pdf / (S * sigma * np.sqrt(T))
    
    return np.nan_to_num(gamma)

@st.cache_data(ttl=300)
def get_option_data(symbol, expiry, spot_price, range_pct=25.0):
    """Scarica e pulisce la chain opzioni."""
    try:
        tk = yf.Ticker(symbol)
        chain = tk.option_chain(expiry)
        calls = chain.calls.copy()
        puts = chain.puts.copy()
    except Exception as e:
        return None, None, f"Errore download dati: {e}"

    for df in [calls, puts]:
        df.fillna(0, inplace=True)
        df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
        df["openInterest"] = pd.to_numeric(df["openInterest"], errors="coerce")
        df["impliedVolatility"] = pd.to_numeric(df["impliedVolatility"], errors="coerce")

    lower_bound = spot_price * (1 - range_pct/100)
    upper_bound = spot_price * (1 + range_pct/100)
    
    calls = calls[(calls["strike"] >= lower_bound) & (calls["strike"] <= upper_bound)]
    puts = puts[(puts["strike"] >= lower_bound) & (puts["strike"] <= upper_bound)]

    return calls, puts, None

def calculate_gex_profile(calls, puts, spot, expiry_date, call_sign=1, put_sign=-1, min_oi_ratio=0.1):
    risk_free = 0.05
    
    try:
        exp_dt = datetime.strptime(expiry_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        exp_dt = datetime.strptime(expiry_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        
    now_dt = datetime.now(timezone.utc)
    dte = (exp_dt - now_dt).total_seconds() / 86400.0
    T = max(dte / 365.0, 1e-4)

    # Filtro OI
    max_oi = max(calls["openInterest"].max(), puts["openInterest"].max()) if not calls.empty and not puts.empty else 0
    threshold = max_oi * min_oi_ratio
    
    calls_clean = calls[calls["openInterest"] >= threshold].copy()
    puts_clean = puts[puts["openInterest"] >= threshold].copy()
    
    if calls_clean.empty: calls_clean = calls.copy()
    if puts_clean.empty: puts_clean = puts.copy()

    # Calcolo Gamma Vettoriale
    c_gamma = vectorized_bs_gamma(spot, calls_clean["strike"].values, T, risk_free, calls_clean["impliedVolatility"].values)
    p_gamma = vectorized_bs_gamma(spot, puts_clean["strike"].values, T, risk_free, puts_clean["impliedVolatility"].values)

    # GEX Formula Standard ($ Gamma)
    calls_clean["GEX"] = call_sign * c_gamma * spot * calls_clean["openInterest"].values * 100
    puts_clean["GEX"] = put_sign * p_gamma * spot * puts_clean["openInterest"].values * 100

    # Ricalcolo completo per plotting (senza filtri OI)
    full_c_gamma = vectorized_bs_gamma(spot, calls["strike"].values, T, risk_free, calls["impliedVolatility"].values)
    full_p_gamma = vectorized_bs_gamma(spot, puts["strike"].values, T, risk_free, puts["impliedVolatility"].values)
    calls["GEX"] = call_sign * full_c_gamma * spot * calls["openInterest"].values * 100
    puts["GEX"] = put_sign * full_p_gamma * spot * puts["openInterest"].values * 100

    gex_df = pd.concat([calls[["strike", "GEX"]], puts[["strike", "GEX"]]])
    gex_by_strike = gex_df.groupby("strike")["GEX"].sum().reset_index().sort_values("strike")

    # Indicatori
    total_call_gex = calls["GEX"].sum()
    total_put_gex = puts["GEX"].sum()
    total_gex = total_call_gex + total_put_gex
    
    abs_total = abs(total_call_gex) + abs(total_put_gex)
    net_gamma_bias = (total_call_gex / abs_total * 100) if abs_total > 0 else 0

    gamma_flip = None
    if call_sign == 1 and put_sign == -1:
        numerator = (calls["strike"] * calls["GEX"]).sum() + (puts["strike"] * puts["GEX"]).sum()
        if total_gex != 0:
            raw_flip = numerator / total_gex
            if 0.5 * spot < raw_flip < 1.5 * spot:
                gamma_flip = raw_flip

    return {
        "calls": calls,
        "puts": puts,
        "gex_by_strike": gex_by_strike,
        "gamma_flip": gamma_flip,
        "net_gamma_bias": net_gamma_bias,
        "total_gex": total_gex,
        "regime": "LONG GAMMA" if total_gex > 0 else "SHORT GAMMA"
    }, None

# -----------------------------------------------------------------------------
# 2. SISTEMA DI REPORTISTICA AVANZATO (ANALISI DIRETTA)
# -----------------------------------------------------------------------------

def find_zero_crossing(df, spot):
    """Trova il Flip visuale (Zero Crossing) più vicino allo spot."""
    try:
        df = df.sort_values("strike")
        sign_change = ((df["GEX"] > 0) != (df["GEX"].shift(1) > 0)) & (~df["GEX"].shift(1).isna())
        crossings = df[sign_change]
        
        if crossings.empty: return None
            
        crossings = crossings.copy()
        crossings["dist"] = abs(crossings["strike"] - spot)
        nearest_flip = crossings.sort_values("dist").iloc[0]["strike"]
        
        # Interpolazione per precisione
        idx = df[df["strike"] == nearest_flip].index[0]
        prev_idx = idx - 1
        if prev_idx >= 0:
            y2, x2 = df.loc[idx, "GEX"], df.loc[idx, "strike"]
            y1, x1 = df.loc[prev_idx, "GEX"], df.loc[prev_idx, "strike"]
            if y2 != y1:
                return x1 + (-y1) * (x2 - x1) / (y2 - y1)
        return nearest_flip
    except:
        return None

def generate_detailed_report(symbol, spot, data, call_walls, put_walls):
    """Genera report esplicito: Scommessa Rialzo/Ribasso."""
    
    tot_gex = data['total_gex']
    net_bias = data['net_gamma_bias']
    global_flip = data['gamma_flip']
    gex_df = data['gex_by_strike']
    
    # Flip Visivo
    local_flip = find_zero_crossing(gex_df, spot)
    effective_flip = local_flip if local_flip else global_flip
    
    # A. Regime & Bias
    regime_status = "LONG GAMMA" if tot_gex > 0 else "SHORT GAMMA"
    
    if net_bias > 40:
        bias_desc = f"Dominanza Call (**{net_bias:.1f}%**)."
    elif net_bias < -40:
        bias_desc = f"Dominanza Put (**{abs(net_bias):.1f}%**)."
    else:
        bias_desc = f"Neutrale ({net_bias:.1f}%)."

    # B. Analisi Zona (Safe vs Danger)
    flip_analysis = ""
    safe_zone = False
    if effective_flip:
        if spot > effective_flip:
            safe_zone = True
            zone_desc = f"Prezzo SOPRA il Flip ({effective_flip:.0f}). Safe Zone."
        else:
            safe_zone = False
            zone_desc = f"Prezzo SOTTO il Flip ({effective_flip:.0f}). Danger Zone."
        flip_analysis = zone_desc
    else:
        flip_analysis = "Nessun Flip chiaro."

    # C. SINTESI ESPLICITA (Logica Direzionale)
    scommessa = ""
    colore_sintesi = "#ffffff"
    bordino = "grey"
    icona = ""

    # Logica Decisionale
    if net_bias > 20:
        direzione_base = "SCOMMETTE AL RIALZO"
        icona = "📈"
    elif net_bias < -20:
        direzione_base = "SCOMMETTE AL RIBASSO"
        icona = "📉"
    else:
        direzione_base = "È LATERALE / INDECISO"
        icona = "⚖️"

    # Analisi Contesto
    if safe_zone and tot_gex > 0:
        if net_bias > 0:
            scommessa = f"{icona} Il mercato {direzione_base}"
            dettaglio = "Contesto **Favorevole**: Siamo in Safe Zone con Dealer 'ammortizzatori'."
            colore_sintesi = "#e8f5e9" # Verde
            bordino = "green"
        else:
            scommessa = "⚠️ Il mercato è CAUTO (Copertura)"
            dettaglio = "Bias Put presente ma il prezzo tiene la Safe Zone. Possibile rimbalzo (Pain Trade)."
            colore_sintesi = "#fff3e0" # Arancio
            bordino = "orange"

    elif not safe_zone:
        if net_bias < 0:
            scommessa = f"{icona} Il mercato {direzione_base}"
            dettaglio = "Contesto **Critico**: Siamo sotto il Flip e i Dealer accelerano i ribassi."
            colore_sintesi = "#ffebee" # Rosso
            bordino = "red"
        else:
            scommessa = "⚠️ Il mercato è INTRAPPOLATO (Bull Trap)"
            dettaglio = f"Tante Call aperte ma prezzo sotto il Flip ({effective_flip:.0f}). Se non recupera, liquidano le Call."
            colore_sintesi = "#ffccbc" # Rosso scuro
            bordino = "darkred"
    else:
        scommessa = f"{icona} Il mercato {direzione_base}"
        dettaglio = "Situazione transitoria."

    # Muri Principali
    cw = min([w for w in call_walls if w > spot], default="N/A")
    pw = max([w for w in put_walls if w < spot], default="N/A")
    cw_txt = f"{int(cw)}" if cw != "N/A" else "-"
    pw_txt = f"{int(pw)}" if pw != "N/A" else "-"

    html = f"""
    <div style="font-family: sans-serif; color: #333;">
        <div style="display: flex; justify-content: space-between; font-size: 13px; color: #555; margin-bottom: 10px; border-bottom: 1px solid #ddd; padding-bottom: 5px;">
            <span>Spot: <strong>{spot:.2f}</strong></span>
            <span>Regime: <strong>{regime_status}</strong></span>
            <span>Bias: <strong>{bias_desc}</strong></span>
        </div>
        
        <div style="font-size: 13px; margin-bottom: 10px;">
            <strong>Analisi Livelli:</strong> {flip_analysis}<br>
            <span style="color: #0d47a1;">Resistenza Call: <strong>{cw_txt}</strong></span> | 
            <span style="color: #e65100;">Supporto Put: <strong>{pw_txt}</strong></span>
        </div>

        <div style="background-color: {colore_sintesi}; padding: 15px; border-radius: 8px; border-left: 6px solid {bordino}; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
            <h3 style="margin: 0; margin-bottom: 5px; color: #222; font-size: 18px;">{scommessa}</h3>
            <p style="font-size: 14px; margin: 0; color: #444;">{dettaglio}</p>
        </div>
    </div>
    """
    return html

# -----------------------------------------------------------------------------
# 3. SISTEMA DI PLOTTING (GRAFICA PULITA)
# -----------------------------------------------------------------------------

def plot_dashboard(symbol, data, spot, expiry, dist_min_pct):
    calls, puts = data["calls"], data["puts"]
    gex_strike = data["gex_by_strike"]
    flip = data["gamma_flip"]
    
    # Identificazione Muri
    calls = calls.copy(); puts = puts.copy()
    calls["WallScore"] = calls["openInterest"]
    puts["WallScore"] = puts["openInterest"]
    
    def get_top_levels(df, min_dist):
        df_s = df.sort_values("WallScore", ascending=False)
        levels = []
        for k in df_s["strike"]:
            if not levels or all(abs(k - x) > min_dist for x in levels):
                levels.append(k)
            if len(levels) >= 3: break
        return levels

    min_dist_val = spot * (dist_min_pct / 100.0)
    call_walls = get_top_levels(calls, min_dist_val)
    put_walls = get_top_levels(puts, min_dist_val)

    # Plot Setup
    fig = plt.figure(figsize=(13, 6))
    ax = fig.add_subplot(111)
    
    bar_width = spot * 0.007
    
    # 1. Barre Sbiadite (Sfondo)
    ax.bar(puts["strike"], -puts["openInterest"], color="#ffcc80", alpha=0.25, width=bar_width)
    ax.bar(calls["strike"], calls["openInterest"], color="#90caf9", alpha=0.25, width=bar_width)
    
    # 2. Muri Intensi
    for w in call_walls:
        val = calls[calls['strike']==w]['openInterest'].sum()
        ax.bar(w, val, color="#0d47a1", alpha=0.9, width=bar_width) # Blu scuro
        
    for w in put_walls:
        val = -puts[puts['strike']==w]['openInterest'].sum()
        ax.bar(w, val, color="#e65100", alpha=0.9, width=bar_width) # Arancio scuro

    # 3. Linea GEX
    ax2 = ax.twinx()
    ax2.fill_between(gex_strike["strike"], gex_strike["GEX"], 0, where=(gex_strike["GEX"]>=0), color="green", alpha=0.10, interpolate=True)
    ax2.fill_between(gex_strike["strike"], gex_strike["GEX"], 0, where=(gex_strike["GEX"]<0), color="red", alpha=0.10, interpolate=True)
    ax2.plot(gex_strike["strike"], gex_strike["GEX"], color="#999999", lw=1.5, ls="-", label="Net GEX")
    
    # 4. Linee Verticali
    ax.axvline(spot, color="blue", ls="--", lw=1.2, label="Spot")
    if flip:
        ax.axvline(flip, color="purple", ls="-.", lw=1.2, label="Flip")

    # 5. Etichette Leggibili (Bbox)
    max_y = calls["openInterest"].max()
    y_offset = max_y * 0.03
    bbox_props = dict(boxstyle="round,pad=0.2", fc="white", ec="#cccccc", alpha=0.9)

    for w in call_walls:
        val = calls[calls['strike']==w]['openInterest'].sum()
        ax.text(w, val + y_offset, f"C {int(w)}", color="#0d47a1", fontsize=9, fontweight='bold', 
                ha='center', va='bottom', bbox=bbox_props, zorder=20)
                
    for w in put_walls:
        val = -puts[puts['strike']==w]['openInterest'].sum()
        ax.text(w, val - y_offset, f"P {int(w)}", color="#e65100", fontsize=9, fontweight='bold', 
                ha='center', va='top', bbox=bbox_props, zorder=20)

    ax.set_xlabel("Strike")
    ax.set_ylabel("Open Interest")
    ax2.set_ylabel("Gamma Exposure")
    ax2.axhline(0, color="grey", lw=0.5)
    
    plt.title(f"{symbol} GEX Matrix | {expiry}", fontsize=12, pad=10)
    plt.tight_layout()
    
    return fig, call_walls, put_walls

# -----------------------------------------------------------------------------
# 4. INTERFACCIA STREAMLIT
# -----------------------------------------------------------------------------

st.title("⚡ GEX Pro v18.2 Explicit")

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### ⚙️ Setup")
    symbol = st.text_input("Ticker", value="SPY").upper()
    spot = get_spot_price(symbol)
    
    sel_exp = None
    if spot:
        st.success(f"Spot: ${spot:.2f}")
        try:
            exps = yf.Ticker(symbol).options
            if exps: sel_exp = st.selectbox("Scadenza", exps[:12])
        except: pass
    
    st.markdown("---")
    # Configurazione Dealer Ibrida (Default Corretto)
    dealer_long_call = st.checkbox("Dealer Long CALL (+)", value=True, help="Default: Dealer comprano da chi vende Covered Calls.")
    dealer_long_put = st.checkbox("Dealer Long PUT (+)", value=False, help="Default: Dealer vendono a chi compra Protezione (-).")
    
    call_sign = 1 if dealer_long_call else -1
    put_sign = 1 if dealer_long_put else -1

    range_pct = st.slider("Range %", 5, 50, 20)
    min_oi_ratio = st.slider("Filtro OI", 5, 50, 15) / 100.0
    dist_min = st.slider("Dist. Muri", 1, 10, 2)

    btn_calc = st.button("🚀 Analizza Mercato", type="primary", use_container_width=True)

with col2:
    if btn_calc and spot and sel_exp:
        with st.spinner("Elaborazione Strategica..."):
            calls, puts, err = get_option_data(symbol, sel_exp, spot, range_pct)
            
            if err:
                st.error(err)
            else:
                data_res, err_calc = calculate_gex_profile(
                    calls, puts, spot, sel_exp, 
                    call_sign=call_sign, put_sign=put_sign, min_oi_ratio=min_oi_ratio
                )
                
                if err_calc:
                    st.error(err_calc)
                else:
                    # Plot
                    fig, c_walls, p_walls = plot_dashboard(symbol, data_res, spot, sel_exp, dist_min)
                    st.pyplot(fig)
                    
                    # Report Esplicito
                    html_rep = generate_detailed_report(symbol, spot, data_res, c_walls, p_walls)
                    st.markdown(html_rep, unsafe_allow_html=True)
                    
                    st.markdown("---")
                    buf = BytesIO()
                    fig.savefig(buf, format="png", dpi=150, bbox_inches='tight')
                    st.download_button("💾 Salva Grafico", buf.getvalue(), f"GEX_{symbol}.png", "image/png")
