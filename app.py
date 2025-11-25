# -*- coding: utf-8 -*-
"""
GEX Focused Pro v19.0 (GEX-Based Walls)
- WALLS LOGIC UPGRADE: Walls are now selected based on Net GEX (OI * Gamma)
  instead of just Open Interest. This highlights active hedging pressure.
- Synced Flip & High-End Report preserved.
- BUG FIX: Gestione corretta errori di formattazione su Flip mancante.
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import norm
from datetime import datetime, timezone
from io import BytesIO
import textwrap

# Configurazione pagina
st.set_page_config(page_title="GEX Pro v19.0", layout="wide", page_icon="⚡")

# -----------------------------------------------------------------------------
# 1. MOTORE MATEMATICO & DATI
# -----------------------------------------------------------------------------

def get_spot_price(ticker):
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

    max_oi = max(calls["openInterest"].max(), puts["openInterest"].max()) if not calls.empty and not puts.empty else 0
    threshold = max_oi * min_oi_ratio
    
    calls_clean = calls[calls["openInterest"] >= threshold].copy()
    puts_clean = puts[puts["openInterest"] >= threshold].copy()
    
    if calls_clean.empty: calls_clean = calls.copy()
    if puts_clean.empty: puts_clean = puts.copy()

    # Calcolo Gamma per tutti gli strike nel range
    full_c_gamma = vectorized_bs_gamma(spot, calls["strike"].values, T, risk_free, calls["impliedVolatility"].values)
    full_p_gamma = vectorized_bs_gamma(spot, puts["strike"].values, T, risk_free, puts["impliedVolatility"].values)
    
    # GEX = OI * Gamma * Spot * 100
    calls["GEX"] = call_sign * full_c_gamma * spot * calls["openInterest"].values * 100
    puts["GEX"] = put_sign * full_p_gamma * spot * puts["openInterest"].values * 100

    gex_df = pd.concat([calls[["strike", "GEX"]], puts[["strike", "GEX"]]])
    gex_by_strike = gex_df.groupby("strike")["GEX"].sum().reset_index().sort_values("strike")

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
        "total_gex": total_gex
    }, None

# -----------------------------------------------------------------------------
# 2. REPORT ANALITICO (CORRETTO)
# -----------------------------------------------------------------------------

def find_zero_crossing(df, spot):
    try:
        df = df.sort_values("strike")
        sign_change = ((df["GEX"] > 0) != (df["GEX"].shift(1) > 0)) & (~df["GEX"].shift(1).isna())
        crossings = df[sign_change]
        if crossings.empty: return None
        crossings = crossings.copy()
        crossings["dist"] = abs(crossings["strike"] - spot)
        nearest_flip = crossings.sort_values("dist").iloc[0]["strike"]
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

def get_analysis_content(spot, data, call_walls, put_walls, synced_flip):
    tot_gex = data['total_gex']
    net_bias = data['net_gamma_bias']
    effective_flip = synced_flip
    
    regime_status = "LONG GAMMA" if tot_gex > 0 else "SHORT GAMMA"
    regime_color = "#1b5e20" if tot_gex > 0 else "#b71c1c"
    
    if net_bias > 40: bias_desc = f"Dominanza Call ({net_bias:.1f}%)"
    elif net_bias < -40: bias_desc = f"Dominanza Put ({abs(net_bias):.1f}%)"
    else: bias_desc = f"Neutrale ({net_bias:.1f}%)"

    safe_zone = False
    if effective_flip is not None:
        if spot > effective_flip:
            safe_zone = True
            flip_desc = f"Prezzo SOPRA il Flip ({effective_flip:.0f}) - Safe Zone"
        else:
            safe_zone = False
            flip_desc = f"Prezzo SOTTO il Flip ({effective_flip:.0f}) - Danger Zone"
    else:
        # Se non c'è un flip chiaro
        flip_desc = "Nessun Flip chiaro rilevato"

    scommessa = ""
    colore_bg = "#ffffff"
    bordino = "grey"
    icona = ""

    if net_bias > 20:
        direzione_base = "SCOMMETTE AL RIALZO"
        icona = "📈"
    elif net_bias < -20:
        direzione_base = "SCOMMETTE AL RIBASSO"
        icona = "📉"
    else:
        direzione_base = "È LATERALE / INDECISO"
        icona = "⚖️"

    if safe_zone and tot_gex > 0:
        if net_bias > 0:
            scommessa = f"{icona} Il mercato {direzione_base}"
            dettaglio = "Contesto Favorevole: Siamo in Safe Zone con Dealer 'ammortizzatori'."
            colore_bg = "#e8f5e9"
            bordino = "#2e7d32"
        else:
            scommessa = "⚠️ Il mercato è CAUTO (Copertura)"
            dettaglio = "Bias Put presente ma il prezzo tiene la Safe Zone. Possibile rimbalzo."
            colore_bg = "#fff3e0"
            bordino = "#ef6c00"
    elif not safe_zone:
        if net_bias < 0:
            scommessa = f"{icona} Il mercato {direzione_base}"
            dettaglio = "Contesto Critico: Siamo sotto il Flip e i Dealer accelerano i ribassi."
            colore_bg = "#ffebee"
            bordino = "#c62828"
        else:
            scommessa = "⚠️ Il mercato è INTRAPPOLATO (Bull Trap)"
            # FIX: Gestione sicura della stampa del Flip se è None
            flip_str = f"{effective_flip:.0f}" if effective_flip is not None else "N/D"
            dettaglio = f"Tante Call aperte ma prezzo sotto il Flip ({flip_str}). Se non recupera, liquidano le Call."
            colore_bg = "#ffccbc"
            bordino = "#bf360c"
    else:
        scommessa = f"{icona} Il mercato {direzione_base}"
        dettaglio = "Situazione transitoria."
        colore_bg = "#f5f5f5"

    cw = min(call_walls) if call_walls else None
    pw = max(put_walls) if put_walls else None
    cw_txt = f"{int(cw)}" if cw else "-"
    pw_txt = f"{int(pw)}" if pw else "-"

    return {
        "spot": spot,
        "regime": regime_status,
        "regime_color": regime_color,
        "bias": bias_desc,
        "flip_desc": flip_desc,
        "cw": cw_txt,
        "pw": pw_txt,
        "scommessa": scommessa,
        "dettaglio": dettaglio,
        "colore_bg": colore_bg,
        "bordino": bordino
    }

# -----------------------------------------------------------------------------
# 3. PLOTTING UNIFICATO (GEX-BASED WALLS)
# -----------------------------------------------------------------------------

def plot_dashboard_unified(symbol, data, spot, expiry, dist_min_pct):
    calls, puts = data["calls"], data["puts"]
    gex_strike = data["gex_by_strike"]
    
    local_flip = find_zero_crossing(gex_strike, spot)
    final_flip = local_flip if local_flip else data["gamma_flip"]

    # --- WALLS BASATI SU GEX ---
    calls = calls.copy(); puts = puts.copy()
    
    # WallScore = Assoluto del GEX (OI * Gamma). Più è alto, più è un muro sensibile.
    calls["WallScore"] = calls["GEX"].abs()
    puts["WallScore"] = puts["GEX"].abs()
    
    def get_top_levels(df, min_dist):
        df_s = df.sort_values("WallScore", ascending=False)
        levels = []
        for k in df_s["strike"]:
            if not levels or all(abs(k - x) > min_dist for x in levels):
                levels.append(k)
            if len(levels) >= 3: break
        return levels

    min_dist_val = spot * (dist_min_pct / 100.0)
    
    calls_above = calls[calls["strike"] > spot].copy()
    puts_below = puts[puts["strike"] < spot].copy()
    
    call_walls = get_top_levels(calls_above, min_dist_val)
    put_walls = get_top_levels(puts_below, min_dist_val)

    rep = get_analysis_content(spot, data, call_walls, put_walls, final_flip)

    # Setup Figura
    fig = plt.figure(figsize=(13, 9.5))
    gs = gridspec.GridSpec(2, 1, height_ratios=[2.8, 1.2], hspace=0.2) 
    
    # --- SUBPLOT 1: GRAFICO ---
    ax = fig.add_subplot(gs[0])
    bar_width = spot * 0.007
    
    # Sfondo: OI (trasparente) per contesto storico
    ax.bar(puts["strike"], -puts["openInterest"], color="#ffcc80", alpha=0.2, width=bar_width)
    ax.bar(calls["strike"], calls["openInterest"], color="#90caf9", alpha=0.2, width=bar_width)
    
    # Walls: GEX BASED (Barre Piene)
    for w in call_walls:
        val = calls[calls['strike']==w]['openInterest'].sum()
        ax.bar(w, val, color="#0d47a1", alpha=0.9, width=bar_width)
    for w in put_walls:
        val = -puts[puts['strike']==w]['openInterest'].sum()
        ax.bar(w, val, color="#e65100", alpha=0.9, width=bar_width)

    ax2 = ax.twinx()
    ax2.fill_between(gex_strike["strike"], gex_strike["GEX"], 0, where=(gex_strike["GEX"]>=0), color="green", alpha=0.10, interpolate=True)
    ax2.fill_between(gex_strike["strike"], gex_strike["GEX"], 0, where=(gex_strike["GEX"]<0), color="red", alpha=0.10, interpolate=True)
    ax2.plot(gex_strike["strike"], gex_strike["GEX"], color="#999999", lw=1.5, ls="-", label="Net GEX")
    
    ax.axvline(spot, color="blue", ls="--", lw=1.2, label="Spot")
    if final_flip:
        ax.axvline(final_flip, color="red", ls="-.", lw=1.5, label="Flip")

    max_y = calls["openInterest"].max()
    y_offset = max_y * 0.03
    bbox_props = dict(boxstyle="round,pad=0.2", fc="white", ec="#cccccc", alpha=0.9)

    for w in call_walls:
        val = calls[calls['strike']==w]['openInterest'].sum()
        ax.text(w, val + y_offset, f"C {int(w)}", color="#0d47a1", fontsize=9, fontweight='bold', ha='center', va='bottom', bbox=bbox_props, zorder=20)
    for w in put_walls:
        val = -puts[puts['strike']==w]['openInterest'].sum()
        ax.text(w, val - y_offset, f"P {int(w)}", color="#e65100", fontsize=9, fontweight='bold', ha='center', va='top', bbox=bbox_props, zorder=20)

    ax.set_ylabel("Open Interest", fontsize=11, fontweight='bold', color="#444")
    ax2.set_ylabel("Gamma Exposure")
    ax2.axhline(0, color="grey", lw=0.5)
    ax.text(0.99, 0.02, "GEX Focused Pro v19.0", transform=ax.transAxes, ha="right", va="bottom", fontsize=14, color="#999999", fontweight="bold", alpha=0.5)

    legend_elements = [
        Patch(facecolor='#90caf9', edgecolor='none', label='Call OI (Mass)'),
        Patch(facecolor='#ffcc80', edgecolor='none', label='Put OI (Mass)'),
        Line2D([0], [0], color='blue', lw=1.5, ls='--', label=f'Spot {spot:.0f}'),
        Line2D([0], [0], color='#999999', lw=1.5, label='Net GEX (Pressure)'),
    ]
    if final_flip: legend_elements.append(Line2D([0], [0], color='red', lw=1.5, ls='-.', label=f'Flip {final_flip:.0f}'))
    ax.legend(handles=legend_elements, loc='upper left', framealpha=0.95, fontsize=10)
    ax.set_title(f"{symbol} GEX Matrix (GEX-Weighted Walls) | {expiry}", fontsize=13, pad=10, fontweight='bold', fontfamily='sans-serif')

    # --- SUBPLOT 2: REPORT ---
    ax_rep = fig.add_subplot(gs[1])
    ax_rep.axis("off")
    
    ax_rep.text(0.02, 0.88, f"SPOT: {rep['spot']:.2f}", fontsize=12, fontweight='bold', color="#222", fontfamily='sans-serif', transform=ax_rep.transAxes)
    ax_rep.text(0.20, 0.88, "|", fontsize=12, color="#aaa", transform=ax_rep.transAxes)
    ax_rep.text(0.22, 0.88, f"REGIME: ", fontsize=12, fontweight='bold', color="#222", fontfamily='sans-serif', transform=ax_rep.transAxes)
    ax_rep.text(0.33, 0.88, rep['regime'], fontsize=12, fontweight='bold', color=rep['regime_color'], fontfamily='sans-serif', transform=ax_rep.transAxes)
    ax_rep.text(0.53, 0.88, "|", fontsize=12, color="#aaa", transform=ax_rep.transAxes)
    ax_rep.text(0.55, 0.88, f"BIAS: {rep['bias']}", fontsize=12, fontweight='bold', color="#222", fontfamily='sans-serif', transform=ax_rep.transAxes)

    flip_text = f"FLIP: {rep['flip_desc']}"
    levels_text = f"RES (1° Call GEX): {rep['cw']}   |   SUP (1° Put GEX): {rep['pw']}"
    
    ax_rep.text(0.02, 0.72, flip_text, fontsize=11, color="#444", fontfamily='sans-serif', transform=ax_rep.transAxes)
    ax_rep.text(0.02, 0.60, levels_text, fontsize=11, color="#444", fontfamily='sans-serif', transform=ax_rep.transAxes)
    
    box_x, box_y = 0.02, 0.05
    box_w, box_h = 0.96, 0.45
    rect = patches.FancyBboxPatch((box_x, box_y), box_w, box_h, boxstyle="round,pad=0.03", linewidth=1, edgecolor="#dddddd", facecolor=rep['colore_bg'], transform=ax_rep.transAxes, zorder=1)
    ax_rep.add_patch(rect)
    accent_rect = patches.Rectangle((box_x, box_y), 0.015, box_h, facecolor=rep['bordino'], edgecolor="none", transform=ax_rep.transAxes, zorder=2)
    ax_rep.add_patch(accent_rect)
    
    sintesi_title = rep['scommessa']
    sintesi_desc = textwrap.fill(rep['dettaglio'], width=105)
    ax_rep.text(box_x + 0.04, box_y + 0.30, sintesi_title, fontsize=15, fontweight='bold', color="#222", fontfamily='sans-serif', transform=ax_rep.transAxes)
    ax_rep.text(box_x + 0.04, box_y + 0.15, sintesi_desc, fontsize=12, color="#333", fontfamily='sans-serif', transform=ax_rep.transAxes)

    plt.tight_layout()
    return fig

# -----------------------------------------------------------------------------
# 4. INTERFACCIA STREAMLIT
# -----------------------------------------------------------------------------

st.title("⚡ GEX Pro v19.0 (GEX Walls)")

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
        with st.spinner("Calcolo GEX e Pressione Volatilità..."):
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
                    fig = plot_dashboard_unified(symbol, data_res, spot, sel_exp, dist_min)
                    st.pyplot(fig)
                    
                    st.markdown("---")
                    buf = BytesIO()
                    fig.savefig(buf, format="png", dpi=150, bbox_inches='tight')
                    st.download_button("💾 Scarica Report GEX-Pro", buf.getvalue(), f"GEX_FULL_{symbol}.png", "image/png", use_container_width=True)
