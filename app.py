# -*- coding: utf-8 -*-
"""
GEX Focused Pro v20.0 (Positioning Edition)
- CORE UPGRADE: Multi-Expiry Aggregation for Swing/Positioning Trading.
- WALLS LOGIC: Net GEX (OI * Gamma) on aggregated open interest.
- VISUALS: Streamlit High-End Report preserved.
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
from matplotlib.collections import LineCollection # <--- NUOVA IMPORTAZIONE NECESSARIA
from scipy.stats import norm
from datetime import datetime, timezone, timedelta
from io import BytesIO
import textwrap
import time

# Configurazione pagina
st.set_page_config(page_title="GEX Positioning Pro v20.0", layout="wide", page_icon="⚡")

# -----------------------------------------------------------------------------
# 1. MOTORE MATEMATICO & DATI (AGGREGATI)
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
    # Evita divisioni per zero o tempi negativi
    T = np.maximum(T, 0.001) 
    sigma = np.maximum(sigma, 0.01)
    S = float(S)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        pdf = norm.pdf(d1)
        gamma = pdf / (S * sigma * np.sqrt(T))
    return np.nan_to_num(gamma)

@st.cache_data(ttl=600) # Cache più lunga (10 min) dato che scarichiamo più dati
def get_aggregated_data(symbol, spot_price, n_expirations=8, range_pct=25.0):
    """
    Scarica le prime n_expirations e le aggrega in un unico DataFrame.
    """
    try:
        tk = yf.Ticker(symbol)
        exps = tk.options
        if not exps: return None, None, "Nessuna scadenza trovata."
        
        # Prendiamo le prime N scadenze (Positioning View)
        target_exps = exps[:n_expirations]
        
        all_calls = []
        all_puts = []
        
        # Barra progresso (visibile solo se chiamato da UI, qui logica interna)
        progress_text = "Scaricamento scadenze in corso..."
        my_bar = st.progress(0, text=progress_text)
        
        for i, exp in enumerate(target_exps):
            try:
                # Aggiorna barra
                pct = int((i / len(target_exps)) * 100)
                my_bar.progress(pct, text=f"Scaricamento scadenza: {exp}")
                
                chain = tk.option_chain(exp)
                c = chain.calls.copy()
                p = chain.puts.copy()
                
                c["expiry"] = exp
                p["expiry"] = exp
                
                all_calls.append(c)
                all_puts.append(p)
                
                # Pausa anti-ban Yahoo
                time.sleep(0.15) 
                
            except Exception:
                continue
        
        my_bar.empty() # Rimuovi barra alla fine
        
        if not all_calls: return None, None, "Errore recupero chain."

        calls = pd.concat(all_calls, ignore_index=True)
        puts = pd.concat(all_puts, ignore_index=True)

    except Exception as e:
        return None, None, f"Errore download dati: {e}"

    # Pulizia Dati
    for df in [calls, puts]:
        df.fillna(0, inplace=True)
        df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
        df["openInterest"] = pd.to_numeric(df["openInterest"], errors="coerce")
        df["impliedVolatility"] = pd.to_numeric(df["impliedVolatility"], errors="coerce")
        # Fix IV a 0 -> mettiamo default 30% per evitare crash gamma
        df["impliedVolatility"] = df["impliedVolatility"].replace(0, 0.3)

    # Filtro Range Prezzo (per alleggerire i calcoli)
    lower_bound = spot_price * (1 - range_pct/100)
    upper_bound = spot_price * (1 + range_pct/100)
    
    calls = calls[(calls["strike"] >= lower_bound) & (calls["strike"] <= upper_bound)]
    puts = puts[(puts["strike"] >= lower_bound) & (puts["strike"] <= upper_bound)]

    return calls, puts, None

def calculate_aggregated_gex(calls, puts, spot, call_sign=1, put_sign=-1, min_oi_ratio=0.05):
    risk_free = 0.05
    now_dt = datetime.now(timezone.utc)

    # Funzione helper per calcolare T (anni) per ogni riga
    def get_time_to_expiry(exp_str):
        try:
            exp_dt = datetime.strptime(str(exp_str), "%Y-%m-%d").replace(tzinfo=timezone.utc)
            # Aggiungiamo 16 ore per simulare la chiusura alle 16:00 NY
            exp_dt = exp_dt + timedelta(hours=16)
            seconds = (exp_dt - now_dt).total_seconds()
            return max(seconds / 31536000.0, 0.001) # Minimo un millesimo di anno
        except:
            return 0.001

    # Calcolo T vettorializzato (applicato a ogni riga in base alla sua scadenza)
    calls["T"] = calls["expiry"].apply(get_time_to_expiry)
    puts["T"] = puts["expiry"].apply(get_time_to_expiry)

    # Filtro OI "Rumore"
    max_oi = max(calls["openInterest"].max(), puts["openInterest"].max()) if not calls.empty else 0
    threshold = max_oi * min_oi_ratio
    
    # Calcolo Gamma Row-by-Row
    calls["gamma_val"] = vectorized_bs_gamma(spot, calls["strike"].values, calls["T"].values, risk_free, calls["impliedVolatility"].values)
    puts["gamma_val"] = vectorized_bs_gamma(spot, puts["strike"].values, puts["T"].values, risk_free, puts["impliedVolatility"].values)

    # GEX = OI * Gamma * Spot * 100
    calls["GEX"] = call_sign * calls["gamma_val"] * spot * calls["openInterest"].values * 100
    puts["GEX"] = put_sign * puts["gamma_val"] * spot * puts["openInterest"].values * 100

    # Aggregazione per Strike (Somma di tutte le scadenze)
    gex_df = pd.concat([calls[["strike", "GEX"]], puts[["strike", "GEX"]]])
    gex_by_strike = gex_df.groupby("strike")["GEX"].sum().reset_index().sort_values("strike")

    # Totali
    total_call_gex = calls["GEX"].sum()
    total_put_gex = puts["GEX"].sum()
    total_gex = total_call_gex + total_put_gex
    
    abs_total = abs(total_call_gex) + abs(total_put_gex)
    net_gamma_bias = (total_call_gex / abs_total * 100) if abs_total > 0 else 0

    # Calcolo Flip (Weighted Average)
    gamma_flip = None
    if call_sign == 1 and put_sign == -1 and abs(total_gex) > 1000:
        relevant_gex = gex_by_strike[gex_by_strike["GEX"].abs() > (gex_by_strike["GEX"].abs().max() * 0.05)]
        if not relevant_gex.empty:
            numerator = (relevant_gex["strike"] * relevant_gex["GEX"]).sum()
            denominator = relevant_gex["GEX"].sum()
            if denominator != 0:
                raw_flip = numerator / denominator
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
# 2. REPORT ANALITICO
# -----------------------------------------------------------------------------

def find_zero_crossing(df, spot):
    try:
        df = df.sort_values("strike")
        df["GEX_MA"] = df["GEX"].rolling(3, center=True, min_periods=1).mean()
        sign_change = ((df["GEX_MA"] > 0) != (df["GEX_MA"].shift(1) > 0)) & (~df["GEX_MA"].shift(1).isna())
        crossings = df[sign_change]
        if crossings.empty: return None
        crossings = crossings.copy()
        crossings["dist"] = abs(crossings["strike"] - spot)
        nearest_flip = crossings.sort_values("dist").iloc[0]["strike"]
        return nearest_flip
    except:
        return None

def get_analysis_content(spot, data, call_walls, put_walls, synced_flip):
    tot_gex = data['total_gex']
    net_bias = data['net_gamma_bias']
    effective_flip = synced_flip
    
    # Logica Regime
    regime_status = "LONG GAMMA" if tot_gex > 0 else "SHORT GAMMA"
    regime_color = "#1b5e20" if tot_gex > 0 else "#b71c1c"
    
    if net_bias > 30: bias_desc = f"Dominanza Call (Strutturale)"
    elif net_bias < -30: bias_desc = f"Dominanza Put (Strutturale)"
    else: bias_desc = f"Equilibrio / Neutrale"

    # Logica Safe Zone
    safe_zone = False
    if effective_flip is not None:
        if spot > effective_flip:
            safe_zone = True
            flip_desc = f"Prezzo SOPRA il Flip ({effective_flip:.0f}) - Safe Zone"
        else:
            safe_zone = False
            flip_desc = f"Prezzo SOTTO il Flip ({effective_flip:.0f}) - Volatility Zone"
    else:
        flip_desc = "Flip indefinito (Mercato confuso)"

    # Logica Narrativa
    scommessa = ""
    colore_bg = "#ffffff"
    bordino = "grey"

    if safe_zone and tot_gex > 0:
        scommessa = "📈 POSITIONING: RIALZISTA / BUY THE DIP"
        dettaglio = "Mercato in regime Long Gamma strutturale. I cali tendono ad essere comprati dai Dealer. I muri Call agiscono da magneti/target."
        colore_bg = "#e8f5e9"
        bordino = "#2e7d32"
    elif not safe_zone and tot_gex < 0:
        scommessa = "📉 POSITIONING: RIBASSISTA / HEDGE"
        dettaglio = "Regime Short Gamma sotto il Flip. I Dealer accelerano i ribassi. Rischio di movimenti violenti verso i Put Wall inferiori."
        colore_bg = "#ffebee"
        bordino = "#c62828"
    elif safe_zone and tot_gex < 0:
        scommessa = "⚠️ POSITIONING: CAUTO (Possibile Top)"
        dettaglio = "Siamo sopra il Flip, ma il Gamma totale è negativo. Indica che la salita è fragile e priva di supporto strutturale. Occhio ai reversal."
        colore_bg = "#fff3e0"
        bordino = "#ef6c00"
    else:
        scommessa = "⚖️ POSITIONING: LATERALE / CHOPPY"
        flip_str = f"{effective_flip:.0f}" if effective_flip is not None else "N/D"
        dettaglio = f"Prezzo intrappolato sotto il Flip ({flip_str}) ma con Gamma positivo residuo. Mercato indeciso, trading range probabile."
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
# 3. PLOTTING UNIFICATO (AGGREGATO)
# -----------------------------------------------------------------------------

def plot_dashboard_unified(symbol, data, spot, n_exps, dist_min_pct):
    calls, puts = data["calls"], data["puts"]
    gex_strike = data["gex_by_strike"]
    
    local_flip = find_zero_crossing(gex_strike, spot)
    final_flip = local_flip if local_flip else data["gamma_flip"]

    # --- WALLS LOGIC ---
    calls_agg = calls.groupby("strike")[["openInterest", "GEX"]].sum().reset_index()
    puts_agg = puts.groupby("strike")[["openInterest", "GEX"]].sum().reset_index()
    
    calls_agg["WallScore"] = calls_agg["GEX"].abs()
    puts_agg["WallScore"] = puts_agg["GEX"].abs()
    
    def get_top_levels(df, min_dist):
        df_s = df.sort_values("WallScore", ascending=False)
        levels = []
        for k in df_s["strike"]:
            if not levels or all(abs(k - x) > min_dist for x in levels):
                levels.append(k)
            if len(levels) >= 3: break
        return levels

    min_dist_val = spot * (dist_min_pct / 100.0)
    
    calls_above = calls_agg[calls_agg["strike"] > spot].copy()
    puts_below = puts_agg[puts_agg["strike"] < spot].copy()
    
    call_walls = get_top_levels(calls_above, min_dist_val)
    put_walls = get_top_levels(puts_below, min_dist_val)

    rep = get_analysis_content(spot, data, call_walls, put_walls, final_flip)

    # Setup Figura
    fig = plt.figure(figsize=(13, 9.5))
    gs = gridspec.GridSpec(2, 1, height_ratios=[2.8, 1.2], hspace=0.2) 
    
    # --- SUBPLOT 1: GRAFICO ---
    ax = fig.add_subplot(gs[0])
    bar_width = spot * 0.007
    
    # OI Aggregato
    ax.bar(puts_agg["strike"], -puts_agg["openInterest"], color="#ffcc80", alpha=0.3, width=bar_width, label="Put OI (Total)")
    ax.bar(calls_agg["strike"], calls_agg["openInterest"], color="#90caf9", alpha=0.3, width=bar_width, label="Call OI (Total)")
    
    for w in call_walls:
        val = calls_agg[calls_agg['strike']==w]['openInterest'].sum()
        ax.bar(w, val, color="#0d47a1", alpha=0.9, width=bar_width)
    for w in put_walls:
        val = -puts_agg[puts_agg['strike']==w]['openInterest'].sum()
        ax.bar(w, val, color="#e65100", alpha=0.9, width=bar_width)

    # Profilo GEX Netto Aggregato (Aree)
    ax2 = ax.twinx()
    ax2.fill_between(gex_strike["strike"], gex_strike["GEX"], 0, where=(gex_strike["GEX"]>=0), color="green", alpha=0.10, interpolate=True)
    ax2.fill_between(gex_strike["strike"], gex_strike["GEX"], 0, where=(gex_strike["GEX"]<0), color="red", alpha=0.10, interpolate=True)
    
    # --- MODIFICA VISUALE: LINEA BICOLORE (Verde Scuro / Rosso Scuro) ---
    x_vals = gex_strike["strike"].values
    y_vals = gex_strike["GEX"].values
    
    # Creazione segmenti per LineCollection
    points = np.array([x_vals, y_vals]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    # Logica Colore: Se il valore medio del segmento è positivo -> Verde Scuro, altrimenti Rosso Scuro
    colors = ["#006400" if (y_vals[i] + y_vals[i+1])/2 >= 0 else "#8B0000" for i in range(len(y_vals)-1)]
    
    lc = LineCollection(segments, colors=colors, linewidth=2.5)
    ax2.add_collection(lc)
    ax2.autoscale_view() # Importante per aggiornare i limiti dell'asse twinx
    # ---------------------------------------------------------------------
    
    ax.axvline(spot, color="blue", ls="--", lw=1.2, label="Spot")
    if final_flip:
        ax.axvline(final_flip, color="red", ls="-.", lw=1.5, label="Flip (Aggregated)")

    # Etichette
    max_y = calls_agg["openInterest"].max()
    y_offset = max_y * 0.03
    bbox_props = dict(boxstyle="round,pad=0.2", fc="white", ec="#cccccc", alpha=0.9)

    for w in call_walls:
        val = calls_agg[calls_agg['strike']==w]['openInterest'].sum()
        ax.text(w, val + y_offset, f"RES {int(w)}", color="#0d47a1", fontsize=9, fontweight='bold', ha='center', va='bottom', bbox=bbox_props, zorder=20)
    for w in put_walls:
        val = -puts_agg[puts_agg['strike']==w]['openInterest'].sum()
        ax.text(w, val - y_offset, f"SUP {int(w)}", color="#e65100", fontsize=9, fontweight='bold', ha='center', va='top', bbox=bbox_props, zorder=20)

    ax.set_ylabel("Aggregated Open Interest", fontsize=11, fontweight='bold', color="#444")
    ax2.set_ylabel("Net Gamma Exposure (Combined)")
    ax2.axhline(0, color="grey", lw=0.5)
    ax.text(0.99, 0.02, "GEX Positioning Pro v20.0", transform=ax.transAxes, ha="right", va="bottom", fontsize=14, color="#999999", fontweight="bold", alpha=0.5)

    legend_elements = [
        Patch(facecolor='#90caf9', edgecolor='none', label='Total Call OI'),
        Patch(facecolor='#ffcc80', edgecolor='none', label='Total Put OI'),
        Line2D([0], [0], color='blue', lw=1.5, ls='--', label=f'Spot {spot:.0f}'),
        Line2D([0], [0], color='#444444', lw=2.5, label='Net GEX Structure'), # Colore legenda neutro
    ]
    if final_flip: legend_elements.append(Line2D([0], [0], color='red', lw=1.5, ls='-.', label=f'Flip {final_flip:.0f}'))
    ax.legend(handles=legend_elements, loc='upper left', framealpha=0.95, fontsize=10)
    ax.set_title(f"{symbol} STRUCTURAL GEX PROFILE (Next {n_exps} Expirations)", fontsize=13, pad=10, fontweight='bold', fontfamily='sans-serif')

    # --- SUBPLOT 2: REPORT ---
    ax_rep = fig.add_subplot(gs[1])
    ax_rep.axis("off")
    
    ax_rep.text(0.02, 0.88, f"SPOT: {rep['spot']:.2f}", fontsize=12, fontweight='bold', color="#222", fontfamily='sans-serif', transform=ax_rep.transAxes)
    ax_rep.text(0.20, 0.88, "|", fontsize=12, color="#aaa", transform=ax_rep.transAxes)
    ax_rep.text(0.22, 0.88, f"REGIME: ", fontsize=12, fontweight='bold', color="#222", fontfamily='sans-serif', transform=ax_rep.transAxes)
    ax_rep.text(0.33, 0.88, rep['regime'], fontsize=12, fontweight='bold', color=rep['regime_color'], fontfamily='sans-serif', transform=ax_rep.transAxes)
    ax_rep.text(0.53, 0.88, "|", fontsize=12, color="#aaa", transform=ax_rep.transAxes)
    ax_rep.text(0.55, 0.88, f"BIAS: {rep['bias']}", fontsize=12, fontweight='bold', color="#222", fontfamily='sans-serif', transform=ax_rep.transAxes)

    flip_text = f"FLIP STRUTTURALE: {rep['flip_desc']}"
    levels_text = f"RESISTENZA CHIAVE: {rep['cw']}   |   SUPPORTO CHIAVE: {rep['pw']}"
    
    ax_rep.text(0.02, 0.72, flip_text, fontsize=11, color="#444", fontfamily='sans-serif', transform=ax_rep.transAxes)
    ax_rep.text(0.02, 0.60, levels_text, fontsize=11, color="#444", fontfamily='sans-serif', transform=ax_rep.transAxes)
    
    box_x, box_y = 0.02, 0.05
    box_w, box_h = 0.96, 0.45
    rect = patches.FancyBboxPatch((box_x, box_y), box_w, box_h, boxstyle="round,pad=0.03", linewidth=1, edgecolor="#dddddd", facecolor=rep['colore_bg'], transform=ax_rep.transAxes, zorder=1)
    ax_rep.add_patch(rect)
    accent_rect = patches.Rectangle((box_x, box_y), 0.015, box_h, facecolor=rep['bordino'], edgecolor="none", transform=ax_rep.transAxes, zorder=2)
    ax_rep.add_patch(accent_rect)
    
    sintesi_title = rep['scommessa']
    sintesi_desc = textwrap.fill(rep['dettaglio'], width=110)
    
    # Coordinate corrette per evitare sovrapposizioni
    ax_rep.text(box_x + 0.04, box_y + 0.35, sintesi_title, fontsize=14, fontweight='bold', color="#222", fontfamily='sans-serif', transform=ax_rep.transAxes)
    ax_rep.text(box_x + 0.04, box_y + 0.28, sintesi_desc, fontsize=11, color="#333", fontfamily='sans-serif', va='top', transform=ax_rep.transAxes)

    plt.tight_layout()
    return fig

# -----------------------------------------------------------------------------
# 4. INTERFACCIA STREAMLIT
# -----------------------------------------------------------------------------

st.title("⚡ GEX Positioning Pro v20.0")
st.markdown("Analisi strutturale Multi-Scadenza per Swing Trading (Lun-Ven).")

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### ⚙️ Setup Positioning")
    symbol = st.text_input("Ticker", value="SPY").upper()
    spot = get_spot_price(symbol)
    
    if spot:
        st.success(f"Spot: ${spot:.2f}")
    
    st.markdown("---")
    n_exps = st.slider("Scadenze da Aggregare", 4, 12, 8, help="8 scadenze coprono solitamente ~30-40 giorni (ideale per swing).")
    
    st.markdown("---")
    dealer_long_call = st.checkbox("Dealer Long CALL (+)", value=True)
    dealer_long_put = st.checkbox("Dealer Long PUT (+)", value=False)
    
    call_sign = 1 if dealer_long_call else -1
    put_sign = 1 if dealer_long_put else -1

    range_pct = st.slider("Range % Prezzo", 10, 40, 20, help="Filtra strike troppo lontani per pulire il grafico.")
    dist_min = st.slider("Dist. Muri", 1, 10, 2)

    btn_calc = st.button("🚀 Analizza Struttura", type="primary", use_container_width=True)

with col2:
    if btn_calc and spot:
        calls, puts, err = get_aggregated_data(symbol, spot, n_exps, range_pct)
        
        if err:
            st.error(err)
        else:
            with st.spinner("Calcolo GEX Strutturale..."):
                data_res, err_calc = calculate_aggregated_gex(
                    calls, puts, spot, 
                    call_sign=call_sign, put_sign=put_sign
                )
                
                if err_calc:
                    st.error(err_calc)
                else:
                    fig = plot_dashboard_unified(symbol, data_res, spot, n_exps, dist_min)
                    st.pyplot(fig)
                    
                    st.markdown("---")
                    buf = BytesIO()
                    fig.savefig(buf, format="png", dpi=150, bbox_inches='tight')
                    st.download_button("💾 Scarica Report Positioning", buf.getvalue(), f"GEX_STRUCT_{symbol}.png", "image/png", use_container_width=True)
