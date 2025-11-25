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
from matplotlib.collections import LineCollection
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
    T = np.maximum(T, 0.001) 
    sigma = np.maximum(sigma, 0.01)
    S = float(S)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        pdf = norm.pdf(d1)
        gamma = pdf / (S * sigma * np.sqrt(T))
    return np.nan_to_num(gamma)

@st.cache_data(ttl=600)
def get_aggregated_data(symbol, spot_price, n_expirations=8, range_pct=25.0):
    try:
        tk = yf.Ticker(symbol)
        exps = tk.options
        if not exps: return None, None, "Nessuna scadenza trovata."
        
        target_exps = exps[:n_expirations]
        all_calls = []
        all_puts = []
        
        progress_text = "Scaricamento scadenze in corso..."
        my_bar = st.progress(0, text=progress_text)
        
        for i, exp in enumerate(target_exps):
            try:
                pct = int((i / len(target_exps)) * 100)
                my_bar.progress(pct, text=f"Scaricamento scadenza: {exp}")
                
                chain = tk.option_chain(exp)
                c = chain.calls.copy()
                p = chain.puts.copy()
                
                c["expiry"] = exp
                p["expiry"] = exp
                
                all_calls.append(c)
                all_puts.append(p)
                time.sleep(0.15) 
            except Exception:
                continue
        
        my_bar.empty()
        if not all_calls: return None, None, "Errore recupero chain."

        calls = pd.concat(all_calls, ignore_index=True)
        puts = pd.concat(all_puts, ignore_index=True)

    except Exception as e:
        return None, None, f"Errore download dati: {e}"

    for df in [calls, puts]:
        df.fillna(0, inplace=True)
        df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
        df["openInterest"] = pd.to_numeric(df["openInterest"], errors="coerce")
        df["impliedVolatility"] = pd.to_numeric(df["impliedVolatility"], errors="coerce")
        df["impliedVolatility"] = df["impliedVolatility"].replace(0, 0.3)

    lower_bound = spot_price * (1 - range_pct/100)
    upper_bound = spot_price * (1 + range_pct/100)
    calls = calls[(calls["strike"] >= lower_bound) & (calls["strike"] <= upper_bound)]
    puts = puts[(puts["strike"] >= lower_bound) & (puts["strike"] <= upper_bound)]

    return calls, puts, None

def calculate_aggregated_gex(calls, puts, spot, call_sign=1, put_sign=-1, min_oi_ratio=0.05):
    risk_free = 0.05
    now_dt = datetime.now(timezone.utc)

    def get_time_to_expiry(exp_str):
        try:
            exp_dt = datetime.strptime(str(exp_str), "%Y-%m-%d").replace(tzinfo=timezone.utc)
            exp_dt = exp_dt + timedelta(hours=16)
            seconds = (exp_dt - now_dt).total_seconds()
            return max(seconds / 31536000.0, 0.001)
        except:
            return 0.001

    calls["T"] = calls["expiry"].apply(get_time_to_expiry)
    puts["T"] = puts["expiry"].apply(get_time_to_expiry)

    calls["gamma_val"] = vectorized_bs_gamma(spot, calls["strike"].values, calls["T"].values, risk_free, calls["impliedVolatility"].values)
    puts["gamma_val"] = vectorized_bs_gamma(spot, puts["strike"].values, puts["T"].values, risk_free, puts["impliedVolatility"].values)

    calls["GEX"] = call_sign * calls["gamma_val"] * spot * calls["openInterest"].values * 100
    puts["GEX"] = put_sign * puts["gamma_val"] * spot * puts["openInterest"].values * 100

    gex_df = pd.concat([calls[["strike", "GEX"]], puts[["strike", "GEX"]]])
    gex_by_strike = gex_df.groupby("strike")["GEX"].sum().reset_index().sort_values("strike")

    total_call_gex = calls["GEX"].sum()
    total_put_gex = puts["GEX"].sum()
    total_gex = total_call_gex + total_put_gex
    
    abs_total = abs(total_call_gex) + abs(total_put_gex)
    net_gamma_bias = (total_call_gex / abs_total * 100) if abs_total > 0 else 0

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
    
    # Colori per il testo/status
    regime_status = "LONG GAMMA" if tot_gex > 0 else "SHORT GAMMA"
    regime_color = "#2E8B57" if tot_gex > 0 else "#C0392B" # Colori più eleganti
    
    if net_bias > 30: bias_desc = f"Dominanza Call (Strutturale)"
    elif net_bias < -30: bias_desc = f"Dominanza Put (Strutturale)"
    else: bias_desc = f"Equilibrio / Neutrale"

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

    scommessa = ""
    # Nota: impostiamo bordino per l'accento, il colore_bg sarà bianco fisso nel plot
    bordino = "grey"

    if safe_zone and tot_gex > 0:
        scommessa = "📈 POSITIONING: RIALZISTA / BUY THE DIP"
        dettaglio = "Mercato in regime Long Gamma strutturale. I cali tendono ad essere comprati dai Dealer. I muri Call agiscono da magneti/target."
        bordino = "#2E8B57" # SeaGreen
    elif not safe_zone and tot_gex < 0:
        scommessa = "📉 POSITIONING: RIBASSISTA / HEDGE"
        dettaglio = "Regime Short Gamma sotto il Flip. I Dealer accelerano i ribassi. Rischio di movimenti violenti verso i Put Wall inferiori."
        bordino = "#C0392B" # Muted Red
    elif safe_zone and tot_gex < 0:
        scommessa = "⚠️ POSITIONING: CAUTO (Possibile Top)"
        dettaglio = "Siamo sopra il Flip, ma il Gamma totale è negativo. Indica che la salita è fragile e priva di supporto strutturale. Occhio ai reversal."
        bordino = "#E67E22" # Muted Orange
    else:
        scommessa = "⚖️ POSITIONING: LATERALE / CHOPPY"
        flip_str = f"{effective_flip:.0f}" if effective_flip is not None else "N/D"
        dettaglio = f"Prezzo intrappolato sotto il Flip ({flip_str}) ma con Gamma positivo residuo. Mercato indeciso, trading range probabile."
        bordino = "#95A5A6" # Grey

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
    
    # 1. VISUAL: OI Aggregato (Colori Professionali)
    # Burlywood (Sabbia/Oro spento) per Puts, SteelBlue (Blu aviazione) per Calls
    ax.bar(puts_agg["strike"], -puts_agg["openInterest"], color="#DEB887", alpha=0.4, width=bar_width, label="Put OI (Total)")
    ax.bar(calls_agg["strike"], calls_agg["openInterest"], color="#4682B4", alpha=0.4, width=bar_width, label="Call OI (Total)")
    
    # Walls (Toni più saturi degli stessi colori)
    for w in call_walls:
        val = calls_agg[calls_agg['strike']==w]['openInterest'].sum()
        ax.bar(w, val, color="#21618C", alpha=0.9, width=bar_width) # Dark Steel Blue
    for w in put_walls:
        val = -puts_agg[puts_agg['strike']==w]['openInterest'].sum()
        ax.bar(w, val, color="#D35400", alpha=0.9, width=bar_width) # Pumpkin/Rust

    # 2. VISUAL: Profilo GEX Netto (Linee più fini ed eleganti)
    ax2 = ax.twinx()
    # Aree molto trasparenti (Alpha 0.05) per non disturbare
    ax2.fill_between(gex_strike["strike"], gex_strike["GEX"], 0, where=(gex_strike["GEX"]>=0), color="#2ECC71", alpha=0.05, interpolate=True)
    ax2.fill_between(gex_strike["strike"], gex_strike["GEX"], 0, where=(gex_strike["GEX"]<0), color="#E74C3C", alpha=0.05, interpolate=True)
    
    # Linea Bicolore Elegante (Verde Bosco / Rosso Mattone) - Spessore 1.8
    x_vals = gex_strike["strike"].values
    y_vals = gex_strike["GEX"].values
    
    points = np.array([x_vals, y_vals]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    # Colori professionali per la linea
    color_pos = "#2E8B57" # SeaGreen
    color_neg = "#C0392B" # Dark Red
    colors = [color_pos if (y_vals[i] + y_vals[i+1])/2 >= 0 else color_neg for i in range(len(y_vals)-1)]
    
    lc = LineCollection(segments, colors=colors, linewidth=1.8)
    ax2.add_collection(lc)
    ax2.autoscale_view()
    
    ax.axvline(spot, color="#2980B9", ls="--", lw=1.0, label="Spot")
    if final_flip:
        ax.axvline(final_flip, color="#7F8C8D", ls="-.", lw=1.2, label="Flip")

    # Etichette (Box bianchi puliti)
    max_y = calls_agg["openInterest"].max()
    y_offset = max_y * 0.03
    bbox_props = dict(boxstyle="round,pad=0.2", fc="white", ec="#D5D8DC", alpha=0.95)

    for w in call_walls:
        val = calls_agg[calls_agg['strike']==w]['openInterest'].sum()
        ax.text(w, val + y_offset, f"RES {int(w)}", color="#21618C", fontsize=8, fontweight='bold', ha='center', va='bottom', bbox=bbox_props, zorder=20)
    for w in put_walls:
        val = -puts_agg[puts_agg['strike']==w]['openInterest'].sum()
        ax.text(w, val - y_offset, f"SUP {int(w)}", color="#D35400", fontsize=8, fontweight='bold', ha='center', va='top', bbox=bbox_props, zorder=20)

    ax.set_ylabel("Aggregated Open Interest", fontsize=10, fontweight='bold', color="#555")
    ax2.set_ylabel("Net Gamma Exposure", fontsize=10, color="#555")
    ax2.axhline(0, color="#BDC3C7", lw=0.5) # Asse zero grigio chiaro
    
    # Watermark discreto
    ax.text(0.99, 0.02, "GEX Positioning Pro v20.0", transform=ax.transAxes, ha="right", va="bottom", fontsize=12, color="#CCCCCC", fontweight="bold")

    legend_elements = [
        Patch(facecolor='#4682B4', edgecolor='none', label='Total Call OI', alpha=0.5),
        Patch(facecolor='#DEB887', edgecolor='none', label='Total Put OI', alpha=0.5),
        Line2D([0], [0], color='#2980B9', lw=1.0, ls='--', label=f'Spot {spot:.0f}'),
        Line2D([0], [0], color='#555555', lw=1.8, label='Net GEX Structure'),
    ]
    if final_flip: legend_elements.append(Line2D([0], [0], color='#7F8C8D', lw=1.2, ls='-.', label=f'Flip {final_flip:.0f}'))
    ax.legend(handles=legend_elements, loc='upper left', framealpha=0.9, fontsize=9, edgecolor="#EEEEEE")
    ax.set_title(f"{symbol} STRUCTURAL GEX PROFILE (Next {n_exps} Expirations)", fontsize=13, pad=10, fontweight='bold', fontfamily='sans-serif', color="#333")

    # --- SUBPLOT 2: REPORT MINIMALISTA ---
    ax_rep = fig.add_subplot(gs[1])
    ax_rep.axis("off")
    
    # Header Dati
    ax_rep.text(0.02, 0.88, f"SPOT: {rep['spot']:.2f}", fontsize=11, fontweight='bold', color="#333", fontfamily='sans-serif', transform=ax_rep.transAxes)
    ax_rep.text(0.20, 0.88, "|", fontsize=11, color="#BDC3C7", transform=ax_rep.transAxes)
    ax_rep.text(0.22, 0.88, f"REGIME: ", fontsize=11, fontweight='bold', color="#333", fontfamily='sans-serif', transform=ax_rep.transAxes)
    ax_rep.text(0.33, 0.88, rep['regime'], fontsize=11, fontweight='bold', color=rep['regime_color'], fontfamily='sans-serif', transform=ax_rep.transAxes)
    ax_rep.text(0.53, 0.88, "|", fontsize=11, color="#BDC3C7", transform=ax_rep.transAxes)
    ax_rep.text(0.55, 0.88, f"BIAS: {rep['bias']}", fontsize=11, fontweight='bold', color="#333", fontfamily='sans-serif', transform=ax_rep.transAxes)

    flip_text = f"FLIP STRUTTURALE: {rep['flip_desc']}"
    levels_text = f"RESISTENZA CHIAVE: {rep['cw']}   |   SUPPORTO CHIAVE: {rep['pw']}"
    
    ax_rep.text(0.02, 0.72, flip_text, fontsize=10, color="#555", fontfamily='sans-serif', transform=ax_rep.transAxes)
    ax_rep.text(0.02, 0.60, levels_text, fontsize=10, color="#555", fontfamily='sans-serif', transform=ax_rep.transAxes)
    
    # Box Riassuntivo: BIANCO con Bordo Sottile (Stile Card)
    box_x, box_y = 0.02, 0.05
    box_w, box_h = 0.96, 0.45
    rect = patches.FancyBboxPatch((box_x, box_y), box_w, box_h, boxstyle="round,pad=0.03", linewidth=0.8, edgecolor="#DDDDDD", facecolor="#FFFFFF", transform=ax_rep.transAxes, zorder=1)
    ax_rep.add_patch(rect)
    
    # Barra di accento laterale (indica il sentiment con il colore)
    accent_rect = patches.Rectangle((box_x, box_y), 0.010, box_h, facecolor=rep['bordino'], edgecolor="none", transform=ax_rep.transAxes, zorder=2)
    ax_rep.add_patch(accent_rect)
    
    sintesi_title = rep['scommessa']
    sintesi_desc = textwrap.fill(rep['dettaglio'], width=115)
    
    ax_rep.text(box_x + 0.035, box_y + 0.35, sintesi_title, fontsize=13, fontweight='bold', color="#2C3E50", fontfamily='sans-serif', transform=ax_rep.transAxes)
    ax_rep.text(box_x + 0.035, box_y + 0.28, sintesi_desc, fontsize=10, color="#444", fontfamily='sans-serif', va='top', transform=ax_rep.transAxes)

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
