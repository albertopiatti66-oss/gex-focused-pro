# -*- coding: utf-8 -*-
"""
GEX Positioning v20.7 (Swing Ready Edition)
- FIX: Layout Report (Scritte avvicinate per evitare sovrapposizioni).
- FIX: Compattamento orizzontale Resistenza/Supporto.
- LOGIC: Livelli chiave basati su POTENZA (Max GEX).
- VISUAL: Livelli chiave colorati (Blu/Rosso) nel report.
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
from datetime import datetime, timezone, timedelta
from io import BytesIO
import textwrap
import time

# Configurazione pagina
st.set_page_config(page_title="GEX Positioning V.20", layout="wide", page_icon="⚡")

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
        if not exps: return None, None, "Nessuna scadenza trovata (Yahoo API)."
        
        # --- FILTRO SCADENZE INTELLIGENTE (SWING TRADING 0-45 GG) ---
        today = datetime.now().date()
        valid_exps = []
        for e in exps:
            try:
                edate = datetime.strptime(e, "%Y-%m-%d").date()
                days_to = (edate - today).days
                if 0 <= days_to <= 45:
                    valid_exps.append(e)
            except:
                continue
        
        if not valid_exps:
            return None, None, "Nessuna scadenza trovata nei prossimi 45 giorni."

        target_exps = valid_exps[:n_expirations]
        
        all_calls = []
        all_puts = []
        
        progress_text = "Analisi scadenze Swing (0-45gg)..."
        my_bar = st.progress(0, text=progress_text)
        
        for i, exp in enumerate(target_exps):
            try:
                pct = int((i / len(target_exps)) * 100)
                my_bar.progress(pct, text=f"Scaricamento scadenza: {exp}")
                
                chain = tk.option_chain(exp)
                c = chain.calls.copy()
                p = chain.puts.copy()
                
                # --- FILTRO DATI SPAZZATURA ---
                c = c[(c['lastPrice'] >= 0.01) | (c['bid'] > 0)]
                p = p[(p['lastPrice'] >= 0.01) | (p['bid'] > 0)]

                c["expiry"] = exp
                p["expiry"] = exp
                
                all_calls.append(c)
                all_puts.append(p)
                time.sleep(0.1) 
            except Exception:
                continue
        
        my_bar.empty()
        if not all_calls: return None, None, "Errore recupero chain (Dati vuoti dopo filtro)."

        calls = pd.concat(all_calls, ignore_index=True)
        puts = pd.concat(all_puts, ignore_index=True)

    except Exception as e:
        return None, None, f"Errore download dati: {e}"

    for df in [calls, puts]:
        df.fillna(0, inplace=True)
        df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
        df["openInterest"] = pd.to_numeric(df["openInterest"], errors="coerce")
        df["impliedVolatility"] = pd.to_numeric(df["impliedVolatility"], errors="coerce")
        
        # --- VOLATILITÀ MEDIA DINAMICA ---
        mean_iv = df[df["impliedVolatility"] > 0.001]["impliedVolatility"].mean()
        fill_val = mean_iv if not pd.isna(mean_iv) else 0.3
        df["impliedVolatility"] = df["impliedVolatility"].replace(0, fill_val)

    lower_bound = spot_price * (1 - range_pct/100)
    upper_bound = spot_price * (1 + range_pct/100)
    calls = calls[(calls["strike"] >= lower_bound) & (calls["strike"] <= upper_bound)]
    puts = puts[(puts["strike"] >= lower_bound) & (puts["strike"] <= upper_bound)]

    return calls, puts, None

def calculate_aggregated_gex(calls, puts, spot, call_sign=1, put_sign=-1, min_oi_ratio=0.05):
    # --- RISK FREE RATE DINAMICO ---
    try:
        irx = yf.Ticker("^IRX").history(period="1d")["Close"].iloc[-1]
        risk_free = irx / 100 
    except:
        risk_free = 0.045

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
    net_gamma_bias = (total_gex / abs_total * 100) if abs_total > 0 else 0

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
        "total_gex": total_gex,
        "risk_free_used": risk_free
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

def get_analysis_content(spot, data, cw_val, pw_val, synced_flip):
    tot_gex = data['total_gex']
    net_bias = data['net_gamma_bias']
    effective_flip = synced_flip
    
    regime_status = "LONG GAMMA" if tot_gex > 0 else "SHORT GAMMA"
    regime_color = "#2E8B57" if tot_gex > 0 else "#C0392B"
    
    if net_bias > 5: 
        bias_desc = f"Dominanza Call (+{net_bias:.1f}%)"
    elif net_bias < -5: 
        bias_desc = f"Dominanza Put ({net_bias:.1f}%)"
    else: 
        bias_desc = f"Equilibrio ({net_bias:.1f}%)"

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

    # Formattazione livelli
    cw_txt = f"{int(cw_val)}" if cw_val else "-"
    pw_txt = f"{int(pw_val)}" if pw_val else "-"

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

def plot_dashboard_unified(symbol, data, spot, n_exps, dist_min_call_pct, dist_min_put_pct):
    calls, puts = data["calls"], data["puts"]
    gex_strike = data["gex_by_strike"]
    total_gex = data['total_gex'] 
    
    local_flip = find_zero_crossing(gex_strike, spot)
    final_flip = local_flip if local_flip else data["gamma_flip"]

    calls_agg = calls.groupby("strike")[["openInterest", "GEX"]].sum().reset_index()
    puts_agg = puts.groupby("strike")[["openInterest", "GEX"]].sum().reset_index()
    
    calls_agg["WallScore"] = calls_agg["GEX"].abs()
    puts_agg["WallScore"] = puts_agg["GEX"].abs()
    
    # --- LOGICA GESTIONE DISTANZA MURI ---
    def get_top_levels(df, min_dist):
        df_s = df.sort_values("WallScore", ascending=False)
        levels = []
        for k in df_s["strike"]:
            if min_dist < 0.01:
                levels.append(k)
            else:
                if not levels or all(abs(k - x) > min_dist for x in levels):
                    levels.append(k)
            if len(levels) >= 3: break
        return levels
    # -------------------------------------

    min_dist_call_val = spot * (dist_min_call_pct / 100.0)
    min_dist_put_val = spot * (dist_min_put_pct / 100.0)
    
    calls_above = calls_agg[calls_agg["strike"] > spot].copy()
    puts_below = puts_agg[puts_agg["strike"] < spot].copy()
    
    # 1. Trova i muri candidati (Top 3 distanziati)
    call_walls_candidates = get_top_levels(calls_above, min_dist_call_val)
    put_walls_candidates = get_top_levels(puts_below, min_dist_put_val)

    # 2. LOGICA "KING WALL": Seleziona il più forte in assoluto tra i candidati
    best_cw = None
    if call_walls_candidates:
        subset = calls_agg[calls_agg['strike'].isin(call_walls_candidates)]
        best_cw = subset.sort_values("WallScore", ascending=False).iloc[0]["strike"]

    best_pw = None
    if put_walls_candidates:
        subset = puts_agg[puts_agg['strike'].isin(put_walls_candidates)]
        best_pw = subset.sort_values("WallScore", ascending=False).iloc[0]["strike"]

    # Genera Report
    rep = get_analysis_content(spot, data, best_cw, best_pw, final_flip)

    # Setup Figura
    fig = plt.figure(figsize=(13, 9.5))
    gs = gridspec.GridSpec(2, 1, height_ratios=[2.8, 1.2], hspace=0.2) 
    
    # --- SUBPLOT 1: GRAFICO ---
    ax = fig.add_subplot(gs[0])
    bar_width = spot * 0.007
    
    x_min = min(calls_agg["strike"].min(), puts_agg["strike"].min())
    x_max = max(calls_agg["strike"].max(), puts_agg["strike"].max())
    
    # SFONDO ADATTIVO
    if final_flip:
        if total_gex > 0:
            ax.axvspan(final_flip, x_max, facecolor='#E8F5E9', alpha=0.45, zorder=0)
        else:
            ax.axvspan(x_min, final_flip, facecolor='#FFEBEE', alpha=0.45, zorder=0)
    
    # OI Aggregato
    ax.bar(puts_agg["strike"], -puts_agg["openInterest"], color="#DEB887", alpha=0.35, width=bar_width, label="Put OI", zorder=2)
    ax.bar(calls_agg["strike"], calls_agg["openInterest"], color="#4682B4", alpha=0.35, width=bar_width, label="Call OI", zorder=2)
    
    # Walls
    for w in call_walls_candidates:
        val = calls_agg[calls_agg['strike']==w]['openInterest'].sum()
        alpha_val = 1.0 if w == best_cw else 0.6
        ax.bar(w, val, color="#21618C", alpha=alpha_val, width=bar_width, zorder=3) 
        
    for w in put_walls_candidates:
        val = -puts_agg[puts_agg['strike']==w]['openInterest'].sum()
        alpha_val = 1.0 if w == best_pw else 0.6
        ax.bar(w, val, color="#D35400", alpha=alpha_val, width=bar_width, zorder=3) 

    # Profilo GEX
    ax2 = ax.twinx()
    gex_clean = gex_strike.dropna().sort_values("strike")
    
    ax2.plot(gex_clean["strike"], gex_clean["GEX"], color='#999999', linestyle=':', linewidth=2, label="Net GEX Structure", zorder=5)
    
    ax.axvline(spot, color="#2980B9", ls="--", lw=1.0, label="Spot", zorder=6)
    if final_flip:
        ax.axvline(final_flip, color="#7F8C8D", ls="-.", lw=1.2, label="Flip", zorder=6)

    # Etichette Grafico
    max_y = calls_agg["openInterest"].max() if not calls_agg.empty else 100
    y_offset = max_y * 0.03
    bbox_props = dict(boxstyle="round,pad=0.2", fc="white", ec="#D5D8DC", alpha=0.95)

    for w in call_walls_candidates:
        val = calls_agg[calls_agg['strike']==w]['openInterest'].sum()
        prefix = "★ RES" if w == best_cw else "RES"
        font_w = 'bold' if w == best_cw else 'normal'
        ax.text(w, val + y_offset, f"{prefix} {int(w)}", color="#21618C", fontsize=8, fontweight=font_w, ha='center', va='bottom', bbox=bbox_props, zorder=20)
        
    for w in put_walls_candidates:
        val = -puts_agg[puts_agg['strike']==w]['openInterest'].sum()
        prefix = "★ SUP" if w == best_pw else "SUP"
        font_w = 'bold' if w == best_pw else 'normal'
        ax.text(w, val - y_offset, f"{prefix} {int(w)}", color="#D35400", fontsize=8, fontweight=font_w, ha='center', va='top', bbox=bbox_props, zorder=20)

    ax.set_ylabel("Aggregated Open Interest", fontsize=10, fontweight='bold', color="#777")
    ax2.set_ylabel("Net Gamma Exposure", fontsize=10, color="#777")
    ax2.axhline(0, color="#BDC3C7", lw=0.5, ls='-') 
    
    ax.grid(True, which='major', axis='both', color='#EEEEEE', linestyle='-', linewidth=0.5, zorder=0)

    legend_elements = [
        Patch(facecolor='#4682B4', edgecolor='none', label='Total Call OI', alpha=0.5),
        Patch(facecolor='#DEB887', edgecolor='none', label='Total Put OI', alpha=0.5),
        Line2D([0], [0], color='#2980B9', lw=1.0, ls='--', label=f'Spot {spot:.0f}'),
        Line2D([0], [0], color='#999999', lw=2, ls=':', label='Net GEX Structure'),
    ]
    if final_flip: legend_elements.append(Line2D([0], [0], color='#7F8C8D', lw=1.2, ls='-.', label=f'Flip {final_flip:.0f}'))
    
    ax.legend(handles=legend_elements, loc='upper left', framealpha=0.95, fontsize=9, edgecolor="#EEEEEE")
    ax.set_title(f"{symbol} STRUCTURAL GEX PROFILE (Next {n_exps} Active Swing Expirations)", fontsize=13, pad=10, fontweight='bold', fontfamily='sans-serif', color="#444")

    # --- SUBPLOT 2: REPORT MINIMALISTA (LAYOUT FIX COMPATTO) ---
    ax_rep = fig.add_subplot(gs[1])
    ax_rep.axis("off")
    
    # Riga 1: Spot / Regime
    ax_rep.text(0.02, 0.88, f"SPOT: {rep['spot']:.2f}", fontsize=11, fontweight='bold', color="#333", fontfamily='sans-serif', transform=ax_rep.transAxes)
    ax_rep.text(0.20, 0.88, "|", fontsize=11, color="#BDC3C7", transform=ax_rep.transAxes)
    ax_rep.text(0.22, 0.88, f"REGIME: ", fontsize=11, fontweight='bold', color="#333", fontfamily='sans-serif', transform=ax_rep.transAxes)
    ax_rep.text(0.33, 0.88, rep['regime'], fontsize=11, fontweight='bold', color=rep['regime_color'], fontfamily='sans-serif', transform=ax_rep.transAxes)
    ax_rep.text(0.53, 0.88, "|", fontsize=11, color="#BDC3C7", transform=ax_rep.transAxes)
    
    # Riga 1 (Destra): Bias
    ax_rep.text(0.55, 0.88, f"BIAS: {rep['bias']}", fontsize=11, fontweight='bold', color="#333", fontfamily='sans-serif', transform=ax_rep.transAxes)

    # Riga 2: Flip
    flip_text = f"FLIP STRUTTURALE: {rep['flip_desc']}"
    ax_rep.text(0.02, 0.72, flip_text, fontsize=10, color="#555", fontfamily='sans-serif', transform=ax_rep.transAxes)
    
    # Riga 3: Key Levels (COMPATTATI A SINISTRA)
    # 1. Resistenza (Sinistra)
    ax_rep.text(0.02, 0.60, f"RESISTENZA CHIAVE: {rep['cw']}", fontsize=10, fontweight='bold', color="#21618C", fontfamily='sans-serif', transform=ax_rep.transAxes)
    
    # 2. Divisore (Avvicinato molto)
    ax_rep.text(0.26, 0.60, "|", fontsize=10, color="#BDC3C7", transform=ax_rep.transAxes)
    
    # 3. Supporto (Avvicinato molto)
    ax_rep.text(0.28, 0.60, f"SUPPORTO CHIAVE: {rep['pw']}", fontsize=10, fontweight='bold', color="#D35400", fontfamily='sans-serif', transform=ax_rep.transAxes)

    # --- TIMESTAMP (ALLINEATO A DESTRA) ---
    all_exps = pd.concat([calls["expiry"], puts["expiry"]]).unique()
    sorted_exps = sorted([pd.to_datetime(d) for d in all_exps])
    range_str = f"{sorted_exps[0].strftime('%d/%m')} - {sorted_exps[-1].strftime('%d/%m/%Y')}" if sorted_exps else "N/D"
    now_str = datetime.now().strftime("%d/%m/%Y %H:%M")
    timestamp_text = f"Report prodotto il {now_str}  |  Range Scadenze: {range_str}"
    
    ax_rep.text(0.98, 0.60, timestamp_text, fontsize=9, color="#888", fontstyle='italic', fontfamily='sans-serif', ha='right', transform=ax_rep.transAxes)
    # ---------------------------------------------
    
    # Box Commento
    box_x, box_y = 0.02, 0.05
    box_w, box_h = 0.96, 0.45
    rect = patches.FancyBboxPatch((box_x, box_y), box_w, box_h, boxstyle="round,pad=0.03", linewidth=0.8, edgecolor="#DDDDDD", facecolor="#FFFFFF", transform=ax_rep.transAxes, zorder=1)
    ax_rep.add_patch(rect)
    
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

st.title("⚡ GEX Positioning v20 (Swing)")
st.markdown("Analisi Strutturale Swing (0-45gg) - Dati Yahoo Filtrati.")

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### ⚙️ Setup Positioning")
    symbol = st.text_input("Ticker", value="SPY").upper()
    spot = get_spot_price(symbol)
    
    if spot:
        st.success(f"Spot: ${spot:.2f}")
    
    st.markdown("---")
    n_exps = st.slider("Scadenze da Aggregare", 4, 12, 8, help="Seleziona fino a X scadenze, ma il sistema filtrerà solo quelle entro 45 giorni.")
    
    st.markdown("---")
    dealer_long_call = st.checkbox("Dealer Long CALL (+)", value=True)
    dealer_long_put = st.checkbox("Dealer Long PUT (+)", value=False)
    
    call_sign = 1 if dealer_long_call else -1
    put_sign = 1 if dealer_long_put else -1

    range_pct = st.slider("Range % Prezzo", 10, 40, 20, help="Filtra strike troppo lontani per pulire il grafico.")
    
    st.write("---")
    st.write("🧩 Filtri Muri (0 = Mostra tutto)")
    dist_call = st.slider("Dist. Min. Muri CALL (%)", 0, 10, 2)
    dist_put = st.slider("Dist. Min. Muri PUT (%)", 0, 10, 2)

    btn_calc = st.button("🚀 Analizza Struttura", type="primary", use_container_width=True)

with col2:
    if btn_calc and spot:
        calls, puts, err = get_aggregated_data(symbol, spot, n_exps, range_pct)
        
        if err:
            st.error(err)
        else:
            with st.spinner("Calcolo GEX Strutturale (Safe Mode)..."):
                data_res, err_calc = calculate_aggregated_gex(
                    calls, puts, spot, 
                    call_sign=call_sign, put_sign=put_sign
                )
                
                if err_calc:
                    st.error(err_calc)
                else:
                    try:
                        # Mostriamo il tasso risk-free usato per trasparenza
                        rf_used = data_res.get('risk_free_used', 0.05)
                        st.caption(f"ℹ️ Parametri calcolati: Risk-Free Rate {rf_used*100:.2f}% (da T-Bill ^IRX)")

                        fig = plot_dashboard_unified(symbol, data_res, spot, n_exps, dist_call, dist_put)
                        st.pyplot(fig)
                        
                        st.markdown("---")
                        buf = BytesIO()
                        fig.savefig(buf, format="png", dpi=150, bbox_inches='tight')
                        st.download_button("💾 Scarica Report Positioning", buf.getvalue(), f"GEX_STRUCT_{symbol}.png", "image/png", use_container_width=True)
                    except Exception as e:
                        st.error(f"Errore durante la creazione del grafico: {e}")
