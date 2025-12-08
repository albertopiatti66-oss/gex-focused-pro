# -*- coding: utf-8 -*-
"""
GEX Positioning v20.9.4 (Final Layout Fix)
- STABILITY: Layout grafico perfezionato per evitare sovrapposizioni con testi lunghi (AI + Scenari).
- LOGIC: Auto-Detect (AI) + Analisi GEX Dealer/Istituzionali.
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
st.set_page_config(page_title="GEX Positioning V.20.9.4", layout="wide", page_icon="⚡")

# -----------------------------------------------------------------------------
# 1. MOTORE MATEMATICO & DATI
# -----------------------------------------------------------------------------

def get_market_data(ticker):
    """Scarica Spot Price, Volume Medio (ADV) e Storico per AI."""
    try:
        tk = yf.Ticker(ticker)
        # 6 mesi per SMA50 e Bollinger
        hist = tk.history(period="6mo")
        
        if hist.empty:
            return None, None, None
        
        spot = hist["Close"].iloc[-1]
        adv = hist["Volume"].tail(20).mean()
        
        return float(spot), float(adv), hist
    except Exception:
        return None, None, None

def suggest_market_context(hist):
    """AI Auto-Detect: Analizza trend e volatilità per suggerire lo scenario."""
    try:
        df = hist.copy()
        df['SMA20'] = df['Close'].rolling(window=20).mean()
        df['SMA50'] = df['Close'].rolling(window=50).mean()
        
        # Bollinger Bands (20, 2)
        df['STD'] = df['Close'].rolling(window=20).std()
        df['Upper'] = df['SMA20'] + (df['STD'] * 2)
        df['Lower'] = df['SMA20'] - (df['STD'] * 2)
        df['BB_Width'] = (df['Upper'] - df['Lower']) / df['SMA20']
        
        last = df.iloc[-1]
        
        # Squeeze Detection (15esimo percentile larghezza bande)
        low_vol_threshold = df['BB_Width'].quantile(0.15)
        is_squeezing = last['BB_Width'] <= low_vol_threshold
        
        # 1. Check Squeeze
        if is_squeezing:
            return ("Long Straddle (Volatile)", 1, "🤖 AI Detect: Compressione Volatilità (Squeeze). Attesa esplosione.")
            
        # 2. Check Trend Bullish
        elif last['Close'] > last['SMA20'] and last['SMA20'] > last['SMA50']:
            return ("Synthetic Long (Bullish)", 3, "🤖 AI Detect: Strong Uptrend (P > SMA20 > SMA50). Istituzionali Long.")
            
        # 3. Check Trend Bearish
        elif last['Close'] < last['SMA20'] and last['SMA20'] < last['SMA50']:
            return ("Synthetic Short (Bearish)", 0, "🤖 AI Detect: Strong Downtrend (P < SMA20 < SMA50). Istituzionali Short.")
            
        # 4. Range / Neutral
        else:
            return ("Short Straddle (Neutral)", 2, "🤖 AI Detect: Fase Laterale/Range. Vendita Volatilità.")
            
    except Exception as e:
        return "Short Straddle (Neutral)", 2, f"Dati insufficienti per AI ({e})"

def vectorized_bs_gamma(S, K, T, r, sigma):
    """Gamma Black-Scholes vettorializzato."""
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
                c, p = chain.calls.copy(), chain.puts.copy()
                c = c[(c['lastPrice'] >= 0.01) | (c['bid'] > 0)]
                p = p[(p['lastPrice'] >= 0.01) | (p['bid'] > 0)]
                c["expiry"], p["expiry"] = exp, exp
                all_calls.append(c)
                all_puts.append(p)
                time.sleep(0.05) 
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
        mean_iv = df[df["impliedVolatility"] > 0.001]["impliedVolatility"].mean()
        df["impliedVolatility"] = df["impliedVolatility"].replace(0, mean_iv if not pd.isna(mean_iv) else 0.3)

    lower_bound = spot_price * (1 - range_pct/100)
    upper_bound = spot_price * (1 + range_pct/100)
    calls = calls[(calls["strike"] >= lower_bound) & (calls["strike"] <= upper_bound)]
    puts = puts[(puts["strike"] >= lower_bound) & (puts["strike"] <= upper_bound)]

    return calls, puts, None

def calculate_aggregated_gex(calls, puts, spot, adv, call_sign=1, put_sign=-1):
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

    total_gex = calls["GEX"].sum() + puts["GEX"].sum()
    abs_total = abs(calls["GEX"].sum()) + abs(puts["GEX"].sum())
    net_gamma_bias = (total_gex / abs_total * 100) if abs_total > 0 else 0

    gpi_val = 0
    if adv > 0:
        gpi_val = (abs(total_gex) * (spot * 0.01) / (adv * spot)) * 100

    gamma_flip = None
    if call_sign == 1 and put_sign == -1 and abs(total_gex) > 1000:
        relevant_gex = gex_by_strike[gex_by_strike["GEX"].abs() > (gex_by_strike["GEX"].abs().max() * 0.05)]
        if not relevant_gex.empty:
            raw_flip = (relevant_gex["strike"] * relevant_gex["GEX"]).sum() / relevant_gex["GEX"].sum()
            if 0.5 * spot < raw_flip < 1.5 * spot:
                gamma_flip = raw_flip

    return {
        "calls": calls, "puts": puts, "gex_by_strike": gex_by_strike, 
        "gamma_flip": gamma_flip, "net_gamma_bias": net_gamma_bias,
        "total_gex": total_gex, "gpi": gpi_val, "risk_free_used": risk_free
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
        return crossings.sort_values("dist").iloc[0]["strike"]
    except:
        return None

def get_analysis_content(spot, data, cw_val, pw_val, synced_flip, scenario_name):
    tot_gex = data['total_gex']
    net_bias = data['net_gamma_bias']
    gpi = data['gpi']
    
    if tot_gex > 0:
        regime_status = "LONG GAMMA (Stabile)"
        regime_color = "#2E8B57" 
    else:
        regime_status = "SHORT GAMMA (Instabile)"
        regime_color = "#C0392B" 

    gpi_txt = f"{gpi:.1f}%"
    gpi_desc_short = "Basso"
    if gpi > 3.0: gpi_desc_short = "MEDIO"
    if gpi > 8.0: gpi_desc_short = "ALTO"
    if gpi > 15.0: gpi_desc_short = "ESTREMO"

    flip_desc = f"Spot vs Flip ({synced_flip:.0f})" if synced_flip else "Flip indefinito"
    safe_zone = True if synced_flip and spot > synced_flip else False

    scommessa = ""
    dettaglio = ""
    bordino = "grey"

    if tot_gex > 0:
        if gpi > 10:
            scommessa = "🛡️ POSITIONING: PINNING"
            dettaglio = f"Long Gamma + GPI alto ({gpi_txt}). Dealer bloccano il prezzo. Scenario '{scenario_name}': bassa volatilità."
        else:
            scommessa = "🛡️ POSITIONING: STABILIZZAZIONE"
            dettaglio = f"Long Gamma classico. GPI contenuto ({gpi_txt}). Mercato ammortizzato. Scenario '{scenario_name}'."
        bordino = "#2E8B57"

    elif tot_gex < 0:
        if gpi > 8.0:
            scommessa = "🔥 POSITIONING: SQUEEZE / CRASH"
            dettaglio = f"ALLARME: Short Gamma + GPI Alto ({gpi_txt}). Sotto scenario '{scenario_name}' rischio esplosione volatilità."
            bordino = "#8B0000"
        else:
            scommessa = "🔥 POSITIONING: ACCELERAZIONE"
            dettaglio = f"Short Gamma + GPI moderato. Dealer non frenano. Scenario '{scenario_name}': possibili trend veloci."
            bordino = "#C0392B"

    elif safe_zone and tot_gex < 0:
        scommessa = "⚠️ POSITIONING: FRAGILE"
        dettaglio = f"Sopra Flip ma Gamma negativo. Scenario '{scenario_name}'. Rischio inversioni rapide."
        bordino = "#E67E22"

    cw_txt = f"{int(cw_val)}" if cw_val else "-"
    pw_txt = f"{int(pw_val)}" if pw_val else "-"
    if net_bias > 5: bias_desc = f"Call (+{net_bias:.0f}%)"
    elif net_bias < -5: bias_desc = f"Put ({net_bias:.0f}%)"
    else: bias_desc = "Neutrale"

    return {
        "spot": spot, "regime": regime_status, "regime_color": regime_color,
        "bias": bias_desc, "flip_desc": flip_desc, "cw": cw_txt, "pw": pw_txt,
        "gpi": gpi_txt, "gpi_desc": gpi_desc_short, "scommessa": scommessa,
        "dettaglio": dettaglio, "bordino": bordino, "scenario_name": scenario_name
    }

# -----------------------------------------------------------------------------
# 3. PLOTTING UNIFICATO
# -----------------------------------------------------------------------------

def plot_dashboard_unified(symbol, data, spot, n_exps, dist_min_call_pct, dist_min_put_pct, scenario_name, ai_explanation):
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
            if min_dist < 0.01:
                levels.append(k)
            else:
                if not levels or all(abs(k - x) > min_dist for x in levels):
                    levels.append(k)
            if len(levels) >= 3: break
        return levels

    min_dist_call_val = spot * (dist_min_call_pct / 100.0)
    min_dist_put_val = spot * (dist_min_put_pct / 100.0)
    call_walls_candidates = get_top_levels(calls_agg[calls_agg["strike"] > spot], min_dist_call_val)
    put_walls_candidates = get_top_levels(puts_agg[puts_agg["strike"] < spot], min_dist_put_val)

    best_cw = calls_agg[calls_agg['strike'].isin(call_walls_candidates)].sort_values("WallScore", ascending=False).iloc[0]["strike"] if call_walls_candidates else None
    best_pw = puts_agg[puts_agg['strike'].isin(put_walls_candidates)].sort_values("WallScore", ascending=False).iloc[0]["strike"] if put_walls_candidates else None

    rep = get_analysis_content(spot, data, best_cw, best_pw, final_flip, scenario_name)

    # Setup Figura
    fig = plt.figure(figsize=(13, 9.5))
    gs = gridspec.GridSpec(2, 1, height_ratios=[2.8, 1.2], hspace=0.2) 
    
    # --- SUBPLOT 1: GRAFICO ---
    ax = fig.add_subplot(gs[0])
    bar_width = spot * 0.007
    
    ax.bar(puts_agg["strike"], -puts_agg["openInterest"], color="#DEB887", alpha=0.35, width=bar_width, label="Put OI", zorder=2)
    ax.bar(calls_agg["strike"], calls_agg["openInterest"], color="#4682B4", alpha=0.35, width=bar_width, label="Call OI", zorder=2)
    
    for w in call_walls_candidates:
        val = calls_agg[calls_agg['strike']==w]['openInterest'].sum()
        ax.bar(w, val, color="#21618C", alpha=1.0 if w == best_cw else 0.6, width=bar_width, zorder=3) 
    for w in put_walls_candidates:
        val = -puts_agg[puts_agg['strike']==w]['openInterest'].sum()
        ax.bar(w, val, color="#D35400", alpha=1.0 if w == best_pw else 0.6, width=bar_width, zorder=3) 

    ax2 = ax.twinx()
    gex_clean = gex_strike.dropna().sort_values("strike")
    ax2.plot(gex_clean["strike"], gex_clean["GEX"], color='#999999', linestyle=':', linewidth=2, label="Net GEX Structure", zorder=5)
    
    ax.axvline(spot, color="#2980B9", ls="--", lw=1.0, label="Spot", zorder=6)
    if final_flip: ax.axvline(final_flip, color="#7F8C8D", ls="-.", lw=1.2, label="Flip", zorder=6)
    
    max_y = calls_agg["openInterest"].max() if not calls_agg.empty else 100
    y_offset = max_y * 0.03
    bbox_props = dict(boxstyle="round,pad=0.2", fc="white", ec="#D5D8DC", alpha=0.95)

    for w in call_walls_candidates:
        val = calls_agg[calls_agg['strike']==w]['openInterest'].sum()
        font_w = 'bold' if w == best_cw else 'normal'
        ax.text(w, val + y_offset, f"RES {int(w)}", color="#21618C", fontsize=8, fontweight=font_w, ha='center', va='bottom', bbox=bbox_props, zorder=20)
    for w in put_walls_candidates:
        val = -puts_agg[puts_agg['strike']==w]['openInterest'].sum()
        font_w = 'bold' if w == best_pw else 'normal'
        ax.text(w, val - y_offset, f"SUP {int(w)}", color="#D35400", fontsize=8, fontweight=font_w, ha='center', va='top', bbox=bbox_props, zorder=20)

    ax.set_ylabel("Aggregated Open Interest", fontsize=10, fontweight='bold', color="#777")
    ax2.set_ylabel("Net Gamma Exposure", fontsize=10, color="#777")
    ax2.axhline(0, color="#BDC3C7", lw=0.5, ls='-') 
    
    legend_elements = [Patch(facecolor='#4682B4', alpha=0.5, label='Call OI'), Patch(facecolor='#DEB887', alpha=0.5, label='Put OI'), Line2D([0], [0], color='#2980B9', ls='--', label=f'Spot {spot:.0f}'), Line2D([0], [0], color='#999999', ls=':', label='Net GEX')]
    if final_flip: legend_elements.append(Line2D([0], [0], color='#7F8C8D', ls='-.', label=f'Flip {final_flip:.0f}'))
    ax.legend(handles=legend_elements, loc='upper left', framealpha=0.95, fontsize=9)
    ax.set_title(f"{symbol} GEX & GPI PROFILE (Next {n_exps} Expirations)", fontsize=13, fontweight='bold', color="#444")

    # --- SUBPLOT 2: REPORT LAYOUT SICURO ---
    ax_rep = fig.add_subplot(gs[1])
    ax_rep.axis("off")
    
    # RIGA 1 (y=0.90)
    ax_rep.text(0.02, 0.90, f"SPOT: {rep['spot']:.2f}", fontsize=11, fontweight='bold', color="#333", transform=ax_rep.transAxes)
    ax_rep.text(0.20, 0.90, "|", fontsize=11, color="#BDC3C7", transform=ax_rep.transAxes)
    ax_rep.text(0.22, 0.90, f"REGIME: ", fontsize=11, fontweight='bold', color="#333", transform=ax_rep.transAxes)
    ax_rep.text(0.33, 0.90, rep['regime'], fontsize=11, fontweight='bold', color=rep['regime_color'], transform=ax_rep.transAxes)
    # BIAS SPOSTATO A DESTRA (0.72) PER SICUREZZA
    ax_rep.text(0.72, 0.90, f"BIAS: {rep['bias']}", fontsize=11, fontweight='bold', color="#333", transform=ax_rep.transAxes)

    # RIGA 2 (y=0.78)
    ax_rep.text(0.02, 0.78, f"FLIP: {rep['flip_desc']}", fontsize=10, color="#555", transform=ax_rep.transAxes)
    ax_rep.text(0.35, 0.78, f"RES: {rep['cw']}", fontsize=10, fontweight='bold', color="#21618C", transform=ax_rep.transAxes)
    ax_rep.text(0.50, 0.78, "|", fontsize=10, color="#BDC3C7", transform=ax_rep.transAxes)
    ax_rep.text(0.52, 0.78, f"SUP: {rep['pw']}", fontsize=10, fontweight='bold', color="#D35400", transform=ax_rep.transAxes)
    
    # RIGA 3 (y=0.66)
    ax_rep.text(0.02, 0.66, f"SCENARIO IST.:", fontsize=10, color="#555", transform=ax_rep.transAxes)
    ax_rep.text(0.18, 0.66, f"{rep['scenario_name']}", fontsize=10, fontweight='bold', color="#2C3E50", transform=ax_rep.transAxes)

    # RIGA 4 (y=0.54)
    gpi_color = "#333" if "Basso" in rep['gpi_desc'] else "#C0392B"
    ax_rep.text(0.02, 0.54, "GPI (Pressure Index):", fontsize=10, color="#555", transform=ax_rep.transAxes)
    ax_rep.text(0.18, 0.54, f"{rep['gpi']} ({rep['gpi_desc']})", fontsize=10, fontweight='bold', color=gpi_color, transform=ax_rep.transAxes)

    now_str = datetime.now().strftime("%d/%m/%Y %H:%M")
    ax_rep.text(0.98, 0.54, f"Report: {now_str}", fontsize=9, color="#888", fontstyle='italic', ha='right', transform=ax_rep.transAxes)
    
    # BOX COMMENTI
    box_x, box_y = 0.02, 0.02
    box_w, box_h = 0.96, 0.48
    rect = patches.FancyBboxPatch((box_x, box_y), box_w, box_h, boxstyle="round,pad=0.03", linewidth=0.8, edgecolor="#DDDDDD", facecolor="#FFFFFF", transform=ax_rep.transAxes, zorder=1)
    ax_rep.add_patch(rect)
    accent_rect = patches.Rectangle((box_x, box_y), 0.010, box_h, facecolor=rep['bordino'], edgecolor="none", transform=ax_rep.transAxes, zorder=2)
    ax_rep.add_patch(accent_rect)
    
    full_detail = rep['dettaglio'] + "\n" + ai_explanation
    sintesi_desc = textwrap.fill(full_detail, width=110)
    
    # Titolo SINTESI
    ax_rep.text(box_x + 0.035, box_y + 0.36, rep['scommessa'], fontsize=13, fontweight='bold', color="#2C3E50", transform=ax_rep.transAxes)
    # Corpo testo (Abbassato a 0.25 per evitare collisione col titolo)
    ax_rep.text(box_x + 0.035, box_y + 0.25, sintesi_desc, fontsize=9, color="#444", va='top', transform=ax_rep.transAxes)

    plt.tight_layout()
    return fig

# -----------------------------------------------------------------------------
# 4. INTERFACCIA STREAMLIT
# -----------------------------------------------------------------------------

st.title("⚡ GEX Positioning v20.9.4 (Final Layout Fix)")
st.markdown("Analisi Strutturale con **AI Auto-Detect** e layout grafico ottimizzato.")

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### ⚙️ Setup")
    symbol = st.text_input("Ticker", value="SPY").upper()
    spot, adv, history_df = get_market_data(symbol)
    
    if spot:
        st.success(f"Spot: ${spot:.2f}")
    
    st.markdown("---")
    n_exps = st.slider("Scadenze", 4, 12, 8)
    
    st.markdown("### 🏦 AI Scenario Detect")
    
    rec_name = "Short Straddle (Neutral)"
    rec_idx = 2
    ai_expl = ""
    
    if history_df is not None:
        rec_name, rec_idx, ai_expl = suggest_market_context(history_df)
        st.info(ai_expl)
    
    scenario_options = [
        "Synthetic Short (Bearish) [L-Put / S-Call]",
        "Long Straddle (Volatile) [L-Put / L-Call]",
        "Short Straddle (Neutral) [S-Put / S-Call]",
        "Synthetic Long (Bullish) [S-Put / L-Call]"
    ]
    
    selected_scenario = st.radio("Scenario Istituzionale:", scenario_options, index=rec_idx)

    if "Synthetic Short" in selected_scenario:
        call_sign, put_sign, short_label = 1, -1, "Bearish (Synth Short)"
    elif "Long Straddle" in selected_scenario:
        call_sign, put_sign, short_label = -1, -1, "Volatile (L-Straddle)"
    elif "Short Straddle" in selected_scenario:
        call_sign, put_sign, short_label = 1, 1, "Neutral (S-Straddle)"
    elif "Synthetic Long" in selected_scenario:
        call_sign, put_sign, short_label = -1, 1, "Bullish (Synth Long)"

    range_pct = st.slider("Range %", 10, 40, 20)
    btn_calc = st.button("🚀 Analizza", type="primary", use_container_width=True)

with col2:
    if btn_calc and spot:
        calls, puts, err = get_aggregated_data(symbol, spot, n_exps, range_pct)
        if err: st.error(err)
        else:
            with st.spinner("Calcolo GEX & AI..."):
                data_res, _ = calculate_aggregated_gex(calls, puts, spot, adv, call_sign, put_sign)
                fig = plot_dashboard_unified(symbol, data_res, spot, n_exps, 2, 2, short_label, ai_expl)
                st.pyplot(fig)
                buf = BytesIO()
                fig.savefig(buf, format="png", dpi=150, bbox_inches='tight')
                st.download_button("💾 Scarica", buf.getvalue(), f"GEX_{symbol}.png", "image/png", use_container_width=True)
