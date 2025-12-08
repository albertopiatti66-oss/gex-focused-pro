# -*- coding: utf-8 -*-
"""
GEX Positioning v20.9.5 (Final Corrected)
- RESTORED: Sliders per distanza muri, Colorazione zone Gamma (Verde/Rosso).
- FIXED: Testo Flip esplicito (Sopra/Sotto), Rimossa ridondanza nome scenario nel testo.
- LOGIC: AI Auto-Detect + Analisi Istituzionale.
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
st.set_page_config(page_title="GEX Positioning V.20.9.5", layout="wide", page_icon="⚡")

# -----------------------------------------------------------------------------
# 1. MOTORE MATEMATICO & DATI
# -----------------------------------------------------------------------------

def get_market_data(ticker):
    """Scarica Spot, Volume e Storico per AI."""
    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(period="6mo")
        if hist.empty: return None, None, None
        spot = hist["Close"].iloc[-1]
        adv = hist["Volume"].tail(20).mean()
        return float(spot), float(adv), hist
    except:
        return None, None, None

def suggest_market_context(hist):
    """AI Auto-Detect."""
    try:
        df = hist.copy()
        df['SMA20'] = df['Close'].rolling(20).mean()
        df['SMA50'] = df['Close'].rolling(50).mean()
        df['STD'] = df['Close'].rolling(20).std()
        df['Upper'] = df['SMA20'] + (df['STD'] * 2)
        df['Lower'] = df['SMA20'] - (df['STD'] * 2)
        df['BB_Width'] = (df['Upper'] - df['Lower']) / df['SMA20']
        
        last = df.iloc[-1]
        is_squeezing = last['BB_Width'] <= df['BB_Width'].quantile(0.15)
        
        if is_squeezing:
            return ("Long Straddle (Volatile)", 1, "🤖 AI: Compressione Volatilità (Squeeze). Attesa esplosione.")
        elif last['Close'] > last['SMA20'] and last['SMA20'] > last['SMA50']:
            return ("Synthetic Long (Bullish)", 3, "🤖 AI: Strong Uptrend (P > SMA20 > SMA50).")
        elif last['Close'] < last['SMA20'] and last['SMA20'] < last['SMA50']:
            return ("Synthetic Short (Bearish)", 0, "🤖 AI: Strong Downtrend (P < SMA20 < SMA50).")
        else:
            return ("Short Straddle (Neutral)", 2, "🤖 AI: Fase Laterale/Range.")
    except Exception as e:
        return "Short Straddle (Neutral)", 2, f"AI Error: {e}"

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
        if not exps: return None, None, "No Expirations found."
        
        today = datetime.now().date()
        valid_exps = []
        for e in exps:
            try:
                edate = datetime.strptime(e, "%Y-%m-%d").date()
                if 0 <= (edate - today).days <= 45: valid_exps.append(e)
            except: continue
            
        if not valid_exps: return None, None, "No exp in 45 days."
        target_exps = valid_exps[:n_expirations]
        
        all_calls, all_puts = [], []
        bar = st.progress(0, "Analisi scadenze...")
        
        for i, exp in enumerate(target_exps):
            try:
                bar.progress(int((i / len(target_exps)) * 100), f"Loading {exp}")
                chain = tk.option_chain(exp)
                c, p = chain.calls.copy(), chain.puts.copy()
                c = c[(c['lastPrice'] >= 0.01) | (c['bid'] > 0)]
                p = p[(p['lastPrice'] >= 0.01) | (p['bid'] > 0)]
                c["expiry"], p["expiry"] = exp, exp
                all_calls.append(c); all_puts.append(p)
                time.sleep(0.05)
            except: continue
        bar.empty()
        
        if not all_calls: return None, None, "Empty data."
        calls = pd.concat(all_calls, ignore_index=True)
        puts = pd.concat(all_puts, ignore_index=True)
    except Exception as e: return None, None, str(e)

    for df in [calls, puts]:
        df.fillna(0, inplace=True)
        df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
        df["openInterest"] = pd.to_numeric(df["openInterest"], errors="coerce")
        df["impliedVolatility"] = pd.to_numeric(df["impliedVolatility"], errors="coerce")
        mean_iv = df[df["impliedVolatility"] > 0.001]["impliedVolatility"].mean()
        df["impliedVolatility"] = df["impliedVolatility"].replace(0, mean_iv if not pd.isna(mean_iv) else 0.3)

    lb = spot_price * (1 - range_pct/100)
    ub = spot_price * (1 + range_pct/100)
    return calls[(calls["strike"] >= lb) & (calls["strike"] <= ub)], puts[(puts["strike"] >= lb) & (puts["strike"] <= ub)], None

def calculate_aggregated_gex(calls, puts, spot, adv, call_sign=1, put_sign=-1):
    try: irx = yf.Ticker("^IRX").history(period="1d")["Close"].iloc[-1] / 100
    except: irx = 0.045
    
    now_dt = datetime.now(timezone.utc)
    def get_tte(exp_str):
        try:
            exp_dt = datetime.strptime(str(exp_str), "%Y-%m-%d").replace(tzinfo=timezone.utc) + timedelta(hours=16)
            return max((exp_dt - now_dt).total_seconds() / 31536000.0, 0.001)
        except: return 0.001

    calls["T"] = calls["expiry"].apply(get_tte)
    puts["T"] = puts["expiry"].apply(get_tte)
    calls["gamma_val"] = vectorized_bs_gamma(spot, calls["strike"].values, calls["T"].values, irx, calls["impliedVolatility"].values)
    puts["gamma_val"] = vectorized_bs_gamma(spot, puts["strike"].values, puts["T"].values, irx, puts["impliedVolatility"].values)

    calls["GEX"] = call_sign * calls["gamma_val"] * spot * calls["openInterest"].values * 100
    puts["GEX"] = put_sign * puts["gamma_val"] * spot * puts["openInterest"].values * 100

    gex_df = pd.concat([calls[["strike", "GEX"]], puts[["strike", "GEX"]]])
    gex_by_strike = gex_df.groupby("strike")["GEX"].sum().reset_index().sort_values("strike")
    
    tot_gex = calls["GEX"].sum() + puts["GEX"].sum()
    net_bias = (tot_gex / (abs(calls["GEX"].sum()) + abs(puts["GEX"].sum())) * 100) if tot_gex != 0 else 0
    gpi = (abs(tot_gex) * (spot * 0.01) / (adv * spot) * 100) if adv > 0 else 0

    gamma_flip = None
    if call_sign == 1 and put_sign == -1 and abs(tot_gex) > 1000:
        rel = gex_by_strike[gex_by_strike["GEX"].abs() > (gex_by_strike["GEX"].abs().max() * 0.05)]
        if not rel.empty:
            rf = (rel["strike"] * rel["GEX"]).sum() / rel["GEX"].sum()
            if 0.5 * spot < rf < 1.5 * spot: gamma_flip = rf

    return {
        "calls": calls, "puts": puts, "gex_by_strike": gex_by_strike,
        "gamma_flip": gamma_flip, "net_gamma_bias": net_bias, "total_gex": tot_gex,
        "gpi": gpi, "risk_free_used": irx
    }, None

# -----------------------------------------------------------------------------
# 2. REPORT ANALITICO
# -----------------------------------------------------------------------------

def find_zero_crossing(df, spot):
    try:
        df = df.sort_values("strike")
        df["GEX_MA"] = df["GEX"].rolling(3, center=True, min_periods=1).mean()
        sign_change = ((df["GEX_MA"] > 0) != (df["GEX_MA"].shift(1) > 0)) & (~df["GEX_MA"].shift(1).isna())
        crossings = df[sign_change].copy()
        if crossings.empty: return None
        crossings["dist"] = abs(crossings["strike"] - spot)
        return crossings.sort_values("dist").iloc[0]["strike"]
    except: return None

def get_analysis_content(spot, data, cw_val, pw_val, synced_flip, scenario_name):
    tot_gex = data['total_gex']
    gpi = data['gpi']
    
    regime_status = "LONG GAMMA (Stabile)" if tot_gex > 0 else "SHORT GAMMA (Instabile)"
    regime_color = "#2E8B57" if tot_gex > 0 else "#C0392B"
    
    gpi_txt = f"{gpi:.1f}%"
    gpi_desc = "Basso"
    if gpi > 3.0: gpi_desc = "MEDIO"
    if gpi > 8.0: gpi_desc = "ALTO"
    if gpi > 15.0: gpi_desc = "ESTREMO"

    # Fix: Testo Flip Esplicito
    if synced_flip:
        cond = "SOPRA" if spot > synced_flip else "SOTTO"
        flip_desc = f"Spot {cond} il Flip ({synced_flip:.0f})"
    else:
        flip_desc = "Flip indefinito"
        
    safe_zone = True if synced_flip and spot > synced_flip else False

    # Fix: Niente ripetizioni del nome scenario nel dettaglio
    if tot_gex > 0:
        if gpi > 10:
            scommessa = "🛡️ POSITIONING: PINNING"
            dettaglio = f"Long Gamma con GPI alto ({gpi_txt}). I Dealer stanno bloccando il prezzo. Volatilità compressa."
        else:
            scommessa = "🛡️ POSITIONING: STABILIZZAZIONE"
            dettaglio = f"Long Gamma classico. GPI contenuto ({gpi_txt}). Il mercato ammortizza i movimenti. 'Buy the Dip' favorito."
        bordino = "#2E8B57"
    elif tot_gex < 0:
        if gpi > 8.0:
            scommessa = "🔥 POSITIONING: SQUEEZE / CRASH"
            dettaglio = f"ALLARME: Short Gamma + GPI Alto ({gpi_txt}). I Dealer accelerano i movimenti. Rischio esplosione o crollo verticale."
            bordino = "#8B0000"
        else:
            scommessa = "🔥 POSITIONING: ACCELERAZIONE"
            dettaglio = f"Short Gamma attivo. Mancano i freni dei Dealer. Possibili trend veloci e direzionali."
            bordino = "#C0392B"
    elif safe_zone and tot_gex < 0:
        scommessa = "⚠️ POSITIONING: FRAGILE"
        dettaglio = "Sopra il Flip ma con Gamma negativo. La salita è priva di fondamenta strutturali."
        bordino = "#E67E22"
    else:
        scommessa = "⚠️ POSITIONING: NEUTRALE/INCERTO"
        dettaglio = "Configurazione mista. Attendere conferme dai livelli chiave."
        bordino = "grey"

    cw_txt = f"{int(cw_val)}" if cw_val else "-"
    pw_txt = f"{int(pw_val)}" if pw_val else "-"
    nb = data['net_gamma_bias']
    if nb > 5: bias = f"Call (+{nb:.0f}%)"
    elif nb < -5: bias = f"Put ({nb:.0f}%)"
    else: bias = "Neutrale"

    return {
        "spot": spot, "regime": regime_status, "regime_color": regime_color,
        "bias": bias, "flip_desc": flip_desc, "cw": cw_txt, "pw": pw_txt,
        "gpi": gpi_txt, "gpi_desc": gpi_desc, "scommessa": scommessa,
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
    
    # Logic Filters
    def get_top_levels(df, min_dist):
        df_s = df.sort_values("WallScore", ascending=False)
        levels = []
        for k in df_s["strike"]:
            if min_dist < 0.01: levels.append(k)
            else:
                if not levels or all(abs(k - x) > min_dist for x in levels): levels.append(k)
            if len(levels) >= 3: break
        return levels

    min_dist_call_val = spot * (dist_min_call_pct / 100.0)
    min_dist_put_val = spot * (dist_min_put_pct / 100.0)
    
    cw_cands = get_top_levels(calls_agg[calls_agg["strike"] > spot], min_dist_call_val)
    pw_cands = get_top_levels(puts_agg[puts_agg["strike"] < spot], min_dist_put_val)

    best_cw = calls_agg[calls_agg['strike'].isin(cw_cands)].sort_values("Score", ascending=False).iloc[0]["strike"] if cw_cands and "Score" in calls_agg else (cw_cands[0] if cw_cands else None)
    best_pw = puts_agg[puts_agg['strike'].isin(pw_cands)].sort_values("Score", ascending=False).iloc[0]["strike"] if pw_cands and "Score" in puts_agg else (pw_cands[0] if pw_cands else None)
    
    # Fix per selezione best se Score non esiste ancora (usiamo WallScore)
    if cw_cands: best_cw = calls_agg[calls_agg['strike'].isin(cw_cands)].sort_values("WallScore", ascending=False).iloc[0]["strike"]
    if pw_cands: best_pw = puts_agg[puts_agg['strike'].isin(pw_cands)].sort_values("WallScore", ascending=False).iloc[0]["strike"]

    rep = get_analysis_content(spot, data, best_cw, best_pw, final_flip, scenario_name)

    # Plot
    fig = plt.figure(figsize=(13, 9.5))
    gs = gridspec.GridSpec(2, 1, height_ratios=[2.8, 1.2], hspace=0.2) 
    ax = fig.add_subplot(gs[0])
    bar_width = spot * 0.007
    
    # FIX: Restore Color Zones
    x_min = min(calls_agg["strike"].min(), puts_agg["strike"].min())
    x_max = max(calls_agg["strike"].max(), puts_agg["strike"].max())
    if final_flip:
        if data['total_gex'] > 0:
            ax.axvspan(final_flip, x_max, facecolor='#E8F5E9', alpha=0.45, zorder=0) # Green Zone
        else:
            ax.axvspan(x_min, final_flip, facecolor='#FFEBEE', alpha=0.45, zorder=0) # Red Zone

    ax.bar(puts_agg["strike"], -puts_agg["openInterest"], color="#DEB887", alpha=0.35, width=bar_width, label="Put OI", zorder=2)
    ax.bar(calls_agg["strike"], calls_agg["openInterest"], color="#4682B4", alpha=0.35, width=bar_width, label="Call OI", zorder=2)
    
    for w in cw_cands:
        val = calls_agg[calls_agg['strike']==w]['openInterest'].sum()
        ax.bar(w, val, color="#21618C", alpha=1.0 if w == best_cw else 0.6, width=bar_width, zorder=3)
    for w in pw_cands:
        val = -puts_agg[puts_agg['strike']==w]['openInterest'].sum()
        ax.bar(w, val, color="#D35400", alpha=1.0 if w == best_pw else 0.6, width=bar_width, zorder=3)

    ax2 = ax.twinx()
    gex_clean = gex_strike.dropna().sort_values("strike")
    ax2.plot(gex_clean["strike"], gex_clean["GEX"], color='#999999', ls=':', lw=2, label="Net GEX", zorder=5)
    
    ax.axvline(spot, color="#2980B9", ls="--", lw=1.0, label="Spot", zorder=6)
    if final_flip: ax.axvline(final_flip, color="#7F8C8D", ls="-.", lw=1.2, label="Flip", zorder=6)
    
    max_y = calls_agg["openInterest"].max() if not calls_agg.empty else 100
    yo = max_y * 0.03
    bbox = dict(boxstyle="round,pad=0.2", fc="white", ec="#D5D8DC", alpha=0.95)

    for w in cw_cands:
        val = calls_agg[calls_agg['strike']==w]['openInterest'].sum()
        fw = 'bold' if w == best_cw else 'normal'
        ax.text(w, val + yo, f"RES {int(w)}", color="#21618C", fontsize=8, fontweight=fw, ha='center', va='bottom', bbox=bbox, zorder=20)
    for w in pw_cands:
        val = -puts_agg[puts_agg['strike']==w]['openInterest'].sum()
        fw = 'bold' if w == best_pw else 'normal'
        ax.text(w, val - yo, f"SUP {int(w)}", color="#D35400", fontsize=8, fontweight=fw, ha='center', va='top', bbox=bbox, zorder=20)

    ax.set_ylabel("OI", fontsize=10, fontweight='bold', color="#777"); ax2.set_ylabel("GEX", fontsize=10, color="#777")
    ax2.axhline(0, color="#BDC3C7", lw=0.5, ls='-')
    
    legs = [Patch(facecolor='#4682B4', alpha=0.5, label='Call OI'), Patch(facecolor='#DEB887', alpha=0.5, label='Put OI'), Line2D([0],[0], color='#2980B9', ls='--', label='Spot'), Line2D([0],[0], color='#999999', ls=':', label='GEX')]
    if final_flip: legs.append(Line2D([0],[0], color='#7F8C8D', ls='-.', label='Flip'))
    ax.legend(handles=legs, loc='upper left', fontsize=9)
    ax.set_title(f"{symbol} GEX & GPI (Next {n_exps} Exps)", fontsize=13, fontweight='bold', color="#444")

    # --- REPORT ---
    axr = fig.add_subplot(gs[1]); axr.axis("off")
    
    # Riga 1
    axr.text(0.02, 0.90, f"SPOT: {rep['spot']:.2f}", fontsize=11, fontweight='bold', color="#333", transform=axr.transAxes)
    axr.text(0.20, 0.90, "|", fontsize=11, color="#BDC3C7", transform=axr.transAxes)
    axr.text(0.22, 0.90, "REGIME:", fontsize=11, fontweight='bold', color="#333", transform=axr.transAxes)
    axr.text(0.33, 0.90, rep['regime'], fontsize=11, fontweight='bold', color=rep['regime_color'], transform=axr.transAxes)
    axr.text(0.72, 0.90, f"BIAS: {rep['bias']}", fontsize=11, fontweight='bold', color="#333", transform=axr.transAxes)

    # Riga 2
    axr.text(0.02, 0.78, f"FLIP: {rep['flip_desc']}", fontsize=10, color="#555", transform=axr.transAxes)
    axr.text(0.35, 0.78, f"RES: {rep['cw']}", fontsize=10, fontweight='bold', color="#21618C", transform=axr.transAxes)
    axr.text(0.50, 0.78, "|", fontsize=10, color="#BDC3C7", transform=axr.transAxes)
    axr.text(0.52, 0.78, f"SUP: {rep['pw']}", fontsize=10, fontweight='bold', color="#D35400", transform=axr.transAxes)
    
    # Riga 3
    axr.text(0.02, 0.66, "SCENARIO IST.:", fontsize=10, color="#555", transform=axr.transAxes)
    axr.text(0.18, 0.66, f"{rep['scenario_name']}", fontsize=10, fontweight='bold', color="#2C3E50", transform=axr.transAxes)

    # Riga 4
    gc = "#333" if "Basso" in rep['gpi_desc'] else "#C0392B"
    axr.text(0.02, 0.54, "GPI (Pressure):", fontsize=10, color="#555", transform=axr.transAxes)
    axr.text(0.18, 0.54, f"{rep['gpi']} ({rep['gpi_desc']})", fontsize=10, fontweight='bold', color=gc, transform=axr.transAxes)
    axr.text(0.98, 0.54, f"Report: {datetime.now().strftime('%d/%m %H:%M')}", fontsize=9, color="#888", fontstyle='italic', ha='right', transform=axr.transAxes)
    
    # Box
    bx, by, bw, bh = 0.02, 0.02, 0.96, 0.48
    rect = patches.FancyBboxPatch((bx, by), bw, bh, boxstyle="round,pad=0.03", ec="#DDD", fc="white", transform=axr.transAxes, zorder=1)
    axr.add_patch(rect)
    axr.add_patch(patches.Rectangle((bx, by), 0.01, bh, fc=rep['bordino'], ec="none", transform=axr.transAxes, zorder=2))
    
    fulld = rep['dettaglio'] + "\n" + ai_explanation
    axr.text(bx+0.035, by+0.36, rep['scommessa'], fontsize=13, fontweight='bold', color="#2C3E50", transform=axr.transAxes)
    axr.text(bx+0.035, by+0.25, textwrap.fill(fulld, 110), fontsize=9, color="#444", va='top', transform=axr.transAxes)

    plt.tight_layout()
    return fig

# -----------------------------------------------------------------------------
# 4. STREAMLIT UI
# -----------------------------------------------------------------------------
st.title("⚡ GEX Positioning v20.9.5")
c1, c2 = st.columns([1, 2])

with c1:
    st.markdown("### ⚙️ Setup")
    sym = st.text_input("Ticker", "SPY").upper()
    spot, adv, hist = get_market_data(sym)
    if spot: st.success(f"Spot: ${spot:.2f}")
    
    st.markdown("---")
    nex = st.slider("Scadenze", 4, 12, 8)
    
    st.markdown("### 🏦 AI Scenario")
    rname, ridx, aiexp = suggest_market_context(hist) if hist is not None else ("Neutral", 2, "")
    if hist is not None: st.info(aiexp)
    
    opts = ["Synthetic Short (Bearish)", "Long Straddle (Volatile)", "Short Straddle (Neutral)", "Synthetic Long (Bullish)"]
    sel = st.radio("Scenario:", opts, index=ridx)
    
    if "Short (Bearish)" in sel: cs, ps, sl = 1, -1, "Bearish (Synth Short)"
    elif "Long Straddle" in sel: cs, ps, sl = -1, -1, "Volatile (L-Straddle)"
    elif "Short Straddle" in sel: cs, ps, sl = 1, 1, "Neutral (S-Straddle)"
    else: cs, ps, sl = -1, 1, "Bullish (Synth Long)"

    rng = st.slider("Range %", 10, 40, 20)
    
    # FIX: Sliders Ripristinati
    st.write("🧩 Filtri Muri")
    dc = st.slider("Dist. Min. Muri CALL (%)", 0, 10, 2)
    dp = st.slider("Dist. Min. Muri PUT (%)", 0, 10, 2)
    
    btn = st.button("🚀 Analizza", type="primary", use_container_width=True)

with c2:
    if btn and spot:
        calls, puts, err = get_aggregated_data(sym, spot, nex, rng)
        if err: st.error(err)
        else:
            with st.spinner("Processing..."):
                res, _ = calculate_aggregated_gex(calls, puts, spot, adv, cs, ps)
                # Passiamo dc (dist_call) e dp (dist_put) alla funzione plot
                fig = plot_dashboard_unified(sym, res, spot, nex, dc, dp, sl, aiexp)
                st.pyplot(fig)

