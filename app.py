# -*- coding: utf-8 -*-
"""
GEX Positioning v20.9.25 (Badges Update)
- NEW: Badge "Relative Strength" (RS) vs SPY.
- NEW: Badge "Expiry Risk" (% Gamma in scadenza venerdì).
- LOGIC: Motore matematico e Obstacle Check (Median) invariati.
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
st.set_page_config(page_title="GEX Positioning V.20.9.25", layout="wide", page_icon="⚡")

# Inizializzazione Session State
if 'shared_ticker' not in st.session_state:
    st.session_state['shared_ticker'] = "NVDA"

# -----------------------------------------------------------------------------
# 1. MOTORE MATEMATICO & DATI (CORE)
# -----------------------------------------------------------------------------

def get_market_data(ticker):
    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(period="6mo")
        if hist.empty: return None, None, None
        spot = hist["Close"].iloc[-1]
        adv = hist["Volume"].tail(20).mean()
        return float(spot), float(adv), hist
    except:
        return None, None, None

def check_earnings_risk(ticker):
    try:
        tk = yf.Ticker(ticker)
        cal = tk.calendar
        if cal is None or not isinstance(cal, dict): return None
        earnings_date = None
        if 'Earnings Date' in cal:
            earnings_list = cal['Earnings Date']
            if len(earnings_list) > 0: earnings_date = earnings_list[0]
        elif 'Earnings High' in cal: earnings_date = cal['Earnings High'][0]
        if earnings_date:
            ed = earnings_date.date() if hasattr(earnings_date, 'date') else earnings_date
            today = datetime.now().date()
            delta = (ed - today).days
            if 0 <= delta <= 7:
                return f"🚨 EARNINGS ALERT: Utili previsti il {ed} (tra {delta} giorni). Rischio binario elevato!"
        return None
    except: return None

def check_volume_obstacle(hist, entry, target):
    """Analizza il Volume Profile usando la MEDIANA (Robust Obstacle Check)."""
    try:
        recent = hist.tail(90)
        price_data = recent['Close']
        vol_data = recent['Volume']
        lower = min(entry, target); upper = max(entry, target)
        full_counts, full_bins = np.histogram(price_data, bins=70, weights=vol_data)
        valid_counts = full_counts[full_counts > 0]
        if len(valid_counts) == 0: return False, None
        baseline_median = np.median(valid_counts)
        threshold = baseline_median * 2.5
        bin_centers = 0.5 * (full_bins[:-1] + full_bins[1:])
        mask = (bin_centers >= lower) & (bin_centers <= upper)
        path_counts = full_counts[mask]; path_prices = bin_centers[mask]
        if len(path_counts) == 0: return False, None
        max_vol_path_idx = np.argmax(path_counts)
        max_vol_path = path_counts[max_vol_path_idx]
        obstacle_price = path_prices[max_vol_path_idx]
        if max_vol_path > threshold:
            if abs(obstacle_price - entry) > (entry * 0.01): return True, obstacle_price
        return False, None
    except: return False, None

def calculate_technical_squeeze(hist):
    try:
        df = hist.copy()
        df['SMA20'] = df['Close'].rolling(20).mean()
        df['STD'] = df['Close'].rolling(20).std()
        df['Upper'] = df['SMA20'] + (df['STD'] * 2)
        df['Lower'] = df['SMA20'] - (df['STD'] * 2)
        df['BB_Width'] = (df['Upper'] - df['Lower']) / df['SMA20']
        is_squeeze = df['BB_Width'].iloc[-1] <= df['BB_Width'].quantile(0.15)
        return is_squeeze
    except: return False

def suggest_market_context(hist):
    try:
        df = hist.copy()
        df['SMA20'] = df['Close'].rolling(20).mean()
        df['SMA50'] = df['Close'].rolling(50).mean()
        is_sqz = calculate_technical_squeeze(df)
        last = df.iloc[-1]
        price = last['Close']
        sma20 = last['SMA20']
        sma50 = last['SMA50']
        if is_sqz: return ("Long Straddle (Volatile)", 1, "🤖 SQUEEZE: Volatilità compressa. Istituzionali comprano gamma in attesa di esplosione.")
        if price > sma20 and sma20 > sma50: return ("Synthetic Long (Bullish)", 3, "🤖 STRONG UPTREND: Prezzo sopra medie allineate. Istituzionali Long Call / Short Put.")
        elif price < sma20 and sma20 < sma50: return ("Synthetic Short (Bearish)", 0, "🤖 STRONG DOWNTREND: Prezzo sotto medie allineate. Istituzionali Long Put / Short Call.")
        elif sma20 > sma50 and sma50 < price < sma20: return ("Synthetic Long (Bullish)", 3, "🤖 BULL PULLBACK / BUY ZONE: Ritracciamento su supporto dinamico. Ist. vendono Put (Floor) per accumulare.")
        elif sma20 < sma50 and sma20 < price < sma50: return ("Synthetic Short (Bearish)", 0, "🤖 BEAR RALLY / FADE ZONE: Rimbalzo tecnico verso la MA50 (Resistenza). Ist. vendono Call (Muro) o comprano Put.")
        elif sma20 > sma50 and price < sma50: return ("Synthetic Long (Bullish)", 3, "⚠️ CRITICAL REVERSAL ALERT: Struttura Bull (20>50) ma Prezzo sotto MA50. Tentativo di supporto estremo o Stop Loss massicci.")
        elif sma20 < sma50 and price > sma50: return ("Synthetic Short (Bearish)", 0, "⚠️ CRITICAL BREAKOUT ALERT: Struttura Bear (20<50) ma Prezzo sopra MA50. Rischio Short Squeeze o Bull Trap finale.")
        else: return ("Short Straddle (Neutral)", 2, "🤖 CHOPPY / UNDEFINED: Nessuna direzione chiara. Vendita volatilità.")
    except Exception as e: return "Short Straddle (Neutral)", 2, f"AI Error: {e}"

def vectorized_bs_gamma(S, K, T, r, sigma):
    T = np.maximum(T, 0.001); sigma = np.maximum(sigma, 0.01); S = float(S)
    with np.errstate(divide='ignore', invalid='ignore'):
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        pdf = np.exp(-0.5 * d1**2) / np.sqrt(2 * np.pi)
        gamma = pdf / (S * sigma * np.sqrt(T))
    return np.nan_to_num(gamma)

@st.cache_data(ttl=600)
def get_aggregated_data(symbol, spot_price, n_expirations=8, range_pct=25.0):
    try:
        tk = yf.Ticker(symbol)
        exps = tk.options
        if not exps: return None, None, "No Data"
        today = datetime.now().date()
        valid_exps = []
        for e in exps:
            try:
                edate = datetime.strptime(e, "%Y-%m-%d").date()
                if 0 <= (edate - today).days <= 45: valid_exps.append(e)
            except: continue
        if not valid_exps: return None, None, "No Exps < 45 days"
        target_exps = valid_exps[:n_expirations]
        all_calls, all_puts = [], []
        for i, exp in enumerate(target_exps):
            try:
                chain = tk.option_chain(exp)
                c, p = chain.calls.copy(), chain.puts.copy()
                c = c[(c['lastPrice'] >= 0.01) | (c['bid'] > 0)]
                p = p[(p['lastPrice'] >= 0.01) | (p['bid'] > 0)]
                c["expiry"], p["expiry"] = exp, exp
                all_calls.append(c); all_puts.append(p)
            except: continue
        if not all_calls: return None, None, "Empty Data"
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

def calculate_gex_metrics(calls, puts, spot, adv, call_sign, put_sign):
    try: irx = 0.045
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
    abs_tot = abs(calls["GEX"].sum()) + abs(puts["GEX"].sum())
    net_bias = (tot_gex / abs_tot * 100) if abs_tot > 0 else 0
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
    }

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

# -----------------------------------------------------------------------------
# 2. NUOVE FUNZIONI BADGES (RS & EXPIRY)
# -----------------------------------------------------------------------------

def get_rs_badge(ticker):
    """Calcola la Forza Relativa (RS) vs SPY."""
    try:
        if ticker == "SPY": return ""
        df = yf.download([ticker, "SPY"], period="10d", progress=False)['Close']
        if df.empty: return ""
        
        # Gestione Multi-Index se scarica entrambi
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(0) # Appiattisci se necessario (dipende versione yfinance)
            
        # Calcolo Performance
        # Nota: yfinance a volte cambia format, usiamo il metodo più sicuro: pct_change
        perf_ticker = (df[ticker].iloc[-1] - df[ticker].iloc[0]) / df[ticker].iloc[0]
        perf_spy = (df["SPY"].iloc[-1] - df["SPY"].iloc[0]) / df["SPY"].iloc[0]
        
        rs = perf_ticker - perf_spy
        
        if rs > 0.01: # +1% vs SPY
            return f"<span style='background-color:#E8F5E9; color:#2E8B57; padding:4px 8px; border-radius:5px; font-weight:bold; border: 1px solid #2E8B57; font-size:12px;'>🚀 LEADER (RS +{rs*100:.1f}%)</span>"
        elif rs < -0.01: # -1% vs SPY
            return f"<span style='background-color:#FFEBEE; color:#C0392B; padding:4px 8px; border-radius:5px; font-weight:bold; border: 1px solid #C0392B; font-size:12px;'>🐢 LAGGARD (RS {rs*100:.1f}%)</span>"
        else:
            return f"<span style='background-color:#F5F5F5; color:#555; padding:4px 8px; border-radius:5px; font-weight:bold; border: 1px solid #CCC; font-size:12px;'>⚖️ IN LINE (RS {rs*100:.1f}%)</span>"
    except:
        return ""

def get_expiry_badge(calls, puts, total_gex):
    """Calcola quanto Gamma scade entro questo Venerdì."""
    try:
        today = datetime.now().date()
        # Trova prossimo venerdì
        days_ahead = 4 - today.weekday()
        if days_ahead < 0: days_ahead += 7
        next_friday = today + timedelta(days=days_ahead)
        next_friday_str = next_friday.strftime('%Y-%m-%d')
        
        # Filtra GEX in scadenza
        c_exp = calls[calls['expiry'] <= next_friday_str]['GEX'].abs().sum()
        p_exp = puts[puts['expiry'] <= next_friday_str]['GEX'].abs().sum()
        
        expiring_gex = c_exp + p_exp
        total_abs_gex = calls['GEX'].abs().sum() + puts['GEX'].abs().sum()
        
        if total_abs_gex == 0: return ""
        
        ratio = expiring_gex / total_abs_gex
        
        if ratio > 0.35: # >35% scade questa settimana
            return f"<span style='background-color:#FFF3E0; color:#EF6C00; padding:4px 8px; border-radius:5px; font-weight:bold; border: 1px solid #EF6C00; font-size:12px;'>⚠️ EXPIRY RISK: {ratio:.0%} scade Ven.</span>"
        else:
            return f"<span style='background-color:#E3F2FD; color:#1565C0; padding:4px 8px; border-radius:5px; font-weight:bold; border: 1px solid #1565C0; font-size:12px;'>🔒 SOLID: Solo {ratio:.0%} scade Ven.</span>"
    except:
        return ""

# -----------------------------------------------------------------------------
# 3. FUNZIONI GRAFICHE & REPORT
# -----------------------------------------------------------------------------

def get_analysis_content(spot, data, cw_val, pw_val, synced_flip, scenario_name):
    tot_gex = data['total_gex']; gpi = data['gpi']
    regime_status = "LONG GAMMA (Stabile)" if tot_gex > 0 else "SHORT GAMMA (Instabile)"
    regime_color = "#2E8B57" if tot_gex > 0 else "#C0392B"
    gpi_txt = f"{gpi:.1f}%"; gpi_desc = "Basso"
    if gpi > 3.0: gpi_desc = "MEDIO"; 
    if gpi > 8.0: gpi_desc = "ALTO"
    if gpi > 15.0: gpi_desc = "ESTREMO"
    if synced_flip: cond = "SOPRA" if spot > synced_flip else "SOTTO"; flip_desc = f"Spot {cond} il Flip ({synced_flip:.0f})"
    else: flip_desc = "Flip indefinito"
    safe_zone = True if synced_flip and spot > synced_flip else False
    if tot_gex > 0:
        if gpi > 10: scommessa = "🛡️ POSITIONING: PINNING"; dettaglio = f"Long Gamma con GPI alto ({gpi_txt}). Dealer bloccano il prezzo."
        else: scommessa = "🛡️ POSITIONING: STABILIZZAZIONE"; dettaglio = f"Long Gamma classico. GPI contenuto ({gpi_txt}). Mercato ammortizzato."
        bordino = "#2E8B57"
    elif tot_gex < 0:
        if gpi > 8.0: scommessa = "🔥 POSITIONING: SQUEEZE / CRASH"; dettaglio = f"ALLARME: Short Gamma + GPI Alto ({gpi_txt}). Rischio esplosione volatilità."; bordino = "#8B0000"
        else: scommessa = "🔥 POSITIONING: ACCELERAZIONE"; dettaglio = f"Short Gamma attivo. Mancano freni. Possibili trend veloci."; bordino = "#C0392B"
    elif safe_zone and tot_gex < 0: scommessa = "⚠️ POSITIONING: FRAGILE"; dettaglio = "Sopra Flip ma Gamma negativo. Salita fragile."; bordino = "#E67E22"
    else: scommessa = "⚠️ POSITIONING: NEUTRALE"; dettaglio = "Configurazione mista."; bordino = "grey"
    cw_txt = f"{int(cw_val)}" if cw_val else "-"; pw_txt = f"{int(pw_val)}" if pw_val else "-"
    nb = data['net_gamma_bias']
    if nb > 5: bias = f"Call (+{nb:.0f}%)"
    elif nb < -5: bias = f"Put ({nb:.0f}%)"
    else: bias = "Neutrale"
    return { "spot": spot, "regime": regime_status, "regime_color": regime_color, "bias": bias, "flip_desc": flip_desc, "cw": cw_txt, "pw": pw_txt, "gpi": gpi_txt, "gpi_desc": gpi_desc, "scommessa": scommessa, "dettaglio": dettaglio, "bordino": bordino, "scenario_name": scenario_name }

def plot_dashboard_unified(symbol, data, spot, hist, n_exps, dist_min_call_pct, dist_min_put_pct, scenario_name, ai_explanation):
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
            if min_dist < 0.01: levels.append(k)
            else:
                if not levels or all(abs(k - x) > min_dist for x in levels): levels.append(k)
            if len(levels) >= 3: break
        return levels
    min_dist_call_val = spot * (dist_min_call_pct / 100.0)
    min_dist_put_val = spot * (dist_min_put_pct / 100.0)
    cw_cands = get_top_levels(calls_agg[calls_agg["strike"] > spot], min_dist_call_val)
    pw_cands = get_top_levels(puts_agg[puts_agg["strike"] < spot], min_dist_put_val)
    best_cw = calls_agg[calls_agg['strike'].isin(cw_cands)].sort_values("WallScore", ascending=False).iloc[0]["strike"] if cw_cands else None
    best_pw = puts_agg[puts_agg['strike'].isin(pw_cands)].sort_values("WallScore", ascending=False).iloc[0]["strike"] if pw_cands else None
    rep = get_analysis_content(spot, data, best_cw, best_pw, final_flip, scenario_name)
    fig = plt.figure(figsize=(13, 12)); gs = gridspec.GridSpec(3, 1, height_ratios=[1.8, 2.0, 1.0], hspace=0.25)
    ax_p = fig.add_subplot(gs[0])
    price_hist = hist['Close']; vol_hist = hist['Volume']; vp_bins = 50
    counts, bins = np.histogram(price_hist, bins=vp_bins, weights=vol_hist)
    df_plot = hist.tail(80).copy(); df_plot.reset_index(drop=True, inplace=True)
    max_vol = counts.max(); scale_factor = (len(df_plot) * 0.3) / max_vol if max_vol > 0 else 0
    for i in range(len(counts)): ax_p.barh(bins[i], counts[i] * scale_factor, height=(bins[i+1]-bins[i]), left=0, color='#D3D3D3', alpha=0.4, zorder=0)
    if 'SMA20' not in df_plot.columns: df_plot['SMA20'] = df_plot['Close'].rolling(20).mean()
    if 'SMA50' not in df_plot.columns: df_plot['SMA50'] = df_plot['Close'].rolling(50).mean()
    up = df_plot[df_plot.Close >= df_plot.Open]; down = df_plot[df_plot.Close < df_plot.Open]
    ax_p.bar(up.index, up.Close - up.Open, 0.6, bottom=up.Open, color='#26A69A', alpha=0.9, zorder=2)
    ax_p.bar(up.index, up.High - up.Close, 0.05, bottom=up.Close, color='#26A69A', zorder=2)
    ax_p.bar(up.index, up.Low - up.Open, 0.05, bottom=up.Open, color='#26A69A', zorder=2)
    ax_p.bar(down.index, down.Close - down.Open, 0.6, bottom=down.Open, color='#EF5350', alpha=0.9, zorder=2)
    ax_p.bar(down.index, down.High - down.Open, 0.05, bottom=down.Open, color='#EF5350', zorder=2)
    ax_p.bar(down.index, down.Low - down.Close, 0.05, bottom=down.Close, color='#EF5350', zorder=2)
    ax_p.plot(df_plot.index, df_plot['SMA20'], color='#2979FF', lw=1.5, label='SMA 20', zorder=3)
    ax_p.plot(df_plot.index, df_plot['SMA50'], color='#FF1744', lw=1.5, label='SMA 50', zorder=3)
    right_idx = df_plot.index[-1] + 1
    if best_cw: ax_p.axhline(best_cw, color='#21618C', linestyle='--', linewidth=1.2, alpha=0.8, zorder=4); ax_p.text(right_idx, best_cw, f" Call Wall ${int(best_cw)} ", color='white', fontsize=8, fontweight='bold', va='center', ha='left', bbox=dict(boxstyle="square,pad=0.2", fc="#21618C", ec="none"), zorder=10)
    if best_pw: ax_p.axhline(best_pw, color='#D35400', linestyle='--', linewidth=1.2, alpha=0.8, zorder=4); ax_p.text(right_idx, best_pw, f" Put Wall ${int(best_pw)} ", color='white', fontsize=8, fontweight='bold', va='center', ha='left', bbox=dict(boxstyle="square,pad=0.2", fc="#D35400", ec="none"), zorder=10)
    ax_p.set_title(f"{symbol} Trend + Volume Profile (Gray) + Option Walls", fontsize=11, fontweight='bold', color='#444'); ax_p.legend(loc='upper left', fontsize=8); ax_p.grid(True, alpha=0.2); ax_p.set_xlim(-1, len(df_plot) + 6); ax_p.set_xticks([])
    ax = fig.add_subplot(gs[1]); bar_width = spot * 0.007; x_min = min(calls_agg["strike"].min(), puts_agg["strike"].min()); x_max = max(calls_agg["strike"].max(), puts_agg["strike"].max())
    if final_flip:
        if data['total_gex'] > 0: ax.axvspan(final_flip, x_max, facecolor='#E8F5E9', alpha=0.45, zorder=0) 
        else: ax.axvspan(x_min, final_flip, facecolor='#FFEBEE', alpha=0.45, zorder=0) 
    ax.bar(puts_agg["strike"], -puts_agg["openInterest"], color="#DEB887", alpha=0.35, width=bar_width, label="Put OI", zorder=2)
    ax.bar(calls_agg["strike"], calls_agg["openInterest"], color="#4682B4", alpha=0.35, width=bar_width, label="Call OI", zorder=2)
    for w in cw_cands: val = calls_agg[calls_agg['strike']==w]['openInterest'].sum(); ax.bar(w, val, color="#21618C", alpha=1.0 if w == best_cw else 0.6, width=bar_width, zorder=3)
    for w in pw_cands: val = -puts_agg[puts_agg['strike']==w]['openInterest'].sum(); ax.bar(w, val, color="#D35400", alpha=1.0 if w == best_pw else 0.6, width=bar_width, zorder=3)
    ax2 = ax.twinx(); gex_clean = gex_strike.dropna().sort_values("strike"); ax2.plot(gex_clean["strike"], gex_clean["GEX"], color='#999999', ls=':', lw=2, label="Net GEX", zorder=5)
    ax.axvline(spot, color="#2980B9", ls="--", lw=1.0, label="Spot", zorder=6); 
    if final_flip: ax.axvline(final_flip, color="red", ls="-.", lw=2.0, label="Flip", zorder=6)
    max_y = calls_agg["openInterest"].max() if not calls_agg.empty else 100; yo = max_y * 0.03; bbox = dict(boxstyle="round,pad=0.2", fc="white", ec="#D5D8DC", alpha=0.95)
    for w in cw_cands: val = calls_agg[calls_agg['strike']==w]['openInterest'].sum(); fw = 'bold' if w == best_cw else 'normal'; ax.text(w, val + yo, f"RES {int(w)}", color="#21618C", fontsize=8, fontweight=fw, ha='center', va='bottom', bbox=bbox, zorder=20)
    for w in pw_cands: val = -puts_agg[puts_agg['strike']==w]['openInterest'].sum(); fw = 'bold' if w == best_pw else 'normal'; ax.text(w, val - yo, f"SUP {int(w)}", color="#D35400", fontsize=8, fontweight=fw, ha='center', va='top', bbox=bbox, zorder=20)
    ax.set_ylabel("OI", fontsize=10, fontweight='bold', color="#777"); ax2.set_ylabel("GEX", fontsize=10, color="#777"); ax2.axhline(0, color="#BDC3C7", lw=0.5, ls='-'); ax.legend(loc='upper left', fontsize=9); ax.set_title(f"Options Positioning (Next {n_exps} Exps)", fontsize=11, fontweight='bold', color="#444")
    axr = fig.add_subplot(gs[2]); axr.axis("off")
    axr.text(0.02, 0.90, f"SPOT: {rep['spot']:.2f}", fontsize=11, fontweight='bold', color="#333", transform=axr.transAxes)
    axr.text(0.20, 0.90, "|", fontsize=11, color="#BDC3C7", transform=axr.transAxes)
    axr.text(0.22, 0.90, "REGIME:", fontsize=11, fontweight='bold', color="#333", transform=axr.transAxes)
    axr.text(0.33, 0.90, rep['regime'], fontsize=11, fontweight='bold', color=rep['regime_color'], transform=axr.transAxes)
    axr.text(0.72, 0.90, f"BIAS: {rep['bias']}", fontsize=11, fontweight='bold', color="#333", transform=axr.transAxes)
    axr.text(0.02, 0.76, f"FLIP: {rep['flip_desc']}", fontsize=10, color="#555", transform=axr.transAxes)
    axr.text(0.35, 0.76, f"RES: {rep['cw']}", fontsize=10, fontweight='bold', color="#21618C", transform=axr.transAxes)
    axr.text(0.50, 0.76, "|", fontsize=10, color="#BDC3C7", transform=axr.transAxes)
    axr.text(0.52, 0.76, f"SUP: {rep['pw']}", fontsize=10, fontweight='bold', color="#D35400", transform=axr.transAxes)
    axr.text(0.02, 0.62, "SCENARIO IST.:", fontsize=10, color="#555", transform=axr.transAxes); axr.text(0.18, 0.62, f"{rep['scenario_name']}", fontsize=10, fontweight='bold', color="#2C3E50", transform=axr.transAxes)
    gc = "#333" if "Basso" in rep['gpi_desc'] else "#C0392B"; axr.text(0.02, 0.48, "GPI (Pressure):", fontsize=10, color="#555", transform=axr.transAxes); axr.text(0.18, 0.48, f"{rep['gpi']} ({rep['gpi_desc']})", fontsize=10, fontweight='bold', color=gc, transform=axr.transAxes); axr.text(0.98, 0.48, f"Report: {datetime.now().strftime('%d/%m %H:%M')}", fontsize=9, color="#888", fontstyle='italic', ha='right', transform=axr.transAxes)
    bx, by, bw, bh = 0.02, 0.02, 0.96, 0.42; rect = patches.FancyBboxPatch((bx, by), bw, bh, boxstyle="round,pad=0.03", ec="#DDD", fc="white", transform=axr.transAxes, zorder=1); axr.add_patch(rect); axr.add_patch(patches.Rectangle((bx, by), 0.01, bh, fc=rep['bordino'], ec="none", transform=axr.transAxes, zorder=2))
    fulld = rep['dettaglio'] + "\n" + ai_explanation; axr.text(bx+0.035, by+0.30, rep['scommessa'], fontsize=12, fontweight='bold', color="#2C3E50", transform=axr.transAxes); axr.text(bx+0.035, by+0.18, textwrap.fill(fulld, 110), fontsize=9, color="#444", va='top', transform=axr.transAxes)
    plt.tight_layout(); return fig

def plot_probability_cone(spot, iv, target_price, days=30):
    t_vals = np.arange(1, days + 1); sigma = iv
    upper1 = spot * np.exp(sigma * np.sqrt(t_vals / 365.0)); lower1 = spot * np.exp(-sigma * np.sqrt(t_vals / 365.0))
    upper2 = spot * np.exp(2 * sigma * np.sqrt(t_vals / 365.0)); lower2 = spot * np.exp(-2 * sigma * np.sqrt(t_vals / 365.0))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t_vals, upper1, color='green', linestyle='--', alpha=0.7, label='1 SD (68%)'); ax.plot(t_vals, lower1, color='green', linestyle='--', alpha=0.7)
    ax.plot(t_vals, upper2, color='orange', linestyle=':', alpha=0.6, label='2 SD (95%)'); ax.plot(t_vals, lower2, color='orange', linestyle=':', alpha=0.6)
    ax.fill_between(t_vals, lower1, upper1, color='green', alpha=0.1); ax.fill_between(t_vals, lower2, upper2, color='orange', alpha=0.05)
    ax.axhline(spot, color='blue', alpha=0.5, label='Spot Price'); ax.axhline(target_price, color='purple', linestyle='-.', linewidth=2, label=f'Target: ${target_price:.2f}')
    ax.set_title(f"Probability Cone (IV: {iv:.1%}) - Next {days} Days"); ax.set_xlabel("Days Forward"); ax.set_ylabel("Price"); ax.legend(loc='upper left')
    final_1sd_up = upper1[-1]; final_1sd_low = lower1[-1]; final_2sd_up = upper2[-1]; final_2sd_low = lower2[-1]
    is_inside_1sd = final_1sd_low <= target_price <= final_1sd_up; is_inside_2sd = final_2sd_low <= target_price <= final_2sd_up
    if is_inside_1sd: explanation = "✅ **TARGET AD ALTA PROBABILITÀ:** Il livello rientra nella prima deviazione standard (68%). È statisticamente molto probabile che venga toccato."
    elif is_inside_2sd: explanation = "⚖️ **TARGET MEDIO/ALTO:** Il livello è nella zona tra 1 e 2 SD. Richiede un movimento deciso ma non impossibile (Probabilità < 30%)."
    else: explanation = "⚠️ **TARGET AMBIZIOSO / IMPROBABILE:** Il livello è oltre le 2 Deviazioni Standard (zona estrema). La probabilità statistica è inferiore al 5%."
    return fig, explanation

# -----------------------------------------------------------------------------
# 4. UI PRINCIPALE
# -----------------------------------------------------------------------------

st.title("⚡ GEX Positioning Suite v20")
tab1, tab2, tab3 = st.tabs(["📊 Analisi Singola", "🧪 Strategy Lab", "🔥 Squeeze Scanner"])

# --- TAB 1: ANALISI SINGOLA ---
with tab1:
    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown("### ⚙️ Setup")
        sym_input = st.text_input("Ticker", st.session_state['shared_ticker'], key="tab1_ticker").upper()
        if sym_input != st.session_state['shared_ticker']: st.session_state['shared_ticker'] = sym_input
        sym = st.session_state['shared_ticker']
        spot, adv, hist = get_market_data(sym)
        if spot: st.success(f"Spot: ${spot:.2f}")
        earn_alert = check_earnings_risk(sym); 
        if earn_alert: st.error(earn_alert)
        
        # BADGE RS (TAB 1)
        if spot:
            rs_html = get_rs_badge(sym)
            if rs_html: st.markdown(rs_html, unsafe_allow_html=True)
            
        nex = st.slider("Scadenze", 4, 12, 8, help="Num. scadenze future")
        st.markdown("### 🤖 AI Market Scenario")
        rname, ridx, aiexp = suggest_market_context(hist) if hist is not None else ("Neutral", 2, "")
        if hist is not None:
            if "⚠️" in aiexp: st.warning(aiexp)
            else: st.info(aiexp)
        opts = ["Synthetic Short (Bearish)", "Long Straddle (Volatile)", "Short Straddle (Neutral)", "Synthetic Long (Bullish)"]
        sel = st.radio("Seleziona Scenario:", opts, index=ridx)
        if "Short (Bearish)" in sel: cs, ps, sl = 1, -1, "Bearish"; st.markdown("""<div style='background-color:#ffebee;padding:8px;border-left:4px solid #d32f2f'><b>🐻 BEARISH:</b> Ist. Long Put. Dealer: Long Call / Short Put.</div>""", unsafe_allow_html=True)
        elif "Long Straddle" in sel: cs, ps, sl = -1, -1, "Volatile"; st.markdown("""<div style='background-color:#fff3e0;padding:8px;border-left:4px solid #f57c00'><b>💥 VOLATILE:</b> Ist. Long Straddle. Dealer: Short Gamma.</div>""", unsafe_allow_html=True)
        elif "Short Straddle" in sel: cs, ps, sl = 1, 1, "Neutral"; st.markdown("""<div style='background-color:#e8f5e9;padding:8px;border-left:4px solid #388e3c'><b>💤 NEUTRAL:</b> Ist. Short Straddle. Dealer: Long Gamma.</div>""", unsafe_allow_html=True)
        else: cs, ps, sl = -1, 1, "Bullish"; st.markdown("""<div style='background-color:#e3f2fd;padding:8px;border-left:4px solid #1976d2'><b>🐂 BULLISH:</b> Ist. Long Call. Dealer: Short Call / Long Put.</div>""", unsafe_allow_html=True)
        rng = st.slider("Range %", 10, 40, 20, help="Zoom grafico")
        st.write("🧩 Filtri Muri")
        dc = st.slider("Dist. Min. Muri CALL (%)", 0, 10, 2)
        dp = st.slider("Dist. Min. Muri PUT (%)", 0, 10, 2)
        btn = st.button("🚀 Analizza Single", type="primary", use_container_width=True)
    with c2:
        if btn and spot:
            calls, puts, err = get_aggregated_data(sym, spot, nex, rng)
            if err: st.error(err)
            else:
                with st.spinner("Processing..."):
                    res = calculate_gex_metrics(calls, puts, spot, adv, cs, ps)
                    # BADGE EXPIRY (TAB 1 - Calcolato dopo GEX)
                    exp_html = get_expiry_badge(calls, puts, res['total_gex'])
                    if exp_html: st.markdown(exp_html, unsafe_allow_html=True)
                    
                    fig = plot_dashboard_unified(sym, res, spot, hist, nex, dc, dp, sl, aiexp)
                    st.pyplot(fig)
                    buf = BytesIO(); fig.savefig(buf, format="png", dpi=150, bbox_inches='tight')
                    st.download_button("💾 Scarica Report", buf.getvalue(), f"GEX_{sym}.png", "image/png", use_container_width=True)

# --- TAB 2: STRATEGY LAB ---
with tab2:
    st.markdown("### 🧪 Strategy Lab: Institutional Trade Architect")
    ls_col1, ls_col2 = st.columns([1, 2])
    with ls_col1:
        t3_sym_input = st.text_input("Ticker Strategy", st.session_state['shared_ticker'], key="tab2_ticker").upper()
        if t3_sym_input != st.session_state['shared_ticker']: st.session_state['shared_ticker'] = t3_sym_input
        t3_spot, t3_adv, t3_hist = get_market_data(t3_sym_input)
        if t3_spot:
            t3_hist['LogRet'] = np.log(t3_hist['Close'] / t3_hist['Close'].shift(1))
            hv_current = t3_hist['LogRet'].tail(20).std() * np.sqrt(252)
            st.metric("Spot Price", f"${t3_spot:.2f}"); st.metric("Historical Vol (HV20)", f"{hv_current:.1%}")
            
            # BADGE RS (TAB 2)
            rs_html_t2 = get_rs_badge(t3_sym_input)
            if rs_html_t2: st.markdown(rs_html_t2, unsafe_allow_html=True)
            
            t3_earn = check_earnings_risk(t3_sym_input)
            if t3_earn: st.error(t3_earn)
            t3_scen = suggest_market_context(t3_hist)
            st.info(f"AI Context: {t3_scen[0]}")
            btn_strat = st.button("🛠️ Genera Trade Setup", type="primary")
    with ls_col2:
        if t3_spot and btn_strat:
            t3_calls, t3_puts, t3_err = get_aggregated_data(t3_sym_input, t3_spot, 6, 20)
            if not t3_err:
                calls_agg = t3_calls.groupby("strike")["openInterest"].sum().reset_index()
                puts_agg = t3_puts.groupby("strike")["openInterest"].sum().reset_index()
                t3_cw = calls_agg[calls_agg['strike'] > t3_spot].sort_values("openInterest", ascending=False).iloc[0]['strike']
                t3_pw = puts_agg[puts_agg['strike'] < t3_spot].sort_values("openInterest", ascending=False).iloc[0]['strike']
                iv_est = t3_calls[t3_calls['strike'].between(t3_spot*0.95, t3_spot*1.05)]['impliedVolatility'].mean()
                if pd.isna(iv_est): iv_est = hv_current
                vol_regime = "HIGH VOL" if iv_est > (hv_current * 1.1) else "LOW VOL"
                strat_type = "CREDIT (Sell Premium)" if vol_regime == "HIGH VOL" else "DEBIT (Buy Premium)"
                bias_bull = "Bull" in t3_scen[0]
                if bias_bull:
                    entry = t3_spot; stop = t3_pw * 0.99; target = t3_cw * 0.99; setup_title = "🐂 BULLISH SETUP"; col_setup = "#e3f2fd"
                else:
                    entry = t3_spot; stop = t3_cw * 1.01; target = t3_pw * 1.01; setup_title = "🐻 BEARISH SETUP"; col_setup = "#ffebee"
                is_obstacle, obst_price = check_volume_obstacle(t3_hist, entry, target)
                obstacle_msg = ""
                if is_obstacle: obstacle_msg = f"""<br><div style='background-color:#FFCCBC; color:#D84315; padding:8px; border-radius:5px;'><b>🚧 OBSTACLE ALERT:</b> Muro di Volume rilevato a <b>${obst_price:.2f}</b>. <br>Il prezzo potrebbe rimbalzare qui prima di raggiungere il target GEX. Considera Take Profit anticipato.</div>"""
                risk = abs(entry - stop); reward = abs(target - entry); rr_ratio = reward / risk if risk > 0 else 0
                st.markdown(f"#### 📊 Volatility Regime: {vol_regime}"); st.write(f"IV Est: **{iv_est:.1%}** vs HV: **{hv_current:.1%}** -> Suggerimento: **{strat_type}**"); st.markdown("---")
                st.markdown(f"""<div style="background-color: {col_setup}; padding: 15px; border-radius: 10px; border: 1px solid #ddd;"><h3 style="margin-top:0">{setup_title}</h3><p><b>📐 ENTRY:</b> ${entry:.2f} (Spot)</p><p><b>🛑 STOP LOSS:</b> ${stop:.2f} (Livello Muro Opzioni: ${t3_pw if bias_bull else t3_cw:.0f})</p><p><b>🎯 TARGET:</b> ${target:.2f} (Livello Muro Opzioni: ${t3_cw if bias_bull else t3_pw:.0f})</p>{obstacle_msg}<hr><p style="font-size: 18px"><b>⚖️ Risk/Reward: 1 : {rr_ratio:.2f}</b></p></div>""", unsafe_allow_html=True)
                fig_cone, txt_explanation = plot_probability_cone(t3_spot, iv_est, target, days=30)
                st.pyplot(fig_cone); st.info(txt_explanation)
            else: st.error("Dati opzioni non disponibili per la strategia.")

# --- TAB 3: SQUEEZE SCANNER ---
with tab3:
    st.markdown("### 🔥 Gamma Squeeze & Volatility Scanner")
    default_tickers = "SPY, QQQ, NVDA, TSLA, AMD, AAPL, MSFT, AMZN, META, COIN, MSTR, AMC, PLTR"
    ticker_input = st.text_area("Lista Tickers (separati da virgola)", default_tickers)
    n_scan_exps = st.slider("Scadenze Scanner (Speed vs Precision)", 2, 6, 4)
    btn_scan = st.button("🔎 Avvia Scansione", type="primary")
    if btn_scan:
        tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
        results = []
        bar = st.progress(0, "Inizio scansione...")
        for i, t in enumerate(tickers):
            bar.progress(int((i / len(tickers)) * 100), f"Analisi {t}...")
            spot_s, adv_s, hist_s = get_market_data(t)
            if not spot_s: continue
            is_sqz = calculate_technical_squeeze(hist_s)
            scen_tuple = suggest_market_context(hist_s)
            scen_name = scen_tuple[0]
            warn_earn = "⚠️ EARNINGS" if check_earnings_risk(t) else ""
            if "Synthetic Short" in scen_name: c_s, p_s, readable_scen = 1, -1, "BEARISH 🐻"
            elif "Long Straddle" in scen_name: c_s, p_s, readable_scen = -1, -1, "VOLATILE 💥"
            elif "Short Straddle" in scen_name: c_s, p_s, readable_scen = 1, 1, "NEUTRAL 💤"
            else: c_s, p_s, readable_scen = -1, 1, "BULLISH 🐂"
            calls_s, puts_s, err_s = get_aggregated_data(t, spot_s, n_scan_exps, 15)
            if not err_s:
                res_s = calculate_gex_metrics(calls_s, puts_s, spot_s, adv_s, c_s, p_s)
                regime = "LONG GAMMA" if res_s['total_gex'] > 0 else "SHORT GAMMA"
                gpi_val = res_s['gpi']
                score = gpi_val
                if regime == "SHORT GAMMA": score += 20
                if is_sqz: score += 15
                results.append({"Ticker": t, "Price": spot_s, "Regime": regime, "GPI %": round(gpi_val, 1), "BB Squeeze": "✅ YES" if is_sqz else "No", "AI Scenario": readable_scen, "Risk": warn_earn, "ScoreVal": score})
            time.sleep(0.1)
        bar.empty()
        if results:
            df_res = pd.DataFrame(results).sort_values("ScoreVal", ascending=False)
            def format_score_col(val):
                if val > 40: return f"{val:.1f} | ⚡ ESPLOSIVO"
                if val > 25: return f"{val:.1f} | 🔥 ALTO"
                if val > 10: return f"{val:.1f} | ⚠️ MEDIO"
                return f"{val:.1f} | ✅ STABILE"
            df_res["Score"] = df_res["ScoreVal"].apply(format_score_col)
            df_display = df_res.drop(columns=["ScoreVal"])
            def color_regime(val): return f'background-color: {"#ffcdd2" if val == "SHORT GAMMA" else "#c8e6c9"}; color: black'
            st.dataframe(df_display.style.applymap(color_regime, subset=['Regime']).format({"Price": "${:.2f}", "GPI %": "{:.1f}%"}), use_container_width=True)
            st.markdown("""**Legenda Score:** > 40 (ESPLOSIVO), 25-40 (ALTO), 10-25 (MEDIO), < 10 (STABILE).""")
        else: st.warning("Nessun risultato.")
