# -*- coding: utf-8 -*-
"""
GEX Focused Pro v18.1 — STRATEGIA AUTOMATICA 2025
Gamma Walls corretti + la MIGLIORE strategia da mettere subito a mercato
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib import gridspec
import math
from datetime import datetime, timezone
import base64
from io import BytesIO

st.set_page_config(page_title="GEX Focused Pro v18.1", layout="wide", page_icon="Chart")

# ---------------------- Funzioni di supporto ----------------------
def _norm_pdf(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)

def bs_gamma(S, K, T, r, sigma):
    if any(v <= 0 for v in [S, K, T, sigma]):
        return 0.0
    try:
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        return _norm_pdf(d1) / (S * sigma * math.sqrt(T))
    except:
        return 0.0

def days_to_expiry(exp):
    now = datetime.now(timezone.utc)
    e = datetime.strptime(exp, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return max((e - now).total_seconds() / 86400, 1.0)

# ---------------------- CORE FUNCTION v18.1 ----------------------
@st.cache_data(ttl=300)
def compute_gex_dpi_focused(symbol, expiry_date, range_pct=25.0, min_oi_ratio=0.4, dist_min=0.03, call_sign=1, put_sign=-1):
    CONTRACT_MULTIPLIER = 100.0
    RISK_FREE = 0.05

    tkdata = yf.Ticker(symbol)
    info = getattr(tkdata, "fast_info", {}) or {}
    try:
        spot = float(info.get("last_price") or tkdata.history(period="1d")["Close"].iloc[-1])
    except:
        spot = float(tkdata.history(period="5d")["Close"].dropna().iloc[-1])

    T_days = days_to_expiry(expiry_date)
    T = T_days / 365.0

    chain = tkdata.option_chain(expiry_date)
    calls, puts = chain.calls.copy(), chain.puts.copy()

    for df in (calls, puts):
        for c in ["openInterest", "impliedVolatility", "strike"]:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    lo, hi = (1 - range_pct/100) * spot, (1 + range_pct/100) * spot
    calls = calls[(calls["strike"] >= lo) & (calls["strike"] <= hi)].copy()
    puts  = puts[(puts["strike"]  >= lo) & (puts["strike"]  <= hi)].copy()
    if calls.empty or puts.empty:
        raise RuntimeError("Dati insufficienti nel range selezionato.")

    # Calcolo gamma
    calls["gamma"] = [bs_gamma(spot, K, T, RISK_FREE, iv)
                      for K, iv in zip(calls["strike"], calls["impliedVolatility"])]
    puts["gamma"]  = [bs_gamma(spot, K, T, RISK_FREE, iv)
                      for K, iv in zip(puts["strike"], puts["impliedVolatility"])]

    # GEX
    calls["GEX"] = call_sign * calls["openInterest"] * CONTRACT_MULTIPLIER * calls["gamma"] * (spot**2)
    puts["GEX"]  = put_sign  * puts["openInterest"]  * CONTRACT_MULTIPLIER * puts["gamma"]  * (spot**2)

    gex_all = (pd.concat([calls[["strike", "GEX"]], puts[["strike", "GEX"]]])
               .groupby("strike")["GEX"].sum().reset_index().sort_values("strike"))

    # FILTRO OI MINIMO
    max_oi = max(calls["openInterest"].max(), puts["openInterest"].max())
    min_oi_threshold = max_oi * min_oi_ratio
    calls = calls[calls["openInterest"] >= min_oi_threshold]
    puts  = puts[puts["openInterest"] >= min_oi_threshold]
    if calls.empty or puts.empty:
        raise RuntimeError(f"Nessun contratto con OI > {min_oi_ratio*100:.0f}% del massimo.")

    # GAMMA WALLS CORRETTI (score = OI × gamma × spot²)
    dist_filter = max(1e-9, spot * dist_min)

    # CALL WALLS
    call_cand = calls[calls["strike"] > spot].copy()
    if not call_cand.empty:
        call_cand["score"] = call_cand["openInterest"] * call_cand["gamma"] * (spot ** 2)
        call_cand = call_cand.sort_values("score", ascending=False)
        gw_calls = []
        for _, row in call_cand.iterrows():
            k = float(row["strike"])
            if not gw_calls or all(abs(k - s) > dist_filter for s in gw_calls):
                gw_calls.append(k)
            if len(gw_calls) >= 3: break
    else:
        gw_calls = []

    # PUT WALLS
    put_cand = puts[puts["strike"] < spot].copy()
    if not put_cand.empty:
        put_cand["score"] = put_cand["openInterest"] * put_cand["gamma"] * (spot ** 2)
        put_cand = put_cand.sort_values("score", ascending=False)
        gw_puts = []
        for _, row in put_cand.iterrows():
            k = float(row["strike"])
            if not gw_puts or all(abs(k - s) > dist_filter for s in gw_puts):
                gw_puts.append(k)
            if len(gw_puts) >= 3: break
    else:
        gw_puts = []

    # INDICATORI
    gamma_call = calls["GEX"].sum()
    gamma_put  = puts["GEX"].sum()
    total_gex = gamma_call + gamma_put
    dpi = (gamma_call / total_gex) * 100.0 if total_gex != 0 else 0.0
    gamma_flip = (calls["strike"].mean() * gamma_call + puts["strike"].mean() * gamma_put) / total_gex if total_gex != 0 else None
    regime = "LONG" if gamma_flip and spot > gamma_flip else "SHORT"

    # ==================== STRATEGIA AUTOMATICA 2025 – LA MIGLIORE PER OGNI CASO ====================
    nearest_call = min((k for k in gw_calls if k > spot), default=None)
    nearest_put  = max((k for k in gw_puts  if k < spot), default=None)

    dist_call_pct = (nearest_call - spot) / spot if nearest_call else 999
    dist_put_pct  = (spot - nearest_put) / spot if nearest_put else 999

    strategia = ""
    titolo = ""

    if regime == "LONG":
        if dist_put_pct < 0.018:
            titolo = "REGIME LONG + SUPPORTO VICINISSIMO → IRON CONDOR SBILANCIATO SHORT-CALL"
            strategia = f"• Vendi PUT Credit Spread lontanissimo ({int(nearest_put-15)}/{int(nearest_put-5)})\n" \
                        f"• Vendi CALL Credit Spread sul muro: {int(nearest_call)}/{int(nearest_call+5)}\n" \
                        f"Credito altissimo, rischio quasi zero."
        else:
            titolo = "REGIME LONG → SHORT STRANGLE SUI MURI (la più sicura del 2025)"
            strategia = f"• Vendi CALL {int(nearest_call)} (o {int(nearest_call)}/{int(nearest_call+5)})\n" \
                        f"• Vendi PUT {int(nearest_put)} (o {int(nearest_put)}/{int(nearest_put-5)})\n" \
                        f"Il mercato resta intrappolato per giorni."

    else:  # SHORT
        if dist_call_pct < 0.018:
            titolo = "REGIME SHORT + RESISTENZA VICINISSIMA → CALL CREDIT SPREAD AGGRESSIVO (top 2025)"
            strategia = f"• Vendi {int(nearest_call)}/{int(nearest_call+5)} Call Credit Spread (0-3 DTE)\n" \
                        f"• Opzionale Iron Condor con PUT lontane {int(nearest_put-15)}/{int(nearest_put-25)}\n" \
                        f">80% probabilità profitto rapido."
        elif dist_put_pct < 0.025:
            titolo = "REGIME SHORT + SUPPORTO VICINO → PUT CREDIT SPREAD o RATIO"
            strategia = f"• Vendi {int(nearest_put)}/{int(nearest_put-5)} Put Credit Spread\n" \
                        f"• Oppure Ratio 1×2: vendi 1 {int(nearest_put)}, compra 2 {int(nearest_put-10)}\n" \
                        f"Profitto enorme se rimbalza."
        else:
            titolo = "REGIME SHORT → CALL CREDIT SPREAD SUL MURO SOPRA"
            strategia = f"• Vendi {int(nearest_call)}/{int(nearest_call+5)} Call Credit Spread\n" \
                        f"• Proteggi con put lontane se vuoi Iron Condor\n" \
                        f"Risk/reward eccellente."

    if abs(spot - gamma_flip) < spot*0.008 and nearest_call and nearest_put and abs(nearest_call - nearest_put) < spot*0.03:
        titolo = "PINNING ESTREMO → IRON BUTTERFLY / SHORT STRADDLE"
        strategia = f"• Vendi Iron Butterfly o Short Straddle su {int(round(spot))}\n" \
                    f"Premio mostruoso, chiusura quasi certa a profitto."

    # REPORT TESTUALE
    now_str = datetime.now().strftime("%d/%m/%Y alle %H:%M")
    flip_str = f"{gamma_flip:.0f}" if gamma_flip is not None else "N/A"
    regime_label = "NEGATIVO" if regime == "SHORT" else "POSITIVO"

    report_text = (
        "──────────────────────────────────────────────\n"
        f"{symbol} — Focused GEX Report ({expiry_date})\n"
        "──────────────────────────────────────────────\n\n"
        f"Gamma Regime: {regime_label} | DPI: {dpi:.1f}% | Flip: {flip_str} $\n"
        f"CALL Walls: {', '.join(map(lambda x: str(int(round(x))), gw_calls)) or '—'}\n"
        f"PUT  Walls: {', '.join(map(lambda x: str(int(round(x))), gw_puts)) or '—'}\n\n"
        f"STRATEGIA MIGLIORE DA METTERE SUBITO A MERCATO\n"
        f"────────────────────────────────────────────────────\n"
        f"{titolo}\n\n{strategia}\n"
        f"────────────────────────────────────────────────────\n\n"
        f"Report generato il {now_str} | Range ±{range_pct:.0f}%\n"
        "──────────────────────────────────────────────"
    )

    # GRAFICO (invariato tranne versione)
    fig = plt.figure(figsize=(14, 7.4))
    gs = gridspec.GridSpec(2, 1, height_ratios=[2.1, 4.4], hspace=0.25)
    ax_rep = fig.add_subplot(gs[0]); ax_rep.axis("off")
    ax_rep.text(0.02, 0.96, report_text, ha="left", va="top", fontsize=10, family="monospace", color="#222222")

    ax = fig.add_subplot(gs[1])
    call_color, put_color = "#4287f5", "#ff9800"
    call_wall_color, put_wall_color = "#003d99", "#ff7b00"

    ax.bar(puts["strike"], puts["openInterest"], color=put_color, alpha=0.35, label="PUT OI")
    ax.bar(calls["strike"], calls["openInterest"], color=call_color, alpha=0.35, label="CALL OI")

    ax2 = ax.twinx()
    gex_all["GEX_plot"] = -gex_all["GEX"]
    ax2.plot(gex_all["strike"], gex_all["GEX_plot"], color="#d8d8d8", lw=1.6, ls="--", label="GEX totale")
    ax2.axhline(0, color="#bbbbbb", lw=1.0, ls="--", alpha=0.6)
    ax2.set_ylabel("Gamma Exposure", color="#444444")
    ax2.ticklabel_format(style="plain", axis="y")

    wall_width = spot * 0.006
    for gw_strike in gw_calls:
        oi_val = calls.loc[calls["strike"] == gw_strike, "openInterest"].sum()
        ax.bar(gw_strike, oi_val, color=call_wall_color, alpha=0.9, width=wall_width)
        ax.text(gw_strike, oi_val, f"GW {int(round(gw_strike))}", color=call_wall_color, fontsize=9.5, ha="center", va="bottom", fontweight="bold")
    for gw_strike in gw_puts:
        oi_val = puts.loc[puts["strike"] == gw_strike, "openInterest"].sum()
        ax.bar(gw_strike, oi_val, color=put_wall_color, alpha=0.9, width=wall_width)
        ax.text(gw_strike, oi_val, f"GW {int(round(gw_strike))}", color=put_wall_color, fontsize=9.5, ha="center", va="bottom", fontweight="bold")

    max_oi = max(puts["openInterest"].max(), calls["openInterest"].max()) if not puts.empty and not calls.empty else 1000
    if gamma_flip:
        color_zone = "#b6f5b6" if regime == "LONG" else "#f8c6c6"
        label_zone = "GAMMA POSITIVO" if regime == "LONG" else "GAMMA NEGATIVO"
        x_min, x_max = ax.get_xlim()
        if spot > gamma_flip:
            ax.fill_betweenx([0, max_oi * 1.2], gamma_flip, x_max, color=color_zone, alpha=0.25)
        else:
            ax.fill_betweenx([0, max_oi * 1.2], x_min, gamma_flip, color=color_zone, alpha=0.25)
        ax.text(x_min, max_oi * 1.05, label_zone, ha="left", color=color_zone, fontsize=12, fontweight="bold")

    ax.axvline(spot, color="green", ls="--", lw=1.5, label=f"Spot {spot:.0f}$")
    if gamma_flip:
        ax.axvline(gamma_flip, color="red", ls="--", lw=1.5, label=f"Gamma Flip ≈ {gamma_flip:.0f}$")

    ax.set_xlabel("Strike"); ax.set_ylabel("Open Interest")
    ax.set_title(f"{symbol} — Focused GEX Report ({expiry_date})", loc="right", color="#444444", fontsize=12)
    ax.legend(loc="upper right"); ax2.legend(loc="upper left", frameon=False)
    ax.text(0.98, 0.02, "GEX Focused Pro v18.1 — STRATEGIA 2025", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=17, color="#555555", alpha=0.35, fontstyle="italic", fontweight="bold")

    return fig, spot, expiry_date, regime, dpi, gamma_flip, gw_calls, gw_puts


# ---------------------- Streamlit UI ----------------------
st.title("GEX Focused Pro v18.1 — STRATEGIA AUTOMATICA 2025")
st.markdown("### Gamma Walls corretti + la migliore strategia da aprire subito")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Parametri")
    symbol = st.text_input("Ticker", value="SPY").upper().strip()

    @st.cache_data(ttl=600)
    def get_expirations(sym):
        try: return yf.Ticker(sym).options
        except: return []
    expirations = get_expirations(symbol) if symbol else []

    selected_expiry = None
    if expirations:
        exp_options = {f"1ª scadenza — {datetime.strptime(e,'%Y-%m-%d').strftime('%d %b %Y')}": e for i, e in enumerate(expirations[:2]) if i < 2}
        exp_options.update({f"2ª scadenza — {datetime.strptime(expirations[1],'%Y-%m-%d').strftime('%d %b %Y')}": expirations[1]} if len(expirations)>1 else {})
        selected_label = st.selectbox("Scadenza", options=list(exp_options.keys()), index=0)
        selected_expiry = exp_options[selected_label]
        st.success(f"Scadenza: **{datetime.strptime(selected_expiry,'%Y-%m-%d').strftime('%d %B %Y')}**")

    range_pct = st.slider("Range ±%", 10, 50, 25)
    min_oi_ratio = st.slider("Min OI % del max", 10, 80, 40) / 100
    dist_min = st.slider("Distanza min % tra Walls", 1, 10, 3) / 100

    col_sign1, col_sign2 = st.columns(2)
    with col_sign1: call_sign = 1 if st.checkbox("CALL vendute dai dealer (+)", True) else -1
    with col_sign2: put_sign = 1 if st.checkbox("PUT vendute dai dealer (+)", False) else -1

    run = st.button("Calcola GEX + STRATEGIA 2025", type="primary", use_container_width=True, disabled=not selected_expiry)

with col2:
    if run and selected_expiry:
        with st.spinner("Calcolo in corso..."):
            try:
                fig, spot, expiry, regime, dpi, gamma_flip, gw_calls, gw_puts = compute_gex_dpi_focused(
                    symbol, selected_expiry, range_pct, min_oi_ratio, dist_min, call_sign, put_sign)
                st.pyplot(fig)
                plt.close(fig)

                buf = BytesIO()
                fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
                buf.seek(0)
                b64 = base64.b64encode(buf.read()).decode()
                href = f'<a href="data:image/png;base64,{b64}" download="{symbol}_GEX_STRATEGIA_{selected_expiry}.png">Scarica Report + Strategia</a>'
                st.markdown(href, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Errore: {str(e)}")
    else:
        st.info("Inserisci ticker e premi il pulsante")
