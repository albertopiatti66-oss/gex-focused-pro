# app.py
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

st.set_page_config(page_title="GEX Focused Pro", layout="wide", page_icon="Chart")

# --- Funzioni ---
def _norm_pdf(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)

def bs_gamma(S, K, T, r, sigma):
    if any(v <= 0 for v in [S, K, T, sigma]): return 0.0
    try:
        d1 = (math.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
        return _norm_pdf(d1) / (S * sigma * math.sqrt(T))
    except:
        return 0.0

def days_to_expiry(exp):
    e = datetime.strptime(exp, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return max((e - datetime.now(timezone.utc)).total_seconds() / 86400, 1.0)

# --- Core ---
@st.cache_data(ttl=300)
def compute_gex_dpi_focused(symbol, range_pct=25.0, min_oi_ratio=0.4, dist_min=0.03, call_sign=1, put_sign=-1):
    try:
        tkdata = yf.Ticker(symbol)
        info = getattr(tkdata, "fast_info", {}) or {}
        spot = float(info.get("last_price") or tkdata.history(period="1d")["Close"].iloc[-1])
        expirations = tkdata.options
        if not expirations: raise RuntimeError("Nessuna scadenza.")
        expiry = expirations[0]
        T = days_to_expiry(expiry) / 365.0
        chain = tkdata.option_chain(expiry)
        calls, puts = chain.calls.copy(), chain.puts.copy()

        for df in (calls, puts):
            df["openInterest"] = pd.to_numeric(df["openInterest"], errors="coerce").fillna(0)
            df["impliedVolatility"] = pd.to_numeric(df["impliedVolatility"], errors="coerce").fillna(0.3)
            df["strike"] = pd.to_numeric(df["strike"])

        lo, hi = spot * (1 - range_pct/100), spot * (1 + range_pct/100)
        calls = calls[(calls["strike"] >= lo) & (calls["strike"] <= hi)]
        puts = puts[(puts["strike"] >= lo) & (puts["strike"] <= hi)]
        if calls.empty or puts.empty: raise RuntimeError("Dati insufficienti nel range.")

        calls["gamma"] = [bs_gamma(spot, K, T, 0.05, iv) for K, iv in zip(calls["strike"], calls["impliedVolatility"])]
        puts["gamma"] = [bs_gamma(spot, K, T, 0.05, iv) for K, iv in zip(puts["strike"], puts["impliedVolatility"])]

        calls["GEX"] = call_sign * calls["openInterest"] * 100 * calls["gamma"] * (spot**2)
        puts["GEX"] = put_sign * puts["openInterest"] * 100 * puts["gamma"] * (spot**2)

        gex_all = pd.concat([calls[["strike", "GEX"]], puts[["strike", "GEX"]]]).groupby("strike")["GEX"].sum().reset_index()

        max_oi = max(calls["openInterest"].max(), puts["openInterest"].max())
        min_oi = max_oi * min_oi_ratio
        calls = calls[calls["openInterest"] >= min_oi]
        puts = puts[puts["openInterest"] >= min_oi]

        dist_filter = spot * dist_min
        call_cand = calls[calls["strike"] > spot].copy()
        call_cand["absGEX"] = call_cand["GEX"].abs()
        call_cand.sort_values("absGEX", ascending=False, inplace=True)

        put_cand = puts[puts["strike"] < spot].copy()
        put_cand["absGEX"] = put_cand["GEX"].abs()
        put_cand.sort_values("absGEX", ascending=False, inplace=True)

        def pick_levels(df, max_levels=3):
            selected = []
            for _, row in df.iterrows():
                k = float(row["strike"])
                if not selected or all(abs(k - s) > dist_filter for s in selected):
                    selected.append(k)
                if len(selected) >= max_levels: break
            return selected

        gw_calls = pick_levels(call_cand)
        gw_puts = pick_levels(put_cand)

        gamma_call = calls["GEX"].sum()
        gamma_put = puts["GEX"].sum()
        total_gex = gamma_call + gamma_put
        dpi = (gamma_call / total_gex) * 100 if total_gex != 0 else 0
        gamma_flip = (calls["strike"].mean() * gamma_call + puts["strike"].mean() * gamma_put) / total_gex if total_gex != 0 else None
        regime = "LONG" if gamma_flip and spot > gamma_flip else "SHORT"

        # --- Grafico ---
        fig = plt.figure(figsize=(14, 7.4))
        gs = gridspec.GridSpec(2, 1, height_ratios=[2.1, 4.4], hspace=0.25)
        ax_rep = fig.add_subplot(gs[0]); ax_rep.axis("off")
        now_str = datetime.now().strftime("%d/%m/%Y %H:%M")
        flip_str = f"{gamma_flip:.0f}" if gamma_flip else "N/A"
        report_text = (
            "──────────────────────────────────────────────\n"
            f"{symbol} — GEX Report ({expiry})\n"
            "──────────────────────────────────────────────\n\n"
            f"Gamma Regime: {'NEGATIVO' if regime == 'SHORT' else 'POSITIVO'} | DPI: {dpi:.1f}% | Flip: {flip_str}$\n"
            f"{'Bearish' if regime == 'SHORT' else 'Bullish'}\n\n"
            f"CALL Walls: {', '.join(map(lambda x: str(int(round(x))), gw_calls)) or '—'}\n"
            f"PUT  Walls: {', '.join(map(lambda x: str(int(round(x))), gw_puts)) or '—'}\n\n"
            f"Generato: {now_str} | Range ±{range_pct:.0f}%\n"
            "──────────────────────────────────────────────"
        )
        ax_rep.text(0.02, 0.96, report_text, ha="left", va="top", fontsize=10, family="monospace", color="#222")

        ax = fig.add_subplot(gs[1])
        ax.bar(puts["strike"], puts["openInterest"], color="#ff9800", alpha=0.35, label="PUT OI")
        ax.bar(calls["strike"], calls["openInterest"], color="#4287f5", alpha=0.35, label="CALL OI")

        ax2 = ax.twinx()
        gex_all["GEX_plot"] = -gex_all["GEX"]
        ax2.plot(gex_all["strike"], gex_all["GEX_plot"], color="#d8d8d8", lw=1.6, ls="--", label="GEX")
        ax2.axhline(0, color="#bbb", lw=1, ls="--", alpha=0.6)

        wall_width = spot * 0.006
        for gw in gw_calls:
            oi = calls.loc[calls["strike"] == gw, "openInterest"].sum()
            ax.bar(gw, oi, color="#003d99", width=wall_width)
            ax.text(gw, oi, f"GW {int(gw)}", color="#003d99", ha="center", va="bottom", fontweight="bold")
        for gw in gw_puts:
            oi = puts.loc[puts["strike"] == gw, "openInterest"].sum()
            ax.bar(gw, oi, color="#ff7b00", width=wall_width)
            ax.text(gw, oi, f"GW {int(gw)}", color="#ff7b00", ha="center", va="bottom", fontweight="bold")

        max_oi = max(puts["openInterest"].max(), calls["openInterest"].max())
        if gamma_flip:
            color_zone = "#b6f5b6" if regime == "LONG" else "#f8c6c6"
            x_min, x_max = ax.get_xlim()
            ax.fill_betweenx([0, max_oi*1.2], gamma_flip if spot > gamma_flip else x_min, x_max if spot > gamma_flip else gamma_flip, color=color_zone, alpha=0.25)
        ax.axvline(spot, color="green", ls="--", lw=1.5, label=f"Spot {spot:.0f}")
        if gamma_flip: ax.axvline(gamma_flip, color="red", ls="--", lw=1.5, label=f"Flip {gamma_flip:.0f}")

        ax.set_xlabel("Strike"); ax.set_ylabel("Open Interest"); ax.set_title(f"{symbol} — GEX Report", loc="right")
        ax.legend(loc="upper right"); ax2.legend(loc="upper left", frameon=False)
        ax.text(0.98, 0.02, "GEX Focused Pro", transform=ax.transAxes, ha="right", va="bottom", fontsize=17, color="#555", alpha=0.35, fontstyle="italic")

        return fig, spot, expiry, regime, dpi, gamma_flip, gw_calls, gw_puts
    except Exception as e:
        st.error(f"Errore: {str(e)}")
        return None, None, None, None, None, None, [], []

# --- UI ---
st.title("GEX Focused Pro v17.8")
col1, col2 = st.columns([1, 2])
with col1:
    symbol = st.text_input("Ticker", "SPY").upper().strip()
    range_pct = st.slider("Range ±%", 10, 50, 25)
    min_oi_ratio = st.slider("Min OI % del max", 10, 80, 40) / 100
    dist_min = st.slider("Distanza min % tra Walls", 1, 10, 3) / 100
    call_sign = 1 if st.checkbox("CALL vendute dai dealer (+)", True) else -1
    put_sign = 1 if st.checkbox("PUT vendute dai dealer (+)", False) else -1
    run = st.button("Calcola GEX Focused", type="primary", use_container_width=True)

with col2:
    if run:
        with st.spinner("Analisi in corso..."):
            result = compute_gex_dpi_focused(symbol, range_pct, min_oi_ratio, dist_min, call_sign, put_sign)
            if result[0]:
                fig = result[0]
                st.pyplot(fig)
                buf = BytesIO()
                fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
                buf.seek(0)
                b64 = base64.b64encode(buf.read()).decode()
                st.markdown(f'<a href="data:image/png;base64,{b64}" download="{symbol}_GEX_report.png">Scarica Report PNG</a>', unsafe_allow_html=True)
