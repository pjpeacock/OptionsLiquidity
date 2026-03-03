# app_options_liquidity.py
# Streamlit app: Options Liquidity Map + Top Contracts table

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Options Liquidity Map", layout="wide")


# ============================
# Helpers
# ============================

def zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").astype(float)
    mu = s.mean()
    sd = s.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mu) / sd


def clean_chain(df: pd.DataFrame, expiry: str, right: str) -> pd.DataFrame:
    df = df.copy()
    df["expiry"] = pd.to_datetime(expiry)
    df["right"] = right

    for col in ["bid", "ask", "lastPrice", "volume", "openInterest", "strike"]:
        if col not in df.columns:
            df[col] = np.nan

    df["bid"] = pd.to_numeric(df["bid"], errors="coerce")
    df["ask"] = pd.to_numeric(df["ask"], errors="coerce")
    df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
    df["openInterest"] = pd.to_numeric(df["openInterest"], errors="coerce").fillna(0)

    df["mid"] = (df["bid"] + df["ask"]) / 2.0
    df["spread_abs"] = (df["ask"] - df["bid"]).clip(lower=0)
    df["spread_pct"] = np.where(df["mid"] > 0, df["spread_abs"] / df["mid"], np.nan)

    df["log_vol"] = np.log1p(df["volume"])
    df["log_oi"] = np.log1p(df["openInterest"])

    return df


def build_liquidity_frame(ticker: str, max_exps: int) -> pd.DataFrame:
    t = yf.Ticker(ticker)
    exps = list(t.options)
    if not exps:
        return pd.DataFrame()

    exps = exps[:max_exps]
    frames = []

    for exp in exps:
        chain = t.option_chain(exp)
        frames.append(clean_chain(chain.calls, exp, "C"))
        frames.append(clean_chain(chain.puts, exp, "P"))

    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["strike", "expiry"]).copy()

    df["z_vol"] = zscore(df["log_vol"])
    df["z_oi"] = zscore(df["log_oi"])

    spread = df["spread_pct"].copy()
    if spread.notna().any():
        spread = spread.fillna(spread.max(skipna=True))
    else:
        spread = spread.fillna(1.0)

    df["z_spread"] = zscore(spread)
    df["liq_score"] = df["z_vol"] + df["z_oi"] - df["z_spread"]

    return df


def pick_threshold_tier(base_df: pd.DataFrame) -> tuple[str, int, int, float]:
    """
    Returns: (tier_name, min_vol, min_oi, max_spread_pct_decimal)
    """
    N = len(base_df)
    if N == 0:
        return ("Normal", 50, 200, 0.1)

    vols = pd.to_numeric(base_df["volume"], errors="coerce").fillna(0)
    ois = pd.to_numeric(base_df["openInterest"], errors="coerce").fillna(0)

    p75_vol = float(np.percentile(vols, 75))
    p75_oi = float(np.percentile(ois, 75))

    # Thin
    if (N < 400) or (p75_vol < 30 and p75_oi < 200):
        return ("Thin", 10, 50, 0.25)

    # Deep
    if (N > 1500) and ((p75_vol > 150) or (p75_oi > 1000)):
        return ("Deep", 100, 500, 0.05)

    # Normal
    return ("Normal", 50, 200, 0.1)

@st.cache_data(ttl=3600)
def get_company_name(ticker: str) -> str:
    try:
        info = yf.Ticker(ticker).info
        return info.get("longName") or info.get("shortName") or ""
    except Exception:
        return ""


@st.cache_data(ttl=30)
def get_quote_snapshot(ticker: str) -> tuple[float | None, float | None]:
    """
    Returns (last_price, prev_close). Robust across yfinance versions and weekends.
    """
    try:
        t = yf.Ticker(ticker)

        # Try fast_info with multiple key variants
        fi = getattr(t, "fast_info", None)
        if fi:
            last = (
                fi.get("last_price")
                or fi.get("lastPrice")
                or fi.get("regularMarketPrice")
            )
            prev = (
                fi.get("previous_close")
                or fi.get("previousClose")
                or fi.get("regularMarketPreviousClose")
            )
            if last is not None and prev is not None:
                return (float(last), float(prev))

        # Fallback: use several days to survive weekends/holidays
        hist = t.history(period="10d", interval="1d", auto_adjust=False)
        if hist is None or hist.empty or "Close" not in hist.columns:
            return (None, None)

        closes = hist["Close"].dropna()
        if len(closes) == 0:
            return (None, None)

        last_px = float(closes.iloc[-1])
        prev_close = float(closes.iloc[-2]) if len(closes) >= 2 else None
        return (last_px, prev_close)

    except Exception:
        return (None, None)
    

@st.cache_data(show_spinner=True, ttl=600)
def load_chain(ticker: str, max_exps: int):
    return build_liquidity_frame(ticker, max_exps)




# ============================
# Density Strip Builder
# ============================

def build_density_strip(values, threshold, label, theme_bg, theme_text,
                        axis_mode="linear", compress_pctl=95):

    v = pd.to_numeric(values, errors="coerce").dropna()
    if len(v) == 0:
        return go.Figure()

    if axis_mode == "log":
        v = v.clip(lower=0)
        v_t = np.log10(1 + v)
        thr_t = np.log10(1 + threshold)
        x_max = max(np.percentile(v_t, compress_pctl), thr_t)
        bins = np.linspace(0, x_max, 101)
        counts, _ = np.histogram(v_t.clip(0, x_max), bins=bins)
        x_centers = (bins[:-1] + bins[1:]) / 2
        thr_x = thr_t
        x_range = [0, x_max]
        tickvals = np.linspace(0, x_max, 5)
        ticktext = [f"{int((10**t)-1):,}" for t in tickvals]
    else:
        v_t = v
        x_max = max(np.percentile(v_t, compress_pctl), threshold)
        bins = np.linspace(0, x_max, 101)
        counts, _ = np.histogram(v_t.clip(0, x_max), bins=bins)
        x_centers = (bins[:-1] + bins[1:]) / 2
        thr_x = threshold
        x_range = [0, x_max]
        tickvals = np.linspace(0, x_max, 5)
        ticktext = [f"{t:.0f}%" if label == "Spread" else f"{int(t):,}" for t in tickvals]

    z = counts / counts.max() if counts.max() > 0 else counts

    fade = np.array([1.0, 0.75, 0.55, 0.38, 0.25, 0.15])
    Z = np.vstack([z * f for f in fade])

    colorscale = (
        (0.0, theme_bg),
        (0.2, "#003300"),
        (0.4, "#006600"),
        (0.6, "#009900"),
        (0.8, "#00CC00"),
        (1.0, "#00FF00"),
    )

    fig = go.Figure(go.Heatmap(
        z=Z,
        x=x_centers,
        colorscale=colorscale,
        showscale=False,
        hoverinfo="skip"
    ))

    fig.add_shape(
        type="line",
        x0=thr_x,
        x1=thr_x,
        y0=-0.5,
        y1=Z.shape[0]-0.5,
        line=dict(color="red", width=3),
    )

    fig.update_layout(
        height=64,
        margin=dict(l=0, r=0, t=0, b=0),
        plot_bgcolor=theme_bg,
        paper_bgcolor=theme_bg,
        shapes=(fig.layout.shapes or ()) + (
            dict(
                type="rect",
                xref="paper",
                yref="paper",
                x0=0,
                y0=0,
                x1=1,
                y1=1,
                line=dict(color=theme_text, width=1),
                fillcolor="rgba(0,0,0,0)",
                layer="above",
            ),
        ),
    )

    fig.update_xaxes(
        range=x_range,
        tickmode="array",
        tickvals=tickvals,
        ticktext=ticktext,
        tickfont=dict(color=theme_text, size=11),
        showgrid=False,
    )

    fig.update_yaxes(visible=False)

    return fig


# ============================
# UI
# ============================
title_slot = st.empty()
meta_slot = st.empty()

title_slot.title("Options Liquidity Map")

st.write("")
st.write("")

ctl1, ctl2, ctl3, ctl4, ctl5 = st.columns([1,1,1.4,1.6,2])

with ctl1:
    ticker = st.text_input("Ticker", "NVDA")

with ctl2:
    max_exps = st.slider("Expirations", 1, 40, 12)

with ctl3:
    cp_filter = st.radio("Calls/Puts", ["Both","Calls","Puts"], horizontal=True)

with ctl4:
    size_mode = st.selectbox("Bubble sizing", ["SQRT(Volume)", "Volume"])

with ctl5:
    color_metric_display = st.selectbox(
        "Color metric",
        ["Liquidity Score", "Spread %", "Open Interest", "Volume"],
    )


ticker_u = (ticker or "").strip().upper()

company_name = get_company_name(ticker_u) if ticker_u else ""
last_px, prev_close = get_quote_snapshot(ticker_u) if ticker_u else (None, None)

# Subtitle: price + % change (if available)
if last_px is not None and prev_close not in (None, 0):
    pct = (last_px - prev_close) / prev_close
    sign = "+" if pct >= 0 else ""

    pct = None
    if last_px is not None and prev_close not in (None, 0):
        pct = (last_px - prev_close) / prev_close

    # Colors
    chg_color = "#00C853" if (pct is not None and pct >= 0) else "#FF1744"  # green / red
    muted = st.get_option("theme.textColor") or "#999999"

    # Display strings
    ticker_disp = ticker_u if ticker_u else "—"
    name_disp = company_name if company_name else ""
    price_disp = f"${last_px:,.2f}" if last_px is not None else "—"
    pct_disp = (f"{'+' if pct is not None and pct >= 0 else ''}{pct*100:.2f}%"
            if pct is not None else "")

    # Render: larger than caption, below title
    meta_slot.markdown(
        f"""
        <div style="font-size:1.05rem; line-height:1.35; margin-top:-6px;">
        <span style="font-weight:700;">{ticker_disp}</span>
        <span style="color:{muted};"> — {name_disp}</span>
        <span style="margin-left:14px; font-weight:700; color:{chg_color};">{price_disp}</span>
        <span style="margin-left:10px; font-weight:700; color:{chg_color};">{pct_disp}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    

# ============================
# Data
# ============================
df = load_chain(ticker, max_exps)

if df.empty:
    st.warning("No options returned.")
    st.stop()

base_df = df.copy()
if cp_filter == "Calls":
    base_df = base_df[base_df["right"]=="C"]
elif cp_filter == "Puts":
    base_df = base_df[base_df["right"]=="P"]


# Auto-seed thresholds by tier when the "universe" changes (ticker/max_exps/calls-puts)
universe_key = f"{ticker}|{max_exps}|{cp_filter}"

if st.session_state.get("universe_key") != universe_key:
    tier_name, tier_vol, tier_oi, tier_spread = pick_threshold_tier(base_df)

    st.session_state["min_vol"] = int(tier_vol)
    st.session_state["min_oi"] = int(tier_oi)
    st.session_state["max_spread_input"] = float(tier_spread * 100.0)  # store as percent for UI
    st.session_state["tier_name"] = tier_name

    st.session_state["universe_key"] = universe_key

st.caption(f"Threshold tier: **{st.session_state.get('tier_name','')}**")


# ============================
# Filters
# ============================

flt1, flt2, flt3 = st.columns([1.1,1.1,2.8])

with flt1:
    min_vol = st.number_input("Min Volume", min_value=0, step=25, key="min_vol")
    vol_hint = st.empty()
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    vol_strip = st.empty()

with flt2:
    min_oi = st.number_input("Min Open Interest", min_value=0, step=25, key="min_oi")
    oi_hint = st.empty()
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    oi_strip = st.empty()

with flt3:
    max_spread_input = st.number_input("Max Spread %", min_value=0.0, step=0.5, format="%.1f", key="max_spread_input")
    max_spread_pct = max_spread_input / 100.0
    spread_hint = st.empty()
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    spread_strip = st.empty()





N = len(base_df)

if N > 0:
    vol_pass = (base_df["volume"] >= min_vol).mean()*100
    oi_pass = (base_df["openInterest"] >= min_oi).mean()*100
    spread_pass = (base_df["spread_pct"].fillna(999) <= max_spread_pct).mean()*100

    vol_hint.caption(f"{vol_pass:.0f}% pass • {100-vol_pass:.0f}% filtered out")
    oi_hint.caption(f"{oi_pass:.0f}% pass • {100-oi_pass:.0f}% filtered out")
    spread_hint.caption(f"{spread_pass:.0f}% pass • {100-spread_pass:.0f}% filtered out")

    theme_bg = st.get_option("theme.backgroundColor")
    theme_text = st.get_option("theme.textColor")

    vol_strip.plotly_chart(
        build_density_strip(base_df["volume"], min_vol, "Volume",
                            theme_bg, theme_text, axis_mode="log"),
        use_container_width=True,
        config={"displayModeBar":False}
    )

    oi_strip.plotly_chart(
        build_density_strip(base_df["openInterest"], min_oi, "Open Interest",
                            theme_bg, theme_text, axis_mode="log"),
        use_container_width=True,
        config={"displayModeBar":False}
    )

    spread_strip.plotly_chart(
        build_density_strip(base_df["spread_pct"]*100, max_spread_input,
                            "Spread", theme_bg, theme_text, axis_mode="linear"),
        use_container_width=True,
        config={"displayModeBar":False}
    )





# ============================
# Apply Filters
# ============================

df_f = base_df[
    (base_df["volume"] >= min_vol) &
    (base_df["openInterest"] >= min_oi) &
    (base_df["spread_pct"].fillna(999) <= max_spread_pct)
]

if df_f.empty:
    st.info("No contracts match filters.")
    st.stop()





# ============================
# Scatter Plot
# ============================

st.subheader("Liquidity map (Expiration vs Strike)")

plot_df = df_f.copy()
plot_df["expiry"] = pd.to_datetime(plot_df["expiry"])

if size_mode == "SQRT(Volume)":
    plot_df["size_val"] = np.sqrt(plot_df["volume"]+1)
else:
    plot_df["size_val"] = plot_df["volume"]

plot_df[color_metric_display] = plot_df[
    {"Liquidity Score":"liq_score",
     "Spread %":"spread_pct",
     "Open Interest":"openInterest",
     "Volume":"volume"}[color_metric_display]
]

# --- X axis formatting: categorical dates, always show all expirations ---
plot_df["expiry_date"] = pd.to_datetime(plot_df["expiry"]).dt.date

# Unique expirations in ascending order
exp_dates = sorted(plot_df["expiry_date"].unique())

# If multiple years, include year in label
years = {d.year for d in exp_dates}
include_year = (len(years) > 1)

if include_year:
    # e.g. "Jan 5, 2026"
    exp_labels = [pd.Timestamp(d).strftime("%b ") + str(pd.Timestamp(d).day) + pd.Timestamp(d).strftime(", %Y") for d in exp_dates]
else:
    # e.g. "Jan 5"
    exp_labels = [pd.Timestamp(d).strftime("%b ") + str(pd.Timestamp(d).day) for d in exp_dates]

label_map = dict(zip(exp_dates, exp_labels))
plot_df["Expiration"] = plot_df["expiry_date"].map(label_map)

# Preserve chronological order on a categorical axis
category_order = exp_labels

num_exps = len(category_order)

if num_exps <= 12:
    tick_angle = 0
elif num_exps <= 24:
    tick_angle = 30
else:
    tick_angle = 45

fig = px.scatter(
    plot_df,
    x="Expiration",
    y="strike",
    size="size_val",
    color=color_metric_display,
    size_max=20
)

fig.update_xaxes(
    type="category",
    categoryorder="array",
    categoryarray=category_order,
    tickmode="array",
    tickvals=category_order,
    tickangle=tick_angle,  # set to 45 if it gets crowded
)

fig.update_traces(marker=dict(symbol="circle", opacity=0.7))
fig.update_layout(height=650)

st.plotly_chart(fig, use_container_width=True)


# ============================
# Top contracts table
# ============================

st.subheader("Top contracts by Liquidity Score")

topn = df_f.sort_values("liq_score", ascending=False).head(500).copy()

topn["Expiration"] = pd.to_datetime(topn["expiry"]).dt.strftime("%Y-%m-%d")
topn["Type"] = topn["right"].map({"C": "Call", "P": "Put"})
topn["Strike"] = pd.to_numeric(topn["strike"], errors="coerce").round(2)

topn["Bid"] = pd.to_numeric(topn["bid"], errors="coerce")
topn["Ask"] = pd.to_numeric(topn["ask"], errors="coerce")
topn["Mid"] = pd.to_numeric(topn["mid"], errors="coerce")

topn["Spread %"] = pd.to_numeric(topn["spread_pct"], errors="coerce")

topn["Volume"] = pd.to_numeric(topn["volume"], errors="coerce").fillna(0).astype(int)
topn["Open Interest"] = pd.to_numeric(topn["openInterest"], errors="coerce").fillna(0).astype(int)
topn["Liquidity Score"] = pd.to_numeric(topn["liq_score"], errors="coerce")

topn["Contract"] = (
    topn["Expiration"]
    + " "
    + topn["Type"].str[0]
    + " "
    + topn["Strike"].astype(str)
)

display_cols = [
    "Contract",
    "Type",
    "Expiration",
    "Strike",
    "Bid",
    "Ask",
    "Mid",
    "Spread %",
    "Volume",
    "Open Interest",
    "Liquidity Score",
]

grid_df = topn[display_cols].copy()

st.dataframe(
    grid_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Bid": st.column_config.NumberColumn(format="%.2f"),
        "Ask": st.column_config.NumberColumn(format="%.2f"),
        "Mid": st.column_config.NumberColumn(format="%.2f"),
        "Spread %": st.column_config.ProgressColumn(
            "Spread %",
            format="%.2f%%",
            min_value=0.0,
            max_value=1.0,
        ),
        "Liquidity Score": st.column_config.NumberColumn(format="%.2f"),
    },
)

st.caption("Data source: Yahoo Finance via yfinance (screening-grade; not guaranteed real-time).")
