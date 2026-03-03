# app_options_liquidity.py
# Streamlit app: Options Liquidity Map + Top Contracts table (Plotly + human-readable UX)

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.express as px

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="Options Liquidity Map", layout="wide")


# ----------------------------
# Helpers
# ----------------------------
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
    df["right"] = right  # "C" or "P"

    # Ensure expected columns exist
    for col in ["bid", "ask", "lastPrice", "volume", "openInterest", "strike"]:
        if col not in df.columns:
            df[col] = np.nan

    # Numerics
    df["bid"] = pd.to_numeric(df["bid"], errors="coerce")
    df["ask"] = pd.to_numeric(df["ask"], errors="coerce")
    df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
    df["openInterest"] = pd.to_numeric(df["openInterest"], errors="coerce").fillna(0)

    df["mid"] = (df["bid"] + df["ask"]) / 2.0
    df["spread_abs"] = (df["ask"] - df["bid"]).clip(lower=0)
    df["spread_pct"] = np.where(df["mid"] > 0, df["spread_abs"] / df["mid"], np.nan)

    # Stabilize for scoring
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

    # Drop unusable rows
    df = df.dropna(subset=["strike", "expiry"]).copy()

    # Score: z(log(1+vol)) + z(log(1+OI)) - z(spread%)
    df["z_vol"] = zscore(df["log_vol"])
    df["z_oi"] = zscore(df["log_oi"])

    spread = df["spread_pct"].copy()
    # Penalize missing spreads by treating as worst observed spread
    if spread.notna().any():
        spread = spread.fillna(spread.max(skipna=True))
    else:
        spread = spread.fillna(1.0)
    df["z_spread"] = zscore(spread)

    df["liq_score"] = df["z_vol"] + df["z_oi"] - df["z_spread"]

    return df


@st.cache_data(show_spinner=True, ttl=60 * 10)
def load_chain(ticker: str, max_exps: int) -> pd.DataFrame:
    return build_liquidity_frame(ticker, max_exps)


def fmt_pct(x: float, decimals: int = 2) -> str:
    """Format 0.0123 -> '1.23%'."""
    if pd.isna(x):
        return ""
    return f"{x * 100:.{decimals}f}%"


# ----------------------------
# UI: Title + compact controls
# ----------------------------
st.title("Options Liquidity Map")

ctl1, ctl2, ctl3, ctl4, ctl5 = st.columns([1, 1, 1.4, 1.6, 2.0])

with ctl1:
    ticker = st.text_input("Ticker", value="NVDA").strip().upper()

with ctl2:
    max_exps = st.slider("Expirations", 1, 40, 12)

with ctl3:
    # Radio (short, horizontal)
    cp_filter = st.radio("Calls/Puts", ["Both", "Calls", "Puts"], horizontal=True, index=0)

with ctl4:
    size_mode = st.selectbox("Bubble sizing", ["SQRT(Volume)", "Volume"], index=0)

with ctl5:
    color_metric_display = st.selectbox(
        "Color metric",
        ["Liquidity Score", "Spread %", "Open Interest", "Volume"],
        index=0,
    )

# Secondary filters row
flt1, flt2, flt3 = st.columns([1.1, 1.1, 2.8])
with flt1:
    min_vol = st.number_input("Min Volume", min_value=0, value=50, step=25)
with flt2:
    min_oi = st.number_input("Min Open Interest", min_value=0, value=100, step=25)
with flt3:
    max_spread_pct = st.number_input(
        "Max Spread % (e.g. 5 = 5%)",
        min_value=0.0,
        value=7.5,
        step=0.5,
        format="%.1f",
    ) / 100.0  # store as decimal

COLOR_MAP = {
    "Liquidity Score": "liq_score",
    "Spread %": "spread_pct",
    "Open Interest": "openInterest",
    "Volume": "volume",
}
color_metric = COLOR_MAP[color_metric_display]

# ----------------------------
# Load data
# ----------------------------
df = load_chain(ticker, max_exps)

if df.empty:
    st.warning("No options data returned (ticker may be invalid or data source unavailable).")
    st.stop()

# ----------------------------
# Apply filters (top filters apply to everything)
# ----------------------------
df_f = df.copy()

df_f = df_f[(df_f["volume"] >= min_vol) & (df_f["openInterest"] >= min_oi)]
df_f = df_f[df_f["spread_pct"].fillna(999) <= max_spread_pct]

if cp_filter == "Calls":
    df_f = df_f[df_f["right"] == "C"]
elif cp_filter == "Puts":
    df_f = df_f[df_f["right"] == "P"]

if df_f.empty:
    st.info("No contracts match the current filters.")
    st.stop()

# ----------------------------
# Plot: Liquidity map (Plotly)
# ----------------------------
st.subheader("Liquidity map (Expiration vs Strike)")

plot_df = df_f.copy()
plot_df["expiry"] = pd.to_datetime(plot_df["expiry"])

# Bubble sizing (smaller + less occlusion)
if size_mode == "SQRT(Volume)":
    plot_df["size_val"] = np.sqrt(plot_df["volume"].clip(lower=0) + 1)
    size_label = "SQRT(Volume)"
else:
    plot_df["size_val"] = plot_df["volume"].clip(lower=0)
    size_label = "Volume"

# Human-readable color field for legend + colorbar title
COLOR_LEGEND_NAME = color_metric_display
plot_df[COLOR_LEGEND_NAME] = plot_df[color_metric]

# Hover helpers
plot_df["spread_pct"] = pd.to_numeric(plot_df["spread_pct"], errors="coerce")
plot_df["Spread %"] = plot_df["spread_pct"].apply(lambda v: fmt_pct(v, 2))
plot_df["spread_bps"] = (plot_df["spread_pct"] * 10_000).round(1)

plot_df["Contract"] = (
    plot_df["expiry"].dt.strftime("%Y-%m-%d")
    + " "
    + plot_df["right"]
    + " "
    + plot_df["strike"].round(0).astype(int).astype(str)
)

fig = px.scatter(
    plot_df,
    x="expiry",
    y="strike",
    size="size_val",
    color=COLOR_LEGEND_NAME,  # uses human-readable name
    hover_name="Contract",
    hover_data={
        "right": True,
        "expiry": True,
        "strike": True,
        "bid": ":.2f",
        "ask": ":.2f",
        "mid": ":.2f",
        "Spread %": True,      # percentage formatted
        "volume": True,
        "openInterest": True,
        "liq_score": ":.2f",   # keep raw score visible on hover
        "size_val": False,
        "Contract": False,
        "spread_bps": False,
        "spread_pct": False,
    },
    size_max=20,  # slightly smaller
)

# Force round markers, add transparency
fig.update_traces(marker=dict(symbol="circle", opacity=0.70))

# If coloring by Spread %, show it as percent on the colorbar ticks
if color_metric_display == "Spread %":
    fig.update_coloraxes(
        colorbar=dict(
            tickformat=".1%",
            title="Spread %",
        )
    )
else:
    fig.update_coloraxes(colorbar=dict(title=COLOR_LEGEND_NAME))

fig.update_layout(
    height=650,
    xaxis_title="Expiration",
    yaxis_title="Strike",
    title=f"{ticker} options — size={size_label}, color={color_metric_display} ({cp_filter})",
)

st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# Top contracts table (no separate table filters; uses top filters)
# ----------------------------
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
    + topn["Type"].str[0]   # C / P
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

# Show percentages in the table as "##%"
# Use Streamlit's PercentColumn, which expects 0-1 decimals and formats as percent.
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
            max_value=1.0,  # spread_pct is decimal
        ),
        "Liquidity Score": st.column_config.NumberColumn(format="%.2f"),
    },
)

st.caption("Data source: Yahoo Finance via yfinance (screening-grade; not guaranteed real-time).")