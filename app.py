# app.py
# ============================================================
# Trades Dashboard — Themes + Reporting Summary + Editor
# With Excel-style STOCKS table (bottom) like your screenshot
# ============================================================
import os
import uuid
from datetime import date, datetime

import pandas as pd
import streamlit as st
import altair as alt

st.set_page_config(page_title="Trades Dashboard", layout="wide")

# -----------------------------
# THEME SYSTEM (Template Options)
# -----------------------------
THEMES = {
    "Crimson": {
        "bg_grad": "linear-gradient(135deg,#19070f 0%,#1a0b13 50%,#120812 100%)",
        "card_grad": "linear-gradient(135deg,rgba(255,255,255,.06),rgba(255,255,255,.02))",
        "acc1": "#f87171",  # red
        "acc2": "#60a5fa",  # blue
        "good": "#34d399",
        "bad":  "#fb7185",
        "muted": "#94a3b8",
    },
    "Forest": {
        "bg_grad": "linear-gradient(135deg,#0c1207 0%,#0e1a0c 50%,#08140a 100%)",
        "card_grad": "linear-gradient(135deg,rgba(255,255,255,.05),rgba(255,255,255,.02))",
        "acc1": "#86efac",
        "acc2": "#7dd3fc",
        "good": "#22c55e",
        "bad":  "#ef4444",
        "muted": "#9ca3af",
    },
    "Slate": {
        "bg_grad": "linear-gradient(135deg,#0e1117 0%,#111827 50%,#0b1020 100%)",
        "card_grad": "linear-gradient(135deg,rgba(255,255,255,.06),rgba(255,255,255,.02))",
        "acc1": "#93c5fd",
        "acc2": "#a78bfa",
        "good": "#10b981",
        "bad":  "#f43f5e",
        "muted": "#94a3b8",
    },
    "Ocean": {
        "bg_grad": "linear-gradient(135deg,#04121a 0%,#071c26 50%,#05131f 100%)",
        "card_grad": "linear-gradient(135deg,rgba(255,255,255,.05),rgba(255,255,255,.02))",
        "acc1": "#38bdf8",
        "acc2": "#22d3ee",
        "good": "#2dd4bf",
        "bad":  "#f87171",
        "muted": "#94a3b8",
    },
    "Amber": {
        "bg_grad": "linear-gradient(135deg,#1a1204 0%,#1e1606 50%,#120b02 100%)",
        "card_grad": "linear-gradient(135deg,rgba(255,255,255,.06),rgba(255,255,255,.02))",
        "acc1": "#fbbf24",
        "acc2": "#fde047",
        "good": "#84cc16",
        "bad":  "#ef4444",
        "muted": "#a3a3a3",
    },
    "Rose": {
        "bg_grad": "linear-gradient(135deg,#1b0a12 0%,#210d18 50%,#120611 100%)",
        "card_grad": "linear-gradient(135deg,rgba(255,255,255,.06),rgba(255,255,255,.02))",
        "acc1": "#fb7185",
        "acc2": "#f472b6",
        "good": "#34d399",
        "bad":  "#f87171",
        "muted": "#cbd5e1",
    },
}

# Sidebar template options
st.sidebar.title("Template Options")
theme_name = st.sidebar.selectbox("Color Theme", list(THEMES.keys()), index=0)
PRIVACY = st.sidebar.toggle("Privacy Mode (mask numbers)", value=False)
st.sidebar.markdown("---")

THEME = THEMES[theme_name]
ACCENT  = THEME["acc1"]
ACCENT2 = THEME["acc2"]
GOOD    = THEME["good"]
BAD     = THEME["bad"]
MUTED   = THEME["muted"]

# Altair theme
def _alt_theme():
    return {
        "config": {
            "background": "transparent",
            "view": {"stroke": "transparent"},
            "axis": {
                "domainColor": "#334155",
                "gridColor": "#1f2937",
                "labelColor": "#e5e7eb",
                "titleColor": "#f3f4f6",
                "grid": True,
            },
            "legend": {"labelColor": "#e5e7eb", "titleColor": "#f3f4f6"},
            "title": {"color": "#ffffff"},
        }
    }
alt.themes.register("vibrant_dark", _alt_theme)
alt.themes.enable("vibrant_dark")

# Global CSS
st.markdown(
    f"""
<style>
[data-testid="stAppViewContainer"] > .main {{ background: {THEME["bg_grad"]}; }}
[data-testid="stSidebar"] > div:first-child {{
  background: rgba(255,255,255,.04);
  border-right: 1px solid rgba(255,255,255,.08);
  backdrop-filter: blur(10px);
}}
.kpi-card {{
  border: 1px solid rgba(255,255,255,.10);
  background: {THEME["card_grad"]};
  border-radius: 16px;
  padding: 12px 16px;
  box-shadow: 0 10px 30px rgba(0,0,0,.25);
}}
.kpi-title {{ color: {MUTED}; font-size:.85rem; margin-bottom:6px; }}
.kpi-value {{ color:#fff; font-size:1.8rem; font-weight:700; line-height:1.1; }}
.badge {{ display:inline-flex; align-items:center; gap:6px; padding:2px 8px; border-radius:999px;
         border:1px solid rgba(255,255,255,.18); background:rgba(255,255,255,.06); color:#e5e7eb; font-size:.78rem;}}
.stTabs [data-baseweb="tab"] {{ background: rgba(255,255,255,.05); border:1px solid rgba(255,255,255,.10);
  border-bottom:none; border-radius:10px 10px 0 0; }}
.stTabs [aria-selected="true"] {{ background: rgba(255,255,255,.10); }}
.dataframe th, .dataframe td {{ border-color: rgba(255,255,255,.08)!important; }}
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------
# STORAGE & SCHEMA (tid)
# -----------------------------
TRADES_CSV = "trades.csv"
TID = "tid"
TRADE_COLUMNS = [
    "id", TID, "symbol", "side", "shares", "entry_date", "company",
    "entry_total", "stop_price", "target1", "target2",
    "exit_date", "exit_price", "fees_total", "strategy", "notes", "created_at",
]
NUMERIC_COLS = {"shares","entry_total","stop_price","target1","target2","exit_price","fees_total"}

def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    for c in TRADE_COLUMNS:
        if c not in df.columns:
            if c in NUMERIC_COLS: df[c] = 0.0
            elif c == TID: df[c] = pd.Series(dtype="Int64")
            else: df[c] = ""
    return df[TRADE_COLUMNS]

def _coerce(df: pd.DataFrame) -> pd.DataFrame:
    for c in NUMERIC_COLS:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    if TID in df.columns: df[TID] = pd.to_numeric(df[TID], errors="coerce").astype("Int64")
    for d in ["entry_date","exit_date"]:
        if d in df.columns: df[d] = df[d].fillna("")
    return df

def _assign_tids(df: pd.DataFrame) -> None:
    if df.empty: return
    m = int(df[TID].dropna().max()) if df[TID].notna().any() else 0
    needs = df[TID].isna()
    if needs.any():
        df.loc[needs, TID] = list(range(m+1, m+1+int(needs.sum())))

def load_trades() -> pd.DataFrame:
    if not os.path.exists(TRADES_CSV):
        return pd.DataFrame(columns=TRADE_COLUMNS)
    df = pd.read_csv(TRADES_CSV, dtype=str)
    if "id" not in df.columns:
        df["id"] = [str(uuid.uuid4()) for _ in range(len(df))]
    df = _ensure_columns(df); df = _coerce(df); _assign_tids(df)
    return df

def save_trades(df: pd.DataFrame) -> None:
    df = _ensure_columns(df.copy()); df = _coerce(df)
    df.to_csv(TRADES_CSV, index=False)

def next_tid(df: pd.DataFrame) -> int:
    if df.empty or df[TID].isna().all(): return 1
    return int(df[TID].dropna().max()) + 1

# -----------------------------
# ANALYTICS & HELPERS
# -----------------------------
def add_trade_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    df = df.copy()
    df["entry_dt"] = pd.to_datetime(df["entry_date"], errors="coerce")
    df["exit_dt"]  = pd.to_datetime(df["exit_date"],  errors="coerce")
    df["entry_price"] = (df["entry_total"]/df["shares"]).where(df["shares"]>0, pd.NA)
    df["realized"] = (~df["exit_dt"].isna()) & (df["exit_price"]>0)

    long_m  = (df["side"].str.lower()=="long")  & df["realized"]
    short_m = (df["side"].str.lower()=="short") & df["realized"]

    df["pnl"] = 0.0
    df.loc[long_m, "pnl"]  = (df.loc[long_m,"exit_price"] - df.loc[long_m,"entry_price"]) * df.loc[long_m,"shares"]
    df.loc[short_m,"pnl"]  = (df.loc[short_m,"entry_price"] - df.loc[short_m,"exit_price"]) * df.loc[short_m,"shares"]
    df.loc[df["realized"],"pnl"] = df.loc[df["realized"],"pnl"] - df.loc[df["realized"],"fees_total"]

    df["ret_pct"] = 0.0
    df.loc[df["realized"] & (df["entry_total"]>0), "ret_pct"] = df.loc[df["realized"],"pnl"]/df.loc[df["realized"],"entry_total"]*100
    df["hold_days"] = (df["exit_dt"] - df["entry_dt"]).dt.days.where(df["realized"], pd.NA)
    df["win"] = (df["pnl"] > 0).where(df["realized"], False)

    df["open"] = ~df["realized"]
    df["open_cost"] = df["entry_total"].where(df["open"], 0.0)
    return df

def mask_money(x: float) -> str:
    return "•••" if PRIVACY else f"${x:,.2f}"

def mask_pct(x: float) -> str:
    return "•••" if PRIVACY else f"{x:.2f}%"

def card(title: str, value_html: str, badge: str|None=None, color: str|None=None):
    badge_html = f'<span class="badge" style="border-color:{color or "rgba(255,255,255,.18)"};color:{color or "#e5e7eb"}">{badge}</span>' if badge else ""
    st.markdown(
        f"""
        <div class="kpi-card">
          <div class="kpi-title">{title} {badge_html}</div>
          <div class="kpi-value">{value_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------- Excel-style STOCKS table builder ----------
def build_stocks_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame with the Excel-like columns:
    Stock (Company • SYMBOL), Symbol, Buy/Sell, Shares, Each, Current (proxy),
    Stop, Target1, Target2, Trade Price, Market Value, Profit/Loss, Change,
    Change(%), Change Total.
    """
    if df.empty:
        return pd.DataFrame(columns=[
            "Stock (Company • Ticker)","Symbol","Buy/Sell","Shares","Each","Current",
            "Stop","Target 1","Target 2","Trade Price","Market Value","Profit/Loss",
            "Change","Change(%)","Change Total"
        ])

    d = df.copy()
    d["Each"] = (d["entry_total"]/d["shares"]).where(d["shares"]>0, 0.0)
    # Current proxy: exit_price if closed, else Each
    d["Current"] = d["exit_price"].where(d["realized"], d["Each"])
    d["Trade Price"] = d["exit_price"].where(d["realized"], 0.0)

    # Profit/Loss (realized pnl if closed; else mark-to-entry)
    unreal = (d["Current"] - d["Each"]) * d["shares"]
    d["Profit/Loss"] = d["pnl"].where(d["realized"], unreal)

    d["Change"] = d["Current"] - d["Each"]
    d["Change(%)"] = (d["Change"] / d["Each"] * 100).where(d["Each"]>0, 0.0)
    d["Change Total"] = d["Change"] * d["shares"]
    d["Market Value"] = d["shares"] * d["Current"]

    # “Buy/Sell” from side (long=Buy, short=Sell)
    d["Buy/Sell"] = d["side"].str.lower().map({"long":"Buy","short":"Sell"}).fillna("")

    # Pretty stock label
    d["Stock (Company • Ticker)"] = d.apply(lambda r: f"{(r['company'] or '').strip() or r['symbol']} • {r['symbol']}", axis=1)

    out = d[
        [
            "Stock (Company • Ticker)","symbol","Buy/Sell","shares","Each","Current",
            "stop_price","target1","target2","Trade Price","Market Value","Profit/Loss",
            "Change","Change(%)","Change Total"
        ]
    ].rename(columns={
        "symbol": "Symbol",
        "stop_price":"Stop",
        "target1":"Target 1",
        "target2":"Target 2",
    })

    # order by Symbol then id desc-ish feel
    out = out.sort_values(["Symbol"], ascending=True).reset_index(drop=True)
    return out

def style_stocks_table(df_stocks: pd.DataFrame) -> pd.io.formats.style.Styler:
    """Return Styler with green/red coloring and money formatting."""
    money_cols = ["Each","Current","Stop","Target 1","Target 2","Trade Price","Market Value","Profit/Loss","Change","Change Total"]
    pct_cols = ["Change(%)"]
    int_cols = ["Shares"]

    styled = df_stocks.style

    # number formats
    for c in money_cols:
        if c in df_stocks.columns:
            styled = styled.format({c: (lambda x: mask_money(float(x)))})
    for c in pct_cols:
        if c in df_stocks.columns:
            styled = styled.format({c: (lambda x: mask_pct(float(x)))})
    if "Shares" in df_stocks.columns:
        styled = styled.format({"Shares": lambda x: f"{float(x):,.4f}"})

    # color helpers
    def posneg(val):
        try:
            v = float(val)
        except Exception:
            return ""
        if v > 0:
            return f"color:{GOOD};"
        if v < 0:
            return f"color:{BAD};"
        return f"color:#e5e7eb;"

    for col in ["Profit/Loss","Change","Change(%)","Change Total"]:
        if col in df_stocks.columns:
            styled = styled.applymap(posneg, subset=pd.IndexSlice[:, [col]])

    # header/row borders subtle
    styled = styled.set_properties(**{"border-color":"rgba(255,255,255,.10)"})
    return styled

# -----------------------------
# SIDEBAR NAV
# -----------------------------
page = st.sidebar.radio("Go to", ["Dashboard", "Reporting Summary", "Add / Manage Trades"], index=0)

# ============================================================
# DASHBOARD
# ============================================================
if page == "Dashboard":
    st.markdown(f"## Portfolio Dashboard — <span style='color:{ACCENT2}'>{theme_name}</span>", unsafe_allow_html=True)

    raw = load_trades()
    df = add_trade_metrics(raw)

    if df.empty:
        st.info("No trades yet. Add some in **Add / Manage Trades**.")
    else:
        with st.expander("Filters", expanded=False):
            valid = df[~df["entry_dt"].isna()]
            if not valid.empty:
                dmin, dmax = valid["entry_dt"].min().date(), valid["entry_dt"].max().date()
                date_rng = st.date_input("Entry Date range", value=(dmin, dmax), format="YYYY/MM/DD")
            else:
                date_rng = None
            symbols = sorted([s for s in df["symbol"].dropna().unique() if s])
            sel_syms = st.multiselect("Symbols", options=symbols, default=symbols)

        df_f = df.copy()
        if date_rng and isinstance(date_rng, tuple) and len(date_rng)==2:
            df_f = df_f[df_f["entry_dt"].between(pd.to_datetime(date_rng[0]), pd.to_datetime(date_rng[1]))]
        if sel_syms: df_f = df_f[df_f["symbol"].isin(sel_syms)]

        realized = df_f[df_f["realized"]]
        open_pos = df_f[df_f["open"]]

        total_invested = float(df_f["entry_total"].sum())
        realized_pnl   = float(realized["pnl"].sum()) if not realized.empty else 0.0
        open_cost      = float(open_pos["open_cost"].sum()) if not open_pos.empty else 0.0
        win_rate       = (realized["win"].mean()*100) if not realized.empty else 0.0
        avg_ret        = realized["ret_pct"].mean() if not realized.empty else 0.0

        c1,c2,c3,c4,c5 = st.columns(5)
        with c1: card("Portfolio Value*", mask_money(total_invested + realized_pnl), "proxy", ACCENT2)
        with c2: card("Open Cost", mask_money(open_cost))
        with c3: card("Realized P&L", mask_money(realized_pnl), "realized", GOOD if realized_pnl>=0 else BAD)
        with c4: card("Win Rate", mask_pct(win_rate), "realized", GOOD)
        with c5: card("Avg Return / Trade", mask_pct(avg_ret), "realized", GOOD)

        st.caption("*) Proxy = invested cash + realized P&L (no live quotes).")

        tab1, tab2, tab3, tab4 = st.tabs(["📈 Overview", "💰 P&L", "🧭 Breakdown", "📋 Trades"])

        # Overview
        with tab1:
            l, r = st.columns([2,1], gap="large")

            with l:
                if realized.empty:
                    st.info("No realized trades yet for the equity curve.")
                else:
                    ec = realized.sort_values("exit_dt")[["exit_dt","pnl"]].dropna()
                    ec = ec.groupby("exit_dt", as_index=False).agg(pnl=("pnl","sum"))
                    ec["cum_pnl"] = ec["pnl"].cumsum()
                    line = (
                        alt.Chart(ec, height=340)
                        .mark_line(point=True, interpolate="monotone", color=ACCENT2)
                        .encode(
                            x=alt.X("exit_dt:T", title="Exit Date"),
                            y=alt.Y("cum_pnl:Q", title="Cumulative P&L ($)"),
                            tooltip=["exit_dt:T", alt.Tooltip("cum_pnl:Q", format=",.2f")],
                        )
                        .properties(title="Equity Curve")
                    )
                    st.altair_chart(line, use_container_width=True)

            with r:
                if open_pos.empty:
                    st.info("No open positions for allocation.")
                else:
                    dv = open_pos.groupby("symbol", as_index=False).agg(open_cost=("open_cost","sum")).sort_values("open_cost", ascending=False)
                    donut = (
                        alt.Chart(dv)
                        .mark_arc(innerRadius=60, outerRadius=120)
                        .encode(
                            theta=alt.Theta("open_cost:Q", title=""),
                            color=alt.Color("symbol:N", title=""),
                            tooltip=[alt.Tooltip("symbol:N"), alt.Tooltip("open_cost:Q", format=",.2f")],
                        )
                        .properties(height=340, title="Allocation by Symbol (Open Cost)")
                    )
                    st.altair_chart(donut, use_container_width=True)

        # P&L
        with tab2:
            a, b = st.columns([1.3,1], gap="large")
            if realized.empty:
                a.info("No realized trades to chart P&L by month.")
                b.info("No realized trades for daily heatmap.")
            else:
                rm = realized.dropna(subset=["exit_dt"]).copy()
                rm["month"] = rm["exit_dt"].dt.to_period("M").dt.to_timestamp()
                month_pnl = rm.groupby("month", as_index=False).agg(total_pnl=("pnl","sum"))
                month_pnl["pos"]=month_pnl["total_pnl"]>=0
                bar = (
                    alt.Chart(month_pnl, height=340)
                    .mark_bar()
                    .encode(
                        x=alt.X("month:T", title="Month"),
                        y=alt.Y("total_pnl:Q", title="P&L ($)"),
                        color=alt.Color("pos:N", scale=alt.Scale(domain=[True,False], range=[GOOD,BAD]), legend=None),
                        tooltip=["month:T", alt.Tooltip("total_pnl:Q", format=",.2f")],
                    ).properties(title="P&L by Month")
                )
                a.altair_chart(bar, use_container_width=True)

                daily = realized.copy()
                daily["day"] = daily["exit_dt"].dt.date
                daily = daily.groupby("day", as_index=False).agg(pnl=("pnl","sum"))
                heat = (
                    alt.Chart(daily, height=340)
                    .mark_rect()
                    .encode(
                        x=alt.X("day:T", title="Date"),
                        y=alt.Y("value:Q", aggregate="count", title="Trades", axis=None),
                        color=alt.Color("pnl:Q", scale=alt.Scale(scheme="redblue", domainMid=0), legend=alt.Legend(title="P&L")),
                        tooltip=[alt.Tooltip("day:T"), alt.Tooltip("pnl:Q", format=",.2f")],
                    ).properties(title="Daily Net P&L (Heat)")
                )
                b.altair_chart(heat, use_container_width=True)

        # Breakdown
        with tab3:
            b1,b2 = st.columns(2, gap="large")
            if realized.empty:
                b1.info("No realized P&L by symbol.")
                b2.info("No realized win rate.")
            else:
                pnl_sym = realized.groupby("symbol", as_index=False).agg(total_pnl=("pnl","sum")).sort_values("total_pnl", ascending=False)
                pnl_sym["pos"]=pnl_sym["total_pnl"]>=0
                chart = (
                    alt.Chart(pnl_sym, height=340)
                    .mark_bar(cornerRadiusEnd=6)
                    .encode(
                        x=alt.X("total_pnl:Q", title="Total P&L ($)"),
                        y=alt.Y("symbol:N", sort="-x", title=""),
                        color=alt.Color("pos:N", scale=alt.Scale(domain=[True,False], range=[GOOD,BAD]), legend=None),
                        tooltip=[alt.Tooltip("symbol:N"), alt.Tooltip("total_pnl:Q", format=",.2f")],
                    ).properties(title="Total Realized P&L by Symbol")
                )
                b1.altair_chart(chart, use_container_width=True)

                wr = realized.groupby("symbol", as_index=False).agg(win_rate=("win","mean"), trades=("id","count")).sort_values("win_rate", ascending=False)
                wr["win_rate_pct"]=wr["win_rate"]*100
                chart2 = (
                    alt.Chart(wr, height=340)
                    .mark_bar(cornerRadiusEnd=6, color=ACCENT)
                    .encode(
                        x=alt.X("win_rate_pct:Q", title="Win Rate (%)", scale=alt.Scale(domain=[0,100])),
                        y=alt.Y("symbol:N", sort="-x", title=""),
                        tooltip=["symbol:N", alt.Tooltip("win_rate_pct:Q", format=".1f"), alt.Tooltip("trades:Q", title="Trades")],
                    ).properties(title="Win Rate by Symbol")
                )
                b2.altair_chart(chart2, use_container_width=True)

        # Trades + Stocks table
        with tab4:
            st.markdown("#### All Trades (filtered)")
            cols = [TID,"entry_date","symbol","side","shares","entry_total","exit_date","exit_price","fees_total","pnl","ret_pct","hold_days","strategy","notes"]
            cols = [c for c in cols if c in df_f.columns]
            t = df_f[cols].sort_values(TID, ascending=False).copy()
            for c in ["entry_total","exit_price","fees_total","pnl"]:
                if c in t.columns: t[c]=t[c].map(lambda x: mask_money(float(x)))
            if "ret_pct" in t.columns: t["ret_pct"]=t["ret_pct"].map(lambda x: mask_pct(float(x)))
            if "shares" in t.columns: t["shares"]=t["shares"].map(lambda x: f"{float(x):,.4f}")
            if "hold_days" in t.columns: t["hold_days"]=t["hold_days"].map(lambda x: "" if pd.isna(x) else int(x))
            st.dataframe(t, use_container_width=True, height=260)

            st.markdown("### Stocks (Excel-style)")
            stock_df = build_stocks_table(df_f)
            if stock_df.empty:
                st.info("No rows to show.")
            else:
                styled = style_stocks_table(stock_df)
                st.dataframe(styled, use_container_width=True, height=420)

# ============================================================
# REPORTING SUMMARY
# ============================================================
elif page == "Reporting Summary":
    st.markdown("## Reporting Summary")

    raw = load_trades()
    df = add_trade_metrics(raw)

    if df.empty:
        st.info("No trades yet.")
    else:
        c1,c2 = st.columns([2,1])
        with c1:
            valid = df[~df["entry_dt"].isna()]
            dmin, dmax = (valid["entry_dt"].min().date(), valid["entry_dt"].max().date()) if not valid.empty else (date.today(), date.today())
            rng = st.date_input("Report Date Range (by Entry Date)", value=(dmin,dmax), format="YYYY/MM/DD")
        with c2:
            syms = sorted([s for s in df["symbol"].dropna().unique() if s])
            pick_syms = st.multiselect("Symbols", options=syms, default=syms)

        rep = df.copy()
        if rng and isinstance(rng, tuple) and len(rng)==2:
            rep = rep[rep["entry_dt"].between(pd.to_datetime(rng[0]), pd.to_datetime(rng[1]))]
        if pick_syms:
            rep = rep[rep["symbol"].isin(pick_syms)]

        realized = rep[rep["realized"]]
        winners  = int((realized["pnl"]>0).sum())
        losers   = int((realized["pnl"]<=0).sum())
        avg_win  = float(realized.loc[realized["pnl"]>0,"pnl"].mean()) if winners else 0.0
        avg_loss = float(realized.loc[realized["pnl"]<=0,"pnl"].mean()) if losers else 0.0
        total_invested = float(rep["entry_total"].sum())
        realized_net   = float(realized["pnl"].sum()) if not realized.empty else 0.0

        eq = realized.sort_values("exit_dt")[["exit_dt","pnl"]].dropna()
        if not eq.empty:
            eq["cum"] = eq["pnl"].cumsum()
            eq["pct"] = (eq["cum"] / total_invested * 100) if total_invested>0 else 0.0
        else:
            eq = pd.DataFrame({"exit_dt":[], "cum":[], "pct":[]})

        k1,k2,k3,k4,k5,k6 = st.columns(6)
        with k1: card("Realized Net Gain/Loss", mask_money(realized_net), color=GOOD if realized_net>=0 else BAD, badge="realized")
        with k2: card("Portfolio Net Gain/Loss", mask_money(realized_net), color=GOOD if realized_net>=0 else BAD)
        with k3: card("Winners", f"{winners}")
        with k4: card("Losers", f"{losers}")
        with k5: card("Average Win", mask_money(avg_win))
        with k6: card("Average Loss", mask_money(avg_loss))

        g1,g2 = st.columns([1.4,1], gap="large")
        with g1:
            if realized.empty:
                st.info("No realized trades in range.")
            else:
                by_sym = (realized.groupby("symbol", as_index=False).agg(pnl=("pnl","sum")).sort_values("pnl", ascending=False))
                by_sym["pos"] = by_sym["pnl"]>=0
                bars = (
                    alt.Chart(by_sym, height=380)
                    .mark_bar(cornerRadiusEnd=6)
                    .encode(
                        x=alt.X("pnl:Q", title="P&L ($)"),
                        y=alt.Y("symbol:N", sort="-x", title="Symbol"),
                        color=alt.Color("pos:N", scale=alt.Scale(domain=[True,False], range=[GOOD,BAD]), legend=None),
                        tooltip=["symbol:N", alt.Tooltip("pnl:Q", format=",.2f")],
                    )
                    .properties(title="Gains by Stock (Realized)")
                )
                st.altair_chart(bars, use_container_width=True)
        with g2:
            if eq.empty:
                st.info("No exits in range to compute return.")
            else:
                line = (
                    alt.Chart(eq, height=380)
                    .mark_line(point=True, interpolate="monotone", color=ACCENT2)
                    .encode(
                        x=alt.X("exit_dt:T", title="Date"),
                        y=alt.Y("pct:Q", title="Cumulative Return (%)"),
                        tooltip=["exit_dt:T", alt.Tooltip("pct:Q", format=".2f")],
                    )
                    .properties(title="Cumulative Return Over Time")
                )
                st.altair_chart(line, use_container_width=True)

        st.markdown("#### Summaries")
        if realized.empty:
            st.info("No realized trades to summarize.")
        else:
            summ = (
                realized.groupby("symbol", as_index=False)
                .agg(
                    trades=("id","count"),
                    wins=("win","sum"),
                    pnl=("pnl","sum"),
                    avg_ret=("ret_pct","mean"),
                    avg_hold=("hold_days","mean"),
                )
                .sort_values("pnl", ascending=False)
            )
            summ["win_rate(%)"] = (summ["wins"]/summ["trades"]*100).round(1)
            summ.rename(columns={"pnl":"pnl($)","avg_ret":"avg_return(%)","avg_hold":"avg_hold(days)"}, inplace=True)
            st.dataframe(summ, use_container_width=True, height=360)

# ============================================================
# ADD / MANAGE TRADES
# ============================================================
else:
    st.markdown("## Add Trade")

    with st.form("add_trade_form", enter_to_submit=False):
        c1,c2,c3 = st.columns(3)
        with c1:
            symbol = st.text_input("Symbol *", value="AMZN").upper().strip()
            entry_total = st.number_input("Entry Total ($) *", min_value=0.0, value=0.00, step=1.0, format="%.2f")
            stop_price  = st.number_input("Stop Price (optional)", min_value=0.0, value=0.0, step=1.0, format="%.4f")
            use_exit    = st.checkbox("Set Exit Date", value=False)
            exit_date_dt= st.date_input("Exit Date (optional)", value=date.today(), format="YYYY/MM/DD", disabled=not use_exit)
        with c2:
            side  = st.selectbox("Side *", options=["long","short"], index=0)
            entry_date_dt = st.date_input("Entry Date *", value=date.today(), format="YYYY/MM/DD")
            target1 = st.number_input("Target 1 (optional)", min_value=0.0, value=0.0, step=1.0, format="%.4f")
            exit_price = st.number_input("Exit Price (optional)", min_value=0.0, value=0.0, step=1.0, format="%.4f")
        with c3:
            shares = st.number_input("Shares (fractional ok) *", min_value=0.0, value=1.0, step=1.0, format="%.6f")
            company= st.text_input("Company", value="")
            target2= st.number_input("Target 2 (optional)", min_value=0.0, value=0.0, step=1.0, format="%.4f")
            fees_total = st.number_input("Fees Total ($)", min_value=0.0, value=0.0, step=1.0, format="%.2f")
        strategy = st.text_input("Strategy / Tag (optional)", value="")
        notes    = st.text_area("Notes", height=120)

        if st.form_submit_button("Save Trade"):
            if not symbol:
                st.error("Symbol is required.")
            else:
                df = load_trades()
                rec = {
                    "id": str(uuid.uuid4()),
                    TID: next_tid(df),
                    "symbol": symbol,
                    "side": side,
                    "shares": float(shares),
                    "entry_date": entry_date_dt.strftime("%Y/%m/%d"),
                    "company": company,
                    "entry_total": float(entry_total),
                    "stop_price": float(stop_price),
                    "target1": float(target1),
                    "target2": float(target2),
                    "exit_date": (exit_date_dt.strftime("%Y/%m/%d") if use_exit else ""),
                    "exit_price": float(exit_price),
                    "fees_total": float(fees_total),
                    "strategy": strategy,
                    "notes": notes,
                    "created_at": datetime.utcnow().isoformat(timespec="seconds")+"Z",
                }
                df = pd.concat([df, pd.DataFrame([rec])], ignore_index=True)
                save_trades(df)
                st.success(f"Trade saved as #{rec[TID]}.")

    st.divider()
    st.markdown("## Manage Trades")

    df_all = load_trades()
    if df_all.empty:
        st.info("No trades saved yet.")
    else:
        df_show = df_all[[TID,"symbol","side","shares","entry_date","entry_total","exit_date","exit_price","fees_total","strategy","notes","created_at"]].sort_values(TID, ascending=False)
        st.dataframe(df_show, use_container_width=True, height=280)

        # Delete selected
        st.subheader("Delete Selected")
        def label_row(r): return f"#{int(r[TID])} · {r['symbol']} · {r['side']} · {r['shares']} · {r['entry_date']}"
        labels = df_all.apply(label_row, axis=1).tolist()
        id_map = dict(zip(df_all[TID].astype(int), labels))
        label_to_tid = {v:k for k,v in id_map.items()}
        sel_labels = st.multiselect("Choose trade(s) to delete:", options=list(id_map.values()), placeholder="Select …")
        d1,d2 = st.columns(2)
        with d1:
            if st.button("Delete Selected"):
                if not sel_labels: st.warning("No trades selected.")
                else:
                    tids = {label_to_tid[l] for l in sel_labels}
                    save_trades(df_all[~df_all[TID].isin(list(tids))].copy())
                    st.success(f"Deleted {len(tids)} trade(s)."); st.rerun()
        with d2:
            with st.popover("Danger Zone: Delete ALL Trades"):
                st.write("Type **DELETE** to purge every trade.")
                confirm = st.text_input("Confirmation")
                if st.button("Delete ALL Trades", type="primary"):
                    if confirm.strip()=="DELETE":
                        save_trades(pd.DataFrame(columns=TRADE_COLUMNS))
                        st.success("All trades deleted."); st.rerun()
                    else:
                        st.error("Confirmation did not match.")

        # Edit / Close
        st.subheader("Edit / Close a Trade")
        pick_label = st.selectbox("Pick trade:", options=[id_map[i] for i in sorted(id_map.keys(), reverse=True)], index=0)
        edit_tid = label_to_tid[pick_label]
        row = df_all[df_all[TID]==edit_tid].iloc[0]

        with st.form("edit_trade_form", enter_to_submit=False):
            e1,e2,e3 = st.columns(3)
            with e1:
                e_symbol = st.text_input("Symbol", value=str(row["symbol"])).upper().strip()
                e_entry_total = st.number_input("Entry Total ($)", min_value=0.0, value=float(row["entry_total"]), step=1.0, format="%.2f")
                e_stop_price  = st.number_input("Stop Price", min_value=0.0, value=float(row["stop_price"]), step=1.0, format="%.4f")
                e_entry_date  = st.date_input("Entry Date", value=pd.to_datetime(row["entry_date"], errors="coerce").date() if row["entry_date"] else date.today(), format="YYYY/MM/DD")
            with e2:
                e_side = st.selectbox("Side", options=["long","short"], index=0 if str(row["side"]).lower()=="long" else 1)
                e_target1 = st.number_input("Target 1", min_value=0.0, value=float(row["target1"]), step=1.0, format="%.4f")
                e_exit_price = st.number_input("Exit Price", min_value=0.0, value=float(row["exit_price"]), step=1.0, format="%.4f")
                e_exit_date  = st.date_input("Exit Date", value=pd.to_datetime(row["exit_date"], errors="coerce").date() if row["exit_date"] else date.today(), format="YYYY/MM/DD", disabled=(float(row["exit_price"])==0.0 and str(row["exit_date"]).strip()==""))
            with e3:
                e_shares = st.number_input("Shares", min_value=0.0, value=float(row["shares"]), step=1.0, format="%.6f")
                e_company= st.text_input("Company", value=str(row["company"]))
                e_target2= st.number_input("Target 2", min_value=0.0, value=float(row["target2"]), step=1.0, format="%.4f")
                e_fees_total = st.number_input("Fees Total ($)", min_value=0.0, value=float(row["fees_total"]), step=1.0, format="%.2f")
            e_strategy = st.text_input("Strategy / Tag", value=str(row["strategy"]))
            e_notes    = st.text_area("Notes", value=str(row["notes"]), height=100)

            b1,b2,b3,b4 = st.columns(4)
            if b1.form_submit_button("Save Changes"):
                df_all.loc[df_all[TID]==edit_tid,:] = [
                    row["id"], edit_tid, e_symbol, e_side, float(e_shares),
                    e_entry_date.strftime("%Y/%m/%d"), e_company,
                    float(e_entry_total), float(e_stop_price),
                    float(e_target1), float(e_target2),
                    e_exit_date.strftime("%Y/%m/%d") if (str(row["exit_date"]).strip() or float(e_exit_price)>0) else "",
                    float(e_exit_price), float(e_fees_total),
                    e_strategy, e_notes, row["created_at"]
                ]
                save_trades(df_all); st.success(f"Trade #{edit_tid} updated."); st.rerun()

            if b2.form_submit_button("Close Trade"):
                if float(e_exit_price)<=0: st.error("Set an Exit Price to close.")
                else:
                    df_all.loc[df_all[TID]==edit_tid, ["exit_price","exit_date","fees_total"]] = [
                        float(e_exit_price), date.today().strftime("%Y/%m/%d"), float(e_fees_total)
                    ]
                    save_trades(df_all); st.success(f"Trade #{edit_tid} closed."); st.rerun()

            if b3.form_submit_button("Reopen Trade"):
                df_all.loc[df_all[TID]==edit_tid, ["exit_price","exit_date"]] = [0.0, ""]
                save_trades(df_all); st.success(f"Trade #{edit_tid} reopened."); st.rerun()

            if b4.form_submit_button("Delete This Trade"):
                save_trades(df_all[df_all[TID]!=edit_tid].copy()); st.success(f"Trade #{edit_tid} deleted."); st.rerun()
