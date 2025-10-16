# app.py
# ============================================================
# Trades Dashboard (Vibrant UI Edition)
# - Sidebar: Dashboard | Add / Manage Trades
# - CSV persistence: trades.csv
# - Simple numeric Trade IDs (tid)
# - Integer-step spinners in forms
# - Edit / Close / Reopen / Delete trades
# - Beautiful, modern dashboard with tabs & rich visuals
# ============================================================

import os
import uuid
from datetime import date, datetime

import pandas as pd
import streamlit as st
import altair as alt

st.set_page_config(page_title="Trades Dashboard", layout="wide")

# -----------------------------
# Color System / Theme
# -----------------------------
BG_GRADIENT = "linear-gradient(135deg, #0e1117 0%, #111827 50%, #0b1020 100%)"
CARD_GRADIENT = "linear-gradient(135deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02))"
ACCENT = "#6EE7B7"     # Mint
ACCENT_2 = "#7AA5FF"   # Blue
DANGER = "#FB7185"     # Rose
WARNING = "#FBBF24"    # Amber
MUTED = "#94A3B8"      # Slate

def altair_dark_theme():
    return {
        "config": {
            "background": "transparent",
            "view": {"stroke": "transparent"},
            "axis": {
                "domainColor": "#374151",
                "gridColor": "#1F2937",
                "labelColor": "#E5E7EB",
                "titleColor": "#F3F4F6",
                "grid": True,
            },
            "legend": {"labelColor": "#E5E7EB", "titleColor": "#F3F4F6"},
            "title": {"color": "#FFFFFF"},
        }
    }

alt.themes.register("vibrant_dark", altair_dark_theme)
alt.themes.enable("vibrant_dark")

# -----------------------------
# CSS – glassmorphism cards & layout polish
# -----------------------------
st.markdown(
    f"""
<style>
/* App background */
[data-testid="stAppViewContainer"] > .main {{
  background: {BG_GRADIENT};
}}
/* Sidebar */
[data-testid="stSidebar"] > div:first-child {{
  background: rgba(255,255,255,0.04);
  backdrop-filter: blur(12px);
  border-right: 1px solid rgba(255,255,255,0.06);
}}
/* Section titles */
h1, h2, h3 {{
  letter-spacing: .3px;
}}
/* Card */
.kpi-card {{
  border: 1px solid rgba(255,255,255,0.10);
  background: {CARD_GRADIENT};
  border-radius: 18px;
  padding: 16px 18px;
  position: relative;
  box-shadow: 0 10px 30px rgba(0,0,0,.25);
}}
.kpi-title {{
  font-size: 0.85rem;
  color: {MUTED};
  margin-bottom: 6px;
  display:flex; align-items:center; gap:8px;
}}
.kpi-value {{
  font-size: 1.8rem;
  font-weight: 700;
  color: #FFFFFF;
  line-height: 1.1;
}}
.kpi-delta {{
  font-size: .9rem;
  color: {MUTED};
}}
.badge {{
  display:inline-flex; align-items:center; gap:6px;
  padding: 2px 10px;
  border-radius: 999px;
  font-size: .78rem;
  border: 1px solid rgba(255,255,255,.14);
  background: rgba(255,255,255,.06);
  color: #E5E7EB;
}}
.win {{ color: {ACCENT}; }}
.lose {{ color: {DANGER}; }}
/* Tables */
.dataframe th, .dataframe td {{
  border-color: rgba(255,255,255,0.08)!important;
}}
/* Tabs */
.stTabs [data-baseweb="tab"] {{
  background: rgba(255,255,255,.04);
  border: 1px solid rgba(255,255,255,.10);
  border-bottom: none;
  border-radius: 10px 10px 0 0;
}}
.stTabs [aria-selected="true"] {{
  background: rgba(255,255,255,.10);
}}
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------
# Storage / schema with tid
# -----------------------------
TRADES_CSV = "trades.csv"
TID_COL = "tid"
TRADE_COLUMNS = [
    "id", TID_COL, "symbol", "side", "shares", "entry_date", "company",
    "entry_total", "stop_price", "target1", "target2",
    "exit_date", "exit_price", "fees_total", "strategy", "notes", "created_at",
]
NUMERIC_COLS = {"shares", "entry_total", "stop_price", "target1", "target2", "exit_price", "fees_total"}

def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    for c in TRADE_COLUMNS:
        if c not in df.columns:
            if c in NUMERIC_COLS: df[c] = 0.0
            elif c == TID_COL: df[c] = pd.Series(dtype="Int64")
            else: df[c] = ""
    return df[TRADE_COLUMNS]

def _coerce(df: pd.DataFrame) -> pd.DataFrame:
    for c in NUMERIC_COLS:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    if TID_COL in df.columns: df[TID_COL] = pd.to_numeric(df[TID_COL], errors="coerce").astype("Int64")
    for d in ["entry_date", "exit_date"]:
        if d in df.columns: df[d] = df[d].fillna("")
    return df

def _assign_missing_tids(df: pd.DataFrame) -> None:
    if df.empty: return
    max_tid = int(df[TID_COL].dropna().max()) if df[TID_COL].notna().any() else 0
    need = df[TID_COL].isna()
    if need.any():
        new_ids = range(max_tid + 1, max_tid + 1 + int(need.sum()))
        df.loc[need, TID_COL] = list(new_ids)

def load_trades() -> pd.DataFrame:
    if not os.path.exists(TRADES_CSV):
        return pd.DataFrame(columns=TRADE_COLUMNS)
    df = pd.read_csv(TRADES_CSV, dtype=str)
    if "id" not in df.columns:
        df["id"] = [str(uuid.uuid4()) for _ in range(len(df))]
    df = _ensure_columns(df)
    df = _coerce(df)
    _assign_missing_tids(df)
    return df

def save_trades(df: pd.DataFrame) -> None:
    df = _ensure_columns(df.copy())
    df = _coerce(df)
    df.to_csv(TRADES_CSV, index=False)

def next_tid(df: pd.DataFrame) -> int:
    if df.empty or df[TID_COL].isna().all(): return 1
    return int(df[TID_COL].dropna().max()) + 1

# -----------------------------
# Analytics helpers
# -----------------------------
def add_trade_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    df = df.copy()
    df["entry_dt"] = pd.to_datetime(df["entry_date"], errors="coerce")
    df["exit_dt"] = pd.to_datetime(df["exit_date"], errors="coerce")
    df["entry_price"] = (df["entry_total"] / df["shares"]).where(df["shares"] > 0, pd.NA)
    df["realized"] = (~df["exit_dt"].isna()) & (df["exit_price"] > 0)

    long_mask = (df["side"].str.lower() == "long") & df["realized"]
    short_mask = (df["side"].str.lower() == "short") & df["realized"]

    df["pnl"] = 0.0
    df.loc[long_mask, "pnl"]  = (df.loc[long_mask,"exit_price"] - df.loc[long_mask,"entry_price"]) * df.loc[long_mask,"shares"]
    df.loc[short_mask, "pnl"] = (df.loc[short_mask,"entry_price"] - df.loc[short_mask,"exit_price"]) * df.loc[short_mask,"shares"]
    df.loc[df["realized"], "pnl"] = df.loc[df["realized"], "pnl"] - df.loc[df["realized"], "fees_total"]

    df["ret_pct"] = 0.0
    df.loc[df["realized"] & (df["entry_total"] > 0), "ret_pct"] = df.loc[df["realized"], "pnl"] / df.loc[df["realized"], "entry_total"] * 100
    df["hold_days"] = (df["exit_dt"] - df["entry_dt"]).dt.days.where(df["realized"], pd.NA)
    df["win"] = (df["pnl"] > 0).where(df["realized"], False)

    df["open"] = ~df["realized"]
    df["open_cost"] = df["entry_total"].where(df["open"], 0.0)
    return df

def card(title, value, badge=None, color=None):
    delta_html = f'<span class="badge" style="border-color:{color or "rgba(255,255,255,.18)"}; color:{color or "#E5E7EB"}">{badge}</span>' if badge else ""
    st.markdown(
        f"""
        <div class="kpi-card">
          <div class="kpi-title">{title} {delta_html}</div>
          <div class="kpi-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------
# Sidebar Nav
# -----------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", options=("Dashboard", "Add / Manage Trades"), index=0)

# ============================================================
# DASHBOARD
# ============================================================
if page == "Dashboard":
    st.markdown("## Portfolio Dashboard")

    raw = load_trades()
    df = add_trade_metrics(raw)

    if df.empty:
        st.info("No trades yet. Add some on **Add / Manage Trades**.")
    else:
        # Filters
        with st.expander("Filters", expanded=False):
            valid_entries = df[~df["entry_dt"].isna()]
            if not valid_entries.empty:
                dmin = valid_entries["entry_dt"].min().date()
                dmax = valid_entries["entry_dt"].max().date()
                date_range = st.date_input("Entry Date range", value=(dmin, dmax), format="YYYY/MM/DD")
            else:
                date_range = None
            symbols = sorted([s for s in df["symbol"].dropna().unique() if s])
            sel_symbols = st.multiselect("Symbols", options=symbols, default=symbols)

        df_f = df.copy()
        if date_range and isinstance(date_range, tuple) and len(date_range) == 2:
            start, end = date_range
            df_f = df_f[df_f["entry_dt"].between(pd.to_datetime(start), pd.to_datetime(end))]
        if sel_symbols:
            df_f = df_f[df_f["symbol"].isin(sel_symbols)]

        realized = df_f[df_f["realized"]].copy()
        open_pos = df_f[df_f["open"]].copy()

        total_invested = float(df_f["entry_total"].sum())
        realized_pnl = float(realized["pnl"].sum()) if not realized.empty else 0.0
        open_cost = float(open_pos["open_cost"].sum()) if not open_pos.empty else 0.0
        win_rate = (realized["win"].mean() * 100.0) if not realized.empty else 0.0
        avg_ret = realized["ret_pct"].mean() if not realized.empty else 0.0

        # Summary ribbon (glassy KPI cards)
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            card("Portfolio Value*", f"${(total_invested + realized_pnl):,.2f}", "proxy", color=ACCENT_2)
        with c2:
            card("Open Cost", f"${open_cost:,.2f}")
        with c3:
            color = ACCENT if realized_pnl >= 0 else DANGER
            card("Realized P&L", f"${realized_pnl:,.2f}", ("gain" if realized_pnl >= 0 else "loss"), color=color)
        with c4:
            card("Win Rate", f"{win_rate:.1f}%", badge="realized", color=ACCENT)
        with c5:
            card("Avg Return / Trade", f"{avg_ret:.2f}%", badge="realized", color=ACCENT)

        st.caption("*) Proxy: invested cash + realized P&L (no live prices).")

        # Tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Overview", "💰 P&L", "🧭 Breakdown", "📋 Trades"])

        # -------- Overview
        with tab1:
            left, right = st.columns([2, 1], gap="large")

            # Equity curve
            with left:
                if realized.empty:
                    st.info("No realized trades yet to plot the Equity Curve.")
                else:
                    ec = realized.sort_values("exit_dt")[["exit_dt", "pnl"]].dropna()
                    ec = ec.groupby("exit_dt", as_index=False).agg(pnl=("pnl", "sum"))
                    ec["cum_pnl"] = ec["pnl"].cumsum()

                    line = (
                        alt.Chart(ec, height=340)
                        .mark_line(point=True, interpolate="monotone")
                        .encode(
                            x=alt.X("exit_dt:T", title="Exit Date"),
                            y=alt.Y("cum_pnl:Q", title="Cumulative P&L ($)"),
                            tooltip=["exit_dt:T", alt.Tooltip("cum_pnl:Q", format=",.2f")],
                        )
                        .properties(title="Equity Curve (Realized)")
                    )
                    st.altair_chart(line, use_container_width=True)

            # Allocation donut (open cost)
            with right:
                if open_pos.empty:
                    st.info("No open positions to show allocation.")
                else:
                    dv = (
                        open_pos.groupby("symbol", as_index=False)
                        .agg(open_cost=("open_cost", "sum"))
                        .sort_values("open_cost", ascending=False)
                    )
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

        # -------- P&L
        with tab2:
            colA, colB = st.columns([1.3, 1], gap="large")

            # Monthly bars
            with colA:
                if realized.empty:
                    st.info("No realized trades for monthly P&L.")
                else:
                    rm = realized.dropna(subset=["exit_dt"]).copy()
                    rm["month"] = rm["exit_dt"].dt.to_period("M").dt.to_timestamp()
                    month_pnl = rm.groupby("month", as_index=False).agg(total_pnl=("pnl", "sum"))
                    month_pnl["pos"] = month_pnl["total_pnl"] >= 0

                    bar = (
                        alt.Chart(month_pnl, height=340)
                        .mark_bar()
                        .encode(
                            x=alt.X("month:T", title="Month"),
                            y=alt.Y("total_pnl:Q", title="P&L ($)"),
                            color=alt.Color("pos:N", scale=alt.Scale(domain=[True, False], range=[ACCENT, DANGER]), legend=None),
                            tooltip=["month:T", alt.Tooltip("total_pnl:Q", format=",.2f")],
                        )
                        .properties(title="P&L by Month")
                    )
                    st.altair_chart(bar, use_container_width=True)

            # Month heatmap (days with wins/losses)
            with colB:
                if realized.empty:
                    st.info("No realized trades to visualize daily P&L.")
                else:
                    cal = realized.copy()
                    cal["day"] = cal["exit_dt"].dt.date
                    cal = cal.groupby("day", as_index=False).agg(pnl=("pnl", "sum"))
                    heat = (
                        alt.Chart(cal, height=340)
                        .mark_rect()
                        .encode(
                            x=alt.X("day:T", title="Date"),
                            y=alt.Y("value:Q", aggregate="count", title="Trades", axis=None),
                            color=alt.Color("pnl:Q", scale=alt.Scale(scheme="redblue", domainMid=0), legend=alt.Legend(title="P&L")),
                            tooltip=[alt.Tooltip("day:T"), alt.Tooltip("pnl:Q", format=",.2f")],
                        )
                        .properties(title="Daily Net P&L (Heat)")
                    )
                    st.altair_chart(heat, use_container_width=True)

        # -------- Breakdown
        with tab3:
            b1, b2 = st.columns(2, gap="large")

            with b1:
                if realized.empty:
                    st.info("No realized trades to show P&L by symbol.")
                else:
                    pnl_sym = (
                        realized.groupby("symbol", as_index=False)
                        .agg(total_pnl=("pnl", "sum"))
                        .sort_values("total_pnl", ascending=False)
                    )
                    pnl_sym["pos"] = pnl_sym["total_pnl"] >= 0
                    chart = (
                        alt.Chart(pnl_sym, height=360)
                        .mark_bar(cornerRadiusEnd=6)
                        .encode(
                            x=alt.X("total_pnl:Q", title="Total P&L ($)"),
                            y=alt.Y("symbol:N", sort="-x", title=""),
                            color=alt.Color("pos:N", scale=alt.Scale(domain=[True, False], range=[ACCENT, DANGER]), legend=None),
                            tooltip=[alt.Tooltip("symbol:N"), alt.Tooltip("total_pnl:Q", format=",.2f")],
                        )
                        .properties(title="Total Realized P&L by Symbol")
                    )
                    st.altair_chart(chart, use_container_width=True)

            with b2:
                if realized.empty:
                    st.info("No realized trades to show win rate.")
                else:
                    wr = (
                        realized.groupby("symbol", as_index=False)
                        .agg(win_rate=("win", "mean"), trades=("id", "count"))
                        .sort_values("win_rate", ascending=False)
                    )
                    wr["win_rate_pct"] = wr["win_rate"] * 100
                    chart = (
                        alt.Chart(wr, height=360)
                        .mark_bar(cornerRadiusEnd=6)
                        .encode(
                            x=alt.X("win_rate_pct:Q", title="Win Rate (%)", scale=alt.Scale(domain=[0, 100])),
                            y=alt.Y("symbol:N", sort="-x", title=""),
                            color=alt.value(ACCENT_2),
                            tooltip=["symbol:N", alt.Tooltip("win_rate_pct:Q", format=".1f"), alt.Tooltip("trades:Q", title="Trades")],
                        )
                        .properties(title="Win Rate by Symbol")
                    )
                    st.altair_chart(chart, use_container_width=True)

            st.markdown("#### Summary by Symbol (Realized)")
            if realized.empty:
                st.info("No realized trades yet for summary.")
            else:
                sym = (
                    realized.groupby("symbol", as_index=False)
                    .agg(
                        trades=("id", "count"),
                        wins=("win", "sum"),
                        shares=("shares", "sum"),
                        total_pnl=("pnl", "sum"),
                        avg_ret_pct=("ret_pct", "mean"),
                        med_ret_pct=("ret_pct", "median"),
                        avg_hold_days=("hold_days", "mean"),
                        fees_total=("fees_total", "sum"),
                    )
                    .sort_values(by=["total_pnl"], ascending=False)
                )
                sym["win_rate(%)"] = (sym["wins"] / sym["trades"] * 100).round(1)
                sym.rename(
                    columns={
                        "total_pnl": "total_pnl($)",
                        "avg_ret_pct": "avg_return(%)",
                        "med_ret_pct": "median_return(%)",
                        "avg_hold_days": "avg_hold(days)",
                        "fees_total": "fees_total($)",
                    },
                    inplace=True,
                )
                show = ["symbol","trades","wins","win_rate(%)","shares","total_pnl($)","avg_return(%)","median_return(%)","avg_hold(days)","fees_total($)"]
                st.dataframe(sym[show], use_container_width=True, height=360)

        # -------- Trades table
        with tab4:
            show_cols = [
                TID_COL,"entry_date","symbol","side","shares","entry_total","exit_date","exit_price","fees_total",
                "pnl","ret_pct","hold_days","company","strategy","notes","created_at"
            ]
            show_cols = [c for c in show_cols if c in df_f.columns]
            table = df_f[show_cols].sort_values(TID_COL, ascending=False).copy()

            money_cols = ["entry_total","exit_price","fees_total","pnl"]
            for c in money_cols:
                if c in table.columns:
                    table[c] = table[c].map(lambda x: f"${x:,.2f}")
            if "ret_pct" in table.columns:
                table["ret_pct"] = table["ret_pct"].map(lambda x: f"{x:.2f}%")
            if "shares" in table.columns:
                table["shares"] = table["shares"].map(lambda x: f"{x:,.4f}")
            if "hold_days" in table.columns:
                table["hold_days"] = table["hold_days"].map(lambda x: "" if pd.isna(x) else int(x))

            st.dataframe(table, use_container_width=True, height=520)

# ============================================================
# ADD / MANAGE TRADES (with editor)
# ============================================================
else:
    st.markdown("## Add Trade")

    with st.form("add_trade_form", enter_to_submit=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            symbol = st.text_input("Symbol *", value="AMZN").upper().strip()
            entry_total = st.number_input("Entry Total ($) *", min_value=0.0, value=0.00, step=1.0, format="%.2f")
            stop_price = st.number_input("Stop Price (optional)", min_value=0.0, value=0.0, step=1.0, format="%.4f")
            use_exit_date = st.checkbox("Set Exit Date", value=False)
            exit_date_dt = st.date_input("Exit Date (optional)", value=date.today(), format="YYYY/MM/DD", disabled=not use_exit_date)
        with c2:
            side = st.selectbox("Side *", options=["long", "short"], index=0)
            entry_date_dt = st.date_input("Entry Date *", value=date.today(), format="YYYY/MM/DD")
            target1 = st.number_input("Target 1 (optional)", min_value=0.0, value=0.0, step=1.0, format="%.4f")
            exit_price = st.number_input("Exit Price (optional)", min_value=0.0, value=0.0, step=1.0, format="%.4f")
        with c3:
            shares = st.number_input("Shares (fractional ok) *", min_value=0.0, value=1.0, step=1.0, format="%.6f")
            company = st.text_input("Company", value="")
            target2 = st.number_input("Target 2 (optional)", min_value=0.0, value=0.0, step=1.0, format="%.4f")
            fees_total = st.number_input("Fees Total ($)", min_value=0.0, value=0.0, step=1.0, format="%.2f")
        strategy = st.text_input("Strategy / Tag (optional)", value="")
        notes = st.text_area("Notes", height=120)

        if st.form_submit_button("Save Trade"):
            if not symbol:
                st.error("Symbol is required.")
            else:
                df = load_trades()
                record = {
                    "id": str(uuid.uuid4()),
                    TID_COL: next_tid(df),
                    "symbol": symbol,
                    "side": side,
                    "shares": float(shares),
                    "entry_date": entry_date_dt.strftime("%Y/%m/%d"),
                    "company": company,
                    "entry_total": float(entry_total),
                    "stop_price": float(stop_price),
                    "target1": float(target1),
                    "target2": float(target2),
                    "exit_date": (exit_date_dt.strftime("%Y/%m/%d") if use_exit_date else ""),
                    "exit_price": float(exit_price),
                    "fees_total": float(fees_total),
                    "strategy": strategy,
                    "notes": notes,
                    "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                }
                df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
                save_trades(df)
                st.success(f"Trade saved as #{record[TID_COL]}.")

    st.divider()
    st.markdown("## Manage Trades")

    df_all = load_trades()
    if df_all.empty:
        st.info("No trades saved yet.")
    else:
        df_show = df_all[[TID_COL,"id","symbol","side","shares","entry_date","entry_total","exit_date","exit_price","fees_total","strategy","notes","created_at"]].sort_values(TID_COL, ascending=False)
        st.dataframe(df_show, use_container_width=True, height=280)

        # Delete Selected
        st.subheader("Delete Selected")
        def label_row(r): return f"#{int(r[TID_COL])} · {r['symbol']} · {r['side']} · {r['shares']} · {r['entry_date']}"
        labels = df_all.apply(label_row, axis=1).tolist()
        tid_map = dict(zip(df_all[TID_COL].astype(int), labels))
        label_to_tid = {v:k for k,v in tid_map.items()}

        sel_labels = st.multiselect("Choose trade(s) to delete:", options=labels, placeholder="Select …")
        cdel1, cdel2 = st.columns([1,1])
        with cdel1:
            if st.button("Delete Selected"):
                if not sel_labels:
                    st.warning("No trades selected.")
                else:
                    tids = {label_to_tid[l] for l in sel_labels}
                    df_new = df_all[~df_all[TID_COL].isin(list(tids))].copy()
                    save_trades(df_new)
                    st.success(f"Deleted {len(tids)} trade(s).")
                    st.rerun()
        with cdel2:
            with st.popover("Danger Zone: Delete ALL Trades"):
                st.write("Type **DELETE** to purge every trade.")
                confirm = st.text_input("Confirmation")
                if st.button("Delete ALL Trades", type="primary"):
                    if confirm.strip() == "DELETE":
                        save_trades(pd.DataFrame(columns=TRADE_COLUMNS))
                        st.success("All trades deleted.")
                        st.rerun()
                    else:
                        st.error("Confirmation did not match.")

        # Edit / Close
        st.subheader("Edit / Close a Trade")
        labels_sorted = [tid_map[i] for i in sorted(tid_map.keys(), reverse=True)]
        pick = st.selectbox("Pick trade:", options=labels_sorted, index=0)
        edit_tid = label_to_tid[pick]
        row = df_all[df_all[TID_COL] == edit_tid].iloc[0]

        with st.form("edit_trade_form", enter_to_submit=False):
            e1, e2, e3 = st.columns(3)
            with e1:
                e_symbol = st.text_input("Symbol", value=str(row["symbol"])).upper().strip()
                e_entry_total = st.number_input("Entry Total ($)", min_value=0.0, value=float(row["entry_total"]), step=1.0, format="%.2f")
                e_stop_price = st.number_input("Stop Price", min_value=0.0, value=float(row["stop_price"]), step=1.0, format="%.4f")
                e_entry_date = st.date_input("Entry Date", value=pd.to_datetime(row["entry_date"], errors="coerce").date() if row["entry_date"] else date.today(), format="YYYY/MM/DD")
            with e2:
                e_side = st.selectbox("Side", options=["long","short"], index=0 if str(row["side"]).lower()=="long" else 1)
                e_target1 = st.number_input("Target 1", min_value=0.0, value=float(row["target1"]), step=1.0, format="%.4f")
                e_exit_price = st.number_input("Exit Price", min_value=0.0, value=float(row["exit_price"]), step=1.0, format="%.4f")
                e_exit_date = st.date_input("Exit Date", value=pd.to_datetime(row["exit_date"], errors="coerce").date() if row["exit_date"] else date.today(), format="YYYY/MM/DD", disabled=(float(row["exit_price"])==0.0 and str(row["exit_date"]).strip()==""))
            with e3:
                e_shares = st.number_input("Shares", min_value=0.0, value=float(row["shares"]), step=1.0, format="%.6f")
                e_company = st.text_input("Company", value=str(row["company"]))
                e_target2 = st.number_input("Target 2", min_value=0.0, value=float(row["target2"]), step=1.0, format="%.4f")
                e_fees_total = st.number_input("Fees Total ($)", min_value=0.0, value=float(row["fees_total"]), step=1.0, format="%.2f")
            e_strategy = st.text_input("Strategy / Tag", value=str(row["strategy"]))
            e_notes = st.text_area("Notes", value=str(row["notes"]), height=100)

            bsave, bclose, breopen, bdel = st.columns([1,1,1,1])
            if bsave.form_submit_button("Save Changes"):
                df_all.loc[df_all[TID_COL] == edit_tid, :] = [
                    row["id"], edit_tid, e_symbol, e_side, float(e_shares),
                    e_entry_date.strftime("%Y/%m/%d"), e_company, float(e_entry_total),
                    float(e_stop_price), float(e_target1), float(e_target2),
                    e_exit_date.strftime("%Y/%m/%d") if (str(row["exit_date"]).strip() or float(e_exit_price) > 0) else "",
                    float(e_exit_price), float(e_fees_total), e_strategy, e_notes, row["created_at"]
                ]
                save_trades(df_all); st.success(f"Trade #{edit_tid} updated."); st.rerun()
            if bclose.form_submit_button("Close Trade"):
                if float(e_exit_price) <= 0:
                    st.error("Set an Exit Price to close this trade.")
                else:
                    df_all.loc[df_all[TID_COL] == edit_tid, ["exit_price","exit_date","fees_total"]] = [
                        float(e_exit_price), date.today().strftime("%Y/%m/%d"), float(e_fees_total)
                    ]
                    save_trades(df_all); st.success(f"Trade #{edit_tid} closed."); st.rerun()
            if breopen.form_submit_button("Reopen Trade"):
                df_all.loc[df_all[TID_COL] == edit_tid, ["exit_price","exit_date"]] = [0.0, ""]
                save_trades(df_all); st.success(f"Trade #{edit_tid} reopened."); st.rerun()
            if bdel.form_submit_button("Delete This Trade"):
                save_trades(df_all[df_all[TID_COL] != edit_tid].copy()); st.success(f"Trade #{edit_tid} deleted."); st.rerun()
