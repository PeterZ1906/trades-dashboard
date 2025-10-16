# app.py
# ============================================================
# Trades Dashboard App — with simple numeric Trade IDs (tid)
# - Sidebar: Dashboard | Add / Manage Trades
# - CSV persistence: trades.csv
# - Integer-step spinners in forms
# - Delete selected / delete-all
# - Edit / Close / Reopen trade
# - Pretty analytics on Dashboard
# - NEW: simple numeric ID (#tid) used in selectors
# ============================================================
import os
import uuid
from datetime import date, datetime

import pandas as pd
import streamlit as st
import altair as alt

st.set_page_config(page_title="Trades Dashboard", layout="wide")

# -----------------------------
# Theme (Altair) & colors
# -----------------------------
ACCENT_GREEN = "#37d67a"
ACCENT_RED = "#f47373"
ACCENT_BLUE = "#7aa5ff"

def _altair_dark_theme():
    return {
        "config": {
            "view": {"stroke": "transparent"},
            "axis": {
                "domainColor": "#666",
                "gridColor": "#222",
                "labelColor": "#ddd",
                "titleColor": "#eee",
                "grid": True,
            },
            "legend": {"labelColor": "#ddd", "titleColor": "#eee"},
            "title": {"color": "#fff"},
            "background": "transparent",
        }
    }

alt.themes.register("custom_dark", _altair_dark_theme)
alt.themes.enable("custom_dark")

# -----------------------------
# Storage / data helpers
# -----------------------------
TRADES_CSV = "trades.csv"
# simple numeric trade id column name
TID_COL = "tid"

TRADE_COLUMNS = [
    "id",          # uuid
    TID_COL,       # simple integer id
    "symbol",
    "side",
    "shares",
    "entry_date",
    "company",
    "entry_total",
    "stop_price",
    "target1",
    "target2",
    "exit_date",
    "exit_price",
    "fees_total",
    "strategy",
    "notes",
    "created_at",
]

NUMERIC_COLS = {"shares", "entry_total", "stop_price", "target1", "target2", "exit_price", "fees_total"}

def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure required columns exist; add empty defaults if missing."""
    for c in TRADE_COLUMNS:
        if c not in df.columns:
            if c in NUMERIC_COLS:
                df[c] = 0.0
            elif c == TID_COL:
                df[c] = pd.Series(dtype="Int64")
            else:
                df[c] = ""
    # enforce order
    return df[TRADE_COLUMNS]

def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # TID as nullable int
    if TID_COL in df.columns:
        df[TID_COL] = pd.to_numeric(df[TID_COL], errors="coerce").astype("Int64")
    # clean dates
    for dcol in ["entry_date", "exit_date"]:
        if dcol in df.columns:
            df[dcol] = df[dcol].fillna("")
    return df

def _assign_missing_tids_inplace(df: pd.DataFrame) -> None:
    """Give any rows without a tid a new sequential one."""
    if df.empty:
        return
    # current max tid (ignore NA)
    max_tid = int(df[TID_COL].dropna().max()) if df[TID_COL].notna().any() else 0
    # rows that need a tid
    needs = df[TID_COL].isna()
    count = int(needs.sum())
    if count > 0:
        new_ids = range(max_tid + 1, max_tid + 1 + count)
        df.loc[needs, TID_COL] = list(new_ids)

def load_trades() -> pd.DataFrame:
    if not os.path.exists(TRADES_CSV):
        return pd.DataFrame(columns=TRADE_COLUMNS)

    df = pd.read_csv(TRADES_CSV, dtype=str)
    # add/migrate uuid
    if "id" not in df.columns:
        df["id"] = [str(uuid.uuid4()) for _ in range(len(df))]
    df = _ensure_columns(df)
    df = _coerce_types(df)
    # assign tids where missing
    _assign_missing_tids_inplace(df)
    return df

def save_trades(df: pd.DataFrame) -> None:
    df = _ensure_columns(df.copy())
    df = _coerce_types(df)
    df.to_csv(TRADES_CSV, index=False)

def next_tid(df: pd.DataFrame) -> int:
    if df.empty or df[TID_COL].isna().all():
        return 1
    return int(df[TID_COL].dropna().max()) + 1

# ---------- Derived metrics helpers ----------
def add_trade_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()
    df["entry_dt"] = pd.to_datetime(df["entry_date"], errors="coerce")
    df["exit_dt"] = pd.to_datetime(df["exit_date"], errors="coerce")
    df["entry_price"] = (df["entry_total"] / df["shares"]).where(df["shares"] > 0, pd.NA)

    df["realized"] = (~df["exit_dt"].isna()) & (df["exit_price"] > 0)

    long_mask = (df["side"].str.lower() == "long") & df["realized"]
    short_mask = (df["side"].str.lower() == "short") & df["realized"]

    df["pnl"] = 0.0
    df.loc[long_mask, "pnl"] = (df.loc[long_mask, "exit_price"] - df.loc[long_mask, "entry_price"]) * df.loc[long_mask, "shares"]
    df.loc[short_mask, "pnl"] = (df.loc[short_mask, "entry_price"] - df.loc[short_mask, "exit_price"]) * df.loc[short_mask, "shares"]
    df.loc[df["realized"], "pnl"] = df.loc[df["realized"], "pnl"] - df.loc[df["realized"], "fees_total"]

    df["ret_pct"] = 0.0
    df.loc[df["realized"] & (df["entry_total"] > 0), "ret_pct"] = (
        df.loc[df["realized"], "pnl"] / df.loc[df["realized"], "entry_total"] * 100.0
    )

    df["hold_days"] = (df["exit_dt"] - df["entry_dt"]).dt.days.where(df["realized"], pd.NA)
    df["win"] = (df["pnl"] > 0).where(df["realized"], False)

    df["open"] = ~df["realized"]
    df["open_cost"] = df["entry_total"].where(df["open"], 0.0)
    return df

def kpi(value, label, delta=None, fmt=None, help_text=None):
    if fmt:
        value = fmt(value)
    st.metric(label, value, delta, help=help_text)

# -----------------------------
# UI: Sidebar navigation
# -----------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    options=("Dashboard", "Add / Manage Trades"),
    index=0,
)

# ============================================================
# Page: Dashboard
# ============================================================
if page == "Dashboard":
    st.title("Portfolio Summary")

    raw = load_trades()
    df = add_trade_metrics(raw)

    if df.empty:
        st.info("No trades yet. Add some on the **Add / Manage Trades** page.")
    else:
        with st.expander("Filters", expanded=False):
            valid_entries = df[~df["entry_dt"].isna()]
            if not valid_entries.empty:
                dmin = valid_entries["entry_dt"].min().date()
                dmax = valid_entries["entry_dt"].max().date()
                date_range = st.date_input("Entry Date range", value=(dmin, dmax), format="YYYY/MM/DD")
            else:
                date_range = None
            symbols = sorted([s for s in df["symbol"].dropna().unique() if s != ""])
            sel_symbols = st.multiselect("Symbols", options=symbols, default=symbols)

        df_f = df.copy()
        if date_range and isinstance(date_range, tuple) and len(date_range) == 2:
            start, end = date_range
            mask = df_f["entry_dt"].between(pd.to_datetime(start), pd.to_datetime(end))
            df_f = df_f[mask]
        if sel_symbols:
            df_f = df_f[df_f["symbol"].isin(sel_symbols)]

        realized = df_f[df_f["realized"]].copy()
        open_pos = df_f[df_f["open"]].copy()

        total_invested = float(df_f["entry_total"].sum())
        realized_pnl = float(realized["pnl"].sum()) if not realized.empty else 0.0
        open_cost = float(open_pos["open_cost"].sum()) if not open_pos.empty else 0.0

        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            portfolio_value_proxy = total_invested + realized_pnl
            kpi(portfolio_value_proxy, "Portfolio Value*", fmt=lambda x: f"${x:,.2f}",
                help_text="Proxy: Invested cash + Realized P&L (no live prices).")
        with c2:
            kpi(open_cost, "Open Cost Basis", fmt=lambda x: f"${x:,.2f}")
        with c3:
            kpi(realized_pnl, "Realized P&L ($)", fmt=lambda x: f"${x:,.2f}")
        with c4:
            win_rate = (realized["win"].mean() * 100.0) if not realized.empty else 0.0
            kpi(win_rate, "Win Rate", fmt=lambda x: f"{x:.1f}%")
        with c5:
            avg_ret = realized["ret_pct"].mean() if not realized.empty else 0.0
            kpi(avg_ret, "Avg Return / Trade", fmt=lambda x: f"{x:.2f}%")

        st.caption("*) Without live prices, this is a proxy. Add quotes later for true MV & unrealized P&L.")

        st.subheader("Performance")
        colA, colB = st.columns([2, 1])
        if realized.empty:
            with colA:
                st.info("No realized trades yet to plot the Equity Curve.")
            with colB:
                st.info("No realized trades for monthly P&L.")
        else:
            ec = realized.sort_values("exit_dt")[["exit_dt", "pnl"]].dropna()
            ec = ec.groupby("exit_dt", as_index=False).agg(pnl=("pnl", "sum"))
            ec["cum_pnl"] = ec["pnl"].cumsum()
            line = (
                alt.Chart(ec, height=320)
                .mark_line(point=True)
                .encode(
                    x=alt.X("exit_dt:T", title="Exit Date"),
                    y=alt.Y("cum_pnl:Q", title="Cumulative P&L ($)"),
                    tooltip=["exit_dt:T", alt.Tooltip("cum_pnl:Q", format=",.2f")],
                )
                .properties(title="Equity Curve (Realized P&L)")
            )
            colA.altair_chart(line, use_container_width=True)

            rm = realized.dropna(subset=["exit_dt"]).copy()
            rm["month"] = rm["exit_dt"].dt.to_period("M").dt.to_timestamp()
            month_pnl = rm.groupby("month", as_index=False).agg(total_pnl=("pnl", "sum"))
            month_pnl["color"] = month_pnl["total_pnl"].apply(lambda v: ACCENT_GREEN if v >= 0 else ACCENT_RED)
            bar = (
                alt.Chart(month_pnl, height=320)
                .mark_bar()
                .encode(
                    x=alt.X("month:T", title="Month"),
                    y=alt.Y("total_pnl:Q", title="P&L ($)"),
                    color=alt.Color("color:N", scale=None, legend=None),
                    tooltip=["month:T", alt.Tooltip("total_pnl:Q", format=",.2f")],
                )
                .properties(title="P&L by Month")
            )
            colB.altair_chart(bar, use_container_width=True)

        st.subheader("Breakdown")
        col1, col2, col3 = st.columns(3)
        if realized.empty:
            col1.info("No realized trades to show P&L by symbol.")
            col2.info("No realized trades to show Win Rate.")
        else:
            pnl_sym = (
                realized.groupby("symbol", as_index=False)
                .agg(total_pnl=("pnl", "sum"))
                .sort_values("total_pnl", ascending=False)
            )
            pnl_sym["color"] = pnl_sym["total_pnl"].apply(lambda v: ACCENT_GREEN if v >= 0 else ACCENT_RED)
            bar1 = (
                alt.Chart(pnl_sym, height=300)
                .mark_bar()
                .encode(
                    x=alt.X("total_pnl:Q", title="Total P&L ($)"),
                    y=alt.Y("symbol:N", sort="-x", title="Symbol"),
                    color=alt.Color("color:N", scale=None, legend=None),
                    tooltip=[alt.Tooltip("total_pnl:Q", format=",.2f"), "symbol:N"],
                )
                .properties(title="Total Realized P&L by Symbol")
            )
            col1.altair_chart(bar1, use_container_width=True)

            wr = (
                realized.groupby("symbol", as_index=False)
                .agg(win_rate=("win", "mean"), trades=("id", "count"))
                .sort_values("win_rate", ascending=False)
            )
            wr["win_rate_pct"] = wr["win_rate"] * 100.0
            bar2 = (
                alt.Chart(wr, height=300)
                .mark_bar()
                .encode(
                    x=alt.X("win_rate_pct:Q", title="Win Rate (%)"),
                    y=alt.Y("symbol:N", sort="-x", title="Symbol"),
                    tooltip=[alt.Tooltip("win_rate_pct:Q", format=".1f"), alt.Tooltip("trades:Q", title="Trades"), "symbol:N"],
                )
                .properties(title="Win Rate by Symbol")
            )
            col2.altair_chart(bar2, use_container_width=True)

        if df_f[df_f["open"]].empty:
            col3.info("No open positions to show diversification.")
        else:
            dv = (
                df_f[df_f["open"]].groupby("symbol", as_index=False)
                .agg(open_cost=("open_cost", "sum"))
                .sort_values("open_cost", ascending=False)
            )
            pie = (
                alt.Chart(dv, height=300)
                .mark_arc(outerRadius=130)
                .encode(
                    theta=alt.Theta("open_cost:Q", stack=True, title=""),
                    color=alt.Color("symbol:N", title="Symbol"),
                    tooltip=[alt.Tooltip("open_cost:Q", format=",.2f"), "symbol:N"],
                )
                .properties(title="Open Cost Diversification")
            )
            col3.altair_chart(pie, use_container_width=True)

        st.subheader("All Trades (filtered)")
        show_cols = [
            TID_COL, "id", "entry_date", "symbol", "side", "shares", "entry_total", "stop_price",
            "target1", "target2", "exit_date", "exit_price", "fees_total", "company",
            "strategy", "notes", "pnl", "ret_pct", "hold_days", "created_at"
        ]
        show_cols = [c for c in show_cols if c in df_f.columns]
        st.dataframe(
            df_f[show_cols].sort_values(by=[TID_COL], ascending=False),
            use_container_width=True,
            height=420,
        )

# ============================================================
# Page: Add / Manage Trades (+ Edit/Close)
# ============================================================
else:
    st.title("Add Trade")

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
                st.json(record)

    st.divider()
    st.header("Manage Trades")

    df_all = load_trades()
    if df_all.empty:
        st.info("No trades saved yet.")
    else:
        df_show = df_all[
            [
                TID_COL, "id", "symbol", "side", "shares", "entry_date", "company", "entry_total", "stop_price",
                "target1", "target2", "exit_date", "exit_price", "fees_total", "strategy", "notes", "created_at"
            ]
        ].sort_values(by=[TID_COL], ascending=False)

        st.dataframe(df_show, use_container_width=True, height=300)

        # ---------- Delete Selected (multi) ----------
        st.subheader("Delete Selected")

        def make_label(row):
            return f"#{int(row[TID_COL])} · {row['symbol']} · {row['side']} · {row['shares']} · {row['entry_date']}"

        labels = df_all.apply(make_label, axis=1).tolist()
        tid_to_label = dict(zip(df_all[TID_COL].astype(int), labels))
        label_to_tid = {v: k for k, v in tid_to_label.items()}

        selected_labels = st.multiselect(
            "Choose one or more trades to delete:",
            options=labels,
            placeholder="Select trade(s)…",
        )

        c_del1, c_del2 = st.columns([1, 1])
        with c_del1:
            if st.button("Delete Selected"):
                if not selected_labels:
                    st.warning("No trades selected.")
                else:
                    tids = {label_to_tid[lbl] for lbl in selected_labels}
                    df_new = df_all[~df_all[TID_COL].isin(list(tids))].copy()
                    save_trades(df_new)
                    st.success(f"Deleted {len(tids)} trade(s).")
                    st.rerun()

        with c_del2:
            with st.popover("Danger Zone: Delete ALL Trades"):
                st.write("Type **DELETE** (all caps) to purge every trade.")
                confirm_text = st.text_input("Confirmation")
                if st.button("Delete ALL Trades", type="primary"):
                    if confirm_text.strip() == "DELETE":
                        save_trades(pd.DataFrame(columns=TRADE_COLUMNS))
                        st.success("All trades deleted.")
                        st.rerun()
                    else:
                        st.error("Confirmation text did not match 'DELETE'. No action taken.")

        # ---------- Edit / Close a Single Trade ----------
        st.subheader("Edit / Close a Trade")

        # chooser sorted by newest first
        labels_sorted = [tid_to_label[i] for i in sorted(tid_to_label.keys(), reverse=True)]
        edit_label = st.selectbox("Choose a trade to edit:", options=labels_sorted, index=0)
        edit_tid = label_to_tid[edit_label]
        row = df_all[df_all[TID_COL] == edit_tid].iloc[0]

        with st.form("edit_trade_form", enter_to_submit=False):
            ec1, ec2, ec3 = st.columns(3)

            with ec1:
                e_symbol = st.text_input("Symbol", value=str(row["symbol"])).upper().strip()
                e_entry_total = st.number_input("Entry Total ($)", min_value=0.0, value=float(row["entry_total"]), step=1.0, format="%.2f")
                e_stop_price = st.number_input("Stop Price", min_value=0.0, value=float(row["stop_price"]), step=1.0, format="%.4f")
                e_entry_date = st.date_input(
                    "Entry Date",
                    value=pd.to_datetime(row["entry_date"], errors="coerce").date() if row["entry_date"] else date.today(),
                    format="YYYY/MM/DD"
                )

            with ec2:
                e_side = st.selectbox("Side", options=["long", "short"], index=0 if str(row["side"]).lower()=="long" else 1)
                e_target1 = st.number_input("Target 1", min_value=0.0, value=float(row["target1"]), step=1.0, format="%.4f")
                e_exit_price = st.number_input("Exit Price", min_value=0.0, value=float(row["exit_price"]), step=1.0, format="%.4f")
                e_exit_date = st.date_input(
                    "Exit Date",
                    value=pd.to_datetime(row["exit_date"], errors="coerce").date() if row["exit_date"] else date.today(),
                    format="YYYY/MM/DD",
                    disabled=(float(row["exit_price"])==0.0 and str(row["exit_date"]).strip()=="")
                )

            with ec3:
                e_shares = st.number_input("Shares", min_value=0.0, value=float(row["shares"]), step=1.0, format="%.6f")
                e_company = st.text_input("Company", value=str(row["company"]))
                e_target2 = st.number_input("Target 2", min_value=0.0, value=float(row["target2"]), step=1.0, format="%.4f")
                e_fees_total = st.number_input("Fees Total ($)", min_value=0.0, value=float(row["fees_total"]), step=1.0, format="%.2f")

            e_strategy = st.text_input("Strategy / Tag", value=str(row["strategy"]))
            e_notes = st.text_area("Notes", value=str(row["notes"]), height=100)

            c_save, c_close, c_reopen, c_delete_one = st.columns([1,1,1,1])

            # Save changes
            if c_save.form_submit_button("Save Changes"):
                df_all.loc[df_all[TID_COL] == edit_tid, :] = [
                    row["id"],
                    edit_tid,
                    e_symbol,
                    e_side,
                    float(e_shares),
                    e_entry_date.strftime("%Y/%m/%d"),
                    e_company,
                    float(e_entry_total),
                    float(e_stop_price),
                    float(e_target1),
                    float(e_target2),
                    e_exit_date.strftime("%Y/%m/%d") if (str(row["exit_date"]).strip() or float(e_exit_price) > 0) else "",
                    float(e_exit_price),
                    float(e_fees_total),
                    e_strategy,
                    e_notes,
                    row["created_at"],
                ]
                save_trades(df_all)
                st.success(f"Trade #{edit_tid} updated.")
                st.rerun()

            # Quick close (sets today's exit date unless changed above)
            with c_close:
                if st.form_submit_button("Close Trade"):
                    if float(e_exit_price) <= 0:
                        st.error("Set an Exit Price to close this trade.")
                    else:
                        df_all.loc[df_all[TID_COL] == edit_tid, "exit_price"] = float(e_exit_price)
                        df_all.loc[df_all[TID_COL] == edit_tid, "exit_date"] = date.today().strftime("%Y/%m/%d")
                        df_all.loc[df_all[TID_COL] == edit_tid, "fees_total"] = float(e_fees_total)
                        save_trades(df_all)
                        st.success(f"Trade #{edit_tid} closed.")
                        st.rerun()

            # Reopen (clear exit price/date)
            with c_reopen:
                if st.form_submit_button("Reopen Trade"):
                    df_all.loc[df_all[TID_COL] == edit_tid, "exit_price"] = 0.0
                    df_all.loc[df_all[TID_COL] == edit_tid, "exit_date"] = ""
                    save_trades(df_all)
                    st.success(f"Trade #{edit_tid} reopened.")
                    st.rerun()

            # Delete this one
            with c_delete_one:
                if st.form_submit_button("Delete This Trade"):
                    df_new = df_all[df_all[TID_COL] != edit_tid].copy()
                    save_trades(df_new)
                    st.success(f"Trade #{edit_tid} deleted.")
                    st.rerun()
