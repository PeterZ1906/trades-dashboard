# app.py
# ============================================================
# Trades Dashboard App (Enhanced)
# - Sidebar nav: Dashboard  |  Add / Manage Trades
# - CSV persistence at repo root: trades.csv
# - Integer-step spinners (step=1) inside the form
# - Delete selected or delete-all with confirmation
# - Beautiful, useful analytics on the Dashboard
# ============================================================
import os
import uuid
from datetime import date, datetime

import pandas as pd
import streamlit as st
import altair as alt

st.set_page_config(page_title="Trades Dashboard", layout="wide")


# -----------------------------
# Storage / data helpers
# -----------------------------
TRADES_CSV = "trades.csv"
TRADE_COLUMNS = [
    "id",
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


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure required columns exist; add empty defaults if missing."""
    for c in TRADE_COLUMNS:
        if c not in df.columns:
            if c in {"shares", "entry_total", "stop_price", "target1", "target2", "exit_price", "fees_total"}:
                df[c] = 0.0
            else:
                df[c] = ""
    return df[TRADE_COLUMNS]


def load_trades() -> pd.DataFrame:
    if not os.path.exists(TRADES_CSV):
        return pd.DataFrame(columns=TRADE_COLUMNS)

    df = pd.read_csv(TRADES_CSV, dtype=str)

    # Migration: ensure there's an ID
    if "id" not in df.columns:
        df["id"] = [str(uuid.uuid4()) for _ in range(len(df))]

    # Types
    for col in ["shares", "entry_total", "stop_price", "target1", "target2", "exit_price", "fees_total"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Normalize dates
    for dcol in ["entry_date", "exit_date"]:
        if dcol in df.columns:
            df[dcol] = df[dcol].fillna("")

    return _ensure_columns(df)


def save_trades(df: pd.DataFrame) -> None:
    df = _ensure_columns(df.copy())
    df.to_csv(TRADES_CSV, index=False)


# ---------- Derived metrics helpers ----------
def add_trade_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns for analytics."""
    if df.empty:
        return df

    df = df.copy()

    # Parse dates
    df["entry_dt"] = pd.to_datetime(df["entry_date"], errors="coerce")
    df["exit_dt"] = pd.to_datetime(df["exit_date"], errors="coerce")

    # Entry price per share (guard div-by-zero)
    df["entry_price"] = (df["entry_total"] / df["shares"]).where(df["shares"] > 0, pd.NA)

    # Realized trades only (have exit price >0 and exit date)
    df["realized"] = (~df["exit_dt"].isna()) & (df["exit_price"] > 0)

    # Per-trade P&L ($) and Return (%)
    long_mask = (df["side"].str.lower() == "long") & df["realized"]
    short_mask = (df["side"].str.lower() == "short") & df["realized"]

    df["pnl"] = 0.0
    df.loc[long_mask, "pnl"] = (df.loc[long_mask, "exit_price"] - df.loc[long_mask, "entry_price"]) * df.loc[long_mask, "shares"]
    df.loc[short_mask, "pnl"] = (df.loc[short_mask, "entry_price"] - df.loc[short_mask, "exit_price"]) * df.loc[short_mask, "shares"]

    # fees_total is total per trade (subtract once)
    df.loc[df["realized"], "pnl"] = df.loc[df["realized"], "pnl"] - df.loc[df["realized"], "fees_total"]

    # % return based on cost basis (entry_total); keep safe
    df["ret_pct"] = 0.0
    df.loc[df["realized"] & (df["entry_total"] > 0), "ret_pct"] = (
        df.loc[df["realized"], "pnl"] / df.loc[df["realized"], "entry_total"] * 100.0
    )

    # Holding period (days) for realized trades
    df["hold_days"] = (df["exit_dt"] - df["entry_dt"]).dt.days.where(df["realized"], pd.NA)

    # Win/Loss
    df["win"] = (df["pnl"] > 0).where(df["realized"], False)

    return df


def kpi(value, label, help_text=None, fmt=None):
    """Small helper for clean KPI blocks."""
    if fmt:
        value = fmt(value)
    st.metric(label, value, help=help_text)


# -----------------------------
# UI: Sidebar navigation
# -----------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    options=("Dashboard", "Add / Manage Trades"),
    index=0,
)


# -----------------------------
# Page: Dashboard
# -----------------------------
if page == "Dashboard":
    st.title("Dashboard")

    raw = load_trades()
    df = add_trade_metrics(raw)

    if df.empty:
        st.info("No trades yet. Add some on the **Add / Manage Trades** page.")
    else:
        # --- Filters
        with st.expander("Filters", expanded=False):
            # date range filter (based on entry_date)
            valid_entries = df[~df["entry_dt"].isna()]
            if not valid_entries.empty:
                dmin = valid_entries["entry_dt"].min().date()
                dmax = valid_entries["entry_dt"].max().date()
                date_range = st.date_input(
                    "Entry Date range",
                    value=(dmin, dmax),
                    format="YYYY/MM/DD",
                )
            else:
                date_range = None

            symbols = sorted([s for s in df["symbol"].dropna().unique() if s != ""])
            sel_symbols = st.multiselect("Symbols", options=symbols, default=symbols)

        df_f = df.copy()
        # Apply filters
        if date_range and isinstance(date_range, tuple) and len(date_range) == 2:
            start, end = date_range
            mask = df_f["entry_dt"].between(pd.to_datetime(start), pd.to_datetime(end))
            df_f = df_f[mask]
        if sel_symbols:
            df_f = df_f[df_f["symbol"].isin(sel_symbols)]

        # Derived subsets
        realized = df_f[df_f["realized"]].copy()

        # --- KPIs row
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            kpi(len(df_f), "Total Trades")
        with c2:
            kpi(df_f["shares"].sum(), "Total Shares", fmt=lambda x: f"{x:,.4f}")
        with c3:
            total_fees = realized["fees_total"].sum() if not realized.empty else 0.0
            kpi(total_fees, "Total Fees ($)", fmt=lambda x: f"{x:,.2f}")
        with c4:
            win_rate = (realized["win"].mean() * 100.0) if not realized.empty else 0.0
            kpi(win_rate, "Win Rate", fmt=lambda x: f"{x:.1f}%")
        with c5:
            avg_ret = realized["ret_pct"].mean() if not realized.empty else 0.0
            kpi(avg_ret, "Avg Return / Trade", fmt=lambda x: f"{x:.2f}%")

        # --- Charts
        st.markdown("### Performance Overview")

        # Equity curve (cumulative P&L)
        if realized.empty:
            st.info("No realized trades yet to plot the Equity Curve.")
        else:
            ec = realized.sort_values("exit_dt")[["exit_dt", "pnl"]].dropna()
            ec = ec.groupby("exit_dt", as_index=False).agg(pnl=("pnl", "sum"))
            ec["cum_pnl"] = ec["pnl"].cumsum()

            line = (
                alt.Chart(ec, height=260)
                .mark_line(point=True)
                .encode(
                    x=alt.X("exit_dt:T", title="Exit Date"),
                    y=alt.Y("cum_pnl:Q", title="Cumulative P&L ($)"),
                    tooltip=["exit_dt:T", alt.Tooltip("cum_pnl:Q", format=",.2f")],
                )
                .properties(title="Equity Curve (Realized P&L)")
            )
            st.altair_chart(line, use_container_width=True)

        st.markdown("### Breakdown")

        col_a, col_b = st.columns(2)

        # P&L by Symbol (bar)
        with col_a:
            if realized.empty:
                st.info("No realized trades to show P&L by symbol.")
            else:
                pnl_sym = (
                    realized.groupby("symbol", as_index=False)
                    .agg(total_pnl=("pnl", "sum"))
                    .sort_values("total_pnl", ascending=False)
                )
                bar = (
                    alt.Chart(pnl_sym, height=300)
                    .mark_bar()
                    .encode(
                        x=alt.X("total_pnl:Q", title="Total P&L ($)"),
                        y=alt.Y("symbol:N", sort="-x", title="Symbol"),
                        tooltip=[alt.Tooltip("total_pnl:Q", format=",.2f"), "symbol:N"],
                    )
                    .properties(title="Total Realized P&L by Symbol")
                )
                st.altair_chart(bar, use_container_width=True)

        # Win rate by symbol (bar)
        with col_b:
            if realized.empty:
                st.info("No realized trades to show Win Rate.")
            else:
                wr = (
                    realized.groupby("symbol", as_index=False)
                    .agg(win_rate=("win", "mean"), trades=("id", "count"))
                    .sort_values("win_rate", ascending=False)
                )
                wr["win_rate_pct"] = wr["win_rate"] * 100.0
                bar_wr = (
                    alt.Chart(wr, height=300)
                    .mark_bar()
                    .encode(
                        x=alt.X("win_rate_pct:Q", title="Win Rate (%)"),
                        y=alt.Y("symbol:N", sort="-x", title="Symbol"),
                        tooltip=[
                            alt.Tooltip("win_rate_pct:Q", format=".1f"),
                            alt.Tooltip("trades:Q", title="Trades"),
                            "symbol:N",
                        ],
                    )
                    .properties(title="Win Rate by Symbol")
                )
                st.altair_chart(bar_wr, use_container_width=True)

        st.markdown("### Distributions")

        col_c, col_d = st.columns(2)

        # Return distribution (hist)
        with col_c:
            if realized.empty:
                st.info("No realized trades to show return distribution.")
            else:
                ret_hist = (
                    alt.Chart(realized.dropna(subset=["ret_pct"]), height=300)
                    .mark_bar()
                    .encode(
                        x=alt.X("ret_pct:Q", bin=alt.Bin(maxbins=30), title="Trade Return (%)"),
                        y=alt.Y("count()", title="Trades"),
                        tooltip=[alt.Tooltip("count()", title="Trades")],
                    )
                    .properties(title="Distribution of Trade Returns")
                )
                st.altair_chart(ret_hist, use_container_width=True)

        # Holding period distribution (hist)
        with col_d:
            if realized.empty:
                st.info("No realized trades to show holding periods.")
            else:
                hold_hist = (
                    alt.Chart(realized.dropna(subset=["hold_days"]), height=300)
                    .mark_bar()
                    .encode(
                        x=alt.X("hold_days:Q", bin=alt.Bin(maxbins=30), title="Holding Period (days)"),
                        y=alt.Y("count()", title="Trades"),
                        tooltip=[alt.Tooltip("count()", title="Trades")],
                    )
                    .properties(title="Distribution of Holding Periods")
                )
                st.altair_chart(hold_hist, use_container_width=True)

        st.markdown("### Summary by Symbol")
        if realized.empty:
            st.info("No realized trades yet for summary.")
        else:
            sym_summary = (
                realized.groupby("symbol", as_index=False)
                .agg(
                    trades=("id", "count"),
                    wins=("win", "sum"),
                    shares=("shares", "sum"),
                    total_pnl=("pnl", "sum"),
                    avg_ret_pct=("ret_pct", "mean"),
                    avg_hold_days=("hold_days", "mean"),
                    fees_total=("fees_total", "sum"),
                )
                .sort_values(by=["total_pnl"], ascending=False)
            )
            sym_summary["win_rate"] = (sym_summary["wins"] / sym_summary["trades"] * 100.0).round(1)
            sym_summary = sym_summary[
                ["symbol", "trades", "wins", "win_rate", "shares", "total_pnl", "avg_ret_pct", "avg_hold_days", "fees_total"]
            ]
            sym_summary.rename(
                columns={
                    "win_rate": "win_rate(%)",
                    "total_pnl": "total_pnl($)",
                    "avg_ret_pct": "avg_return(%)",
                    "avg_hold_days": "avg_hold(days)",
                    "fees_total": "fees_total($)",
                },
                inplace=True,
            )
            st.dataframe(sym_summary, use_container_width=True)

        st.markdown("### All Trades (filtered)")
        show_cols = [
            "id", "entry_date", "symbol", "side", "shares", "entry_total", "stop_price",
            "target1", "target2", "exit_date", "exit_price", "fees_total", "company",
            "strategy", "notes", "created_at"
        ]
        show_cols = [c for c in show_cols if c in df_f.columns]
        st.dataframe(
            df_f[show_cols].sort_values(by=["entry_date", "symbol"], ascending=[False, True]),
            use_container_width=True,
            height=380,
        )

# -----------------------------
# Page: Add / Manage Trades
# -----------------------------
else:
    st.title("Add Trade")

    with st.form("add_trade_form", enter_to_submit=False):
        c1, c2, c3 = st.columns(3)

        # ----- Column 1 -----
        with c1:
            symbol = st.text_input("Symbol *", value="AMZN").upper().strip()
            entry_total = st.number_input(
                "Entry Total ($) *", min_value=0.0, value=0.00, step=1.0, format="%.2f",
                help="Spinner moves by $1; type decimals if needed"
            )
            stop_price = st.number_input(
                "Stop Price (optional)", min_value=0.0, value=0.0, step=1.0, format="%.4f",
                help="Spinner moves by 1; type decimals if needed"
            )
            use_exit_date = st.checkbox("Set Exit Date", value=True)
            exit_date_dt = st.date_input(
                "Exit Date (optional)", value=date.today(), format="YYYY/MM/DD", disabled=not use_exit_date
            )

        # ----- Column 2 -----
        with c2:
            side = st.selectbox("Side *", options=["long", "short"], index=0)
            entry_date_dt = st.date_input("Entry Date *", value=date.today(), format="YYYY/MM/DD")
            target1 = st.number_input(
                "Target 1 (optional)", min_value=0.0, value=0.0, step=1.0, format="%.4f"
            )
            exit_price = st.number_input(
                "Exit Price (optional)", min_value=0.0, value=0.0, step=1.0, format="%.4f"
            )

        # ----- Column 3 -----
        with c3:
            shares = st.number_input(
                "Shares (fractional ok) *", min_value=0.0, value=1.0, step=1.0, format="%.6f",
                help="Spinner moves by 1; type fractional shares if needed"
            )
            company = st.text_input("Company", value="")
            target2 = st.number_input(
                "Target 2 (optional)", min_value=0.0, value=0.0, step=1.0, format="%.4f"
            )
            fees_total = st.number_input(
                "Fees Total ($)", min_value=0.0, value=0.0, step=1.0, format="%.2f"
            )

        strategy = st.text_input("Strategy / Tag (optional)", value="", help="e.g., 'breakout', 'earnings scalp', 'DCA'")
        notes = st.text_area("Notes", height=140)

        if st.form_submit_button("Save Trade"):
            if not symbol:
                st.error("Symbol is required.")
            else:
                record = {
                    "id": str(uuid.uuid4()),
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
                df = load_trades()
                df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
                save_trades(df)
                st.success("Trade saved.")
                st.json(record)

    st.divider()
    st.header("Manage Trades")

    df_all = load_trades()
    if df_all.empty:
        st.info("No trades saved yet.")
    else:
        df_show = df_all[
            [
                "id", "symbol", "side", "shares", "entry_date", "company", "entry_total", "stop_price",
                "target1", "target2", "exit_date", "exit_price", "fees_total", "strategy", "notes", "created_at"
            ]
        ].sort_values(by=["entry_date", "symbol"], ascending=[False, True])

        st.dataframe(df_show, use_container_width=True, height=340)

        # Build human labels for delete picker
        def make_label(row):
            ed = row.get("entry_date", "")
            sym = row.get("symbol", "")
            side_ = row.get("side", "")
            sh = row.get("shares", "")
            return f"{ed} • {sym} • {side_} • {sh}"

        labels = df_all.apply(make_label, axis=1).tolist()
        id_to_label = dict(zip(df_all["id"], labels))
        label_to_id = {v: k for k, v in id_to_label.items()}

        st.subheader("Delete Selected")
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
                    selected_ids = {label_to_id[lbl] for lbl in selected_labels}
                    df_new = df_all[~df_all["id"].isin(selected_ids)].copy()
                    save_trades(df_new)
                    st.success(f"Deleted {len(selected_ids)} trade(s).")
                    st.rerun()

        with c_del2:
            with st.popover("Danger Zone: Delete ALL Trades"):
                st.write("Type **DELETE** (all caps) and press the button to purge every trade.")
                confirm_text = st.text_input("Confirmation")
                if st.button("Delete ALL Trades", type="primary"):
                    if confirm_text.strip() == "DELETE":
                        save_trades(pd.DataFrame(columns=TRADE_COLUMNS))
                        st.success("All trades deleted.")
                        st.rerun()
                    else:
                        st.error("Confirmation text did not match 'DELETE'. No action taken.")
