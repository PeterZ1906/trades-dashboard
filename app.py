# app.py
# ============================================================
# Trades Dashboard App
# - Sidebar nav: Dashboard  |  Add / Manage Trades
# - CSV persistence at repo root: trades.csv
# - Integer-step spinners (step=1) inside the form
# - Delete selected or delete-all with confirmation
# ============================================================
import os
import uuid
from datetime import date, datetime

import pandas as pd
import streamlit as st

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

    # Normalize dates to YYYY/MM/DD for display/sort
    for dcol in ["entry_date", "exit_date"]:
        if dcol in df.columns:
            df[dcol] = df[dcol].fillna("")
    return _ensure_columns(df)


def save_trades(df: pd.DataFrame) -> None:
    df = _ensure_columns(df.copy())
    df.to_csv(TRADES_CSV, index=False)


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

    df = load_trades()
    if df.empty:
        st.info("No trades yet. Add some on the **Add / Manage Trades** page.")
    else:
        # --- Filters
        with st.expander("Filters", expanded=False):
            # date range filter (based on entry_date)
            min_date = df["entry_date"].replace("", pd.NA).dropna()
            if not min_date.empty:
                dmin = pd.to_datetime(min_date).min().date()
                dmax = pd.to_datetime(min_date).max().date()
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
            mask = pd.to_datetime(df_f["entry_date"], errors="coerce").between(pd.to_datetime(start), pd.to_datetime(end))
            df_f = df_f[mask]

        if sel_symbols:
            df_f = df_f[df_f["symbol"].isin(sel_symbols)]

        # --- KPIs
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Total Trades", len(df_f))
        with c2:
            st.metric("Total Shares", f"{df_f['shares'].sum():,.4f}")
        with c3:
            st.metric("Total Fees ($)", f"{df_f['fees_total'].sum():,.2f}")
        with c4:
            avg_entry = df_f["entry_total"].mean() if len(df_f) else 0.0
            st.metric("Avg Entry ($)", f"{avg_entry:,.2f}")

        st.divider()

        # --- By Symbol summary
        st.subheader("Summary by Symbol")
        by_sym = (
            df_f.groupby("symbol", as_index=False)
            .agg(
                trades=("id", "count"),
                shares=("shares", "sum"),
                entry_total_avg=("entry_total", "mean"),
                fees_total=("fees_total", "sum"),
            )
            .sort_values(by=["trades", "shares"], ascending=[False, False])
        )
        st.dataframe(by_sym, use_container_width=True)

        st.subheader("All Trades (filtered)")
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
