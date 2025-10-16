# app.py
# ------------------------------------------------------------
# Trades Dashboard: Add & Manage Trades
# - Integer step +/- controls
# - CSV persistence (trades.csv at repo root)
# - Delete selected / delete all with confirmation
# ------------------------------------------------------------
import os
import uuid
from datetime import date, datetime

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Trades Dashboard", layout="wide")

# ==============================
# Storage config
# ==============================
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
            # numeric-ish defaults vs string defaults
            if c in {"shares", "entry_total", "stop_price", "target1", "target2", "exit_price", "fees_total"}:
                df[c] = 0.0
            else:
                df[c] = ""
    # enforce column order
    df = df[TRADE_COLUMNS]
    return df


def load_trades() -> pd.DataFrame:
    if not os.path.exists(TRADES_CSV):
        return pd.DataFrame(columns=TRADE_COLUMNS)
    df = pd.read_csv(TRADES_CSV, dtype=str)

    # Migration: add ids if missing
    if "id" not in df.columns:
        df["id"] = [str(uuid.uuid4()) for _ in range(len(df))]

    # Coerce numerics
    for col in ["shares", "entry_total", "stop_price", "target1", "target2", "exit_price", "fees_total"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = _ensure_columns(df)
    return df


def save_trades(df: pd.DataFrame) -> None:
    df = _ensure_columns(df.copy())
    df.to_csv(TRADES_CSV, index=False)


# ==============================
# Integer-step helper
# ==============================
def int_step_number_input(
    label: str,
    key: str,
    value: float = 0.0,
    step: int = 1,
    min_value: float | None = None,
    max_value: float | None = None,
    format: str | None = None,
    help: str | None = None,
):
    """
    Number input with explicit – / + buttons that move in *integer* steps.
    Users can still TYPE fractional values; the buttons always change by whole integers.
    """
    if key not in st.session_state:
        st.session_state[key] = value

    c_input, c_minus, c_plus = st.columns([10, 1, 1], vertical_alignment="bottom")
    with c_input:
        st.session_state[key] = st.number_input(
            label,
            key=f"{key}_num",
            value=float(st.session_state[key]),
            min_value=min_value,
            max_value=max_value,
            help=help,
            format=(format if format else None),
        )
    with c_minus:
        if st.button("–", key=f"{key}_minus"):
            st.session_state[key] = float(st.session_state[key]) - int(step)
            if min_value is not None:
                st.session_state[key] = max(st.session_state[key], float(min_value))
            st.rerun()
    with c_plus:
        if st.button("+", key=f"{key}_plus"):
            st.session_state[key] = float(st.session_state[key]) + int(step)
            if max_value is not None:
                st.session_state[key] = min(st.session_state[key], float(max_value))
            st.rerun()

    return float(st.session_state[key])


# ==============================
# UI
# ==============================
st.title("Add Trade")

with st.form("add_trade_form", enter_to_submit=False):
    c1, c2, c3 = st.columns(3)

    # ----- Column 1 -----
    with c1:
        symbol = st.text_input("Symbol *", value="AMZN").upper().strip()
        entry_total = int_step_number_input(
            "Entry Total ($) *", key="entry_total", value=0.00, step=1, min_value=0.0, format="%.2f"
        )
        stop_price = int_step_number_input(
            "Stop Price (optional)", key="stop_price", value=0.0, step=1, min_value=0.0, format="%.4f"
        )
        use_exit_date = st.checkbox("Set Exit Date", value=True)
        exit_date_dt = st.date_input(
            "Exit Date (optional)",
            value=date.today(),
            format="YYYY/MM/DD",
            disabled=not use_exit_date,
        )

    # ----- Column 2 -----
    with c2:
        side = st.selectbox("Side *", options=["long", "short"], index=0)
        entry_date_dt = st.date_input("Entry Date *", value=date.today(), format="YYYY/MM/DD")
        target1 = int_step_number_input(
            "Target 1 (optional)", key="target1", value=0.0, step=1, min_value=0.0, format="%.4f"
        )
        exit_price = int_step_number_input(
            "Exit Price (optional)", key="exit_price", value=0.0, step=1, min_value=0.0, format="%.4f"
        )

    # ----- Column 3 -----
    with c3:
        shares = int_step_number_input(
            "Shares (fractional ok) *", key="shares", value=1.0, step=1, min_value=0.0, format="%.6f"
        )
        company = st.text_input("Company", value="")
        target2 = int_step_number_input(
            "Target 2 (optional)", key="target2", value=0.0, step=1, min_value=0.0, format="%.4f"
        )
        fees_total = int_step_number_input(
            "Fees Total ($)", key="fees_total", value=0.0, step=1, min_value=0.0, format="%.2f"
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
    # Pretty dataframe
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
