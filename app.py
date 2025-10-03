
import streamlit as st
import pandas as pd
import altair as alt
from datetime import date
from dateutil import tz
from pathlib import Path

st.set_page_config(page_title="Simple Trades Dashboard", layout="wide")

DATA_FILE = Path(__file__).parent / "trades.csv"
COLUMNS = ["id","symbol","company","side","shares","entry_total","entry_date",
           "stop_price","target1_price","target2_price","exit_date","exit_price",
           "fees_total","strategy_tag","notes"]

def load_trades():
    if DATA_FILE.exists():
        df = pd.read_csv(DATA_FILE)
        df["entry_date"] = pd.to_datetime(df["entry_date"], errors="coerce")
        df["exit_date"] = pd.to_datetime(df["exit_date"], errors="coerce")
        return df
    return pd.DataFrame(columns=COLUMNS)

def save_trades(df):
    df.to_csv(DATA_FILE, index=False)

def next_id(df):
    return 1 if df.empty else int(df["id"].max()) + 1

def compute_derived(df_raw):
    df = df_raw.copy()
    if df.empty:
        for c in ["entry_price","close_total","gross_pl","net_pl","roi","r_multiple","days_in_market","status","profitable_label"]:
            df[c] = pd.Series(dtype=float if c not in ["status","profitable_label"] else "object")
        return df
    df["shares"] = pd.to_numeric(df["shares"], errors="coerce")
    df["entry_total"] = pd.to_numeric(df["entry_total"], errors="coerce")
    df["fees_total"] = pd.to_numeric(df["fees_total"], errors="coerce").fillna(0.0)
    df["stop_price"] = pd.to_numeric(df["stop_price"], errors="coerce")
    df["exit_price"] = pd.to_numeric(df["exit_price"], errors="coerce")
    df["entry_date"] = pd.to_datetime(df["entry_date"], errors="coerce")
    df["exit_date"]  = pd.to_datetime(df["exit_date"], errors="coerce")
    df["entry_price"] = df.apply(lambda r: (r["entry_total"]/r["shares"]) if (pd.notna(r["entry_total"]) and pd.notna(r["shares"]) and r["shares"]>0) else pd.NA, axis=1)
    df["status"] = df["exit_date"].apply(lambda x: "Closed" if pd.notna(x) else "Open")
    df["close_total"] = df.apply(lambda r: (r["exit_price"] * r["shares"]) if (pd.notna(r["exit_price"]) and pd.notna(r["shares"])) else pd.NA, axis=1)
    df["gross_pl"] = df.apply(lambda r: (r["close_total"] - r["entry_total"]) if pd.notna(r["close_total"]) else pd.NA, axis=1)
    df["net_pl"] = df.apply(lambda r: (r["gross_pl"] - r["fees_total"]) if pd.notna(r["gross_pl"]) else pd.NA, axis=1)
    df["roi"] = df.apply(lambda r: (r["net_pl"]/r["entry_total"]) if (pd.notna(r["net_pl"]) and pd.notna(r["entry_total"]) and r["entry_total"]>0) else pd.NA, axis=1)
    df["days_in_market"] = df.apply(lambda r: (r["exit_date"] - r["entry_date"]).days + 1 if (pd.notna(r["exit_date"]) and pd.notna(r["entry_date"])) else pd.NA, axis=1)
    df["profitable_label"] = df["net_pl"].apply(lambda x: "Profitable" if (pd.notna(x) and x>0) else ("Lost/Losing" if pd.notna(x) else ""))
    def r_mult(r):
        stop = r.get("stop_price"); entry = r.get("entry_price"); exitp = r.get("exit_price"); side = r.get("side")
        if pd.isna(stop) or pd.isna(entry) or pd.isna(exitp) or not side: return pd.NA
        if side == "long":
            denom = entry - stop; return (exitp - entry)/denom if denom else pd.NA
        elif side == "short":
            denom = stop - entry; return (entry - exitp)/denom if denom else pd.NA
        return pd.NA
    df["r_multiple"] = df.apply(r_mult, axis=1)
    return df

def kpis(df):
    d = df[df["status"]=="Closed"].copy()
    n = len(d); wins = int((d["net_pl"]>0).sum()); losses = int((d["net_pl"]<0).sum())
    win_rate = (wins/n) if n else 0.0
    avg_r = d["r_multiple"].dropna().astype(float).mean() if "r_multiple" in d else None
    total_wins = d.loc[d["net_pl"]>0,"net_pl"].sum(); total_losses = d.loc[d["net_pl"]<0,"net_pl"].sum()
    profit_factor = (total_wins/abs(total_losses)) if losses else None
    avg_win = d.loc[d["net_pl"]>0,"net_pl"].mean(); avg_loss = d.loc[d["net_pl"]<0,"net_pl"].mean()
    expectancy = (avg_win*win_rate + avg_loss*(1-win_rate)) if (pd.notna(avg_win) and pd.notna(avg_loss)) else None
    if d.empty: max_dd = 0.0
    else:
        eq = d.sort_values("exit_date")["net_pl"].cumsum()
        peaks = eq.cummax(); max_dd = (eq - peaks).min()
    return {"closed_trades":int(n),"wins":wins,"losses":losses,"win_rate":win_rate,
            "avg_r": (float(avg_r) if avg_r==avg_r else None),
            "profit_factor": (float(profit_factor) if profit_factor and profit_factor==profit_factor else None),
            "expectancy": (float(expectancy) if expectancy is not None and expectancy==expectancy else None),
            "max_drawdown": float(max_dd) if max_dd==max_dd else 0.0,
            "net_pl_total": float(d["net_pl"].sum()) if n else 0.0}

def page_dashboard():
    st.header("Dashboard")

    # --- Filters (friendlier) ---
    df_all = compute_derived(load_trades())
    with st.expander("Filters", expanded=True):
        colf1, colf2, colf3, colf4 = st.columns([1,1,1,1])
        start = colf1.date_input("Start date", value=None)
        end   = colf2.date_input("End date", value=None)
        syms  = colf3.multiselect("Symbols", sorted([s for s in df_all["symbol"].dropna().unique()]))
        tags  = colf4.multiselect("Strategy tags", sorted([t for t in df_all["strategy_tag"].dropna().unique()]))

    mask = pd.Series([True]*len(df_all))
    if start is not None:
        mask &= (pd.to_datetime(df_all["entry_date"]).dt.date >= pd.to_datetime(start).date())
    if end is not None:
        mask &= (pd.to_datetime(df_all["exit_date"].fillna(df_all["entry_date"])).dt.date <= pd.to_datetime(end).date())
    if syms:
        mask &= df_all["symbol"].isin(syms)
    if tags:
        mask &= df_all["strategy_tag"].isin(tags)

    df = df_all[mask].copy()

    # --- KPIs (card-like row) ---
    stats = kpis(df)
    c1,c2,c3,c4,c5,c6,c7 = st.columns(7)
    c1.metric("Closed Trades", stats["closed_trades"])
    c2.metric("Win Rate", f"{stats['win_rate']*100:.1f}%")
    c3.metric("Avg R", f"{stats['avg_r']:.2f}" if stats['avg_r'] is not None else "—")
    c4.metric("Profit Factor", f"{stats['profit_factor']:.2f}" if stats['profit_factor'] is not None else "—")
    c5.metric("Expectancy ($)", f"{stats['expectancy']:.2f}" if stats['expectancy'] is not None else "—")
    c6.metric("Max Drawdown ($)", f"{stats['max_drawdown']:.2f}")
    c7.metric("Net P/L ($)", f"{stats['net_pl_total']:.2f}")

    st.markdown("---")

    # --- Charts (colorful & readable via Altair) ---
    # Prepare frames
    closed = df[df["status"]=="Closed"].copy()
    openpos = df[df["status"]=="Open"].copy()

    colA, colB = st.columns([2,1])

    # Equity curve (layered area + line, no gradient)
with colA:
    st.subheader("Equity Curve (Closed)")
    if not closed.empty:
        curve = closed.sort_values("exit_date")[["exit_date","net_pl"]].copy()
        curve["equity"] = curve["net_pl"].fillna(0).cumsum()
        curve = curve.dropna(subset=["exit_date"])

        area = (
            alt.Chart(curve)
            .mark_area(color="#FF4B4B", opacity=0.25)
            .encode(
                x=alt.X("exit_date:T", title="Date"),
                y=alt.Y("equity:Q", title="Cumulative Net P/L ($)")
            )
            .properties(height=280)
        )

        line = (
            alt.Chart(curve)
            .mark_line(color="#FF4B4B", strokeWidth=2)
            .encode(
                x="exit_date:T",
                y="equity:Q",
                tooltip=[alt.Tooltip("exit_date:T", title="Date"),
                         alt.Tooltip("equity:Q", title="Equity")]
            )
        )

        st.altair_chart(area + line, use_container_width=True)
    else:
        st.info("No closed trades yet.")


    # Open positions allocation (donut)
    with colB:
        st.subheader("Open Allocation")
        if not openpos.empty:
            alloc = openpos.groupby("symbol", dropna=False)["entry_total"].sum().reset_index()
            alloc["entry_total"] = alloc["entry_total"].fillna(0)
            base = alt.Chart(alloc).encode(theta=alt.Theta("entry_total:Q"), color=alt.Color("symbol:N"), tooltip=["symbol","entry_total"])
            donut = base.mark_arc(innerRadius=60)
            st.altair_chart(donut.properties(height=280), use_container_width=True)
        else:
            st.info("No open positions.")

    col1, col2 = st.columns(2)

    # Monthly P/L (bar by YearMonth)
    with col1:
        st.subheader("Monthly P/L (Closed)")
        if not closed.empty:
            m = closed.copy()
            m["ym"] = pd.to_datetime(m["exit_date"]).dt.to_period("M").astype(str)
            m = m.groupby("ym", as_index=False)["net_pl"].sum()
            bar = (
                alt.Chart(m)
                .mark_bar()
                .encode(x=alt.X("ym:N", title="Month"), y=alt.Y("net_pl:Q", title="Net P/L ($)"), color=alt.condition("datum.net_pl >= 0", alt.value("#2ECC71"), alt.value("#E74C3C")), tooltip=["ym","net_pl"])
                .properties(height=240)
            )
            st.altair_chart(bar, use_container_width=True)
        else:
            st.info("Close trades to see monthly results.")

    # P/L by Strategy (top 10)
    with col2:
        st.subheader("P/L by Strategy")
        if not closed.empty:
            s = closed.groupby("strategy_tag", dropna=False)["net_pl"].sum().reset_index().rename(columns={"strategy_tag":"strategy"})
            s = s.sort_values("net_pl", ascending=False).head(10)
            bar = (
                alt.Chart(s)
                .mark_bar()
                .encode(y=alt.Y("strategy:N", sort="-x", title="Strategy"), x=alt.X("net_pl:Q", title="Net P/L ($)"),
                        color=alt.condition("datum.net_pl >= 0", alt.value("#2ECC71"), alt.value("#E74C3C")),
                        tooltip=["strategy","net_pl"])
                .properties(height=240)
            )
            st.altair_chart(bar, use_container_width=True)
        else:
            st.info("No closed trades yet.")

    col3, col4 = st.columns(2)

    # R Multiple Distribution
    with col3:
        st.subheader("R Multiple Distribution (Closed)")
        r = closed["r_multiple"].dropna().astype(float)
        if len(r):
            rdf = pd.DataFrame({"R": r})
            hist = (
                alt.Chart(rdf)
                .transform_bin("bin_R", "R", bin=alt.Bin(maxbins=25))
                .mark_bar()
                .encode(x=alt.X("bin_R:Q", title="R multiple"), y="count()", color=alt.value("#8E7CFF"), tooltip=[alt.Tooltip("count()", title="Count")])
                .properties(height=240)
            )
            st.altair_chart(hist, use_container_width=True)
        else:
            st.info("Add stops & exits to compute R.")

    # Win / Loss streak (sequence)
    with col4:
        st.subheader("Win/Loss Sequence (Closed)")
        if not closed.empty:
            seq = closed.sort_values("exit_date")[["exit_date","net_pl"]].copy()
            seq["result"] = (seq["net_pl"] >= 0).map({True:"Win", False:"Loss"})
            seq["n"] = range(1, len(seq)+1)
            dot = (
                alt.Chart(seq)
                .mark_circle(size=90)
                .encode(
                    x=alt.X("n:O", title="Trade # (by close)"),
                    y=alt.Y("net_pl:Q", title="Net P/L ($)"),
                    color=alt.Color("result:N", scale=alt.Scale(domain=["Win","Loss"], range=["#2ECC71","#E74C3C"])),
                    tooltip=["exit_date:T","net_pl:Q","result:N"]
                )
                .properties(height=240)
            )
            st.altair_chart(dot, use_container_width=True)
        else:
            st.info("No closed trades yet.")

    # Quick insights
    st.markdown("---")
    st.subheader("Quick Insights")
    if not closed.empty:
        top_sym = closed.groupby("symbol")["net_pl"].sum().sort_values(ascending=False).head(3)
        worst_sym = closed.groupby("symbol")["net_pl"].sum().sort_values(ascending=True).head(3)
        st.write(f"**Top symbols:** {', '.join([f'{i} (${v:.0f})' for i,v in top_sym.items()])}" if len(top_sym) else "—")
        st.write(f"**Tough symbols:** {', '.join([f'{i} (${v:.0f})' for i,v in worst_sym.items()])}" if len(worst_sym) else "—")
    else:
        st.write("Add and close a few trades to see insights.")

def page_add_trade():
    st.header("Add Trade")
    with st.form("add_trade"):
        c1,c2,c3 = st.columns(3)
        symbol = c1.text_input("Symbol *").upper().strip()
        side = c2.selectbox("Side *", ["long","short"])
        shares = c3.number_input("Shares (fractional ok) *", min_value=0.0, step=0.0001, format="%.6f")
        c4,c5,c6 = st.columns(3)
        entry_total = c4.number_input("Entry Total ($) *", min_value=0.0, step=0.01, format="%.2f")
        entry_date = c5.date_input("Entry Date *", value=date.today())
        company = c6.text_input("Company")
        c7,c8,c9 = st.columns(3)
        stop_price = c7.number_input("Stop Price (optional)", min_value=0.0, step=0.01, format="%.4f")
        target1 = c8.number_input("Target 1 (optional)", min_value=0.0, step=0.01, format="%.4f")
        target2 = c9.number_input("Target 2 (optional)", min_value=0.0, step=0.01, format="%.4f")
        c10,c11,c12 = st.columns(3)
        exit_date = c10.date_input("Exit Date (optional)")
        exit_price = c11.number_input("Exit Price (optional)", min_value=0.0, step=0.01, format="%.4f")
        fees_total = c12.number_input("Fees Total ($)", min_value=0.0, step=0.01, value=0.0, format="%.2f")
        strategy_tag = st.text_input("Strategy / Tag (optional)")
        notes = st.text_area("Notes", height=70)
        submitted = st.form_submit_button("Save Trade")
        if submitted:
            if not symbol or shares <= 0 or entry_total <= 0:
                st.error("Please fill required fields: Symbol, Shares > 0, Entry Total > 0."); return
            df = load_trades()
            new = {
                "id": next_id(df),
                "symbol": symbol,
                "company": company or None,
                "side": side,
                "shares": float(shares),
                "entry_total": float(entry_total),
                "entry_date": pd.to_datetime(entry_date).date().isoformat(),
                "stop_price": float(stop_price) if stop_price else None,
                "target1_price": float(target1) if target1 else None,
                "target2_price": float(target2) if target2 else None,
                "exit_date": pd.to_datetime(exit_date).date().isoformat() if exit_date else None,
                "exit_price": float(exit_price) if exit_price else None,
                "fees_total": float(fees_total) if fees_total else 0.0,
                "strategy_tag": strategy_tag or None,
                "notes": notes or None,
            }
            df = pd.concat([df, pd.DataFrame([new])], ignore_index=True)
            save_trades(df)
            st.success(f"Trade saved (ID: {new['id']}).")

def page_trades():
    st.header("Trades")
    df_raw = load_trades()
    df = compute_derived(df_raw)
    with st.expander("Filters", expanded=True):
        c1,c2,c3 = st.columns(3)
        start = c1.date_input("Start date", value=None)
        end = c2.date_input("End date", value=None)
        status = c3.multiselect("Status", ["Open","Closed"], default=[])
        symbol = st.text_input("Symbol contains")
        strategy = st.text_input("Strategy/Tag contains")
    mask = pd.Series([True]*len(df))
    if start is not None:
        mask &= (pd.to_datetime(df["entry_date"]).dt.date >= pd.to_datetime(start).date())
    if end is not None:
        last_dt = df["exit_date"].fillna(df["entry_date"])
        mask &= (pd.to_datetime(last_dt).dt.date <= pd.to_datetime(end).date())
    if status:
        mask &= df["status"].isin(status)
    if symbol:
        mask &= df["symbol"].str.contains(symbol.upper(), na=False)
    if strategy:
        mask &= df["strategy_tag"].str.contains(strategy, na=False)
    view = df[mask].copy()
    st.dataframe(view.fillna(""), use_container_width=True)
    st.download_button("Export CSV", view.to_csv(index=False).encode("utf-8"),
                       file_name="trades_export.csv", mime="text/csv")
    st.subheader("Quick Close Trade")
    open_df = df[df["status"]=="Open"]
    if open_df.empty:
        st.info("No open trades to close.")
    else:
        ids = open_df["id"].astype(int).tolist()
        cid = st.selectbox("Trade ID", ids)
        c1,c2,c3 = st.columns(3)
        exit_date = c1.date_input("Exit Date", value=date.today())
        exit_price = c2.number_input("Exit Price", min_value=0.0, step=0.01, format="%.4f")
        extra_fees = c3.number_input("Additional Fees (optional)", min_value=0.0, step=0.01, format="%.2f")
        if st.button("Close Trade"):
            df_raw.loc[df_raw["id"]==cid, "exit_date"] = pd.to_datetime(exit_date).date().isoformat()
            df_raw.loc[df_raw["id"]==cid, "exit_price"] = float(exit_price)
            if extra_fees and extra_fees>0:
                cur = pd.to_numeric(df_raw.loc[df_raw["id"]==cid, "fees_total"], errors="coerce").fillna(0.0)
                df_raw.loc[df_raw["id"]==cid, "fees_total"] = (cur + float(extra_fees)).values
            save_trades(df_raw)
            st.success(f"Trade {cid} closed.")

# Sidebar nav
st.sidebar.title("Navigation")
page = st.sidebar.radio("", ["Dashboard","Add Trade","Trades"])

# Ensure data file exists
if not DATA_FILE.exists():
    pd.DataFrame(columns=COLUMNS).to_csv(DATA_FILE, index=False)

if page == "Dashboard":
    page_dashboard()
elif page == "Add Trade":
    page_add_trade()
elif page == "Trades":
    page_trades()
