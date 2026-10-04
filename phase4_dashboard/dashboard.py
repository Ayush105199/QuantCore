"""
QuantCore — Streamlit Dashboard
---------------------------------
Run: streamlit run phase4_dashboard/dashboard.py
URL: http://localhost:8501

Four pages:
  1. Order Book    — live L2 book + OFI simulation
  2. Options Pricer — BS pricing, all Greeks, MC validation
  3. Market Making — AS vs Naive P&L comparison
  4. Vol Surface   — 3D interactive implied vol surface
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import sys, os, math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from phase1_orderbook.order              import Side
from phase1_orderbook.orderbook          import OrderBook
from phase1_orderbook.order_flow_generator import OrderFlowGenerator, GeneratorConfig
from phase2_options.black_scholes        import price_option, bs_call, bs_put
from phase2_options.iv_solver            import implied_volatility_safe
from phase2_options.vol_surface          import generate_vol_surface
from phase2_options.monte_carlo          import mc_price_option
from phase3_marketmaking.avellaneda_stoikov import AvellanedaStoikov, ASParameters
from phase3_marketmaking.naive_mm        import NaiveMarketMaker, NaiveMMParams
from phase3_marketmaking.mm_simulator    import MMSimulator, compare_strategies


# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="QuantCore",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .metric-label { font-size: 13px; color: #90CAF9; }
  .stMetric > div { background: #1E1E2E; border-radius: 8px; padding: 6px; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("QuantCore")
st.sidebar.caption("HFT Quant Systems Toolkit")
st.sidebar.divider()

page = st.sidebar.radio("Navigate", [
    "Order Book Simulator",
    "Options Pricer",
    "Market Making",
    "Vol Surface",
])

st.sidebar.divider()
st.sidebar.markdown("""
**Quick Run:**
```bash
# Each phase standalone:
python phase1_orderbook/orderbook.py
python phase2_options/black_scholes.py
python phase3_marketmaking/mm_simulator.py
```
""")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — ORDER BOOK SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════
if page == "Order Book Simulator":
    st.title("L2 Order Book Simulator")
    st.markdown("Simulates a realistic limit order book with Poisson order arrivals, "
                "GBM price dynamics, and real-time Order Flow Imbalance (OFI) computation.")
    st.divider()

    col_ctrl, col_info = st.columns([1, 3])

    with col_ctrl:
        st.subheader("Controls")
        mid_price   = st.slider("Mid Price",      80.0, 120.0, 100.0, 0.5)
        order_rate  = st.slider("Order Rate (lambda)", 1.0, 15.0, 6.0, 0.5)
        cancel_rate = st.slider("Cancel Rate",    1.0, 10.0, 3.0, 0.5)
        sim_steps   = st.slider("Simulation Steps", 100, 1500, 500, 100)
        vol         = st.slider("Volatility (sigma)", 0.005, 0.05, 0.015, 0.005)
        run_btn     = st.button("Run Simulation", type="primary", use_container_width=True)

    if run_btn or "ob_history" not in st.session_state:
        with st.spinner("Running order book simulation..."):
            cfg  = GeneratorConfig(mid_price=mid_price, lambda_bid=order_rate,
                                   lambda_ask=order_rate, lambda_cancel=cancel_rate,
                                   sigma=vol)
            gen  = OrderFlowGenerator(config=cfg, seed=42)
            book = OrderBook()
            hist = gen.run_simulation(n_steps=sim_steps, book=book)
            st.session_state["ob_history"] = hist
            st.session_state["ob_book"]    = book

    hist = st.session_state["ob_history"]
    book = st.session_state["ob_book"]

    with col_info:
        # KPI row
        c1, c2, c3, c4, c5 = st.columns(5)
        bb = book.best_bid; ba = book.best_ask
        sp = book.spread;   mp = book.mid_price
        ofi_now = hist["ofi"].iloc[-1]

        c1.metric("Best Bid",  f"{bb:.4f}" if bb else "N/A")
        c2.metric("Best Ask",  f"{ba:.4f}" if ba else "N/A")
        c3.metric("Spread",    f"{sp:.4f}" if sp else "N/A")
        c4.metric("Mid Price", f"{mp:.4f}" if mp else "N/A")
        c5.metric("OFI",       f"{ofi_now:+.3f}",
                  delta="Buy" if ofi_now > 0.2 else ("Sell" if ofi_now < -0.2 else "Neutral"))

    st.divider()

    # ── Depth Chart ───────────────────────────────────────────────────────────
    col_d, col_ts = st.columns(2)

    with col_d:
        st.subheader("Order Book Depth")
        bids = book.bid_depth(12)
        asks = book.ask_depth(12)

        fig_depth = go.Figure()
        if bids:
            b_p, b_q = zip(*bids)
            fig_depth.add_trace(go.Bar(
                x=list(b_q), y=[f"{p:.4f}" for p in b_p],
                orientation="h", name="Bids",
                marker_color="#00C853", opacity=0.8
            ))
        if asks:
            a_p, a_q = zip(*asks)
            fig_depth.add_trace(go.Bar(
                x=[-q for q in a_q], y=[f"{p:.4f}" for p in a_p],
                orientation="h", name="Asks",
                marker_color="#FF1744", opacity=0.8
            ))
        if mp:
            fig_depth.add_hline(y=f"{mp:.4f}", line_color="#2196F3",
                                 line_dash="dash",
                                 annotation_text=f"Mid: {mp:.4f}")
        fig_depth.update_layout(
            template="plotly_dark", height=380, barmode="overlay",
            xaxis_title="Volume", yaxis_title="Price",
            title="Bid (green) vs Ask (red) Depth"
        )
        st.plotly_chart(fig_depth, use_container_width=True)

    with col_ts:
        st.subheader("Mid Price + OFI Over Time")
        fig_ts = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                row_heights=[0.6, 0.4], vertical_spacing=0.05)

        mid_data   = hist["step_mid"].dropna()
        ofi_data   = hist["ofi"].fillna(0)
        ofi_colors = ["#00E676" if v >= 0 else "#FF5252" for v in ofi_data.values]

        fig_ts.add_trace(go.Scatter(
            y=mid_data.values, mode="lines",
            line=dict(color="#2196F3", width=1.5), name="Mid Price"
        ), row=1, col=1)
        fig_ts.add_trace(go.Bar(
            y=ofi_data.values, name="OFI",
            marker_color=ofi_colors, opacity=0.8
        ), row=2, col=1)
        fig_ts.update_layout(template="plotly_dark", height=380,
                              showlegend=False)
        fig_ts.update_yaxes(title_text="Mid Price", row=1, col=1)
        fig_ts.update_yaxes(title_text="OFI", row=2, col=1,
                             range=[-1.1, 1.1])
        st.plotly_chart(fig_ts, use_container_width=True)

    # Spread and volume flow
    col_s, col_v = st.columns(2)
    with col_s:
        spread_data = hist["spread_bps"].dropna()
        fig_spread  = go.Figure()
        fig_spread.add_trace(go.Scatter(
            y=spread_data.values, fill="tozeroy",
            fillcolor="rgba(255,193,7,0.15)",
            line=dict(color="#FFC107", width=1.2), name="Spread (bps)"
        ))
        fig_spread.add_hline(y=spread_data.mean(), line_color="white",
                              line_dash="dash",
                              annotation_text=f"Avg: {spread_data.mean():.2f} bps")
        fig_spread.update_layout(template="plotly_dark", height=220,
                                  title="Bid-Ask Spread (bps)",
                                  yaxis_title="bps")
        st.plotly_chart(fig_spread, use_container_width=True)

    with col_v:
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Scatter(
            y=hist["total_bid_qty"].values, fill="tozeroy",
            fillcolor="rgba(0,200,83,0.2)",
            line=dict(color="#00C853"), name="Bid Volume"
        ))
        fig_vol.add_trace(go.Scatter(
            y=hist["total_ask_qty"].values, fill="tozeroy",
            fillcolor="rgba(255,23,68,0.2)",
            line=dict(color="#FF1744"), name="Ask Volume"
        ))
        fig_vol.update_layout(template="plotly_dark", height=220,
                               title="Total Bid vs Ask Volume")
        st.plotly_chart(fig_vol, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — OPTIONS PRICER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Options Pricer":
    st.title("Options Pricing Engine — Black-Scholes")
    st.markdown("Full Black-Scholes implementation from scratch. All 5 Greeks. "
                "Newton-Raphson IV solver. Monte Carlo cross-validation.")
    st.divider()

    col_ctrl, col_out = st.columns([1, 2])

    with col_ctrl:
        st.subheader("Parameters")
        S      = st.slider("Spot Price (S)",        50.0, 200.0, 100.0, 0.5)
        K      = st.slider("Strike (K)",            50.0, 200.0, 100.0, 0.5)
        T_days = st.slider("Days to Expiry",         1,    730,    90,   1)
        r_pct  = st.slider("Risk-Free Rate (%)",     0.0,  15.0,   6.5,  0.1)
        sig_pct= st.slider("Volatility / IV (%)",    1.0,  100.0,  20.0, 0.5)
        opt_t  = st.radio("Option Type", ["call", "put"], horizontal=True)

        T     = T_days / 365.0
        r     = r_pct  / 100.0
        sigma = sig_pct / 100.0

    result = price_option(S, K, T, r, sigma, opt_t)

    with col_out:
        st.subheader("Pricing Results")

        # Moneyness banner
        mness   = result.moneyness()
        m_color = "#00C853" if mness == "ITM" else ("#FFC107" if mness == "ATM" else "#90CAF9")
        st.markdown(f"<span style='background:{m_color};padding:4px 12px;border-radius:4px;"
                    f"color:black;font-weight:bold'>{mness}</span>", unsafe_allow_html=True)
        st.write("")

        r1c1, r1c2, r1c3 = st.columns(3)
        r1c1.metric("Option Price",  f"{result.price:.4f}")
        r1c2.metric("Intrinsic",     f"{result.intrinsic_value():.4f}")
        r1c3.metric("Time Value",    f"{result.time_value():.4f}")

        st.divider()
        st.subheader("The Greeks")
        r2c1, r2c2, r2c3 = st.columns(3)
        r2c1.metric("Delta", f"{result.delta:+.4f}", help="Rs change per Rs1 spot move")
        r2c2.metric("Gamma", f"{result.gamma:.6f}",  help="Delta change per Rs1 spot move")
        r2c3.metric("Theta", f"{result.theta:+.4f}", help="Rs change per day (time decay)")
        r2c4, r2c5, _ = st.columns(3)
        r2c4.metric("Vega",  f"{result.vega:.4f}",   help="Rs change per 1% vol move")
        r2c5.metric("Rho",   f"{result.rho:+.4f}",   help="Rs change per 1% rate move")

    st.divider()

    # ── Charts ────────────────────────────────────────────────────────────────
    col_pay, col_greek = st.columns(2)

    with col_pay:
        st.subheader("Payoff Diagram")
        S_range = np.linspace(K * 0.65, K * 1.35, 300)
        if opt_t == "call":
            payoff = np.maximum(S_range - K, 0) - result.price
            be     = K + result.price
        else:
            payoff = np.maximum(K - S_range, 0) - result.price
            be     = K - result.price

        fig_pay = go.Figure()
        fig_pay.add_trace(go.Scatter(
            x=S_range, y=payoff, mode="lines",
            line=dict(color="#2196F3", width=2.5), name="P&L"
        ))
        fig_pay.add_hline(y=0, line_color="white", line_dash="dash")
        fig_pay.add_vline(x=K,  line_color="yellow", line_dash="dot",
                           annotation_text=f"K={K}")
        fig_pay.add_vline(x=be, line_color="#FF9800", line_dash="dot",
                           annotation_text=f"BE={be:.2f}")
        fig_pay.add_traces([
            go.Scatter(x=S_range[payoff >= 0], y=payoff[payoff >= 0],
                       fill="tozeroy", fillcolor="rgba(0,200,83,0.15)",
                       line=dict(color="rgba(0,0,0,0)"), showlegend=False),
            go.Scatter(x=S_range[payoff < 0],  y=payoff[payoff < 0],
                       fill="tozeroy", fillcolor="rgba(255,23,68,0.15)",
                       line=dict(color="rgba(0,0,0,0)"), showlegend=False),
        ])
        fig_pay.update_layout(template="plotly_dark", height=320,
                               xaxis_title="Spot at Expiry", yaxis_title="P&L")
        st.plotly_chart(fig_pay, use_container_width=True)

    with col_greek:
        st.subheader("Delta vs Spot Price")
        S_range2 = np.linspace(K * 0.60, K * 1.40, 200)
        deltas   = [price_option(s, K, T, r, sigma, opt_t).delta for s in S_range2]
        gammas   = [price_option(s, K, T, r, sigma, opt_t).gamma for s in S_range2]

        fig_dg = make_subplots(specs=[[{"secondary_y": True}]])
        fig_dg.add_trace(go.Scatter(x=S_range2, y=deltas,
                                     line=dict(color="#00C853", width=2), name="Delta"),
                          secondary_y=False)
        fig_dg.add_trace(go.Scatter(x=S_range2, y=gammas,
                                     line=dict(color="#FFC107", width=2, dash="dash"),
                                     name="Gamma"),
                          secondary_y=True)
        fig_dg.add_vline(x=K, line_color="white", line_dash="dot",
                          annotation_text="Strike")
        fig_dg.add_vline(x=S, line_color="#2196F3", line_dash="dot",
                          annotation_text="Spot")
        fig_dg.update_layout(template="plotly_dark", height=320)
        fig_dg.update_yaxes(title_text="Delta", secondary_y=False)
        fig_dg.update_yaxes(title_text="Gamma", secondary_y=True)
        st.plotly_chart(fig_dg, use_container_width=True)

    # ── MC Validation ─────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Monte Carlo Cross-Validation")
    col_mc1, col_mc2 = st.columns([1, 2])
    with col_mc1:
        n_paths = st.select_slider("Paths", [10000, 25000, 50000, 100000], value=50000)
        run_mc  = st.button("Run Monte Carlo", type="primary")

    if run_mc:
        with st.spinner(f"Simulating {n_paths:,} GBM paths..."):
            mc = mc_price_option(S, K, T, r, sigma, opt_t,
                                  n_paths=n_paths, antithetic=True)
        with col_mc2:
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("MC Price",   f"{mc['price']:.6f}")
            mc2.metric("Std Error",  f"±{mc['std_error']:.6f}")
            mc3.metric("BS Price",   f"{mc['bs_price']:.6f}")
            mc4.metric("Difference", f"{mc['diff_pct']:+.4f}%")
            st.info(f"95% Confidence Interval: [{mc['conf_95'][0]:.6f}, {mc['conf_95'][1]:.6f}]")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MARKET MAKING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Market Making":
    st.title("Market Making Simulator — Avellaneda-Stoikov")
    st.markdown("Simulates an optimal inventory-aware market maker vs a naive fixed-spread baseline. "
                "Shows why inventory management is the key to profitable market making.")
    st.divider()

    col_ctrl, col_eq = st.columns([1, 2])

    with col_ctrl:
        st.subheader("AS Model Parameters")
        gamma_val = st.slider("Risk Aversion (gamma)", 0.01, 1.0, 0.10, 0.01,
                               help="Higher = wider spreads, more conservative")
        sigma_val = st.slider("Volatility (sigma)",    0.5,  5.0, 2.0,  0.1,
                               help="Asset vol per unit time")
        n_steps   = st.slider("Simulation Steps",     1000, 10000, 4000, 500)
        run_mm    = st.button("Run Simulation", type="primary", use_container_width=True)

        st.divider()
        st.info("""
        **What to look for:**
        - AS P&L should be higher (green > red)
        - AS inventory should stay near zero
        - Naive inventory drifts and hits limits
        """)

    if run_mm or "mm_results" not in st.session_state:
        with st.spinner("Running market making simulation..."):
            res = compare_strategies(n_steps=n_steps, seed=42,
                                      as_gamma=gamma_val, as_sigma=sigma_val)
            st.session_state["mm_results"] = res

    res     = st.session_state["mm_results"]
    as_m    = res["as"].metrics
    naive_m = res["naive"].metrics

    with col_eq:
        # Metrics comparison table
        st.subheader("Performance Comparison")
        comp_data = {
            "Metric"         : ["Final P&L", "Sharpe Ratio", "Max Drawdown",
                                 "Total Fills", "Spread Earned",
                                 "Max Inventory", "Avg Inventory"],
            "AS Optimal"     : [as_m["final_pnl"], as_m["sharpe_ratio"],
                                  as_m["max_drawdown"], as_m["total_fills"],
                                  as_m["total_spread_earned"], as_m["max_inventory"],
                                  as_m["avg_inventory"]],
            "Naive Fixed"    : [naive_m["final_pnl"], naive_m["sharpe_ratio"],
                                  naive_m["max_drawdown"], naive_m["total_fills"],
                                  naive_m["total_spread_earned"], naive_m["max_inventory"],
                                  naive_m["avg_inventory"]],
        }
        st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)

    st.divider()

    # Charts
    col_pnl, col_inv = st.columns(2)

    with col_pnl:
        st.subheader("P&L Comparison")
        fig_pnl = go.Figure()
        fig_pnl.add_trace(go.Scatter(
            y=res["as"].pnl,    mode="lines",
            line=dict(color="#00C853", width=2), name="AS Optimal"
        ))
        fig_pnl.add_trace(go.Scatter(
            y=res["naive"].pnl, mode="lines",
            line=dict(color="#FF1744", width=2), name="Naive Fixed"
        ))
        fig_pnl.add_hline(y=0, line_color="white", line_dash="dash")
        fig_pnl.update_layout(template="plotly_dark", height=300,
                               yaxis_title="Cumulative P&L")
        st.plotly_chart(fig_pnl, use_container_width=True)

    with col_inv:
        st.subheader("Inventory Management")
        fig_inv = go.Figure()
        fig_inv.add_trace(go.Scatter(
            y=res["as"].inventories,    mode="lines",
            line=dict(color="#00C853", width=2), name="AS Inventory",
            fill="tozeroy", fillcolor="rgba(0,200,83,0.1)"
        ))
        fig_inv.add_trace(go.Scatter(
            y=res["naive"].inventories, mode="lines",
            line=dict(color="#FF1744", width=2), name="Naive Inventory",
            fill="tozeroy", fillcolor="rgba(255,23,68,0.1)"
        ))
        fig_inv.add_hline(y=0, line_color="white", line_dash="dash")
        fig_inv.update_layout(template="plotly_dark", height=300,
                               yaxis_title="Inventory (shares)")
        st.plotly_chart(fig_inv, use_container_width=True)

    # AS quote activity chart
    st.subheader("AS Quote Activity (first 400 steps)")
    n_show = min(400, len(res["as"].mid_prices))
    fig_qa = go.Figure()
    fig_qa.add_trace(go.Scatter(
        y=res["as"].mid_prices[:n_show], mode="lines",
        line=dict(color="#2196F3", width=2), name="Mid Price"
    ))
    bids = res["as"].bid_quotes[:n_show]
    asks = res["as"].ask_quotes[:n_show]
    valid = ~(np.isnan(bids) | np.isnan(asks))
    fig_qa.add_trace(go.Scatter(
        y=bids, mode="lines",
        line=dict(color="#00C853", width=1, dash="dash"), name="Bid Quote"
    ))
    fig_qa.add_trace(go.Scatter(
        y=asks, mode="lines", fill="tonexty",
        fillcolor="rgba(158,158,158,0.12)",
        line=dict(color="#F44336", width=1, dash="dash"), name="Ask Quote"
    ))
    # Mark fills
    fills_show = res["as"].fills[res["as"].fills["step"] < n_show]
    if len(fills_show):
        bf = fills_show[fills_show["side"] == "bid"]
        af = fills_show[fills_show["side"] == "ask"]
        if len(bf):
            fig_qa.add_trace(go.Scatter(
                x=bf["step"], y=bf["price"], mode="markers",
                marker=dict(color="#00C853", size=8, symbol="triangle-up"),
                name="Buy Fill"
            ))
        if len(af):
            fig_qa.add_trace(go.Scatter(
                x=af["step"], y=af["price"], mode="markers",
                marker=dict(color="#FF1744", size=8, symbol="triangle-down"),
                name="Sell Fill"
            ))
    fig_qa.update_layout(template="plotly_dark", height=320,
                          xaxis_title="Step", yaxis_title="Price")
    st.plotly_chart(fig_qa, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — VOL SURFACE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Vol Surface":
    st.title("Implied Volatility Surface")
    st.markdown("Interactive 3D volatility surface showing how IV varies by strike and expiry. "
                "Demonstrates the vol smile, negative skew, and term structure of volatility.")
    st.divider()

    col_ctrl, col_explain = st.columns([1, 2])

    with col_ctrl:
        st.subheader("Surface Parameters")
        spot        = st.slider("Spot Price",       80.0, 120.0, 100.0, 1.0)
        atm_vol     = st.slider("ATM Vol (%)",       5.0,  60.0,  20.0, 0.5) / 100
        skew        = st.slider("Skew Factor",       0.0,   0.5,   0.15, 0.01,
                                  help="Higher = more negative skew (put wing elevated)")
        smile       = st.slider("Smile Curvature",   0.0,   0.4,   0.12, 0.01,
                                  help="Higher = wider smile")
        term_str    = st.slider("Term Structure",    0.0,   0.15,  0.04, 0.005,
                                  help="Upward slope of vol by expiry")

    with col_explain:
        st.info("""
        **Reading the Vol Surface:**

        - **Smile**: IV curves upward away from ATM — market prices tail risk
        - **Skew**: Left wing (OTM puts) higher than right wing (OTM calls) — crash insurance premium
        - **Term Structure**: IV rises with expiry — uncertainty grows over time

        In real markets, traders arbitrage between model-implied and observed IV.
        """)

    with st.spinner("Generating vol surface..."):
        surface = generate_vol_surface(
            S=spot, atm_vol=atm_vol,
            skew_factor=skew, smile_factor=smile, term_factor=term_str, seed=42
        )

    # 3D Surface
    strikes  = surface.index.values.astype(float)
    expiries = surface.columns.values.astype(float)
    Z        = surface.values.astype(float) * 100   # to %

    fig_3d = go.Figure(data=[go.Surface(
        x=expiries, y=strikes, z=Z,
        colorscale="Viridis",
        colorbar=dict(title="IV (%)", x=1.01),
        contours={
            "z": {"show": True, "usecolormap": True,
                  "highlightcolor": "white", "project_z": True}
        }
    )])
    fig_3d.update_layout(
        title="Implied Volatility Surface (3D)",
        scene=dict(
            xaxis_title="Expiry (years)",
            yaxis_title="Strike",
            zaxis_title="IV (%)",
            camera=dict(eye=dict(x=1.8, y=-1.8, z=0.9)),
            bgcolor="#0E1117"
        ),
        template="plotly_dark",
        height=520,
        margin=dict(l=0, r=0, t=50, b=0)
    )
    st.plotly_chart(fig_3d, use_container_width=True)

    st.divider()

    # Smile slices
    col_smile, col_term = st.columns(2)

    with col_smile:
        st.subheader("Vol Smile by Expiry")
        fig_smile = go.Figure()
        palette   = px.colors.qualitative.Plotly
        for i, exp in enumerate(expiries):
            fig_smile.add_trace(go.Scatter(
                x=strikes, y=surface[exp].values * 100,
                mode="lines+markers", marker=dict(size=4),
                line=dict(color=palette[i % len(palette)], width=1.5),
                name=f"T={exp:.2f}yr"
            ))
        fig_smile.add_vline(x=spot, line_color="white", line_dash="dash",
                             annotation_text="Spot")
        fig_smile.update_layout(
            template="plotly_dark", height=310,
            xaxis_title="Strike", yaxis_title="IV (%)",
            title="Volatility Smile (each line = one expiry)"
        )
        st.plotly_chart(fig_smile, use_container_width=True)

    with col_term:
        st.subheader("Term Structure at ATM")
        atm_idx   = len(strikes) // 2
        ts_values = surface.iloc[atm_idx].values * 100
        fig_ts    = go.Figure()
        fig_ts.add_trace(go.Scatter(
            x=expiries, y=ts_values, mode="lines+markers",
            line=dict(color="#FFC107", width=2),
            marker=dict(size=7), name="ATM IV"
        ))
        fig_ts.add_hline(y=atm_vol * 100, line_color="white",
                          line_dash="dash",
                          annotation_text=f"Base: {atm_vol*100:.1f}%")
        fig_ts.update_layout(
            template="plotly_dark", height=310,
            xaxis_title="Expiry (years)", yaxis_title="IV (%)",
            title=f"Vol Term Structure at K={strikes[atm_idx]:.1f}"
        )
        st.plotly_chart(fig_ts, use_container_width=True)

    # Option chain table
    st.divider()
    st.subheader("Option Chain (6-month expiry)")
    exp_col = surface.columns[3] if len(surface.columns) > 3 else surface.columns[-1]
    chain_data = []
    for k in strikes[::2]:   # every other strike to keep table compact
        iv      = float(surface.loc[k, exp_col])
        c_price = bs_call(spot, k, float(exp_col), 0.05, iv)
        p_price = bs_put (spot, k, float(exp_col), 0.05, iv)
        mness   = "ATM" if abs(k - spot) < 2 else ("ITM" if k < spot else "OTM")
        chain_data.append({
            "Strike": k, "Moneyness": mness,
            "IV (%)": round(iv * 100, 2),
            "Call Price": round(c_price, 4),
            "Put Price" : round(p_price, 4),
        })
    st.dataframe(pd.DataFrame(chain_data), use_container_width=True, hide_index=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.sidebar.divider()
st.sidebar.caption("QuantCore")
