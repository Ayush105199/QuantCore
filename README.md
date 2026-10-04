# QuantCore — HFT Quant Systems Toolkit

> "The only way to fight an algorithm is to build one yourself."

QuantCore is a self-contained Python toolkit implementing four core systems
that quantitative professionals build at HFT firms, hedge funds, and
market makers — all from scratch, with zero internet dependency.

---

## What This Project Demonstrates

| Phase | System | Skill Domain |
|-------|--------|-------------|
| 1 | L2 Order Book Simulator | Market Microstructure, Data Structures |
| 2 | Options Pricing Engine (BS + Greeks + IV + MC) | Derivatives, Math, Stochastic Calculus |
| 3 | Market Making Simulator (Avellaneda-Stoikov) | Optimal Control, Risk Management |
| 4 | Live Streamlit Dashboard | Systems Integration, Visualization |

---

## Tech Stack

| Technology | Used For |
|-----------|----------|
| Python 3.12 | Core language |
| NumPy | Vectorized math, GBM simulation |
| Pandas | Time-series data, result tables |
| Matplotlib | Static analysis charts |
| Plotly | Interactive dashboard charts |
| Streamlit | Web dashboard (pure Python) |
| SciPy | Statistical utilities |
| dataclasses | Clean, typed data models |

All packages pre-installed. No internet required. No configuration needed.

---

## Project Structure

```
QuantCore/
|
|-- phase1_orderbook/
|   |-- order.py                  # Order dataclass + Side/Status enums
|   |-- orderbook.py              # L2 OrderBook engine with OFI analytics
|   |-- order_flow_generator.py   # Poisson-process synthetic order stream
|   └-- ob_visualizer.py          # Depth chart + spread + OFI charts
|
|-- phase2_options/
|   |-- black_scholes.py          # BS pricing + all 5 Greeks from scratch
|   |-- iv_solver.py              # Newton-Raphson IV solver
|   |-- vol_surface.py            # Strike x expiry IV surface generator
|   |-- monte_carlo.py            # GBM Monte Carlo pricer + antithetic variates
|   └-- options_visualizer.py     # Payoff diagrams, Greeks charts, vol surface
|
|-- phase3_marketmaking/
|   |-- avellaneda_stoikov.py     # AS optimal quoting model (2008)
|   |-- naive_mm.py               # Fixed-spread baseline for comparison
|   |-- mm_simulator.py           # Full simulation loop + metrics engine
|   └-- mm_visualizer.py          # P&L, inventory, quote activity charts
|
|-- phase4_dashboard/
|   └-- dashboard.py              # Streamlit app — 4 interactive pages
|
|-- requirements.txt
|-- README.md
└-- PROJECT_DEEP_DIVE.md          # Full technical documentation
```

---

## How to Run

```bash
# Install dependencies (one time)
pip install -r requirements.txt

# Run the full dashboard
streamlit run phase4_dashboard/dashboard.py
# Open browser: http://localhost:8501

# Or run each phase standalone:
python phase1_orderbook/orderbook.py
python phase2_options/black_scholes.py
python phase3_marketmaking/mm_simulator.py
```

---

## Interview Talking Points

**On the Order Book:**
"I implemented a full L2 limit order book with price-time priority queues and
real-time Order Flow Imbalance computation. OFI is a documented short-term
price predictor per Cont, Kukanov, Stoikov (2014)."

**On Options:**
"I implemented Black-Scholes and all 5 Greeks from scratch — including the
Abramowitz-Stegun CDF approximation, a Newton-Raphson IV solver that converges
in 4-6 iterations, and a Monte Carlo pricer with antithetic variates."

**On Market Making:**
"I implemented the full Avellaneda-Stoikov model. The key insight is the
reservation price — shifting quotes based on inventory to mean-revert
position risk. My simulation shows AS outperforms naive fixed-spread
especially in trending markets with toxic flow."

---

See `PROJECT_DEEP_DIVE.md` for full technical documentation.
