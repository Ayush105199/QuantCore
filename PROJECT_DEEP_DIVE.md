# QuantCore — Project Deep Dive

Complete technical documentation: why each system was built, how it works,
what every formula means, and what it demonstrates to an interviewer.

---

## 1. Why These Four Systems?

HFT firms evaluate candidates across three dimensions:

**1. Market Microstructure** — Do you understand how exchanges actually work?
  Not just "price goes up and down" but: what is an order book, what is a
  fill, what is adverse selection, what is the bid-ask spread and why does
  it exist?

**2. Mathematical Depth** — Can you implement quantitative models from scratch?
  Not just call scipy.stats.norm.cdf — implement the CDF yourself.
  Not just call QuantLib — derive the Greeks analytically.

**3. Systems Thinking** — Can you build something that works end-to-end?
  A model that only runs in a Jupyter notebook is not useful.
  A working simulation with metrics, charts, and a UI is.

QuantCore addresses all three:
- Order Book: microstructure layer
- Options: mathematical depth
- Market Making: optimal control + risk management
- Dashboard: end-to-end system

---

## 2. Phase 1: L2 Order Book — Deep Dive

### What Is an Order Book?

Every exchange (NSE, NYSE, Binance) maintains a central limit order book (CLOB).
It stores all resting limit orders from all participants, sorted by price.

- **Bid side**: all buyers, sorted descending (best bid = highest)
- **Ask side**: all sellers, sorted ascending (best ask = lowest)

When a new market order arrives, it matches against the best resting opposite-side orders.
This is price-time priority (FIFO): at the same price, earlier orders fill first.

### L1 vs L2 vs L3 Data

- **L1**: just best bid + best ask + sizes. This is what Zerodha shows you.
- **L2**: all price levels with total quantity. HFTs pay crores for this.
- **L3**: individual orders at each level. Only the exchange itself has this.

QuantCore implements a full L2 book.

### Order Flow Imbalance (OFI)

OFI is the single most important microstructure signal for short-horizon prediction.

```
OFI_t = (delta_BidQty - delta_AskQty) / (|delta_BidQty| + |delta_AskQty| + epsilon)

Where:
  delta_BidQty = change in total bid-side quantity vs previous snapshot
  delta_AskQty = change in total ask-side quantity vs previous snapshot
  epsilon      = 1e-9 (prevents division by zero)

Range: [-1, +1]
  +1 = pure buy pressure (bids added, asks removed)
  -1 = pure sell pressure (asks added, bids removed)
   0 = balanced order flow
```

**Why OFI predicts price:**
When many buyers are adding limit orders and sellers are cancelling, the book
is becoming bid-heavy. A market sell order will find thin ask liquidity and
push price up. OFI captures this imbalance before the price move occurs.

Academic reference: Cont, Kukanov, Stoikov (2014), "The Price Impact of Order Book
Events", Journal of Financial Econometrics.

### Implementation Details

The OrderBook class uses:
- `dict[price -> PriceLevel]` for both bid and ask sides
- Plain Python `sorted(keys)` for read operations — O(n_levels) but sufficient for simulation
- `_id_map: dict[order_id -> (side, price)]` for O(1) cancellation by ID
- `deque(maxlen=50)` for rolling OFI window

In production C++ HFT systems, this is a red-black tree (O(log n) insert/delete)
or a flat array indexed by price tick (O(1) for fixed tick sizes).

### Order Flow Generator

Realistic synthetic LOB feed using:
- **Poisson arrivals**: inter-arrival times are exponentially distributed. λ = 6 means avg 6 orders/step.
- **GBM mid-price**: `mid *= exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)` — standard financial model.
- **Exponential price clustering**: most orders arrive near the top of book, fewer arrive deep.
- **Log-normal order sizes**: reflects real-world distribution of order quantities.

---

## 3. Phase 2: Options Pricing — Deep Dive

### Black-Scholes Assumptions

1. Underlying follows GBM: `dS = mu*S*dt + sigma*S*dW`
2. Constant volatility sigma
3. Constant risk-free rate r
4. No dividends
5. European exercise only
6. No transaction costs

In reality: volatility is not constant (hence the vol surface), returns have fat tails,
dividends exist. But BS remains the industry-standard benchmark. Everything is quoted
in terms of BS IV.

### The Black-Scholes Formula Derivation (sketch)

Starting from GBM and applying Ito's Lemma to the option price V(S,t):

```
dV = (dV/dt + 0.5*sigma^2*S^2*d^2V/dS^2)*dt + (dV/dS)*dS

Delta-hedging argument: form portfolio Pi = V - Delta*S
  dPi = dV - Delta*dS = (dV/dt + 0.5*sigma^2*S^2*d^2V/dS^2)*dt

Since Pi is risk-free, dPi = r*Pi*dt:
  dV/dt + 0.5*sigma^2*S^2*d^2V/dS^2 + r*S*dV/dS - r*V = 0

This is the Black-Scholes PDE. Solving with boundary condition V(S,T) = max(S-K,0):

  C = S*N(d1) - K*exp(-rT)*N(d2)

where:
  d1 = [ln(S/K) + (r + 0.5*sigma^2)*T] / (sigma*sqrt(T))
  d2 = d1 - sigma*sqrt(T)
  N(x) = standard normal CDF
```

### The CDF Approximation

QuantCore uses Abramowitz & Stegun (1964) approximation 26.2.17:

```python
def _norm_cdf(x):
    if x >= 0:
        t = 1 / (1 + 0.2316419 * x)
        poly = t*(0.319381530 + t*(-0.356563782 + t*(1.781477937
                     + t*(-1.821255978 + t*1.330274429))))
        return 1 - pdf(x) * poly
    else:
        return 1 - _norm_cdf(-x)
```

Maximum error: 7.5e-8. This is the same level of precision used in many
production trading systems. It avoids calling `math.erfc()` and is faster
on older CPUs.

### The Five Greeks — Full Explanation

**Delta (dC/dS)**
```
Call: N(d1)       Put: N(d1) - 1
```
Rate of change of option price with respect to spot price.
- ATM call delta ≈ 0.5 (50 paise gain per Rs1 spot move)
- Deep ITM call delta ≈ 1.0 (moves 1:1 with underlying)
- Deep OTM call delta ≈ 0.0 (barely reacts)

Use: Delta-neutral hedging. To hedge a long call (delta=0.6), short 0.6 shares.
Options desks rebalance delta hedges continuously ("delta hedging").

**Gamma (d^2C/dS^2 = d(Delta)/dS)**
```
N'(d1) / (S * sigma * sqrt(T))   [same for calls and puts]
```
Curvature of option price curve. Rate of change of delta.
- Highest near ATM and near expiry
- "Gamma scalping": buy gamma, delta-hedge frequently → earn from realized vol

Use: Gamma risk is the primary concern on options desks. A gamma-short position
loses money in volatile markets even if delta-hedged.

**Theta (dC/dt) — Time Decay**
```
Call: -[S*N'(d1)*sigma/(2*sqrt(T))] - r*K*exp(-rT)*N(d2)
Put:  -[S*N'(d1)*sigma/(2*sqrt(T))] + r*K*exp(-rT)*N(-d2)
Divided by 365 for daily theta.
```
Almost always negative — options lose value every day that passes.
- ATM short-dated options have the steepest theta

Use: Selling options (writing calls/puts) earns theta. Iron condors,
covered calls, and cash-secured puts are theta-positive strategies.

**Vega (dC/d(sigma))**
```
S * N'(d1) * sqrt(T)   [same for calls and puts]
Divided by 100 for per-1% vol change.
```
Sensitivity to implied volatility changes. Longer-dated options have more vega.

Use: When IV spikes (market fear), long options gain (positive vega exposure).
Volatility traders specifically trade vega — buying options when IV is low,
selling when IV is elevated.

**Rho (dC/dr)**
```
Call: K*T*exp(-rT)*N(d2)
Put: -K*T*exp(-rT)*N(-d2)
Divided by 100 for per-1% rate change.
```
Sensitivity to interest rate changes. Smallest Greek in practice for short-dated options.
Matters for long-dated LEAPS and deep ITM options.

### Implied Volatility — Newton-Raphson

IV cannot be solved analytically from BS — we must invert numerically.

Newton-Raphson iteration:
```
sigma_{n+1} = sigma_n - f(sigma_n) / f'(sigma_n)
            = sigma_n - (BS_price(sigma_n) - market_price) / vega(sigma_n)
```

Convergence: quadratic — errors square each iteration. Typically 4-6 iterations.
Initial guess: Brenner-Subrahmanyam (1988) approximation for near-ATM options.
Edge cases handled: near-zero vega (deep ITM/OTM or near expiry), no-arbitrage bounds.

### Monte Carlo Pricing

```
GBM discretization (Euler-Maruyama):
  S_{t+dt} = S_t * exp((r - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
  Z ~ N(0,1)

Terminal payoff:
  Call: max(S_T - K, 0)
  Put:  max(K - S_T, 0)

Price = exp(-rT) * E[payoff]
      ≈ exp(-rT) * (1/N) * sum(payoffs)

Variance reduction: Antithetic variates
  For each Z, also simulate -Z.
  Price = exp(-rT) * mean([payoff(Z), payoff(-Z)])
  Reduces variance by ~50% for same computational cost.
```

MC is fully vectorized: `np.random.standard_normal((n_paths//2, n_steps))` — 
100k paths in under 2 seconds.

### The Volatility Surface

If BS assumptions held, all options on the same underlying would have the same IV.
They don't. The surface of IVs across strikes and expiries has three features:

1. **Smile**: IV higher for OTM options than ATM — market prices fat tails
2. **Negative Skew**: OTM puts have higher IV than OTM calls — crash insurance
3. **Term Structure**: IV generally rises with expiry — uncertainty grows over time

QuantCore's parametric surface:
```
IV(K, T) = atm_vol
           + smile_factor * (K/S - 1)^2     [symmetric smile]
           - skew_factor  * (K/S - 1)        [negative skew]
           + term_factor  * sqrt(T)           [term structure]
           + noise                            [realistic imperfections]
```

---

## 4. Phase 3: Market Making — Deep Dive

### What Is a Market Maker?

A market maker simultaneously posts a bid (buy) and ask (sell) quote.
When both sides fill, they earn the bid-ask spread.

Revenue model:
```
If bid = 99.90 and ask = 100.10:
  Buy from seller at 99.90
  Sell to buyer at 100.10
  Profit: 0.20 per round-trip
```

The problem: They accumulate inventory.
If only buyers arrive (trending market), the MM keeps selling → grows short.
If price continues rising, their short position loses money faster than spread earned.

This is **adverse selection**: informed traders (HFTs, institutions) know price will
move and trade against the MM. Uninformed retail flow is the MM's bread and butter.

### The Avellaneda-Stoikov Model (2008)

The stochastic control problem:
- Maximize expected utility of terminal wealth
- Subject to: GBM price dynamics, Poisson order arrivals

The solution has two key formulas:

**Reservation Price:**
```
r*(s, q, t) = s - q * gamma * sigma^2 * (T - t)

Where:
  s     = current mid price
  q     = inventory (+ = long, - = short)
  gamma = risk aversion coefficient
  sigma = volatility
  T-t   = remaining trading horizon
```

When q > 0 (long inventory): r* < s → quotes shift DOWN → easier to sell → inventory reduces
When q < 0 (short): r* > s → quotes shift UP → easier to buy → inventory reduces
As T-t → 0: inventory penalty vanishes → quotes revert to symmetric

**Optimal Spread:**
```
delta* = gamma*sigma^2*(T-t) + (2/gamma)*ln(1 + gamma/kappa)

Component 1: gamma*sigma^2*(T-t)
  Inventory risk premium — wider when vol is high or time horizon is long

Component 2: (2/gamma)*ln(1 + gamma/kappa)
  Adverse selection compensation — the MM's edge over informed flow
  kappa controls how quickly fill probability decays with distance from mid
```

**Fill Probability:**
```
P(fill at depth d in time dt) = A * exp(-kappa * d) * dt

Where:
  d     = |quote_price - mid|  (how far from mid we're quoting)
  A     = market order arrival intensity
  kappa = depth sensitivity (higher = fills drop off faster with distance)
```

This is a Poisson process model: market orders arrive at rate A,
and probability of hitting our quote decays exponentially with distance.

### Why AS Beats Naive in the Simulation

Naive MM:
- Fixed spread, symmetric quotes, ignores inventory
- In trending markets: keeps buying as price falls (or selling as price rises)
- Inventory grows without bound until hit limit → forced to stop quoting
- Large unrealized P&L loss from inventory

AS Model:
- Shifts quotes based on inventory — naturally mean-reverts position
- Widens spread when inventory is extreme — takes less risk when already exposed
- Stops quoting (or quotes very wide) near time horizon end — reduces residual risk
- Result: lower max inventory, more consistent P&L, higher Sharpe ratio

---

## 5. Architecture and Design Decisions

### Why Pure Python (not C++)?

For a portfolio project, clarity of implementation is more important than speed.
Every formula is readable. Every step is commented. An interviewer can follow the logic.
In production: order books run in C++ with lock-free queues and DPDK networking.
Here: the algorithm is identical, just at demonstration speed.

### Why No External Finance Libraries?

To prove the mathematical understanding is real, not copy-pasted.
- No scipy.stats.norm.cdf → implemented with A&S approximation
- No QuantLib → BS and Greeks derived from first principles
- No backtrader → simulation loop written from scratch

### Why Streamlit for the Dashboard?

- Zero HTML/CSS/JS needed — pure Python
- Sliders, radio buttons, metrics tiles, charts all built-in
- `@st.cache_data` prevents recomputing on every widget interaction
- `st.session_state` persists simulation results across slider changes
- Renders Plotly charts natively in browser (no file I/O)

### Why Plotly Instead of Matplotlib in the Dashboard?

- Interactive (zoom, hover, pan) — much better for 3D vol surface
- `go.Surface` renders a proper 3D interactive chart
- No display backend issues on Windows (matplotlib can fail headless)
- Renders directly in Streamlit without savefig

### Data Flow Through the Project

```
[No external data needed — everything is generated in-memory]

Phase 1: OrderFlowGenerator → OrderBook → Snapshot DataFrame
Phase 2: Parameters (S,K,T,r,sigma) → price_option() → OptionResult
         Parametric model → generate_vol_surface() → DataFrame
         GBM paths → mc_price_option() → dict
Phase 3: ASParameters → AvellanedaStoikov → (bid, ask) quotes
         MMSimulator → SimulationResult → metrics dict
Phase 4: Dashboard → imports all phases → renders in Streamlit
```

---

## 6. Key Academic References

1. **Black, F. & Scholes, M. (1973)** — "The Pricing of Options and Corporate Liabilities"
   Journal of Political Economy. The foundational options pricing paper.

2. **Avellaneda, M. & Stoikov, S. (2008)** — "High-frequency trading in a limit order book"
   Quantitative Finance. The market making model implemented in Phase 3.

3. **Cont, R., Kukanov, A., Stoikov, S. (2014)** — "The Price Impact of Order Book Events"
   Journal of Financial Econometrics. The OFI predictive signal in Phase 1.

4. **Abramowitz, M. & Stegun, I. (1964)** — "Handbook of Mathematical Functions"
   The CDF approximation used in the BS implementation.

5. **Brenner, M. & Subrahmanyam, M. (1988)** — "A Simple Formula to Compute the Implied
   Standard Deviation" Financial Analysts Journal. The IV initial guess formula.

---

## 7. Interview Preparation — Role-Specific Talking Points

### Quant Developer Role
- "I implemented price-time priority FIFO matching in the order book"
- "OFI computation uses a rolling deque — O(1) append, O(1) read of previous snapshot"
- "The N-R IV solver has quadratic convergence and handles degenerate cases: zero vega, out-of-bounds"
- "Monte Carlo is fully vectorized — np.random.standard_normal((n//2, steps)) — no Python loops"
- "The AS simulator runs 5000 steps in milliseconds because all operations are numpy-native"

### Quant Researcher Role
- "OFI has documented short-term predictive power — Cont et al. (2014) show cross-impact on adjacent prices"
- "The vol surface has negative skew — consistent with the crash risk premium literature"
- "I validated MC against BS analytically — differences are within 2 standard errors at 100k paths"
- "AS's gamma parameter corresponds to the risk aversion in a CRRA utility function"

### Trader Role
- "Theta explains why I'd sell options going into a weekend — 3 days of decay for 2 days of premium"
- "Gamma risk is why market makers widen spreads during volatile openings"
- "OFI above 0.5 is my signal that a large institutional order is sweeping the book"
- "The AS reservation price is exactly what I do mentally when I'm long and quote slightly below mid"

---

## 8. What to Build Next (Forward Vision)

These are the next logical extensions — knowing them shows depth in interviews:

1. **Longstaff-Schwartz MC** — American option pricing with early exercise boundary
2. **Heston Stochastic Vol** — Let sigma itself follow a mean-reverting process
3. **Queue Position Modeling** — Probability of fill vs. position in FIFO queue
4. **Glosten-Milgrom Model** — Adverse selection with informed trader probability
5. **VWAP Execution** — Optimal order splitting to minimize market impact
6. **Tick Data Calibration** — Fit AS parameters to real NSE tick data (needs internet)

---

Built entirely in Python. No paid data. No Bloomberg. No QuantLib.
Every formula derived. Every line readable.
