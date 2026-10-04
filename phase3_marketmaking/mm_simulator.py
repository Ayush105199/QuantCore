"""
Market Making Simulator
------------------------
Runs a discrete-time simulation of a market maker operating in a
limit order book with stochastic price dynamics.

At each timestep dt:
  1. Mid price evolves via GBM
  2. Market maker computes bid/ask quotes from its model (AS or Naive)
  3. Fill engine: each quote fills with probability P(fill) = A*exp(-kappa*depth)*dt
  4. Toxic flow events: 10% chance per step of a large directional order
     that fills the quote adversely and moves price against inventory
  5. Update: inventory, P&L, spread_earned, mark-to-market

P&L components:
  Spread P&L   : (ask_fill_price - bid_fill_price) per round-trip
  Inventory P&L: inventory * (current_mid - avg_entry_price)
  Total P&L    : spread_earned + unrealized_inventory_pnl
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, List
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from phase3_marketmaking.avellaneda_stoikov import AvellanedaStoikov, ASParameters
from phase3_marketmaking.naive_mm import NaiveMarketMaker, NaiveMMParams


@dataclass
class SimulationResult:
    """All time-series outputs from one simulation run."""
    timestamps    : np.ndarray
    mid_prices    : np.ndarray
    inventories   : np.ndarray
    pnl           : np.ndarray         # total mark-to-market P&L
    spread_earned : np.ndarray         # cumulative spread income
    bid_quotes    : np.ndarray         # posted bid prices (nan if not quoting)
    ask_quotes    : np.ndarray         # posted ask prices (nan if not quoting)
    fills         : pd.DataFrame       # every fill event
    metrics       : dict = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({
            "mid"          : self.mid_prices,
            "inventory"    : self.inventories,
            "pnl"          : self.pnl,
            "spread_earned": self.spread_earned,
            "bid_quote"    : self.bid_quotes,
            "ask_quote"    : self.ask_quotes,
        })


def _compute_metrics(result: SimulationResult, model_name: str) -> dict:
    """Compute summary performance metrics from a simulation result."""
    pnl     = result.pnl
    inv     = result.inventories
    fills   = result.fills
    spread  = result.spread_earned

    # P&L metrics
    final_pnl   = float(pnl[-1])
    pnl_changes = np.diff(pnl)
    sharpe = (pnl_changes.mean() / (pnl_changes.std() + 1e-9)
              * np.sqrt(len(pnl_changes)))

    # Drawdown
    running_max  = np.maximum.accumulate(pnl)
    drawdowns    = pnl - running_max
    max_drawdown = float(drawdowns.min())

    # Fill stats
    n_fills   = len(fills)
    n_bid_fills = len(fills[fills["side"] == "bid"]) if n_fills > 0 else 0
    n_ask_fills = len(fills[fills["side"] == "ask"]) if n_fills > 0 else 0

    # Inventory stats
    avg_inv      = float(np.abs(inv).mean())
    max_inv      = int(np.abs(inv).max())
    inv_risk     = float(np.abs(inv).std())
    n_flat       = int((inv == 0).sum())

    # Spread stats
    final_spread = float(spread[-1]) if len(spread) > 0 else 0.0
    avg_spread   = float(np.diff(spread).mean()) if len(spread) > 1 else 0.0

    return {
        "model"              : model_name,
        "final_pnl"          : round(final_pnl, 4),
        "sharpe_ratio"       : round(sharpe, 4),
        "max_drawdown"       : round(max_drawdown, 4),
        "total_fills"        : n_fills,
        "bid_fills"          : n_bid_fills,
        "ask_fills"          : n_ask_fills,
        "total_spread_earned": round(final_spread, 4),
        "avg_inventory"      : round(avg_inv, 2),
        "max_inventory"      : max_inv,
        "inventory_risk"     : round(inv_risk, 2),
        "steps_flat"         : n_flat,
    }


class MMSimulator:
    """
    Discrete-time market making simulation engine.
    Works with any market maker that implements .optimal_quotes() and .fill_probability().
    """

    def __init__(self,
                 model,
                 mid_price     : float = 100.0,
                 dt            : float = 0.001,
                 n_steps       : int   = 5000,
                 sigma         : float = 0.02,
                 drift         : float = 0.0,
                 toxic_prob    : float = 0.05,
                 toxic_size    : float = 0.10,
                 seed          : int   = 42,
                 quote_size    : int   = 10):
        self.model       = model
        self.mid         = mid_price
        self.dt          = dt
        self.n_steps     = n_steps
        self.sigma       = sigma
        self.drift       = drift
        self.toxic_prob  = toxic_prob
        self.toxic_size  = toxic_size
        self.rng         = np.random.default_rng(seed)
        self.quote_size  = quote_size

    def run(self, model_name: str = "model") -> SimulationResult:
        """
        Execute the full simulation.

        Returns a SimulationResult with all time-series data and metrics.
        """
        n = self.n_steps

        timestamps    = np.arange(n) * self.dt
        mid_prices    = np.zeros(n)
        inventories   = np.zeros(n, dtype=int)
        pnl           = np.zeros(n)
        spread_earned = np.zeros(n)
        bid_quotes    = np.full(n, np.nan)
        ask_quotes    = np.full(n, np.nan)
        fill_records  = []

        mid           = self.mid
        inventory     = 0
        cash          = 0.0    # net cash: sum(sell proceeds) - sum(buy costs)
        cumul_spread  = 0.0    # cumulative spread earned from round-trips

        for step in range(n):
            t = step / n   # normalized time [0, 1]

            # 1. Evolve mid price
            z    = self.rng.standard_normal()
            mid *= np.exp((self.drift - 0.5 * self.sigma ** 2) * self.dt
                          + self.sigma * np.sqrt(self.dt) * z)
            mid  = max(mid, 0.01)
            mid_prices[step] = mid

            # 2. Get quotes
            bid, ask = self.model.optimal_quotes(mid, inventory, t)
            bid_quotes[step] = bid
            ask_quotes[step] = ask

            if bid is None or ask is None:
                # Not quoting — carry forward P&L
                inventories[step]   = inventory
                pnl[step]           = cash + inventory * mid
                spread_earned[step] = cumul_spread
                continue

            # 3. Check fills (bid side = we buy, ask side = we sell)
            bid_fill_prob = self.model.fill_probability(bid, mid, "bid", self.dt)
            ask_fill_prob = self.model.fill_probability(ask, mid, "ask", self.dt)

            bid_filled = self.rng.random() < bid_fill_prob
            ask_filled = self.rng.random() < ask_fill_prob

            # 4. Toxic flow event — large directional order
            is_toxic = self.rng.random() < self.toxic_prob
            if is_toxic:
                toxic_direction = 1 if self.rng.random() < 0.5 else -1
                if toxic_direction > 0:   # price will rise — adverse for shorts
                    bid_filled = True
                    mid *= (1 + self.toxic_size * self.rng.random())
                    mid_prices[step] = mid
                else:                     # price will fall — adverse for longs
                    ask_filled = True
                    mid *= (1 - self.toxic_size * self.rng.random())
                    mid_prices[step] = mid

            # 5. Process fills — update cash and inventory
            if bid_filled:
                cash      -= bid * self.quote_size   # we paid to buy
                inventory += self.quote_size
                fill_records.append({"step": step, "side": "bid",
                                      "price": bid, "qty": self.quote_size,
                                      "inventory_after": inventory})

            if ask_filled:
                cash      += ask * self.quote_size   # we received from selling
                inventory -= self.quote_size
                fill_records.append({"step": step, "side": "ask",
                                      "price": ask, "qty": self.quote_size,
                                      "inventory_after": inventory})

            # Track spread earned (only when both sides filled = round-trip)
            if bid_filled and ask_filled:
                cumul_spread += (ask - bid) * self.quote_size

            # 6. Mark-to-market P&L = realized cash + unrealized inventory value
            total_pnl = cash + inventory * mid

            inventories[step]   = inventory
            pnl[step]           = total_pnl
            spread_earned[step] = cumul_spread

        fills_df = pd.DataFrame(fill_records) if fill_records else pd.DataFrame(
            columns=["step", "side", "price", "qty", "inventory_after"])

        result = SimulationResult(
            timestamps    = timestamps,
            mid_prices    = mid_prices,
            inventories   = inventories,
            pnl           = pnl,
            spread_earned = spread_earned,
            bid_quotes    = bid_quotes,
            ask_quotes    = ask_quotes,
            fills         = fills_df,
        )
        result.metrics = _compute_metrics(result, model_name)
        return result


def compare_strategies(
    n_steps    : int   = 5000,
    mid_price  : float = 100.0,
    sigma      : float = 0.02,
    seed       : int   = 42,
    as_gamma   : float = 0.1,
    as_sigma   : float = 2.0,
) -> dict:
    """
    Run both AS and Naive MM on identical price paths.
    Returns dict with 'as' and 'naive' SimulationResults for comparison.
    """
    as_model    = AvellanedaStoikov(ASParameters(gamma=as_gamma, sigma=as_sigma,
                                                   kappa=1.5, A=140.0))
    naive_model = NaiveMarketMaker(NaiveMMParams(fixed_spread=0.30))

    as_sim    = MMSimulator(as_model,    mid_price=mid_price, n_steps=n_steps,
                            sigma=sigma, seed=seed)
    naive_sim = MMSimulator(naive_model, mid_price=mid_price, n_steps=n_steps,
                            sigma=sigma, seed=seed)   # same seed = same price path

    as_result    = as_sim.run("Avellaneda-Stoikov")
    naive_result = naive_sim.run("Naive Fixed-Spread")

    return {"as": as_result, "naive": naive_result}


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("="*60)
    print("  MARKET MAKING SIMULATOR")
    print("  Avellaneda-Stoikov vs Naive Fixed-Spread")
    print("="*60 + "\n")

    results = compare_strategies(n_steps=5000, seed=42)

    print("  PERFORMANCE COMPARISON:")
    print("  " + "-"*55)

    keys_to_show = ["final_pnl", "sharpe_ratio", "max_drawdown",
                    "total_fills", "total_spread_earned",
                    "avg_inventory", "max_inventory"]

    print(f"  {'Metric':30s}  {'AS Model':>12}  {'Naive MM':>12}")
    print("  " + "-"*58)
    for k in keys_to_show:
        as_v    = results["as"].metrics.get(k, "N/A")
        naive_v = results["naive"].metrics.get(k, "N/A")
        print(f"  {k:30s}  {str(as_v):>12}  {str(naive_v):>12}")

    print("\n  AS model advantages:")
    as_m    = results["as"].metrics
    naive_m = results["naive"].metrics

    pnl_diff  = as_m["final_pnl"] - naive_m["final_pnl"]
    inv_diff  = naive_m["max_inventory"] - as_m["max_inventory"]
    print(f"    P&L advantage       : {pnl_diff:+.4f}")
    print(f"    Inventory reduction : {inv_diff:+d} shares max")
    print(f"    Sharpe improvement  : {as_m['sharpe_ratio'] - naive_m['sharpe_ratio']:+.4f}")
    print("\nPhase 3 simulation complete.")
