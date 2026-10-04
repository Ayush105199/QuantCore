"""
Order Flow Generator
---------------------
Generates a realistic synthetic stream of limit orders using:
  - Poisson process for order arrival times
  - GBM (Geometric Brownian Motion) for mid-price evolution
  - Log-normal distribution for order sizes
  - Price clustering around mid (orders placed near the market)

This simulates what a real exchange feed looks like before it hits
the HFT's market data handler.
"""

import numpy as np
import pandas as pd
import uuid
from dataclasses import dataclass, field
from typing import List, Optional

from phase1_orderbook.order import Order, Side, make_order
from phase1_orderbook.orderbook import OrderBook


@dataclass
class GeneratorConfig:
    mid_price     : float = 100.0    # starting mid price
    tick_size     : float = 0.01     # minimum price increment
    lambda_bid    : float = 5.0      # avg bid orders per step
    lambda_ask    : float = 5.0      # avg ask orders per step
    lambda_cancel : float = 3.0      # avg cancellations per step
    sigma         : float = 0.02     # GBM volatility per step
    drift         : float = 0.0001   # slight upward drift
    price_levels  : int   = 10       # how many levels away from mid to quote
    qty_mean      : float = 200.0    # avg order size (log-normal)
    qty_std       : float = 100.0    # std of order size


class OrderFlowGenerator:
    """
    Simulates a realistic LOB order stream.

    At each time step:
      1. Evolve mid price via GBM
      2. Generate N_bid new bid orders ~ Poisson(lambda_bid)
      3. Generate N_ask new ask orders ~ Poisson(lambda_ask)
      4. Cancel K random existing orders ~ Poisson(lambda_cancel)
      5. Record book snapshot: mid, spread, OFI, bid_vol, ask_vol
    """

    def __init__(self, config: GeneratorConfig = None, seed: int = 42):
        self.cfg    = config or GeneratorConfig()
        self.mid    = self.cfg.mid_price
        self.rng    = np.random.default_rng(seed)
        self._active_order_ids: List[str] = []

    def _evolve_mid(self) -> float:
        """GBM step: mid = mid * exp((drift - 0.5*sigma^2) + sigma*Z)"""
        z          = self.rng.standard_normal()
        log_return = (self.cfg.drift - 0.5 * self.cfg.sigma ** 2) + self.cfg.sigma * z
        self.mid   = max(self.mid * np.exp(log_return), 0.01)
        return self.mid

    def _random_price(self, side: Side) -> float:
        """
        Generate a limit price near the mid.
        Bids below mid, asks above mid.
        Price offset drawn from exponential distribution (most orders near top of book).
        """
        offset = self.rng.exponential(scale=self.cfg.tick_size * 3)
        offset = max(self.cfg.tick_size, round(offset / self.cfg.tick_size) * self.cfg.tick_size)

        if side == Side.BID:
            price = self.mid - offset
        else:
            price = self.mid + offset

        return round(max(price, self.cfg.tick_size), 4)

    def _random_qty(self) -> int:
        """Log-normal order size. Ensures always positive."""
        mu    = np.log(self.cfg.qty_mean ** 2 /
                       np.sqrt(self.cfg.qty_std ** 2 + self.cfg.qty_mean ** 2))
        sigma = np.sqrt(np.log(1 + (self.cfg.qty_std / self.cfg.qty_mean) ** 2))
        return max(1, int(self.rng.lognormal(mu, sigma)))

    MAX_ACTIVE_ORDERS = 80   # cap total resting orders to prevent stale book

    def _purge_stale_orders(self, book: OrderBook) -> None:
        """
        Cancel oldest orders when pool gets too large, AND cancel any orders
        that are now on the wrong side of mid (would create crossed book).
        """
        crossed = []
        for oid in list(self._active_order_ids):
            if oid in book._id_map:
                side, price = book._id_map[oid]
                if (side.value == "bid" and price >= self.mid) or \
                   (side.value == "ask" and price <= self.mid):
                    crossed.append(oid)

        for oid in crossed:
            self._active_order_ids.remove(oid)
            book.cancel_order(oid)

        # Also trim oldest orders if pool is too large
        while len(self._active_order_ids) > self.MAX_ACTIVE_ORDERS:
            oid = self._active_order_ids.pop(0)
            book.cancel_order(oid)

    def step(self, book: OrderBook) -> dict:
        """
        Execute one simulation step.
        Returns a snapshot dict: mid, spread, ofi, bid_vol, ask_vol, n_new_orders
        """
        self._evolve_mid()
        self._purge_stale_orders(book)

        n_bids    = self.rng.poisson(self.cfg.lambda_bid)
        n_asks    = self.rng.poisson(self.cfg.lambda_ask)
        n_cancels = self.rng.poisson(self.cfg.lambda_cancel)

        # Add new orders
        for _ in range(n_bids):
            o = make_order(Side.BID, self._random_price(Side.BID), self._random_qty())
            book.add_order(o)
            self._active_order_ids.append(o.order_id)

        for _ in range(n_asks):
            o = make_order(Side.ASK, self._random_price(Side.ASK), self._random_qty())
            book.add_order(o)
            self._active_order_ids.append(o.order_id)

        # Cancel random existing orders (simulates HFT quote stuffing / normal cancels)
        for _ in range(min(n_cancels, len(self._active_order_ids))):
            idx = self.rng.integers(0, len(self._active_order_ids))
            oid = self._active_order_ids.pop(idx)
            book.cancel_order(oid)

        snap = book.snapshot()
        snap["step_mid"]      = self.mid
        snap["n_new_bids"]    = n_bids
        snap["n_new_asks"]    = n_asks
        snap["n_cancels"]     = n_cancels
        return snap

    def run_simulation(self, n_steps: int = 500,
                       book: Optional[OrderBook] = None) -> pd.DataFrame:
        """
        Run a full simulation for n_steps and return a time-series DataFrame.

        Columns: mid_price, best_bid, best_ask, spread, spread_bps, ofi,
                 total_bid_qty, total_ask_qty, n_new_bids, n_new_asks
        """
        if book is None:
            book = OrderBook()

        records = []
        for step in range(n_steps):
            snap = self.step(book)
            snap["step"] = step
            records.append(snap)

        df = pd.DataFrame(records).set_index("step")
        return df


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("="*55)
    print("  ORDER FLOW GENERATOR — Simulation Demo")
    print("="*55 + "\n")

    cfg  = GeneratorConfig(mid_price=100.0, lambda_bid=6, lambda_ask=6,
                           lambda_cancel=4, sigma=0.015)
    gen  = OrderFlowGenerator(config=cfg, seed=42)
    book = OrderBook()

    print("Running 300-step simulation...\n")
    history = gen.run_simulation(n_steps=300, book=book)

    print("Final Order Book State:")
    book.print_book(n_levels=5)

    print("Simulation Statistics:")
    print(f"  Steps simulated   : {len(history)}")
    print(f"  Mid price range   : {history['step_mid'].min():.4f} to {history['step_mid'].max():.4f}")
    valid_spread = history['spread'].dropna()
    if len(valid_spread):
        print(f"  Avg spread (bps)  : {history['spread_bps'].dropna().mean():.2f}")
    print(f"  OFI range         : {history['ofi'].min():.3f} to {history['ofi'].max():.3f}")
    print(f"  Total orders added: {history['n_new_bids'].sum() + history['n_new_asks'].sum()}")
    print(f"  Total cancels     : {history['n_cancels'].sum()}")

    # OFI predictive check: does positive OFI lead to price increase?
    history["ofi_signal"]   = (history["ofi"] > 0.3).astype(int) - (history["ofi"] < -0.3).astype(int)
    history["future_return"] = history["step_mid"].pct_change(3).shift(-3)
    strong = history[history["ofi_signal"] != 0].dropna()
    if len(strong) > 10:
        corr = strong["ofi_signal"].corr(strong["future_return"])
        print(f"\n  OFI vs 3-step forward return correlation: {corr:.4f}")
        print(f"  (Positive = OFI predicts price direction)")

    print("\nPhase 1 complete. Run phase2_options/black_scholes.py next.")
