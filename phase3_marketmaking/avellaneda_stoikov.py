"""
Avellaneda-Stoikov Market Making Model (2008)
----------------------------------------------
Reference: "High-frequency trading in a limit order book"
           Marco Avellaneda & Sasha Stoikov, Quantitative Finance, 2008.

The problem:
  A market maker posts bid and ask quotes continuously.
  They earn the bid-ask spread when both sides fill.
  But they accumulate inventory risk — if they buy 1000 shares and price drops,
  they lose more than they earned in spread.

  Key insight: The optimal quotes depend on current INVENTORY.
  If you are long 500 shares, you want to SELL more — so you lower your ask
  to attract sellers. This shifts your quotes to reduce inventory risk.

The solution (Avellaneda-Stoikov):
  1. Define a "reservation price" r* — the fair value adjusted for inventory:
       r*(s, q, t) = s - q * gamma * sigma^2 * (T - t)
     where:
       s     = current mid price
       q     = inventory (positive=long, negative=short)
       gamma = risk aversion coefficient
       sigma = asset volatility
       T - t = time remaining in trading horizon

  2. Compute optimal spread delta*:
       delta* = gamma*sigma^2*(T-t) + (2/gamma)*ln(1 + gamma/kappa)
     where kappa controls how quickly fill probability decays with distance.

  3. Post quotes:
       bid* = r* - delta*/2
       ask* = r* + delta*/2

  Why this works:
    - When q > 0 (long): r* < s → bid moves down, ask moves down
      → easier to sell, harder to buy → inventory naturally reduces
    - When q = 0: symmetric quotes around mid
    - Larger gamma = more risk-averse = wider spreads = slower fills
"""

import math
from dataclasses import dataclass
from typing import Tuple


@dataclass
class ASParameters:
    """
    Avellaneda-Stoikov model parameters.

    gamma  : risk aversion [0.01 = almost risk-neutral, 1.0 = very risk-averse]
             Higher gamma = wider spreads, slower trading, more conservative
    sigma  : asset price volatility per unit time
             (in price units per sqrt(second) for intraday, or annualized)
    kappa  : order book depth / market order arrival sensitivity
             Higher kappa = fills become less likely as you move away from mid
    T      : trading horizon (normalized 0 to 1)
    A      : market order arrival intensity (orders per unit time)
             Higher A = more flow = fills more likely
    max_inv: maximum allowed inventory before halting quotes
    """
    gamma   : float = 0.1
    sigma   : float = 2.0
    kappa   : float = 1.5
    T       : float = 1.0
    A       : float = 140.0
    max_inv : int   = 500


class AvellanedaStoikov:
    """
    Full implementation of the Avellaneda-Stoikov market making model.

    Provides:
      - Reservation price (inventory-adjusted mid)
      - Optimal spread
      - Optimal bid/ask quotes
      - Fill probability per time step
    """

    def __init__(self, params: ASParameters = None):
        self.p = params or ASParameters()

    def reservation_price(self, mid: float, inventory: int, t: float) -> float:
        """
        The market maker's subjective fair value, adjusted for inventory risk.

        r*(s, q, t) = s - q * gamma * sigma^2 * (T - t)

        Intuition:
          - q=0: reservation price = mid (symmetric)
          - q>0 (long): reservation price < mid (want to sell cheaper to reduce position)
          - q<0 (short): reservation price > mid (want to buy higher to reduce position)
          - As T-t → 0: reservation price → mid (inventory risk vanishes at horizon end)
        """
        time_remaining = max(self.p.T - t, 0.0)
        return mid - inventory * self.p.gamma * (self.p.sigma ** 2) * time_remaining

    def optimal_spread(self, t: float) -> float:
        """
        Optimal total bid-ask spread.

        delta* = gamma*sigma^2*(T-t) + (2/gamma)*ln(1 + gamma/kappa)

        Two components:
          1. gamma*sigma^2*(T-t)   : inventory risk component
             → larger when vol is high or time horizon is long
          2. (2/gamma)*ln(1+gamma/kappa) : adverse selection / fill-rate component
             → represents the market maker's edge over uninformed flow

        Note: Spread narrows as T-t → 0. Market maker quotes tighter near close.
        """
        time_remaining = max(self.p.T - t, 0.0)
        inv_risk  = self.p.gamma * (self.p.sigma ** 2) * time_remaining
        info_edge = (2.0 / self.p.gamma) * math.log(1.0 + self.p.gamma / self.p.kappa)
        return inv_risk + info_edge

    def optimal_quotes(self, mid: float, inventory: int,
                       t: float) -> Tuple[float, float]:
        """
        Compute optimal bid and ask quotes.

        bid* = r*(s, q, t) - delta*/2
        ask* = r*(s, q, t) + delta*/2

        Returns: (bid_price, ask_price)

        If inventory exceeds max_inv threshold, returns None, None
        (market maker stops quoting — too much risk).
        """
        if abs(inventory) >= self.p.max_inv:
            return None, None

        r_star   = self.reservation_price(mid, inventory, t)
        d_star   = self.optimal_spread(t)
        half_d   = d_star / 2.0

        bid = r_star - half_d
        ask = r_star + half_d

        return max(bid, 0.001), max(ask, bid + 0.001)

    def fill_probability(self, quote_price: float, mid: float,
                          side: str, dt: float) -> float:
        """
        Probability that a limit order gets filled in time dt.

        P(fill) = A * exp(-kappa * depth) * dt

        where depth = |quote_price - mid|

        Intuition:
          - Quotes at mid: filled almost instantly (depth=0)
          - Quotes 1 tick away: lower fill probability
          - Exponential decay — each tick further halves the fill probability
          - Larger A = more market order flow = higher fill prob overall
        """
        depth = abs(quote_price - mid)
        prob  = self.p.A * math.exp(-self.p.kappa * depth) * dt
        return min(1.0, max(0.0, prob))

    def describe(self) -> str:
        """Print model parameters in a readable format."""
        sample_spread = self.optimal_spread(t=0.5)
        return (
            f"\n  Avellaneda-Stoikov Parameters:"
            f"\n    gamma    : {self.p.gamma}  (risk aversion)"
            f"\n    sigma    : {self.p.sigma}  (volatility)"
            f"\n    kappa    : {self.p.kappa}  (order book depth)"
            f"\n    T        : {self.p.T}  (horizon)"
            f"\n    A        : {self.p.A}  (flow intensity)"
            f"\n    max_inv  : {self.p.max_inv}"
            f"\n  Sample spread (t=0.5T) : {sample_spread:.4f}"
        )


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("="*55)
    print("  AVELLANEDA-STOIKOV MARKET MAKING MODEL")
    print("="*55)

    model = AvellanedaStoikov(ASParameters(gamma=0.1, sigma=2.0, kappa=1.5))
    print(model.describe())

    mid = 100.0
    print(f"\n  Quote behavior as inventory changes (mid={mid}, t=0.5T):")
    print(f"  {'Inventory':>12}  {'Reservation':>14}  {'Bid':>10}  {'Ask':>10}  {'Spread':>10}")
    print("  " + "-"*62)
    for inv in [-200, -100, -50, 0, 50, 100, 200]:
        r_star        = model.reservation_price(mid, inv, t=0.5)
        bid, ask      = model.optimal_quotes(mid, inv, t=0.5)
        spread        = (ask - bid) if bid else 0
        bid_s         = f"{bid:.4f}" if bid else "HALTED"
        ask_s         = f"{ask:.4f}" if ask else "HALTED"
        spread_s      = f"{spread:.4f}" if bid else "HALTED"
        print(f"  {inv:>12}  {r_star:>14.4f}  {bid_s:>10}  {ask_s:>10}  {spread_s:>10}")

    print("\n  Spread evolution over time (zero inventory):")
    print(f"  {'t/T':>6}  {'Spread':>10}")
    for t_frac in [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0]:
        spread = model.optimal_spread(t=t_frac)
        print(f"  {t_frac:>6.2f}  {spread:>10.4f}")
