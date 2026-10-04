"""
Naive Fixed-Spread Market Maker (Baseline)
-------------------------------------------
Always quotes mid +/- fixed_spread/2, regardless of inventory.
No inventory adjustment. No risk awareness.

Used as a comparison benchmark to show how much better the
Avellaneda-Stoikov model is in trending/volatile markets.
"""

from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class NaiveMMParams:
    fixed_spread : float = 0.20    # total bid-ask spread (in price units)
    max_inventory: int   = 500     # halt quoting above this threshold
    quote_size   : int   = 10      # shares per quote


class NaiveMarketMaker:
    """
    Fixed-spread market maker.
    Quotes symmetrically around mid at all times.
    No inventory awareness — this is the key weakness.

    In a trending market:
      - Keeps buying as price falls (keeps losing)
      - Inventory builds up on the wrong side
      - Eventually hits max_inventory and is forced to stop
    """

    def __init__(self, params: NaiveMMParams = None):
        self.p = params or NaiveMMParams()

    def optimal_quotes(self, mid: float, inventory: int,
                       t: float = 0.0) -> Tuple[Optional[float], Optional[float]]:
        """
        Always returns symmetric quotes around mid.
        Ignores inventory except for the hard limit.

        Returns (None, None) if inventory limit exceeded.
        """
        if abs(inventory) >= self.p.max_inventory:
            return None, None

        half = self.p.fixed_spread / 2.0
        return max(mid - half, 0.001), mid + half

    def fill_probability(self, quote_price: float, mid: float,
                          side: str, dt: float) -> float:
        """
        Naive fixed fill probability.
        Does not model depth or market order flow properly.
        """
        depth = abs(quote_price - mid)
        if depth <= self.p.fixed_spread / 2.0 + 0.001:
            return min(1.0, 0.5 * dt * 100)
        return 0.0
