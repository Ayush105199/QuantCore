"""
L2 Order Book — Core Engine
-----------------------------
The order book is the central data structure of every exchange.
It maintains all resting limit orders, sorted by price-time priority.

L1 data: just best bid + best ask (what retail sees on Zerodha)
L2 data: all price levels with total quantity (what HFTs pay crores for)
L3 data: individual orders at each level (only the exchange itself has this)

This implements a full L2 book with:
  - O(1) order insertion
  - O(1) order cancellation (by ID)
  - O(levels) depth snapshot
  - Real-time OFI, spread, mid-price computation
"""

from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple
import time

from phase1_orderbook.order import Order, Side, OrderStatus, make_order


class PriceLevel:
    """
    All orders resting at a single price point.
    Maintains FIFO (price-time priority): earlier orders fill first.
    """

    def __init__(self, price: float):
        self.price  = price
        self.orders : Dict[str, Order] = {}   # order_id -> Order
        self._queue : deque = deque()          # FIFO insertion order

    def add(self, order: Order) -> None:
        self.orders[order.order_id] = order
        self._queue.append(order.order_id)

    def cancel(self, order_id: str) -> bool:
        if order_id in self.orders:
            self.orders[order_id].cancel()
            del self.orders[order_id]
            return True
        return False

    @property
    def total_quantity(self) -> int:
        return sum(o.remaining_qty for o in self.orders.values() if o.is_active)

    @property
    def order_count(self) -> int:
        return len([o for o in self.orders.values() if o.is_active])

    def is_empty(self) -> bool:
        return self.total_quantity == 0

    def __repr__(self) -> str:
        return f"PriceLevel({self.price:.4f}, qty={self.total_quantity}, orders={self.order_count})"


class OrderBook:
    """
    Full L2 Limit Order Book.

    Internal representation:
        bids : dict[price -> PriceLevel]  — keyed by price descending (best bid = max)
        asks : dict[price -> PriceLevel]  — keyed by price ascending  (best ask = min)
        _id_map : dict[order_id -> (side, price)] for O(1) cancellation

    In production HFT code (C++), this is implemented as a red-black tree or
    a flat array indexed by price tick (when tick size is fixed).
    Here we use sorted Python dict reads — sufficient for simulation.
    """

    def __init__(self):
        self.bids   : Dict[float, PriceLevel] = {}
        self.asks   : Dict[float, PriceLevel] = {}
        self._id_map: Dict[str, Tuple[Side, float]] = {}

        # OFI history: rolling window of (total_bid_qty, total_ask_qty)
        self._ofi_window : deque = deque(maxlen=50)
        self._total_bid_qty = 0
        self._total_ask_qty = 0

    # ── Order Management ──────────────────────────────────────────────────────

    def add_order(self, order: Order) -> None:
        """Add a new limit order to the book."""
        if order.side == Side.BID:
            if order.price not in self.bids:
                self.bids[order.price] = PriceLevel(order.price)
            self.bids[order.price].add(order)
            self._total_bid_qty += order.quantity
        else:
            if order.price not in self.asks:
                self.asks[order.price] = PriceLevel(order.price)
            self.asks[order.price].add(order)
            self._total_ask_qty += order.quantity

        self._id_map[order.order_id] = (order.side, order.price)
        self._snapshot_ofi()

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order by ID. Returns True if found and cancelled."""
        if order_id not in self._id_map:
            return False

        side, price = self._id_map.pop(order_id)
        book_side   = self.bids if side == Side.BID else self.asks

        if price in book_side:
            level = book_side[price]
            qty_before = level.total_quantity
            cancelled  = level.cancel(order_id)
            if cancelled:
                qty_removed = qty_before - level.total_quantity
                if side == Side.BID:
                    self._total_bid_qty -= qty_removed
                else:
                    self._total_ask_qty -= qty_removed
            if level.is_empty():
                del book_side[price]
            self._snapshot_ofi()
            return cancelled
        return False

    def _snapshot_ofi(self) -> None:
        """Save current bid/ask totals for OFI rolling computation."""
        self._ofi_window.append((self._total_bid_qty, self._total_ask_qty))

    # ── Real-Time Analytics ───────────────────────────────────────────────────

    @property
    def best_bid(self) -> Optional[float]:
        """Highest price any buyer is willing to pay."""
        active = [p for p, lvl in self.bids.items() if not lvl.is_empty()]
        return max(active) if active else None

    @property
    def best_ask(self) -> Optional[float]:
        """Lowest price any seller is willing to accept."""
        active = [p for p, lvl in self.asks.items() if not lvl.is_empty()]
        return min(active) if active else None

    @property
    def mid_price(self) -> Optional[float]:
        """Average of best bid and ask. The 'fair value' reference price."""
        bb, ba = self.best_bid, self.best_ask
        return (bb + ba) / 2.0 if bb and ba else None

    @property
    def spread(self) -> Optional[float]:
        """
        Bid-ask spread = best_ask - best_bid.
        This is the market maker's profit per round-trip trade.
        Tight spread = liquid market. Wide spread = illiquid market.
        """
        bb, ba = self.best_bid, self.best_ask
        return (ba - bb) if bb and ba else None

    @property
    def spread_bps(self) -> Optional[float]:
        """Spread in basis points (bps). 1 bps = 0.01%."""
        s, m = self.spread, self.mid_price
        return (s / m * 10000) if s and m else None

    def order_flow_imbalance(self) -> float:
        """
        OFI measures net buying vs selling pressure from order book changes.

        OFI = (bid_qty_change - ask_qty_change) / (|bid_delta| + |ask_delta| + eps)

        Range: [-1, +1]
          +1 = pure buy pressure (bids added, asks removed)
          -1 = pure sell pressure (asks added, bids removed)
           0 = balanced flow

        Academic reference: Cont, Kukanov, Stoikov (2014)
        "The Price Impact of Order Book Events" — Journal of Financial Econometrics
        """
        if len(self._ofi_window) < 2:
            return 0.0

        prev_bid, prev_ask = self._ofi_window[-2]
        curr_bid, curr_ask = self._ofi_window[-1]

        bid_delta = curr_bid - prev_bid
        ask_delta = curr_ask - prev_ask

        denom = abs(bid_delta) + abs(ask_delta) + 1e-9
        return (bid_delta - ask_delta) / denom

    def bid_depth(self, n_levels: int = 10) -> List[Tuple[float, int]]:
        """
        Top N bid price levels sorted descending (best first).
        Returns list of (price, total_quantity) tuples.
        """
        active = [(p, lvl.total_quantity)
                  for p, lvl in self.bids.items()
                  if not lvl.is_empty()]
        return sorted(active, key=lambda x: -x[0])[:n_levels]

    def ask_depth(self, n_levels: int = 10) -> List[Tuple[float, int]]:
        """
        Top N ask price levels sorted ascending (best first).
        Returns list of (price, total_quantity) tuples.
        """
        active = [(p, lvl.total_quantity)
                  for p, lvl in self.asks.items()
                  if not lvl.is_empty()]
        return sorted(active, key=lambda x: x[0])[:n_levels]

    def snapshot(self) -> dict:
        """
        Serializable snapshot for dashboard / logging.
        Contains all analytics at current moment in time.
        """
        return {
            "best_bid"   : self.best_bid,
            "best_ask"   : self.best_ask,
            "mid_price"  : self.mid_price,
            "spread"     : self.spread,
            "spread_bps" : self.spread_bps,
            "ofi"        : self.order_flow_imbalance(),
            "bid_levels" : len([p for p, l in self.bids.items() if not l.is_empty()]),
            "ask_levels" : len([p for p, l in self.asks.items() if not l.is_empty()]),
            "total_bid_qty": self._total_bid_qty,
            "total_ask_qty": self._total_ask_qty,
        }

    def print_book(self, n_levels: int = 5) -> None:
        """Pretty-print the order book (for debugging / demos)."""
        asks = self.ask_depth(n_levels)
        bids = self.bid_depth(n_levels)

        print("\n" + "="*40)
        print(f"{'ASKS (sellers)':^40}")
        print("="*40)
        for price, qty in reversed(asks):
            bar = "#" * min(int(qty / 100), 30)
            print(f"  {price:>10.4f}  |  {qty:>8,}  {bar}")
        print(f"{'--- SPREAD: ' + (f'{self.spread:.4f}' if self.spread else 'N/A') + ' ---':^40}")
        for price, qty in bids:
            bar = "#" * min(int(qty / 100), 30)
            print(f"  {price:>10.4f}  |  {qty:>8,}  {bar}")
        print(f"{'BIDS (buyers)':^40}")
        print("="*40)
        print(f"  Mid price : {self.mid_price:.4f}" if self.mid_price else "  Mid: N/A")
        print(f"  OFI       : {self.order_flow_imbalance():.4f}")
        print()


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import random
    random.seed(42)

    print("Building a sample order book...\n")

    book = OrderBook()
    mid  = 100.0

    # Add 30 bid orders spread across 5 price levels
    for _ in range(30):
        price = round(mid - random.uniform(0.01, 0.25), 2)
        qty   = random.randint(50, 500)
        book.add_order(make_order(Side.BID, price, qty))

    # Add 30 ask orders spread across 5 price levels
    for _ in range(30):
        price = round(mid + random.uniform(0.01, 0.25), 2)
        qty   = random.randint(50, 500)
        book.add_order(make_order(Side.ASK, price, qty))

    book.print_book(n_levels=6)

    snap = book.snapshot()
    print("Snapshot:")
    for k, v in snap.items():
        print(f"  {k:20s}: {v}")
