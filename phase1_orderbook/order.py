"""
Order — Core Data Types
------------------------
Defines the building blocks of every trade: an Order, its Side (bid/ask),
and its Status (active/filled/cancelled).

In a real exchange, millions of these objects exist simultaneously.
HFT systems represent them as packed C structs for cache efficiency.
Here we use Python dataclasses for clarity.
"""

from dataclasses import dataclass, field
from enum import Enum
import time
import uuid


class Side(Enum):
    BID = "bid"   # buyer — wants to pay as little as possible
    ASK = "ask"   # seller — wants to receive as much as possible


class OrderStatus(Enum):
    ACTIVE    = "active"
    FILLED    = "filled"
    CANCELLED = "cancelled"
    PARTIAL   = "partial"


@dataclass
class Order:
    """
    A single resting limit order in the book.

    Fields:
        order_id   : unique identifier (UUID string)
        side       : BID or ASK
        price      : limit price — order will only fill at this price or better
        quantity   : total shares/contracts requested
        timestamp  : UNIX timestamp when order was placed (nanoseconds in real HFT)
        status     : current lifecycle state
        filled_qty : how many shares have been filled so far
    """
    order_id   : str
    side       : Side
    price      : float
    quantity   : int
    timestamp  : float = field(default_factory=time.time)
    status     : OrderStatus = OrderStatus.ACTIVE
    filled_qty : int = 0

    @property
    def remaining_qty(self) -> int:
        """Shares still waiting to be filled."""
        return self.quantity - self.filled_qty

    @property
    def is_active(self) -> bool:
        return self.status in (OrderStatus.ACTIVE, OrderStatus.PARTIAL)

    def fill(self, qty: int) -> None:
        """Record a partial or full fill."""
        self.filled_qty += qty
        if self.filled_qty >= self.quantity:
            self.status = OrderStatus.FILLED
        else:
            self.status = OrderStatus.PARTIAL

    def cancel(self) -> None:
        self.status = OrderStatus.CANCELLED

    def __repr__(self) -> str:
        return (f"Order({self.side.value.upper()} {self.remaining_qty}"
                f"@{self.price:.4f} [{self.status.value}])")


def make_order(side: Side, price: float, quantity: int) -> Order:
    """Convenience constructor — generates a unique order ID automatically."""
    return Order(
        order_id  = str(uuid.uuid4())[:8],
        side      = side,
        price     = price,
        quantity  = quantity,
    )
