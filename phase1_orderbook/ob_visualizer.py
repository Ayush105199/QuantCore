"""
Order Book Visualizer
----------------------
Static matplotlib charts for the order book and simulation history.
Used for standalone demos. The dashboard uses Plotly for interactivity.
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from phase1_orderbook.orderbook import OrderBook
from phase1_orderbook.order_flow_generator import OrderFlowGenerator, GeneratorConfig

plt.style.use("dark_background")
COLORS = {"bid": "#00C853", "ask": "#FF1744", "mid": "#2196F3",
          "spread": "#FFC107", "ofi_pos": "#00E676", "ofi_neg": "#FF5252"}


def plot_depth_chart(book: OrderBook, n_levels: int = 10,
                     title: str = "L2 Order Book Depth",
                     ax=None) -> plt.Axes:
    """
    Horizontal bar chart showing bid and ask depth.

    Bids on the left (green), asks on the right (red).
    Height = price level, width = cumulative volume at that level.
    This is the chart professional traders watch all day.
    """
    bids = book.bid_depth(n_levels)
    asks = book.ask_depth(n_levels)

    if not ax:
        _, ax = plt.subplots(figsize=(10, 6))

    if bids:
        b_prices = [p for p, _ in bids]
        b_qtys   = [q for _, q in bids]
        ax.barh(b_prices, b_qtys, color=COLORS["bid"], alpha=0.8,
                label="Bids (Buyers)", height=0.005)

    if asks:
        a_prices = [p for p, _ in asks]
        a_qtys   = [-q for _, q in asks]   # negative to go left
        ax.barh(a_prices, a_qtys, color=COLORS["ask"], alpha=0.8,
                label="Asks (Sellers)", height=0.005)

    if book.mid_price:
        ax.axhline(book.mid_price, color=COLORS["mid"], linestyle="--",
                   linewidth=1.5, label=f"Mid: {book.mid_price:.4f}")

    ax.axvline(0, color="white", linewidth=0.5)
    ax.set_xlabel("Volume (shares)", fontsize=11)
    ax.set_ylabel("Price", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)

    spread_text = f"Spread: {book.spread:.4f}" if book.spread else "Spread: N/A"
    ax.text(0.02, 0.98, spread_text, transform=ax.transAxes,
            fontsize=9, va="top", color=COLORS["spread"])
    return ax


def plot_spread_over_time(history: pd.DataFrame, ax=None) -> plt.Axes:
    """Line chart of bid-ask spread (bps) over the simulation."""
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 3))

    spread_data = history["spread_bps"].dropna()
    ax.plot(spread_data.index, spread_data.values,
            color=COLORS["spread"], linewidth=1.2, alpha=0.9)
    ax.fill_between(spread_data.index, spread_data.values,
                    alpha=0.2, color=COLORS["spread"])
    ax.axhline(spread_data.mean(), color="white", linestyle="--",
               alpha=0.5, label=f"Avg: {spread_data.mean():.2f} bps")
    ax.set_ylabel("Spread (bps)", fontsize=10)
    ax.set_title("Bid-Ask Spread Over Time", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    return ax


def plot_ofi_vs_midprice(history: pd.DataFrame, ax1=None) -> plt.Axes:
    """
    Dual-axis chart: OFI (bars) vs mid price (line).
    The key insight: OFI predicts short-term price direction.
    Positive OFI (green bars) → price tends to rise.
    Negative OFI (red bars) → price tends to fall.
    """
    if ax1 is None:
        _, ax1 = plt.subplots(figsize=(12, 4))

    ofi_data = history["ofi"].fillna(0)
    mid_data = history["step_mid"].dropna()

    colors = [COLORS["ofi_pos"] if v >= 0 else COLORS["ofi_neg"]
              for v in ofi_data.values]
    ax1.bar(ofi_data.index, ofi_data.values, color=colors, alpha=0.7, label="OFI")
    ax1.axhline(0, color="white", linewidth=0.5)
    ax1.set_ylabel("OFI", fontsize=10, color=COLORS["ofi_pos"])
    ax1.set_ylim(-1.1, 1.1)

    ax2 = ax1.twinx()
    ax2.plot(mid_data.index, mid_data.values,
             color=COLORS["mid"], linewidth=1.5, label="Mid Price")
    ax2.set_ylabel("Mid Price", fontsize=10, color=COLORS["mid"])

    ax1.set_title("Order Flow Imbalance vs Mid Price", fontsize=11)
    ax1.set_xlabel("Step", fontsize=10)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
    ax1.grid(alpha=0.2)
    return ax1


def plot_volume_flow(history: pd.DataFrame, ax=None) -> plt.Axes:
    """Stacked area chart: bid volume vs ask volume over time."""
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 3))

    ax.fill_between(history.index, history["total_bid_qty"],
                    alpha=0.4, color=COLORS["bid"], label="Total Bid Volume")
    ax.fill_between(history.index, history["total_ask_qty"],
                    alpha=0.4, color=COLORS["ask"], label="Total Ask Volume")
    ax.set_ylabel("Total Volume", fontsize=10)
    ax.set_title("Order Book Volume (Bid vs Ask)", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    return ax


def plot_full_dashboard(book: OrderBook, history: pd.DataFrame,
                        save_path: str = None) -> plt.Figure:
    """
    Full 4-panel summary chart:
      Top-left : Order book depth
      Top-right: OFI vs mid price
      Bot-left : Spread over time
      Bot-right: Volume flow
    """
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("QuantCore — L2 Order Book Analysis", fontsize=15, fontweight="bold")

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    plot_depth_chart(book, ax=ax1)
    plot_ofi_vs_midprice(history, ax1=ax2)
    plot_spread_over_time(history, ax=ax3)
    plot_volume_flow(history, ax=ax4)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Chart saved: {save_path}")

    return fig


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Running simulation for visualization...\n")

    cfg     = GeneratorConfig(mid_price=100.0, lambda_bid=6, lambda_ask=6, sigma=0.012)
    gen     = OrderFlowGenerator(config=cfg, seed=42)
    book    = OrderBook()
    history = gen.run_simulation(n_steps=400, book=book)

    out_dir  = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "orderbook_dashboard.png")

    fig = plot_full_dashboard(book, history, save_path=out_path)
    plt.show()
    print("Phase 1 visualization complete.")
