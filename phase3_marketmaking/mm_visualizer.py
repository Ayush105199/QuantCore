"""
Market Making Visualizer
-------------------------
Charts for comparing AS vs Naive market making strategies.
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from phase3_marketmaking.mm_simulator import compare_strategies, SimulationResult

plt.style.use("dark_background")
COLORS = {"as": "#00C853", "naive": "#FF1744", "mid": "#2196F3",
          "bid": "#4CAF50", "ask": "#F44336", "fill": "#FFC107"}


def plot_pnl_comparison(as_r: SimulationResult, naive_r: SimulationResult,
                         save_path: str = None) -> plt.Figure:
    """Side-by-side cumulative P&L curves."""
    fig, ax = plt.subplots(figsize=(13, 5))

    ax.plot(as_r.pnl,    color=COLORS["as"],    linewidth=1.5, label="AS Optimal", alpha=0.9)
    ax.plot(naive_r.pnl, color=COLORS["naive"], linewidth=1.5, label="Naive Fixed-Spread", alpha=0.9)
    ax.axhline(0, color="white", linewidth=0.5, linestyle="--")
    ax.fill_between(range(len(as_r.pnl)), as_r.pnl, naive_r.pnl,
                    where=(as_r.pnl > naive_r.pnl), alpha=0.12, color=COLORS["as"],
                    label="AS outperformance")

    ax.set_xlabel("Simulation Step", fontsize=11)
    ax.set_ylabel("Cumulative P&L", fontsize=11)
    ax.set_title("Market Maker P&L: Avellaneda-Stoikov vs Naive Fixed-Spread",
                  fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.2)

    as_final    = as_r.pnl[-1]
    naive_final = naive_r.pnl[-1]
    ax.annotate(f"AS: {as_final:+.2f}", xy=(len(as_r.pnl)-1, as_final),
                xytext=(-80, 10), textcoords="offset points",
                color=COLORS["as"], fontsize=9, arrowprops=dict(arrowstyle="->", color=COLORS["as"]))
    ax.annotate(f"Naive: {naive_final:+.2f}", xy=(len(naive_r.pnl)-1, naive_final),
                xytext=(-80, -20), textcoords="offset points",
                color=COLORS["naive"], fontsize=9, arrowprops=dict(arrowstyle="->", color=COLORS["naive"]))

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_inventory_management(as_r: SimulationResult, naive_r: SimulationResult,
                               save_path: str = None) -> plt.Figure:
    """
    The key chart showing why AS beats Naive:
    AS model keeps inventory small and mean-reverting.
    Naive model lets inventory drift dangerously.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    fig.suptitle("Inventory Management: AS vs Naive", fontsize=13, fontweight="bold")

    # Mid price
    ax1.plot(as_r.mid_prices, color=COLORS["mid"], linewidth=1, alpha=0.7, label="Mid Price")
    ax1.set_ylabel("Mid Price", fontsize=10)
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.2)

    # Inventory comparison
    ax2.plot(as_r.inventories,    color=COLORS["as"],    linewidth=1.2,
             label=f"AS Inventory (max={np.abs(as_r.inventories).max()})", alpha=0.9)
    ax2.plot(naive_r.inventories, color=COLORS["naive"], linewidth=1.2,
             label=f"Naive Inventory (max={np.abs(naive_r.inventories).max()})", alpha=0.9)
    ax2.axhline(0, color="white", linewidth=0.5, linestyle="--")
    ax2.fill_between(range(len(as_r.inventories)), as_r.inventories,
                     alpha=0.2, color=COLORS["as"])
    ax2.set_ylabel("Inventory (shares)", fontsize=10)
    ax2.set_xlabel("Simulation Step", fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.2)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_quote_activity(result: SimulationResult, n_show: int = 300,
                         title: str = "Quote Activity",
                         save_path: str = None) -> plt.Figure:
    """
    Most visually impressive chart: mid price + bid/ask spread bands + fills.
    Shows exactly what the market maker is doing second-by-second.
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    mid   = result.mid_prices[:n_show]
    bids  = result.bid_quotes[:n_show]
    asks  = result.ask_quotes[:n_show]
    steps = np.arange(n_show)

    ax.plot(steps, mid,  color=COLORS["mid"],  linewidth=1.5, label="Mid Price", zorder=3)
    ax.plot(steps, bids, color=COLORS["bid"],  linewidth=0.8, label="Bid Quote", alpha=0.7, linestyle="--")
    ax.plot(steps, asks, color=COLORS["ask"],  linewidth=0.8, label="Ask Quote", alpha=0.7, linestyle="--")
    ax.fill_between(steps, bids, asks, alpha=0.12, color="#9E9E9E", label="Spread band")

    # Mark fills on the chart
    fills = result.fills[result.fills["step"] < n_show]
    if len(fills) > 0:
        bid_fills = fills[fills["side"] == "bid"]
        ask_fills = fills[fills["side"] == "ask"]
        if len(bid_fills):
            ax.scatter(bid_fills["step"], bid_fills["price"],
                       color=COLORS["bid"], s=40, zorder=5, marker="^", label="Buy Fill")
        if len(ask_fills):
            ax.scatter(ask_fills["step"], ask_fills["price"],
                       color=COLORS["ask"], s=40, zorder=5, marker="v", label="Sell Fill")

    ax.set_xlabel("Step", fontsize=11)
    ax.set_ylabel("Price", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_full_comparison(as_r: SimulationResult, naive_r: SimulationResult,
                          save_path: str = None) -> plt.Figure:
    """4-panel summary comparison."""
    fig = plt.figure(figsize=(16, 11))
    fig.suptitle("QuantCore — Market Making Analysis: AS vs Naive",
                  fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)

    # P&L comparison
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(as_r.pnl,    color=COLORS["as"],    linewidth=1.5, label="AS")
    ax1.plot(naive_r.pnl, color=COLORS["naive"], linewidth=1.5, label="Naive")
    ax1.axhline(0, color="white", linewidth=0.5, linestyle="--")
    ax1.set_title("P&L Comparison"); ax1.legend(); ax1.grid(alpha=0.2)

    # Inventory
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(as_r.inventories,    color=COLORS["as"],    linewidth=1.2, label="AS")
    ax2.plot(naive_r.inventories, color=COLORS["naive"], linewidth=1.2, label="Naive")
    ax2.axhline(0, color="white", linewidth=0.5, linestyle="--")
    ax2.set_title("Inventory Over Time"); ax2.legend(); ax2.grid(alpha=0.2)

    # AS quote activity
    ax3 = fig.add_subplot(gs[1, 0])
    n   = min(300, len(as_r.mid_prices))
    ax3.plot(as_r.mid_prices[:n],  color=COLORS["mid"], linewidth=1.5)
    ax3.plot(as_r.bid_quotes[:n],  color=COLORS["bid"], linewidth=0.8, linestyle="--")
    ax3.plot(as_r.ask_quotes[:n],  color=COLORS["ask"], linewidth=0.8, linestyle="--")
    ax3.fill_between(range(n), as_r.bid_quotes[:n], as_r.ask_quotes[:n],
                     alpha=0.12, color="#9E9E9E")
    ax3.set_title("AS Quote Activity (first 300 steps)"); ax3.grid(alpha=0.2)

    # Metrics bar chart
    ax4   = fig.add_subplot(gs[1, 1])
    metrics_keys = ["final_pnl", "sharpe_ratio", "total_spread_earned"]
    labels       = ["Final P&L", "Sharpe", "Spread Earned"]
    as_vals   = [as_r.metrics.get(k, 0) for k in metrics_keys]
    naive_vals = [naive_r.metrics.get(k, 0) for k in metrics_keys]

    x  = np.arange(len(labels))
    w  = 0.35
    ax4.bar(x - w/2, as_vals,    w, color=COLORS["as"],    label="AS",    alpha=0.8)
    ax4.bar(x + w/2, naive_vals, w, color=COLORS["naive"], label="Naive", alpha=0.8)
    ax4.set_xticks(x); ax4.set_xticklabels(labels, fontsize=9)
    ax4.axhline(0, color="white", linewidth=0.5)
    ax4.set_title("Key Metrics Comparison"); ax4.legend(); ax4.grid(alpha=0.2)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Running simulation for visualization...\n")

    results = compare_strategies(n_steps=3000, seed=42)
    out_dir  = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(out_dir, exist_ok=True)

    fig = plot_full_comparison(
        results["as"], results["naive"],
        save_path=os.path.join(out_dir, "mm_comparison.png")
    )
    plt.show()
    print("Market making visualization complete.")
