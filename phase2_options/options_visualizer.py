"""
Options Visualizer
-------------------
Static matplotlib charts for options analysis.
Dashboard uses Plotly for interactivity.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from phase2_options.black_scholes import price_option, bs_call, bs_put
from phase2_options.vol_surface   import generate_vol_surface

plt.style.use("dark_background")


def plot_payoff_diagram(K: float, option_type: str, premium: float,
                        S_range: np.ndarray = None, ax=None) -> plt.Axes:
    """Classic hockey-stick payoff diagram with breakeven line."""
    if S_range is None:
        S_range = np.linspace(K * 0.7, K * 1.3, 300)
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))

    if option_type == "call":
        payoff = np.maximum(S_range - K, 0) - premium
        be     = K + premium
    else:
        payoff = np.maximum(K - S_range, 0) - premium
        be     = K - premium

    ax.plot(S_range, payoff, color="#2196F3", linewidth=2)
    ax.fill_between(S_range, payoff, 0,
                    where=(payoff > 0), alpha=0.25, color="#00C853", label="Profit zone")
    ax.fill_between(S_range, payoff, 0,
                    where=(payoff <= 0), alpha=0.25, color="#FF1744", label="Loss zone")
    ax.axhline(0, color="white", linewidth=0.8, linestyle="--")
    ax.axvline(K,  color="yellow", linewidth=1, linestyle=":", label=f"Strike K={K}")
    ax.axvline(be, color="#FF9800", linewidth=1, linestyle=":",
               label=f"Breakeven={be:.2f}")
    ax.set_xlabel("Spot Price at Expiry", fontsize=11)
    ax.set_ylabel("P&L", fontsize=11)
    ax.set_title(f"Payoff Diagram — {option_type.upper()} (K={K}, Premium={premium:.4f})",
                 fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)
    return ax


def plot_greeks_vs_spot(K: float, T: float, r: float, sigma: float,
                        option_type: str = "call",
                        S_range: np.ndarray = None) -> plt.Figure:
    """
    2x3 grid of charts showing each Greek as a function of spot price.
    The most comprehensive options analysis chart.
    """
    if S_range is None:
        S_range = np.linspace(K * 0.60, K * 1.40, 200)

    greeks = {"Price": [], "Delta": [], "Gamma": [],
              "Theta": [], "Vega": [],  "Rho":   []}

    for s in S_range:
        res = price_option(s, K, T, r, sigma, option_type)
        greeks["Price"].append(res.price)
        greeks["Delta"].append(res.delta)
        greeks["Gamma"].append(res.gamma)
        greeks["Theta"].append(res.theta)
        greeks["Vega"].append(res.vega)
        greeks["Rho"].append(res.rho)

    colors = {"Price": "#2196F3", "Delta": "#00C853", "Gamma": "#FFC107",
              "Theta": "#FF1744", "Vega":  "#9C27B0", "Rho":   "#00BCD4"}

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(
        f"{option_type.upper()} Greeks vs Spot  (K={K}, T={T:.2f}yr, sigma={sigma*100:.0f}%)",
        fontsize=14, fontweight="bold"
    )
    axes = axes.flatten()

    for ax, (name, vals) in zip(axes, greeks.items()):
        ax.plot(S_range, vals, color=colors[name], linewidth=2)
        ax.axvline(K, color="white", linestyle="--", alpha=0.5, linewidth=1)
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.set_xlabel("Spot Price")
        ax.grid(alpha=0.2)
        ax.axhline(0, color="white", linewidth=0.4, linestyle=":")

    plt.tight_layout()
    return fig


def plot_vol_surface_3d(surface_df, save_path: str = None) -> plt.Figure:
    """3D matplotlib surface plot of the IV surface."""
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.cm as cm

    strikes  = surface_df.index.values.astype(float)
    expiries = surface_df.columns.values.astype(float)
    Z        = surface_df.values.astype(float) * 100   # to percentage

    X, Y = np.meshgrid(expiries, strikes)

    fig = plt.figure(figsize=(13, 8))
    ax  = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.85,
                            linewidth=0, antialiased=True)
    fig.colorbar(surf, ax=ax, shrink=0.5, label="IV (%)")

    ax.set_xlabel("Expiry (years)", fontsize=11, labelpad=10)
    ax.set_ylabel("Strike",         fontsize=11, labelpad=10)
    ax.set_zlabel("IV (%)",         fontsize=11, labelpad=10)
    ax.set_title("Implied Volatility Surface", fontsize=14, fontweight="bold")
    ax.view_init(elev=25, azim=225)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(out_dir, exist_ok=True)

    print("Generating options charts...\n")

    # 1. Payoff diagrams for call and put
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    call_res = price_option(100, 105, 0.25, 0.05, 0.20, "call")
    put_res  = price_option(100, 95,  0.25, 0.05, 0.20, "put")
    plot_payoff_diagram(105, "call", call_res.price, ax=ax1)
    plot_payoff_diagram(95,  "put",  put_res.price,  ax=ax2)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "payoff_diagrams.png"), dpi=150)
    plt.show()
    print("  Payoff diagrams saved.")

    # 2. Greeks vs spot
    fig2 = plot_greeks_vs_spot(K=100, T=0.5, r=0.05, sigma=0.20, option_type="call")
    fig2.savefig(os.path.join(out_dir, "greeks_vs_spot.png"), dpi=150)
    plt.show()
    print("  Greeks chart saved.")

    # 3. Vol surface
    surface = generate_vol_surface()
    fig3    = plot_vol_surface_3d(surface, save_path=os.path.join(out_dir, "vol_surface.png"))
    plt.show()
    print("  Vol surface saved.\n")
    print("Phase 2 visualization complete.")
