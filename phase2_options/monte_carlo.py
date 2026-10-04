"""
Monte Carlo Options Pricing
-----------------------------
Prices options by simulating thousands of random price paths and
averaging the payoffs. Cross-validates the Black-Scholes formula.

Why Monte Carlo when BS is exact?
  1. Exotic options: Asian, barrier, lookback — no closed-form solution
  2. Path-dependent payoffs require full path simulation
  3. Stochastic vol (Heston), jump-diffusion — BS doesn't apply
  4. American options need Longstaff-Schwartz MC

Here we use standard European MC as a learning/validation tool.

Variance reduction technique: Antithetic Variates
  For every random path Z, also simulate -Z.
  Since E[payoff(Z) + payoff(-Z)] / 2 has lower variance than E[payoff(Z)],
  this halves the required number of paths for the same accuracy.
  Cost: 2x paths per simulation. Benefit: ~50% variance reduction.
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from phase2_options.black_scholes import bs_call, bs_put


def mc_price_option(
    S          : float,
    K          : float,
    T          : float,
    r          : float,
    sigma      : float,
    option_type: str   = "call",
    n_paths    : int   = 100_000,
    n_steps    : int   = 252,
    seed       : int   = 42,
    antithetic : bool  = True,
) -> dict:
    """
    Price a European option via Monte Carlo simulation (GBM paths).

    GBM discretization (Euler-Maruyama):
        S_{t+dt} = S_t * exp((r - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
        where Z ~ N(0, 1)

    Antithetic variates: simulate Z and -Z together.

    Args:
        S, K, T, r, sigma : standard BS parameters
        option_type       : "call" or "put"
        n_paths           : number of simulation paths (more = more accurate)
        n_steps           : timesteps per path (252 = daily for 1 year)
        seed              : random seed for reproducibility
        antithetic        : use antithetic variates for variance reduction

    Returns:
        dict with:
            price      : MC option price
            std_error  : standard error of the estimate
            conf_95    : 95% confidence interval (lower, upper)
            bs_price   : Black-Scholes analytical price (for comparison)
            diff_pct   : percentage difference from BS
            n_paths    : actual paths simulated
    """
    rng = np.random.default_rng(seed)
    dt  = T / n_steps

    # Precompute constants
    drift_step = (r - 0.5 * sigma ** 2) * dt
    vol_step   = sigma * math.sqrt(dt) if False else sigma * np.sqrt(dt)

    if antithetic:
        half = n_paths // 2
        # Shape: (half, n_steps) — one row per path, one column per step
        Z      = rng.standard_normal((half, n_steps))
        Z_anti = -Z   # antithetic counterpart

        # Simulate terminal prices (only need S_T for European options)
        log_returns      = drift_step + vol_step * Z
        log_returns_anti = drift_step + vol_step * Z_anti

        S_T      = S * np.exp(log_returns.sum(axis=1))
        S_T_anti = S * np.exp(log_returns_anti.sum(axis=1))

        all_S_T = np.concatenate([S_T, S_T_anti])
    else:
        Z           = rng.standard_normal((n_paths, n_steps))
        log_returns = drift_step + vol_step * Z
        all_S_T     = S * np.exp(log_returns.sum(axis=1))

    # Compute payoffs
    if option_type == "call":
        payoffs = np.maximum(all_S_T - K, 0.0)
    else:
        payoffs = np.maximum(K - all_S_T, 0.0)

    # Discount to present value
    discount    = np.exp(-r * T)
    pv_payoffs  = discount * payoffs

    # Statistics
    price       = pv_payoffs.mean()
    std_error   = pv_payoffs.std() / np.sqrt(len(pv_payoffs))
    conf_lower  = price - 1.96 * std_error
    conf_upper  = price + 1.96 * std_error

    # BS analytical price for comparison
    bs_price_fn = bs_call if option_type == "call" else bs_put
    bs_val      = bs_price_fn(S, K, T, r, sigma)
    diff_pct    = (price - bs_val) / bs_val * 100 if bs_val > 0 else float("nan")

    # Sample paths for visualization (first 20 paths only)
    sample_paths = None
    if antithetic:
        sample_Z   = rng.standard_normal((20, n_steps))
        cumulative = drift_step + vol_step * sample_Z
        sample_paths = S * np.exp(np.cumsum(cumulative, axis=1))
    else:
        sample_paths = S * np.exp(np.cumsum(
            drift_step + vol_step * rng.standard_normal((20, n_steps)), axis=1
        ))

    return {
        "price"       : float(price),
        "std_error"   : float(std_error),
        "conf_95"     : (float(conf_lower), float(conf_upper)),
        "bs_price"    : float(bs_val),
        "diff_pct"    : float(diff_pct),
        "n_paths"     : len(pv_payoffs),
        "sample_paths": sample_paths,
    }


def mc_convergence_study(
    S: float, K: float, T: float, r: float, sigma: float,
    option_type: str = "call",
    path_counts: list = None,
    seed: int = 42,
) -> dict:
    """
    Show how MC price converges to BS price as n_paths increases.
    Returns results for each path count for plotting convergence.
    """
    if path_counts is None:
        path_counts = [500, 1000, 2000, 5000, 10000, 25000, 50000, 100000]

    bs_val   = (bs_call if option_type == "call" else bs_put)(S, K, T, r, sigma)
    results  = []

    for n in path_counts:
        res = mc_price_option(S, K, T, r, sigma, option_type,
                               n_paths=n, seed=seed)
        results.append({
            "n_paths"  : n,
            "mc_price" : res["price"],
            "std_error": res["std_error"],
            "diff_pct" : res["diff_pct"],
        })

    return {"bs_price": bs_val, "results": results}


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import math
    print("="*55)
    print("  MONTE CARLO OPTIONS PRICER")
    print("  Cross-validation against Black-Scholes")
    print("="*55 + "\n")

    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20

    for opt_type in ["call", "put"]:
        print(f"  {opt_type.upper()} option (S=100, K=100, T=1yr, r=5%, sigma=20%):")
        result = mc_price_option(S, K, T, r, sigma, opt_type,
                                  n_paths=200_000, antithetic=True)
        print(f"    MC Price  : {result['price']:.6f} +/- {result['std_error']:.6f}")
        print(f"    95% CI    : [{result['conf_95'][0]:.6f}, {result['conf_95'][1]:.6f}]")
        print(f"    BS Price  : {result['bs_price']:.6f}")
        print(f"    Diff      : {result['diff_pct']:+.4f}%  ({result['n_paths']:,} paths)")
        print()

    # Convergence study
    print("  Convergence study (call, ATM, 1yr):")
    print(f"  {'Paths':>10}  {'MC Price':>12}  {'BS Price':>12}  {'Diff %':>10}")
    print("  " + "-"*48)
    conv = mc_convergence_study(S, K, T, r, sigma, "call")
    for row in conv["results"]:
        print(f"  {row['n_paths']:>10,}  {row['mc_price']:>12.6f}"
              f"  {conv['bs_price']:>12.6f}  {row['diff_pct']:>+10.4f}%")

    print("\nMonte Carlo complete.")
