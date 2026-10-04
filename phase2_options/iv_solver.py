"""
Implied Volatility Solver
--------------------------
Given a market-observed option price, find the volatility (sigma) that
the Black-Scholes model would need to produce that price.

This is called "implied volatility" (IV) because it is implied by the market price.

Why this matters:
  - Traders don't quote options by price — they quote by IV.
  - "Nifty 22500 Call is trading at 32 IV" means IV = 32%.
  - Comparing IVs across strikes reveals the volatility smile/skew.
  - IV > realized vol = options are expensive. IV < realized = cheap.

Method: Newton-Raphson iteration.
  - Starts with an initial guess (Brenner-Subrahmanyam approximation)
  - Each step: sigma_{n+1} = sigma_n - (BS_price - market_price) / vega
  - Quadratic convergence: errors squares every step
  - Typically converges in 4-6 iterations
"""

import math
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from phase2_options.black_scholes import (
    bs_call, bs_put, vega as bs_vega, price_option, _d1
)


def _brenner_subrahmanyam_guess(market_price: float, S: float, K: float,
                                 T: float, option_type: str) -> float:
    """
    Initial guess for IV using Brenner & Subrahmanyam (1988) approximation.
    Works well for near-ATM options.

    Formula: sigma_0 ≈ sqrt(2*pi/T) * (C / S)  [for ATM call]
    """
    if T <= 0:
        return 0.2
    atm_approx = math.sqrt(2 * math.pi / T) * (market_price / S)
    return max(0.01, min(atm_approx, 5.0))


def implied_volatility(
    market_price : float,
    S            : float,
    K            : float,
    T            : float,
    r            : float = 0.05,
    option_type  : str   = "call",
    initial_guess: float = None,
    max_iter     : int   = 100,
    tolerance    : float = 1e-7,
) -> float:
    """
    Compute implied volatility via Newton-Raphson iteration.

    Algorithm:
        sigma_0 = initial_guess (or Brenner-Subrahmanyam approximation)

        For i = 0, 1, 2, ..., max_iter:
            model_price = BS_price(sigma_i)
            v = vega(sigma_i)
            if |v| < 1e-10: bail out (vega near zero = unstable)
            sigma_{i+1} = sigma_i - (model_price - market_price) / v
            sigma_{i+1} = clip to [0.001, 20.0]  (prevent divergence)
            if |sigma_{i+1} - sigma_i| < tolerance: converged, return

    Args:
        market_price : observed option price in the market
        S            : current spot price
        K            : strike price
        T            : time to expiry (years)
        r            : risk-free rate (decimal)
        option_type  : "call" or "put"
        initial_guess: starting sigma (default: Brenner-Subrahmanyam)
        max_iter     : maximum Newton-Raphson iterations
        tolerance    : convergence threshold

    Returns:
        Implied volatility as a decimal (e.g. 0.20 for 20%)

    Raises:
        ValueError : if market_price violates no-arbitrage bounds
        RuntimeError: if solver fails to converge
    """
    if T <= 0:
        raise ValueError("Time to expiry T must be positive")

    # No-arbitrage bounds check
    disc = K * math.exp(-r * T)
    if option_type == "call":
        lower = max(S - disc, 0.0)
        upper = S
    else:
        lower = max(disc - S, 0.0)
        upper = disc

    if market_price < lower - 1e-4:
        raise ValueError(
            f"Market price {market_price:.4f} below intrinsic value {lower:.4f}. "
            f"Arbitrage opportunity — check inputs."
        )
    if market_price > upper + 1e-4:
        raise ValueError(
            f"Market price {market_price:.4f} above no-arbitrage upper bound {upper:.4f}."
        )

    # Initial guess
    sigma = initial_guess or _brenner_subrahmanyam_guess(market_price, S, K, T, option_type)
    sigma = max(0.001, min(sigma, 5.0))

    price_fn = bs_call if option_type == "call" else bs_put

    for i in range(max_iter):
        model_price = price_fn(S, K, T, r, sigma)
        v = bs_vega(S, K, T, r, sigma) * 100   # vega() returns per-1%, restore to per-unit

        if abs(v) < 1e-10:
            # Vega is essentially zero (deep ITM/OTM or very close to expiry)
            # Fall back to bisection for this step
            if model_price < market_price:
                sigma *= 1.1
            else:
                sigma *= 0.9
            continue

        new_sigma = sigma - (model_price - market_price) / v
        new_sigma = max(0.001, min(new_sigma, 20.0))   # safety clamp

        if abs(new_sigma - sigma) < tolerance:
            return new_sigma

        sigma = new_sigma

    raise RuntimeError(
        f"IV solver did not converge after {max_iter} iterations. "
        f"Final sigma={sigma:.6f}, price diff={abs(price_fn(S,K,T,r,sigma) - market_price):.6f}"
    )


def implied_volatility_safe(
    market_price : float,
    S            : float,
    K            : float,
    T            : float,
    r            : float = 0.05,
    option_type  : str   = "call",
) -> float:
    """
    Safe wrapper — returns float('nan') instead of raising on failure.
    Useful for batch computation (vol surface building).
    """
    try:
        return implied_volatility(market_price, S, K, T, r, option_type)
    except Exception:
        return float("nan")


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("="*55)
    print("  IMPLIED VOLATILITY SOLVER — Newton-Raphson")
    print("="*55 + "\n")

    # Round-trip test: price an option, then recover IV from that price
    S, K, T, r, sigma = 100.0, 105.0, 0.5, 0.05, 0.22

    call_price = bs_call(S, K, T, r, sigma)
    put_price  = bs_put (S, K, T, r, sigma)

    iv_call = implied_volatility(call_price, S, K, T, r, "call")
    iv_put  = implied_volatility(put_price,  S, K, T, r, "put")

    print(f"  Round-trip test:")
    print(f"    True sigma     : {sigma:.6f} ({sigma*100:.2f}%)")
    print(f"    Call price     : {call_price:.6f}")
    print(f"    Recovered IV   : {iv_call:.6f}  (error: {abs(iv_call - sigma):.2e})")
    print(f"    Put price      : {put_price:.6f}")
    print(f"    Recovered IV   : {iv_put:.6f}  (error: {abs(iv_put - sigma):.2e})")

    # Test across a range of moneyness
    print(f"\n  IV across moneyness (S=100, T=0.5, r=5%, sigma=20%):")
    print(f"  {'Strike':>8}  {'Moneyness':>10}  {'Call Price':>12}  {'Recovered IV':>14}  {'Error':>10}")
    print("  " + "-"*60)
    for K_test in [80, 90, 95, 100, 105, 110, 120]:
        c_price = bs_call(100, K_test, 0.5, 0.05, 0.20)
        try:
            iv_rec = implied_volatility(c_price, 100, K_test, 0.5, 0.05, "call")
            err    = abs(iv_rec - 0.20)
            mness  = "ATM" if abs(K_test - 100) < 2 else ("ITM" if K_test < 100 else "OTM")
            print(f"  {K_test:>8}  {mness:>10}  {c_price:>12.6f}  {iv_rec:>14.6f}  {err:>10.2e}")
        except Exception as e:
            print(f"  {K_test:>8}  {'?':>10}  {c_price:>12.6f}  {'FAILED':>14}  {str(e)[:20]}")

    print("\nIV Solver complete.")
