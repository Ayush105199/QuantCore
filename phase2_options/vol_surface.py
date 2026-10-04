"""
Volatility Surface Generator
------------------------------
Generates a realistic implied volatility surface: a grid of IVs
indexed by (strike, expiry).

In reality, every option on the same underlying has a different IV.
This violates the BS assumption of constant vol — but the market
doesn't care about BS assumptions.

Why the vol surface is not flat:
  1. Volatility Smile: OTM options have higher IV than ATM
     → Market prices tail risk (fat tails, crash risk)
  2. Negative Skew: OTM puts have higher IV than OTM calls
     → Investors pay premium for downside protection (crash insurance)
  3. Term Structure: longer expiry usually has higher IV (uncertainty grows)
     → But can invert during stress events (short-term vol spikes)

This surface is used by:
  - Options desks to mark their book consistently
  - Arbitrageurs to find mispriced options
  - Risk managers for scenario analysis
  - Exotic option pricers (local vol, stochastic vol models)
"""

import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from phase2_options.iv_solver import implied_volatility_safe
from phase2_options.black_scholes import bs_call


def generate_vol_surface(
    S           : float       = 100.0,
    r           : float       = 0.05,
    strikes     : np.ndarray  = None,
    expiries    : np.ndarray  = None,
    atm_vol     : float       = 0.20,
    skew_factor : float       = 0.15,
    smile_factor: float       = 0.12,
    term_factor : float       = 0.04,
    seed        : int         = 42,
) -> pd.DataFrame:
    """
    Generate a realistic IV surface using a parametric model.

    IV(K, T) = atm_vol
               + smile_factor * ((K/S - 1)^2)       [symmetric smile]
               - skew_factor  * (K/S - 1)            [negative skew]
               + term_factor  * sqrt(T)               [upward term structure]
               + noise (small random perturbation)

    Args:
        S           : current spot price
        r           : risk-free rate
        strikes     : array of strike prices
        expiries    : array of expiries in years
        atm_vol     : base ATM volatility (decimal)
        skew_factor : size of negative skew (higher = more skewed)
        smile_factor: curvature of the smile (higher = wider smile)
        term_factor : slope of term structure
        seed        : for reproducibility

    Returns:
        DataFrame with shape (n_strikes, n_expiries)
        Index = strikes, Columns = expiries
        Values = IV (decimal, e.g. 0.25 = 25%)
    """
    rng = np.random.default_rng(seed)

    if strikes is None:
        strikes = np.linspace(S * 0.70, S * 1.30, 25)
    if expiries is None:
        expiries = np.array([1/12, 2/12, 3/12, 6/12, 9/12, 1.0, 1.5, 2.0])

    iv_grid = np.zeros((len(strikes), len(expiries)))

    for j, T in enumerate(expiries):
        for i, K in enumerate(strikes):
            moneyness = K / S - 1.0   # 0 = ATM, negative = ITM call, positive = OTM call

            smile     = smile_factor * (moneyness ** 2)
            skew      = -skew_factor * moneyness
            term      = term_factor  * np.sqrt(T)
            noise     = rng.normal(0, 0.002)   # tiny realistic imperfection

            iv = atm_vol + smile + skew + term + noise
            iv = max(0.01, iv)   # IV must be positive
            iv_grid[i, j] = iv

    df = pd.DataFrame(
        iv_grid,
        index   = np.round(strikes, 4),
        columns = np.round(expiries, 4),
    )
    df.index.name   = "strike"
    df.columns.name = "expiry_yr"
    return df


def surface_to_prices(surface_df: pd.DataFrame, S: float,
                       r: float = 0.05) -> pd.DataFrame:
    """
    Convert IV surface to option price surface using Black-Scholes.
    Useful for generating a realistic option chain.
    """
    prices = pd.DataFrame(index=surface_df.index, columns=surface_df.columns)
    for expiry in surface_df.columns:
        for strike in surface_df.index:
            iv    = surface_df.loc[strike, expiry]
            price = bs_call(S, float(strike), float(expiry), r, float(iv))
            prices.loc[strike, expiry] = round(price, 4)
    return prices.astype(float)


def smile_at_expiry(surface_df: pd.DataFrame, expiry_idx: int = 2) -> pd.Series:
    """Extract a single expiry slice (the classic vol smile plot)."""
    col = surface_df.columns[expiry_idx]
    return surface_df[col].rename(f"IV at T={col:.3f}yr")


def term_structure_at_strike(surface_df: pd.DataFrame,
                              strike_idx: int = None) -> pd.Series:
    """Extract the vol term structure at the ATM strike."""
    if strike_idx is None:
        # Find ATM-ish row
        strikes = surface_df.index.values
        mid_idx = len(strikes) // 2
    else:
        mid_idx = strike_idx
    row = surface_df.iloc[mid_idx]
    return row.rename(f"IV term structure at K={surface_df.index[mid_idx]:.2f}")


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("="*55)
    print("  VOLATILITY SURFACE GENERATOR")
    print("="*55 + "\n")

    surface = generate_vol_surface(S=100.0, atm_vol=0.20,
                                    skew_factor=0.15, smile_factor=0.12)

    print(f"  Surface shape : {surface.shape} (strikes x expiries)")
    print(f"  Strike range  : {surface.index.min():.1f} to {surface.index.max():.1f}")
    print(f"  Expiry range  : {surface.columns.min():.3f} to {surface.columns.max():.3f} yr")
    print(f"  IV range      : {surface.values.min()*100:.1f}% to {surface.values.max()*100:.1f}%")
    print(f"  ATM IV (1yr)  : {surface.loc[100.0, 1.0]*100 if 100.0 in surface.index else surface.iloc[len(surface)//2, -1]*100:.2f}%")

    print("\n  Vol smile at 6-month expiry (IV % by strike):")
    smile = smile_at_expiry(surface, expiry_idx=3)
    print("  " + "-"*40)
    for strike, iv in smile.items():
        bar = "#" * int(iv * 100)
        mness = "OTM" if float(strike) > 100 else ("ITM" if float(strike) < 100 else "ATM")
        print(f"  K={float(strike):6.1f}  [{mness}]  IV={iv*100:5.1f}%  {bar}")

    print("\n  Term structure at ATM strike:")
    ts = term_structure_at_strike(surface)
    for T, iv in ts.items():
        print(f"  T={float(T):.3f}yr  IV={iv*100:.2f}%")

    print("\nVol surface complete.")
